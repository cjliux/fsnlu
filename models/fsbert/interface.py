#coding: utf-8
"""
    @author: cjliux@gmail.com
"""
import os, sys
import tqdm
import copy
import torch
import torch.nn as nn
import numpy as np
import pickle
import logging
from collections import defaultdict

from .model import Model
from .data import get_dataloader_for_fs_train, get_dataloader_for_fs_eval
from .tokenization import BertTokenizer
from .optimization import BertAdam, WarmupLinearSchedule
from utils.vocab import Vocab
from utils.scaffold import get_output_dir, init_logger
from utils.conll2002_metrics import conll2002_measure as conll_eval

logger = logging.getLogger()


def collect_named_entities(self, labels):
    named_entities = []
    start_offset, end_offset, ent_type = None, None, None
        
    for offset, token_tag in enumerate(labels):
        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append((ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None
        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset
        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):
            end_offset = offset - 1
            named_entities.append((ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        named_entities.append((ent_type, start_offset, len(labels)-1))
    return named_entities


def evaluate(model, eval_loader, verbose=True):
    # dom_preds, dom_golds = [], []
    int_preds, int_golds = [], []
    lbl_preds, lbl_golds = [], []

    model.eval()
    pbar = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader))
    for i, batch in pbar:
        fwd_dict = model(batch)
        dom_pred, int_pred, lbl_pred = model.predict(batch, fwd_dict)    
        
        # dom_preds.extend(dom_pred); dom_golds.extend(batch["dom_idx"])
        int_preds.extend(int_pred); int_golds.extend(batch["int_idx"])
        lbl_preds.extend(lbl_pred); lbl_golds.extend(batch["label_ids"])

    scores = {}
    # int f1 score
    ma_icnt = defaultdict(lambda: defaultdict(int))
    for pint, gint in zip(int_preds, int_golds):
        pint = model.intent_map.index2word[pint]
        gint = model.intent_map.index2word[gint]
        if pint == gint:
            ma_icnt[pint]['tp'] += 1
        else:
            ma_icnt[pint]['fp'] += 1
            ma_icnt[gint]['fn'] += 1
    scores['ma_if1'] = sum(
        2 * float(ic['tp']) / float(2 * ic['tp'] + ic['fp'] + ic['fn']) 
        for ic in ma_icnt.values()) / len(ma_icnt)

    # lbl f1 score
    lbl_preds = np.concatenate(lbl_preds)
    lbl_golds = np.concatenate(lbl_golds)
    lines = [ "w " + model.label_vocab.index2word[plb] 
                + " " + model.label_vocab.index2word[glb] 
                    for plb, glb in zip(lbl_preds, lbl_golds)]
    scores['lbl_conll_f1'] = conll_eval(lines)['fb1']

    score = (2 * scores['ma_if1'] * scores['lbl_conll_f1']) / (
                    scores['ma_if1'] + scores['lbl_conll_f1'] + 1e-10)

    if verbose:
        buf = "[Eval] "
        for k, v in scores.items():
            buf += "{}: {:.6f}; ".format(k, v)
        logger.info(buf)

    return score


def save_model(model, optimizer, save_path):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    return ckpt


def train_model(args):
    args.dump_path = get_output_dir(args.dump_path, args.exp_name, args.exp_id)
    os.makedirs(args.dump_path, exist_ok=True)
    with open(os.path.join(args.dump_path, "args.pkl"), "wb") as fd:
        pickle.dump(args, fd)
    init_logger(os.path.join(args.dump_path, args.log_file))

    ## def model
    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(args.bert_dir, "vocab.txt"),
        do_lower_case=args.do_lower_case)
    model = Model(args, tokenizer)
    model.cuda()

    ## def data
    train_loader = get_dataloader_for_fs_train(args.data_path, args.raw_data_path,
        args.evl_dm.split(','), args.batch_size, args.n_shots, tokenizer)
    eval_loader = get_dataloader_for_fs_eval(args.data_path, args.raw_data_path,
        args.evl_dm.split(','), args.batch_size, args.n_shots, tokenizer)

    ## def optim
    num_train_optim_steps = len(train_loader) #// args.grad_acc_steps
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters, 
                         lr=args.lr,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optim_steps)

    global_step = 0
    best_score, patience = 0, 0
    stop_training_flag = False
    for epo in range(args.max_epoch):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        qry_loss_list = []

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for step, batch in pbar:
            # sup: omitted

            # qry
            fwd_dict = model(batch)
            qry_loss = model.compute_loss(batch, fwd_dict)

            optimizer.zero_grad()
            qry_loss.backward()
            optimizer.step()

            tr_loss += qry_loss.item()
            nb_tr_examples += len(batch["token"])
            nb_tr_steps += 1  

            qry_loss_list.append(qry_loss.item())
            pbar.set_description("(Epo {}) qry_loss:{:.4f}".format(epo+1, np.mean(qry_loss_list)))

        # epo eval
        logger.info("============== Evaluate Epoch {} ==============".format(epo+1))
        score = evaluate(model, eval_loader)
        if score > best_score:
            best_score, patience = score, 0
            logger.info("Found better model!!")
            save_path = os.path.join(args.dump_path, "best_model.pth")
            save_model(model, optimizer, save_path)
            logger.info("Best model has been saved to %s" % save_path)
        else:
            patience += 1
            logger.info("No better model found (%d/%d)" % (patience, args.early_stop))

        if patience >= args.early_stop:
            stop_training_flag = True
            break
    
    logger.info("Done!")


