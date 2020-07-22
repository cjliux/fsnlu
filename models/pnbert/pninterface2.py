#coding: utf-8
"""
    @author: cjliux@gmail.com
"""
import os, sys
import tqdm
import copy
import json 
import torch
import torch.nn as nn
import numpy as np
import pickle
import logging
from collections import defaultdict

from .model_3 import Model
from .data import (get_dataloader, get_dataloader_for_fs_train, 
    get_dataloader_for_fs_eval, get_dataloader_for_fs_test)
from .tokenization import BertTokenizer
from .optimization import BertAdam, WarmupLinearSchedule
from utils.nertools import collect_named_entities
from utils.vocab import Vocab
from utils.scaffold import get_output_dir, init_logger
from utils.conll2002_metrics import conll2002_measure as conll_eval

logger = logging.getLogger()


from .interface import evaluate


def pn_evaluate(model, eval_loader, verbose=True):
    # dom_preds, dom_golds = [], []
    int_preds, int_golds = [], []
    lbl_preds, lbl_golds = [], []

    model.eval()
    pbar = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader))
    for i, batch in pbar:
        qry_input = batch["model_input"]
        padded_seqs = qry_input["padded_seqs"].cuda()
        seq_lengths = qry_input["seq_lengths"].cuda()
        dom_idx, int_idx = qry_input["dom_idx"].cuda(), qry_input["int_idx"].cuda()
        padded_y = qry_input["padded_y"].cuda()
        padded_bin_y = qry_input["padded_bin_y"].cuda()
        segids = qry_input["segids"].cuda()

        # sup
        batch_size = padded_seqs.size(0)

        for i_sam in range(batch_size):
            sup_batch = batch["support"][i_sam]
            sup_input = sup_batch["model_input"]

            sup_seqs = sup_input["padded_seqs"].cuda()
            sup_slens = sup_input["seq_lengths"].cuda()
            sup_idom = sup_input["dom_idx"].cuda()
            sup_iint = sup_input["int_idx"].cuda()
            sup_y = sup_input["padded_y"].cuda()
            sup_biny = sup_input["padded_bin_y"].cuda()
            sup_seg = sup_input["segids"].cuda()

            sam_seqs = padded_seqs[i_sam:i_sam+1]
            sam_slens = seq_lengths[i_sam:i_sam+1]
            sam_idom = dom_idx[i_sam:i_sam+1]
            sam_iint = int_idx[i_sam:i_sam+1]
            sam_y = padded_y[i_sam:i_sam+1]
            sam_biny = padded_bin_y[i_sam:i_sam+1]
            sam_seg = segids[i_sam:i_sam+1]

            with torch.no_grad():
                proto_dict = model.get_proto_dict(
                    sup_seqs, sup_slens, sup_seg, sup_idom, sup_iint, sup_y, sup_biny)
                sam_fwd = model.encode_with_proto(sam_seqs, sam_slens, sam_seg, proto_dict)
                # sam_loss = model.compute_loss(sam_idom, sam_iint, sup_y, sam_fwd)

                dom_pred, int_pred, lbl_pred = model.predict_postr(sam_slens, sam_idom, sam_fwd)
            
                # dom_preds.extend(dom_pred); dom_golds.extend(batch["dom_idx"])
                int_preds.extend(int_pred); int_golds.extend(batch["int_idx"])
                lbl_preds.extend(lbl_pred); lbl_golds.extend(batch["label_ids"])

    ## compute scores
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


def save_model(model, save_path):
    ckpt = {
        "model": model.state_dict(),
    }
    torch.save(ckpt, save_path)


def do_pn_train(args):
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
    train_loader, train_suploader = get_dataloader_for_fs_train(
                    args.data_path, args.raw_data_path,
                    args.evl_dm.split(','), args.batch_size, 
                    args.max_sup_ratio, args.max_sup_size, args.n_shots, 
                    tokenizer, return_suploader=True)
    eval_loader, eval_suploader = get_dataloader_for_fs_eval(
                    args.data_path, args.raw_data_path,
                    args.evl_dm.split(','), args.batch_size, 
                    args.max_sup_ratio, args.max_sup_size, args.n_shots, 
                    tokenizer, return_suploader=True)
    # test_loaders, test_suploaders = get_dataloader_for_fs_test(
    #     args.data_path, args.raw_data_path, args.batch_size, 
    #     args.n_shots, tokenizer, sep_dom=True, return_suploader=True)

    ## def optim
    num_train_optim_steps = len(train_loader) * args.max_epoch #// args.grad_acc_steps
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

    # global_step = 0
    best_score, patience = 0, 0
    stop_training_flag = False
    for epo in range(args.max_epoch):
        model.train()

        sup_loss_list = []

        comb_sup_loader = get_dataloader(
            train_suploader.dataset.merge(eval_suploader.dataset), 
            args.batch_size, True)

        score = evaluate(model, comb_sup_loader)

        pbar = tqdm.tqdm(enumerate(comb_sup_loader), total=len(comb_sup_loader))
        for step, supbatch in pbar:
            sup_input = supbatch["model_input"]

            sup_seqs = sup_input["padded_seqs"].cuda()
            sup_slens = sup_input["seq_lengths"].cuda()
            sup_idom = sup_input["dom_idx"].cuda()
            sup_iint = sup_input["int_idx"].cuda()
            sup_y = sup_input["padded_y"].cuda()
            sup_biny = sup_input["padded_bin_y"].cuda()
            sup_seg = sup_input["segids"].cuda()
            
            proto_dict = model.get_proto_dict(
                    sup_seqs, sup_slens, sup_seg, sup_idom, sup_iint, sup_y, sup_biny)
            sup_fwd = model.encode_without_proto(sup_seqs, sup_slens, sup_seg)
            sup_loss = model.compute_loss(sup_idom, sup_iint, sup_y, sup_fwd)

            optimizer.zero_grad()
            sup_loss.backward()
            optimizer.step()

            sup_loss_list.append(sup_loss.item())
            pbar.set_description("(Epo {}) sup_loss:{:.4f}".format(epo+1, np.mean(sup_loss_list)))

        score = evaluate(model, comb_sup_loader)

        model.train()

        qry_loss_list = []
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for step, batch in pbar:
            qry_input = batch["model_input"]
            padded_seqs = qry_input["padded_seqs"].cuda()
            seq_lengths = qry_input["seq_lengths"].cuda() 
            dom_idx, int_idx = qry_input["dom_idx"].cuda(), qry_input["int_idx"].cuda()
            padded_y = qry_input["padded_y"].cuda()
            padded_bin_y = qry_input["padded_bin_y"].cuda()
            segids = qry_input["segids"].cuda()

            # sup
            qry_loss = 0
            batch_size = padded_seqs.size(0)
            
            for i_sam in range(batch_size):
                sup_batch = batch["support"][i_sam]
                sup_input = sup_batch["model_input"]
                
                sup_seqs = sup_input["padded_seqs"].cuda()
                sup_slens = sup_input["seq_lengths"].cuda()
                sup_idom = sup_input["dom_idx"].cuda()
                sup_iint = sup_input["int_idx"].cuda()
                sup_y = sup_input["padded_y"].cuda()
                sup_biny = sup_input["padded_bin_y"].cuda()
                sup_seg = sup_input["segids"].cuda()
                
                # qry
                sam_seqs = padded_seqs[i_sam:i_sam+1]
                sam_slens = seq_lengths[i_sam:i_sam+1]
                sam_idom = dom_idx[i_sam:i_sam+1]
                sam_iint = int_idx[i_sam:i_sam+1]
                sam_y = padded_y[i_sam:i_sam+1]
                sam_biny = padded_bin_y[i_sam:i_sam+1]
                sam_seg = segids[i_sam:i_sam+1]

                proto_dict = model.get_proto_dict(
                    sup_seqs, sup_slens, sup_seg, sup_idom, sup_iint, sup_y, sup_biny)
                sam_fwd = model.encode_with_proto(sam_seqs, sam_slens, sam_seg, proto_dict)
                sam_loss = model.compute_postr_loss(sam_idom, sam_iint, sam_y, sam_fwd)

                qry_loss += sam_loss
            # qry_loss /= batch_size
            
            optimizer.zero_grad()
            qry_loss.backward()
            optimizer.step()

            qry_loss_list.append(qry_loss.item())
            pbar.set_description("(Epo {}) qry_loss:{:.4f}".format(epo+1, np.mean(qry_loss_list)))

        # epo eval
        logger.info("============== Evaluate Epoch {} ==============".format(epo+1))
        score = pn_evaluate(model, eval_loader)
        if score > best_score:
            best_score, patience = score, 0
            logger.info("Found better model!!")
            save_path = os.path.join(args.dump_path, "best_model.pth")
            save_model(model, save_path)
            logger.info("Best model has been saved to %s" % save_path)
        else:
            patience += 1
            logger.info("No better model found (%d/%d)" % (patience, args.early_stop))

        if args.early_stop > 0 and patience >= args.early_stop:
            stop_training_flag = True
            break

    logger.info("Done!")


def do_pn_predict(args):
    args.dump_path = get_output_dir(args.dump_path, args.exp_name, args.exp_id)
    model_path = os.path.join(args.dump_path, args.target)
    assert os.path.isfile(model_path)
    save_dir = args.dump_path

    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(args.bert_dir, "vocab.txt"),
        do_lower_case=args.do_lower_case)
    model = Model(args, tokenizer)
    model.cuda()

    test_loaders, test_suploaders = get_dataloader_for_fs_test(
        args.data_path, args.raw_data_path, args.batch_size, 
        args.n_shots, tokenizer, sep_dom=True, return_suploader=True)

    for dom in test_loaders.keys():
        test_loader, test_suploader = test_loaders[dom], test_suploader[dom]

        final_items = []
        int_preds, int_golds = [], []
        lbl_preds, lbl_golds = [], []

        pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
        for i, batch in pbar:
            qry_input = batch["model_input"]
            padded_seqs = qry_input["padded_seqs"].cuda()
            seq_lengths = qry_input["seq_lengths"].cuda()
            dom_idx, int_idx = qry_input["dom_idx"].cuda(), qry_input["int_idx"].cuda()
            padded_y = qry_input["padded_y"].cuda()
            padded_bin_y = qry_input["padded_bin_y"].cuda()
            segids = qry_input["segids"].cuda()

            batch_size = padded_seqs.size(0)

            for i_sam in range(batch_size):
                model.train()
                sup_batch = batch["support"][i_sam]
                sup_input = sup_batch["model_input"]

                sup_seqs = sup_input["padded_seqs"].cuda()
                sup_slens = sup_input["seq_lengths"].cuda()
                sup_idom = sup_input["dom_idx"].cuda()
                sup_iint = sup_input["int_idx"].cuda()
                sup_y = sup_input["padded_y"].cuda()
                sup_biny = sup_input["padded_bin_y"].cuda()
                sup_seg = sup_input["segids"].cuda()

                sam_seqs = padded_seqs[i_sam:i_sam+1]
                sam_slens = seq_lengths[i_sam:i_sam+1]
                sam_idom = dom_idx[i_sam:i_sam+1]
                sam_iint = int_idx[i_sam:i_sam+1]
                sam_y = padded_y[i_sam:i_sam+1]
                sam_biny = padded_bin_y[i_sam:i_sam+1]
                sam_seg = segids[i_sam:i_sam+1]

                with torch.no_grad():
                    proto_dict = model.get_proto_dict(
                        sup_seqs, sup_slens, sup_seg, sup_idom, sup_iint, sup_y, sup_biny)
                    sam_fwd = model.encode_with_proto(sam_seqs, sam_slens, sam_seg, proto_dict)
                    # sam_loss = model.compute_loss(sam_idom, sam_iint, sup_y, sam_fwd)

                    dom_pred, int_pred, lbl_pred = model.predict_postr(sam_slens, sam_idom, sam_fwd)

                    int_preds.extend(int_pred); int_golds.extend(batch["int_idx"])
                    lbl_preds.extend(lbl_pred); lbl_golds.extend(batch["label_ids"])
                
                    tokens = batch["token"][i_sam]
                    labels = [model.label_vocab.index2word[l] for l in lbl_pred[0]]
                    ents = collect_named_entities(labels)

                    slvals = {}
                    for etype, start, end in ents:
                        val = ''.join(tokens[start:end+1]).replace('#', '')
                        if etype not in slvals:
                            slvals[etype] = val
                        elif isinstance(slvals[etype], str):
                            slvals[etype] = [slvals[etype], val]
                        else:
                            slvals[etype].append(val)

                    item = {}
                    item["id"] = batch["id"][i_sam]
                    item["domain"] = batch["domain"][i_sam]
                    item["text"] = batch["text"][i_sam]
                    item["intent"] = model.intent_map.index2word[int_pred[0]]
                    item["slots"] = slvals
                    final_items.append(item)
        
        # save file
        with open(os.path.join(save_dir, "predict_{}.json".format(dom)),
                                                    'w', encoding='utf8') as fd:
            json.dump(final_items, fd, ensure_ascii=False, indent=2)

