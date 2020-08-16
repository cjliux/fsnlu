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

from .model import Model
from .data import (get_dataloader, get_dataloader_for_fs_train, 
    get_dataloader_for_fs_eval, get_dataloader_for_fs_test)
from .tokenization import BertTokenizer
from .optimization import BertAdam, WarmupLinearSchedule
from utils.nertools import collect_named_entities
from utils.vocab import Vocab
from utils.scaffold import get_output_dir, init_logger
from utils.conll2002_metrics import conll2002_measure as conll_eval

logger = logging.getLogger()


def maml_evaluate(model, optimizer, eval_loader, verbose=True):
    # dom_preds, dom_golds = [], []
    int_preds, int_golds = [], []
    lbl_preds, lbl_golds = [], []

    pbar = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader))
    for i, batch in pbar:
        qry_input = batch["model_input"]
        padded_seqs = qry_input["padded_seqs"].cuda()
        seq_lengths = qry_input["seq_lengths"].cuda()
        dom_idx, int_idx = qry_input["dom_idx"].cuda(), qry_input["int_idx"].cuda()
        padded_y = qry_input["padded_y"].cuda()

        # sup
        batch_size = padded_seqs.size(0)
        model_origin = copy.deepcopy(model.state_dict())
        optim_origin = copy.deepcopy(optimizer.state_dict())

        for i_sam in range(batch_size):
            model.train()
            sup_batch = batch["support"][i_sam]
            sup_input = sup_batch["model_input"]

            sup_seqs = sup_input["padded_seqs"].cuda()
            sup_slens = sup_input["seq_lengths"].cuda()
            sup_idom = sup_input["dom_idx"].cuda()
            sup_iint = sup_input["int_idx"].cuda()
            sup_y = sup_input["padded_y"].cuda()

            sup_fwd = model(sup_seqs, sup_slens)
            sup_loss = model.compute_loss(sup_idom, sup_iint, sup_y, sup_fwd)

            optimizer.zero_grad()
            sup_loss.backward()
            optimizer.step()

            # qry
            model.eval()
            with torch.no_grad():
                sam_seqs = padded_seqs[i_sam:i_sam+1]
                sam_slens = seq_lengths[i_sam:i_sam+1]
                sam_idom = dom_idx[i_sam:i_sam+1]
                sam_iint = int_idx[i_sam:i_sam+1]
                sam_y = padded_y[i_sam:i_sam+1]

                sam_fwd = model(sam_seqs, sam_slens)
                dom_pred, int_pred, lbl_pred = model.predict(sam_slens, sam_fwd)
            
                # dom_preds.extend(dom_pred); dom_golds.extend(batch["dom_idx"])
                int_preds.extend(int_pred); int_golds.extend(batch["int_idx"])
                lbl_preds.extend(lbl_pred); lbl_golds.extend(batch["label_ids"])

            model.load_state_dict(model_origin)
            optimizer.load_state_dict(optim_origin)

        del model_origin
        del optim_origin

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


def save_model(model, optimizer, save_path):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, save_path)


def do_maml_train(args):
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
    train_loader = get_dataloader_for_fs_train(
                    args.data_path, args.raw_data_path,
                    args.evl_dm.split(','), args.batch_size, 
                    args.max_sup_ratio, args.max_sup_size args.n_shots, 
                    tokenizer)
    eval_loader = get_dataloader_for_fs_eval(
                    args.data_path, args.raw_data_path,
                    args.evl_dm.split(','), args.batch_size, 
                    args.max_sup_ratio, args.max_sup_size, args.n_shots, 
                    tokenizer)

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
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        qry_loss_list = []
        
        model.train()
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for step, batch in pbar:
            qry_input = batch["model_input"]
            padded_seqs = qry_input["padded_seqs"].cuda()
            seq_lengths = qry_input["seq_lengths"].cuda() 
            dom_idx, int_idx = qry_input["dom_idx"].cuda(), qry_input["int_idx"].cuda()
            padded_y = qry_input["padded_y"].cuda()

            # sup
            meta_loss = 0
            batch_size = padded_seqs.size(0)
            model_origin = copy.deepcopy(model.state_dict())
            optim_origin = copy.deepcopy(optimizer.state_dict())

            for i_sam in range(batch_size):
                sup_batch = batch["support"][i_sam]
                sup_input = sup_batch["model_input"]
                
                sup_seqs = sup_input["padded_seqs"].cuda()
                sup_slens = sup_input["seq_lengths"].cuda()
                sup_idom = sup_input["dom_idx"].cuda()
                sup_iint = sup_input["int_idx"].cuda()
                sup_y = sup_input["padded_y"].cuda()
                
                sup_fwd = model(sup_seqs, sup_slens)
                sup_loss = model.compute_loss(sup_idom, sup_iint, sup_y, sup_fwd)

                optimizer.zero_grad()
                sup_loss.backward(retain_graph=True)
                optimizer.step()

                # qry
                sam_seqs = padded_seqs[i_sam:i_sam+1]
                sam_slens = seq_lengths[i_sam:i_sam+1]
                sam_idom = dom_idx[i_sam:i_sam+1]
                sam_iint = int_idx[i_sam:i_sam+1]
                sam_y = padded_y[i_sam:i_sam+1]

                sam_fwd = model(sam_seqs, sam_slens)
                qry_loss = model.compute_loss(sam_idom, sam_iint, sam_y, sam_fwd)

                meta_loss += qry_loss - 0.5 * sup_loss

                model.load_state_dict(model_origin)
                optimizer.load_state_dict(optim_origin)

            meta_loss /= batch_size
            optimizer.zero_grad()
            meta_loss.backward()
            optimizer.step()

            del model_origin
            del optim_origin

            tr_loss += qry_loss.item()
            nb_tr_examples += len(batch["token"])
            nb_tr_steps += 1  

            qry_loss_list.append(qry_loss.item())
            pbar.set_description("(Epo {}) qry_loss:{:.4f}".format(epo+1, np.mean(qry_loss_list)))
            
            # break

        # epo eval
        logger.info("============== Evaluate Epoch {} ==============".format(epo+1))
        score = maml_evaluate(model, optimizer, eval_loader)
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


def evaluate(model, eval_loader, verbose=True):
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

        # qry
        with torch.no_grad():
            fwd = model(padded_seqs, seq_lengths)
            dom_pred, int_pred, lbl_pred = model.predict(seq_lengths, fwd)
        
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


def ewc_finetune(args, model, optimizer, 
        fisher_loader, train_loader, eval_loader):
    ## fisher    
    fisher, optpar = {}, {}
    for n, p in model.named_parameters():
        optpar[n] = torch.Tensor(p.cpu().data).cuda()
        p.data.zero_()
        fisher[n] = torch.Tensor(p.cpu().data).cuda()

    model.train()
    pbar = tqdm.tqdm(enumerate(fisher_loader), total=len(fisher_loader))
    for i, batch_o in pbar:
        sup_input = batch_o["model_input"]
        sup_seqs = sup_input["padded_seqs"].cuda()
        sup_slens = sup_input["seq_lengths"].cuda()
        sup_idom = sup_input["dom_idx"].cuda()
        sup_iint = sup_input["int_idx"].cuda()
        sup_y = sup_input["padded_y"].cuda()
        
        sup_fwd = model(sup_seqs, sup_slens)
        sup_loss = model.compute_loss(sup_idom, sup_iint, sup_y, sup_fwd)
        sup_loss.backward()        
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n].data += p.grad.data ** 2

    for n, _ in fisher.items():
        fisher[n] /= len(fisher_loader)

    ## fine tune
    sup_loss_list = []
    best_score, patience = 0, 0
    stop_training_flag = False
    weights_best = copy.deepcopy(model.state_dict())
    for epo in range(args.max_epoch):
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in pbar:
            sup_input = batch["model_input"]
            sup_seqs = sup_input["padded_seqs"].cuda()
            sup_slens = sup_input["seq_lengths"].cuda()
            sup_idom = sup_input["dom_idx"].cuda()
            sup_iint = sup_input["int_idx"].cuda()
            sup_y = sup_input["padded_y"].cuda()    

            sup_fwd = model(sup_seqs, sup_slens)
            sup_loss = model.compute_loss(sup_idom, sup_iint, sup_y, sup_fwd)

            for n, p in model.named_parameters():
                if p.grad is not None:
                    l = args.lambda_ewc * fisher[n].cuda() * (p - optpar[n].cuda()).pow(2)
                    sup_loss += l.sum()

            optimizer.zero_grad()
            sup_loss.backward()
            optimizer.step()

            sup_loss_list.append(sup_loss.item())
            pbar.set_description("(Epo {}) sup_loss: {:.4f}".format(
                epo, np.mean(sup_loss_list)))
        # epo eval
        logger.info("============== Evaluate Epoch {} ==============".format(epo+1))
        score = evaluate(model, eval_loader)
        if score > best_score:
            best_score, patience = score, 0
            logger.info("Found better model!!")
            # save_path = os.path.join(args.dump_path, "best_ewc_model.pth")
            # save_model(model, optimizer, save_path)
            # logger.info("Best model has been saved to %s" % save_path)
            weights_best = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            logger.info("No better model found (%d/%d)" % (patience, args.early_stop))

        if patience >= args.early_stop:
            stop_training_flag = True
            break

    return weights_best


def do_ewc_finetune(args):
    args.dump_path = get_output_dir(args.dump_path, args.exp_name, args.exp_id)
    os.makedirs(args.dump_path, exist_ok=True)
    with open(os.path.join(args.dump_path, "ewc_args.pkl"), "wb") as fd:
        pickle.dump(args, fd)
    init_logger(os.path.join(args.dump_path, args.log_file))

    ## def model
    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(args.bert_dir, "vocab.txt"),
        do_lower_case=args.do_lower_case)
    model = Model(args, tokenizer)
    model.cuda()

    ## def data
    eval_loader, eval_suploader = get_dataloader_for_fs_eval(
            args.data_path, args.raw_data_path,
            args.evl_dm.split(','), args.batch_size, 
            args.sup_ratio, args.n_shots, tokenizer, return_suploader=True)
    test_loaders, test_suploaders = get_dataloader_for_fs_test(
            args.data_path, args.raw_data_path, args.batch_size, 
            args.n_shots, tokenizer, sep_dom=True, return_suploader=True)

    for dom in test_loaders.keys():
        logger.info("[EWC] finetune model for test domain {}".format(dom))
        test_loader, test_suploader = test_loaders[dom], test_suploaders[dom]
        suploader = get_dataloader(
            eval_suploader.dataset.merge(test_suploader.dataset), 
            args.batch_size, True)

        ## def optim
        num_train_optim_steps = len(suploader) * args.max_epoch #// args.grad_acc_steps
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

        weights_best = ewc_finetune(model, optimizer, 
                            eval_suploader, suploader, eval_loader)
        save_path = os.path.join(args.dump_path, "best_ewc_model_dom_{}.pth".format(dom))
        save_model(model, optimizer, save_path)
        logger.info("Best model has been saved to %s" % save_path)


def do_predict(args):
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

            with torch.no_grad():
                qry_fwd = model(padded_seqs, seq_lengths)
                dom_pred, int_pred, lbl_pred = model.predict(seq_lengths, qry_fwd)

                int_preds.extend(int_pred); int_golds.extend(batch["int_idx"])
                lbl_preds.extend(lbl_pred); lbl_golds.extend(batch["label_ids"])

                for j in range(padded_seqs.size(0)):
                    tokens = batch["token"]
                    labels = [model.label_vocab.index2word[l] for l in lbl_pred[j]]
                    ents = collect_named_entities(labels)

                    slvals = {}
                    for etype, start, end in ents:
                        val = ''.join(tokens[j][start:end+1]).replace('#', '')
                        if etype not in slvals:
                            slvals[etype] = val
                        elif isinstance(slvals[etype], str):
                            slvals[etype] = [slvals[etype], val]
                        else:
                            slvals[etype].append(val)

                    item = {}
                    item["id"] = batch["id"][j]
                    item["domain"] = batch["domain"][j]
                    item["text"] = batch["text"][j]
                    item["intent"] = model.intent_map.index2word[int_pred[j]]
                    item["slots"] = slvals
                    final_items.append(item)
        
        # save file
        with open(os.path.join(save_dir, "predict_{}.json".format(dom)),
                                                    'w', encoding='utf8') as fd:
            json.dump(final_items, fd, ensure_ascii=False, indent=2)

