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
# from utils.conll2002_metrics import conll2002_measure as conll_eval
from evaluation import cal_sentence_acc

logger = logging.getLogger()


def save_model(model, save_path):
    ckpt = {
        "model": model.state_dict(),
    }
    torch.save(ckpt, save_path)


def evaluate(args, model, eval_loader, verbose=True):
    final_items = []
    int_preds, int_golds = [], []
    lbl_preds, lbl_golds = [], []

    model.eval()
    if args.no_pbar:
        pbar = enumerate(eval_loader)
    else:
        pbar = tqdm.tqdm(enumerate(eval_loader), total=len(eval_loader))
    for i, batch in pbar:
        qry_input = batch["model_input"]
        for k, v in qry_input.items():
            qry_input[k] = v.cuda()
        batch["model_input"] = qry_input

        # qry
        with torch.no_grad():
            fwd = model(batch)
            dom_pred, int_pred, lbl_pred = model.predict(batch, fwd)
        
            int_preds.extend(int_pred); int_golds.extend(batch["int_idx"])
            lbl_preds.extend(lbl_pred); lbl_golds.extend(batch["label_ids"])

            for i_sam in range(len(int_pred)):
                tokens = batch["token"][i_sam]
                labels = [model.label_vocab.index2word[l] for l in lbl_pred[i_sam]]
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
                item["intent"] = model.intent_map.index2word[int_pred[i_sam]]
                item["slots"] = slvals
                final_items.append(item)

    (sent_acc, macro_intent_acc, micro_intent_acc, 
        macro_f1, micro_f1) = cal_sentence_acc(
            eval_loader.dataset.qry_data, final_items)

    if verbose:
        buf = "[Eval] "
        buf += "se_acc {:.6f}; ma_int {:.6f} | mi_int {:.6f}; ma_sl {:.6f} | mi_sl {:.6f}".format(
            sent_acc, macro_intent_acc, micro_intent_acc, macro_f1, micro_f1)
        logger.info(buf)

    return sent_acc 


def normal_train(args, model, optimizer, 
        train_loader, eval_loader, save_path):
    do_eval = (eval_loader != None)

    sup_loss_list = []
    best_score, patience = 0, 0
    stop_training_flag = False
    for epo in range(args.max_epoch):
        if args.no_pbar:
            pbar = enumerate(train_loader)
            period = int(len(train_loader) * 0.1)
        else:
            pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in pbar:
            sup_input = batch["model_input"]
            for k, v in sup_input.items():
                sup_input[k] = v.cuda()
            batch["model_input"] = sup_input

            sup_fwd = model(batch)
            sup_loss = model.compute_loss(batch, sup_fwd)

            optimizer.zero_grad()
            sup_loss.backward()
            optimizer.step()

            sup_loss_list.append(sup_loss.item())

            msg = "(Epo {}) sup_loss: {:.4f}".format(epo, np.mean(sup_loss_list))
            if not args.no_pbar:
                pbar.set_description(msg)
            elif (i+1) % period == 0:
                logger.info(msg + " [{}/{}({:.2f}%)]".format(
                    i+1, len(train_loader), 100 * float(i+1) /len(train_loader)))

        # epo eval
        if do_eval:
            logger.info("============== Evaluate Epoch {} ==============".format(epo+1))
            score = evaluate(args, model, eval_loader)
            if score > best_score:
                best_score, patience = score, 0
                logger.info("Found better model!!")
                # save_path = os.path.join(args.dump_path, "best_model.pth")
                save_model(model, save_path)
                logger.info("Best model has been saved to %s" % save_path)
            else:
                patience += 1
                logger.info("No better model found (%d/%d)" % (patience, args.early_stop))

            if args.early_stop > 0 and patience >= args.early_stop:
                stop_training_flag = True
                break
        else:
            logger.info("Saving model...")
            save_model(model, save_path)
            logger.info("Saved.")


def do_comb_train(args):
    args.dump_path = get_output_dir(args.dump_path, args.exp_name, args.exp_id)
    os.makedirs(args.dump_path, exist_ok=True)
    with open(os.path.join(args.dump_path, "args.pkl"), "wb") as fd:
        pickle.dump(args, fd)
    init_logger(os.path.join(args.dump_path, args.log_file))

    ## def model
    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(args.bert_dir, "vocab.txt"),
        do_lower_case=args.do_lower_case)
    
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
    test_loaders, test_suploaders = get_dataloader_for_fs_test(
        args.data_path, args.raw_data_path, args.batch_size, 
        args.n_shots, tokenizer, sep_dom=True, return_suploader=True)

    for dom in test_loaders.keys():
        logger.info("[Train] train model for test domain {}".format(dom))
        test_loader, test_suploader = test_loaders[dom], test_suploaders[dom]
        
        comb_train_loader = get_dataloader(
            train_loader.dataset.merge(
                train_suploader.dataset).merge(
                    eval_suploader.dataset).merge(
                        test_suploader.dataset), 
            args.batch_size, True)
        
        model = Model(args, tokenizer)
        model.cuda()

        ## def optim
        num_train_optim_steps = len(comb_train_loader) * args.max_epoch #// args.grad_acc_steps
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


        normal_train(args, model, optimizer, 
            comb_train_loader, eval_loader,
            os.path.join(args.dump_path, "best_model_{}.pth".format(dom)))


def do_comb_train_balanced_eval(args):
    args.dump_path = get_output_dir(args.dump_path, args.exp_name, args.exp_id)
    os.makedirs(args.dump_path, exist_ok=True)
    with open(os.path.join(args.dump_path, "args.pkl"), "wb") as fd:
        pickle.dump(args, fd)
    init_logger(os.path.join(args.dump_path, args.log_file))

    ## def model
    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(args.bert_dir, "vocab.txt"),
        do_lower_case=args.do_lower_case)
    
    ## def data
    assert len(args.evl_dm.strip()) == 0
    train_loader, train_suploader = get_dataloader_for_fs_train(
        args.data_path, args.raw_data_path,
        args.evl_dm.split(','), args.batch_size, 
        args.max_sup_ratio, args.max_sup_size, args.n_shots, 
        tokenizer, return_suploader=True)
    # eval_loader, eval_suploader = get_dataloader_for_fs_eval(
    #     args.data_path, args.raw_data_path,
    #     args.evl_dm.split(','), args.batch_size, 
    #     args.max_sup_ratio, args.max_sup_size, args.n_shots, 
    #     tokenizer, return_suploader=True)
    test_loaders, test_suploaders = get_dataloader_for_fs_test(
        args.data_path, args.raw_data_path, args.batch_size, 
        args.n_shots, tokenizer, sep_dom=True, return_suploader=True)

    for dom in test_loaders.keys():
        logger.info("[Train] train model for test domain {}".format(dom))
        test_loader, test_suploader = test_loaders[dom], test_suploaders[dom]
        
        comb_train_loader = get_dataloader(
            train_loader.dataset.merge(test_suploader.dataset), 
            args.batch_size, True)
        eval_loader = train_suploader

        model = Model(args, tokenizer)
        model.cuda()

        ## def optim
        num_train_optim_steps = len(comb_train_loader) * args.max_epoch #// args.grad_acc_steps
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


        normal_train(args, model, optimizer, 
            comb_train_loader, eval_loader,
            os.path.join(args.dump_path, "best_model_{}.pth".format(dom)))


def do_comb_train_no_eval(args):
    args.dump_path = get_output_dir(args.dump_path, args.exp_name, args.exp_id)
    os.makedirs(args.dump_path, exist_ok=True)
    with open(os.path.join(args.dump_path, "args.pkl"), "wb") as fd:
        pickle.dump(args, fd)
    init_logger(os.path.join(args.dump_path, args.log_file))

    ## def model
    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(args.bert_dir, "vocab.txt"),
        do_lower_case=args.do_lower_case)
    
    ## def data
    assert len(args.evl_dm.strip()) == 0
    train_loader, train_suploader = get_dataloader_for_fs_train(
        args.data_path, args.raw_data_path, [], args.batch_size, 
        args.max_sup_ratio, args.max_sup_size, args.n_shots, 
        tokenizer, return_suploader=True)
    test_loaders, test_suploaders = get_dataloader_for_fs_test(
        args.data_path, args.raw_data_path, args.batch_size, 
        args.n_shots, tokenizer, sep_dom=True, return_suploader=True)

    for dom in test_loaders.keys():
        logger.info("[Train] train model for test domain {}".format(dom))
        test_loader, test_suploader = test_loaders[dom], test_suploaders[dom]
        
        comb_train_loader = get_dataloader(
            train_loader.dataset.merge(
                train_suploader.dataset).merge(
                        test_suploader.dataset), 
            args.batch_size, True)
        
        model = Model(args, tokenizer)
        model.cuda()

        ## def optim
        num_train_optim_steps = len(comb_train_loader) * args.max_epoch #// args.grad_acc_steps
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

        normal_train(args, model, optimizer, 
            comb_train_loader, test_loader,
            os.path.join(args.dump_path, "best_model_{}.pth".format(dom)))


def do_predict(args):
    args.dump_path = get_output_dir(args.dump_path, args.exp_name, args.exp_id)
    model_path = (os.path.join(args.model_path, args.target) 
                                if args.model_path is not None 
                                else os.path.join(args.dump_path, args.target))
    save_dir = args.save_dir if args.save_dir is not None else args.dump_path

    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(args.bert_dir, "vocab.txt"),
        do_lower_case=args.do_lower_case)
    model = Model(args, tokenizer)
    model.cuda()

    test_loaders, test_suploaders = get_dataloader_for_fs_test(
        args.data_path, args.raw_data_path, args.batch_size, 
        args.n_shots, tokenizer, sep_dom=True, return_suploader=True)

    for dom in test_loaders.keys():
        logger.info("[Test] test model for test domain {}".format(dom))
        test_loader, test_suploader = test_loaders[dom], test_suploaders[dom]

        state_dict = torch.load(model_path.format(dom))
        model.load_state_dict(state_dict["model"])
        model.eval()

        final_items = []
        int_preds, int_golds = [], []
        lbl_preds, lbl_golds = [], []

        if args.no_pbar:
            pbar = enumerate(test_loader)
        else:
            pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
        for i, batch in pbar:
            qry_input = batch["model_input"]
            for k, v in qry_input.items():
                qry_input[k] = v.cuda()
            batch["model_input"] = qry_input
            
            with torch.no_grad():
                qry_fwd = model(batch)
                dom_pred, int_pred, lbl_pred = model.predict(batch, qry_fwd)

                int_preds.extend(int_pred)
                lbl_preds.extend(lbl_pred)

                for j in range(len(batch["token"])):
                    tokens = batch["token"][j]
                    labels = [model.label_vocab.index2word[l] for l in lbl_pred[j]]
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
                    item["id"] = batch["id"][j]
                    item["domain"] = batch["domain"][j]
                    item["text"] = batch["text"][j]
                    item["intent"] = model.intent_map.index2word[int_pred[j]]
                    item["slots"] = slvals
                    final_items.append(item)
            
        # save file
        os.makedirs(os.path.join(save_dir, "predict"), exist_ok=True)
        with open(os.path.join(save_dir, "predict", "predict_{}.json".format(dom)),
                                                    'w', encoding='utf8') as fd:
            json.dump(final_items, fd, ensure_ascii=False, indent=2)

        # os.makedirs(os.path.join(save_dir, "predict_pproc"), exists_ok=True)
        # with open(os.path.join(save_dir, "predict_pproc", "predict_{}.json".format(dom)),
        #                                             'w', encoding='utf8') as fd:
        #     json.dump(final_items, fd, ensure_ascii=False, indent=2)

