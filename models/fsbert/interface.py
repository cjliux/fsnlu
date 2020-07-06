#coding: utf-8
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

logger = logging.getLogger()


class Manager(object):

    def __init__(self, args, model, optimizer=None):
        super().__init__()
        self.args = args
        os.makedirs(self.params.dump_path, exist_ok=True)

        self.model = Model(args)


def evaluate(model, eval_loader):
    pass


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
    num_train_optim_steps = len(train_loader) // args.grad_acc_steps
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters, 
                         lr=args.lr,
                         warmup=args.warm_proportion,
                         t_total=num_train_optim_steps)

    for _ in range(args.max_epoch):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_loader):
            qry_loss = model(batch)
            
            optimizer.zero_grad()
            qry_loss.backward()
            optimizer.step()

            print()


