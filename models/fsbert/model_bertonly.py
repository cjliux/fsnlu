#coding: utf-8
"""
    @author: cjliux@gmail.com
"""
import os, sys
import copy
import logging
import torch
import torch.nn as nn
import numpy as np
from .modeling import BertForTaskNLU
# from .tokenization import BertTokenizer
from .optimization import BertAdam, WarmupLinearSchedule
from utils.vocab import Vocab

logger = logging.getLogger()


class Model(nn.Module):

    def __init__(self, args, tokenizer=None):
        super().__init__()
        self.args = args

        self.domain_map = Vocab.from_file(os.path.join(args.data_path, "domains.txt"))
        self.intent_map = Vocab.from_file(os.path.join(args.data_path, "intents.txt"))
        self.slots_map = Vocab.from_file(os.path.join(args.data_path, "slots.txt"))
        self.label_vocab = Vocab.from_file(os.path.join(args.data_path, "label_vocab.txt"))
        self.bin_label_vocab = Vocab.from_file(os.path.join(args.data_path, "bin_label_vocab.txt"))
        with open(os.path.join(args.data_path, "domain2slots.json"), 'r', encoding='utf8') as fd:
            self.dom2slots = json.load(fd)

        self.tokenizer = tokenizer

        label_list = {
            "domain": copy.deepcopy(self.domain_map.index2word),
            "intent": copy.deepcopy(self.intent_map.index2word),
            "slots": copy.deepcopy(self.slots_map.index2word),
            "label_vocab": self.label_vocab,
        }

        state_dict = torch.load(os.path.join(args.bert_dir, "pytorch_model.bin"))
        self.bert_nlu = BertForTaskNLU.from_pretrained(
            os.path.join(args.bert_dir, "pytorch_model.bin"),
            os.path.join(args.bert_dir, "bert_config.json"),
            # state_dict=state_dict,
            label_list=label_list,
            max_seq_len=args.max_seq_length)
        # self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

        # self.crf_layer = CRF_coach(self.label_vocab, self.dom2slots)

    def forward(self, padded_seqs, seq_lengths):
        # inputs = batch["model_input"]
        # padded_seqs = inputs["padded_seqs"].cuda() 
        # seq_lengths = inputs["seq_lengths"].cuda() 

        # seq_lengths
        max_len = seq_lengths.max().item()
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        attn_mask = (idxes.cuda() < seq_lengths.unsqueeze(1)).float()

        if padded_seqs.size(1) > max_len:
            padded_seqs = padded_seqs[:, :max_len]

        dom_logits, int_logits, sl_logits = self.bert_nlu(
            input_ids=padded_seqs, 
            #token_type_ids=inputs["segment_ids"],
            attention_mask=attn_mask)
        return { "dom_logits": dom_logits, 
            "int_logits": int_logits, "sl_logits": sl_logits}

    def compute_loss(self, dom_idx, int_idx, padded_y, fwd_dict):
        # inputs = batch["model_input"]
        # dom_idx, int_idx = inputs["dom_idx"].cuda(), inputs["int_idx"].cuda()
        loss = (5 * self.loss_fct(fwd_dict["dom_logits"], dom_idx) 
                + 3 * self.loss_fct(fwd_dict["int_logits"], int_idx))

        seq_len = fwd_dict["sl_logits"].size(1)

        # padded_y = inputs["padded_y"].cuda()
        loss += 2 * self.crf_layer.compute_loss(
            fwd_dict["sl_logits"][:,1:], padded_y[:,1:seq_len])
        return loss.mean()

    def predict(self, seq_lengths, fwd_dict):
        dom_pred = torch.argmax(fwd_dict["dom_logits"], dim=-1).detach().cpu().numpy()
        int_pred = torch.argmax(fwd_dict["int_logits"], dim=-1).detach().cpu().numpy()
        crf_pred = self.crf_layer(fwd_dict["sl_logits"][:, 1:])

        ipreds = torch.argmax(crf_pred, -1)
        

        lbl_pred = [[self.label_vocab.word2index['O']] 
                        + crf_pred[i, :ln-1].data.tolist()
                            for i, ln in enumerate(seq_lengths.tolist())]
        return dom_pred, int_pred, lbl_pred

