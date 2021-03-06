#coding: utf-8
"""
    @author: cjliux@gmail.com
    @elems: bert, mtl, feat_mask, crf, cdt, sep label
    @status: 
"""
import os, sys
import copy
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from .modeling import BertModel, BertConfig
# from .tokenization import BertTokenizer
from .optimization import BertAdam, WarmupLinearSchedule
from utils.vocab import Vocab

logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):

    def __init__(self, args, tokenizer=None):
        super().__init__()
        self.args = args

        self.domain_map = Vocab.from_file(os.path.join(args.data_path, "domains.txt"))
        self.intent_map = Vocab.from_file(os.path.join(args.data_path, "intents.txt"))
        self.slots_map = Vocab.from_file(os.path.join(args.data_path, "slots.txt"))
        self.label_vocab = Vocab.from_file(os.path.join(args.data_path, "label_vocab.txt"))
        self.bin_label_vocab = Vocab.from_file(os.path.join(args.data_path, "bin_label_vocab.txt"))
        with open(os.path.join(args.data_path, "dom2intents.json"), 'r', encoding='utf8') as fd:
            self.dom2intents = json.load(fd)
        with open(os.path.join(args.data_path, "dom2slots.json"), 'r', encoding='utf8') as fd:
            self.dom2slots = json.load(fd)

        dom_int_mask = {}
        for i_dom, dom in enumerate(self.domain_map._vocab):
            dom_int_mask[i_dom] = torch.ByteTensor([
                0 if self.intent_map.index2word[i] in self.dom2intents[dom] else 1 
                    for i in range(self.intent_map.n_words)]).to(device)
        self.dom_int_mask = dom_int_mask

        dom_label_mask = {}
        for i_dom, dom in enumerate(self.domain_map._vocab):
            cand_labels = [self.label_vocab.word2index['O']]
            for sl in self.dom2slots[dom]:
                cand_labels.extend([
                    self.label_vocab.word2index['B-' + sl], 
                    self.label_vocab.word2index['I-' + sl]])
            dom_label_mask[i_dom] = torch.LongTensor([
                0 if i in cand_labels else 1 for i in range(self.label_vocab.n_words)
            ]).bool().to(device)
        self.dom_label_mask = dom_label_mask

        self.tokenizer = tokenizer

        self.bert_enc = BertModel.from_pretrained(
            pretrained_model_path=os.path.join(args.bert_dir, "pytorch_model.bin"),
            config_path=os.path.join(args.bert_dir, "bert_config.json"))

        self.domain_outputs = nn.Linear(
            self.bert_enc.config.hidden_size, self.domain_map.n_words)
        self.intent_outputs = nn.Linear(
            self.bert_enc.config.hidden_size, self.intent_map.n_words)
        
        self.sltype_outputs = nn.Linear(
            self.bert_enc.config.hidden_size, self.slots_map.n_words+1)
        self.bio_outputs = nn.Linear(self.bert_enc.config.hidden_size, 3)
        
        self.sltype_map = torch.LongTensor(
            [self.slots_map.word2index[lbl[2:]] + 1 if lbl != 'O' else 0
                    for lbl in self.label_vocab._vocab ]).to(device)
        m = {'O': 0, 'B': 1, 'I':2}
        self.bio_map = torch.LongTensor(
            [m[lbl[0]] for lbl in self.label_vocab._vocab]).to(device)

        self.dropout = nn.Dropout(p = 0.1)
        
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

        self.crf_layer = CRF(self.label_vocab)

    def map_seq_feature(self, seq_output):
        sltype_logits = self.sltype_outputs(seq_output)
        bio_logits = self.bio_outputs(seq_output)

        batch_size, seq_len, _ = sltype_logits.size()
        feats = (sltype_logits.index_select(-1, self.sltype_map)
                            + bio_logits.index_select(-1, self.bio_map))
        return feats

    def forward(self, batch):
        mdl_input = batch["model_input"]
        padded_seqs, seq_lengths, dom_idx, segids = (mdl_input["padded_seqs"],
            mdl_input["seq_lengths"], mdl_input["dom_idx"], mdl_input["segids"])

        # seq_lengths
        max_len = seq_lengths.max().item()
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        attn_mask = (idxes.to(device) < seq_lengths.unsqueeze(1)).float()

        if padded_seqs.size(1) > max_len:
            padded_seqs = padded_seqs[:, :max_len]

        seq_output, cls_output = self.bert_enc(
            input_ids=padded_seqs,
            token_type_ids=segids,
            attention_mask=attn_mask,
            output_all_encoded_layers=False)
        cls_output = self.dropout(cls_output)
        seq_output = self.dropout(seq_output)

        dom_logits = self.domain_outputs(cls_output)
        int_logits = self.intent_outputs(cls_output)
        
        sltype_logits = self.sltype_outputs(seq_output)
        bio_logits = self.bio_outputs(seq_output)
        feats = self.map_seq_feature(seq_output)
        
        return { "dom_idx": dom_idx,
            "attn_mask": attn_mask, "dom_logits": dom_logits, 
            "int_logits": int_logits, "sl_logits": feats}

    def compute_loss(self, batch, fwd_dict):
        mdl_input = batch["model_input"]
        dom_idx, int_idx, padded_y = (mdl_input["dom_idx"], 
                            mdl_input["int_idx"], mdl_input["padded_y"])

        loss = (2 * self.loss_fct(fwd_dict["dom_logits"], dom_idx) 
                + 4 * self.loss_fct(fwd_dict["int_logits"], int_idx)).sum()

        seq_len = fwd_dict["sl_logits"].size(1)

        seq_logits, seq_mask = fwd_dict["sl_logits"][:,1:], fwd_dict["attn_mask"][:,1:]
        seq_label = padded_y[:,1:seq_len]

        log_slprob = F.log_softmax(seq_logits, -1)
        sl_loss = - log_slprob.gather(
                                2, seq_label.unsqueeze(-1)).squeeze(-1)
        loss = loss + sl_loss[seq_mask.bool()].sum()
        loss = loss + self.crf_layer.compute_loss(
                                log_slprob, seq_mask, seq_label).sum()

        # sl_loss = - F.log_softmax(seq_logits, -1).gather(
        #                         2, seq_label.unsqueeze(-1)).squeeze(-1)
        # loss = loss + sl_loss[seq_mask.bool()].sum()
        # loss = loss + self.crf_layer.compute_loss(
        #                         seq_logits, seq_mask, seq_label).sum()
        return loss

    def predict(self, batch, fwd_dict):
        mdl_input = batch["model_input"]
        seq_lengths, dom_idx = mdl_input["seq_lengths"], mdl_input["dom_idx"]

        for i_sam, i_dom in enumerate(fwd_dict["dom_idx"].tolist()):
            fwd_dict["int_logits"][i_sam].masked_fill_(self.dom_int_mask[i_dom].bool(), -1e9)
            fwd_dict["sl_logits"][i_sam].masked_fill_(self.dom_label_mask[i_dom].bool(), -1e9)
            
        dom_pred = torch.argmax(fwd_dict["dom_logits"], dim=-1).detach().cpu().numpy()
        int_pred = torch.argmax(fwd_dict["int_logits"], dim=-1).detach().cpu().numpy()

        _, crf_pred = self.crf_layer.inference(
                        F.log_softmax(fwd_dict["sl_logits"][:,1:], -1), 
                        fwd_dict["attn_mask"][:,1:])
        # _, crf_pred = self.crf_layer.inference(
        #     fwd_dict["sl_logits"][:,1:], fwd_dict["attn_mask"][:,1:])

        lbl_pred = [[self.label_vocab.word2index['O']] 
                        + crf_pred[i, :ln-1].data.tolist() 
                            for i, ln in enumerate(seq_lengths.tolist())]
        return dom_pred, int_pred, lbl_pred


class CRF(nn.Module):

    def __init__(self, label_vocab):
        """
        config:
            target_size: int, target size = dim_state_space + 2
        """
        super().__init__()
        # sep trans matrix
        self.label_vocab = label_vocab
        self.num_tags = label_vocab.n_words
        # self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))
        self.cdt_transitions = nn.Parameter(torch.Tensor([
                [0.6, 0.4, 1e-3, 1e-3, 1e-3], 
                [0.3, 0.1, 0.1, 0.5, 1e-3], 
                [0.6, 0.1, 0.1, 0.2, 1e-3]
            ]).log())
        self.start_transitions = nn.Parameter(torch.randn(self.num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(self.num_tags))
        # nn.init.xavier_normal_(self.transitions)

    def build_transitions(self, label_vocab):
        trans = torch.zeros(self.num_tags, self.num_tags)
        m1 = {'O':0, 'B':1, 'I':2}
        m2 = {'O':0, 'B':3, 'I':4}
        types = [[[m1[li[0]], m2[lj[0]]] 
                if li != 'O' and li[2:] != lj[2:] 
                else [m1[li[0]], m1[lj[0]]] for lj in label_vocab._vocab] 
                                            for li in label_vocab._vocab]
        types = torch.LongTensor(types).to(device).permute(2,0,1)
        transitions = self.cdt_transitions[types[0], types[1]]
        return transitions.contiguous().view(self.num_tags, self.num_tags)

    def _forward_alg(self, feats, mask, transitions):
        """
        Do the forward algorithm to compute the partition function (batched).

        Args:
            feats: size=(batch_size, seq_len, self.target_size)
            mask: size=(batch_size, seq_len)

        Returns:
            final_partition: (batch_size)
        """
        mask = mask.bool()
        batch_size, seq_len, tag_size = feats.size()
        
        mask = mask.transpose(1, 0).contiguous()
        feats = feats.transpose(1, 0).contiguous()
        partition = (feats[0] + self.start_transitions.unsqueeze(0))
        # transitions = self.transitions.unsqueeze(0)
        transitions = transitions.unsqueeze(0)

        # seq_iter = enumerate(feats[1:], 1)
        for idx in range(1, seq_len):
            cur_values = feats[idx].unsqueeze(-2) + partition.unsqueeze(-1) + transitions
            cur_partition = torch.logsumexp(cur_values, dim=1)

            mask_idx = mask[idx, :].unsqueeze(-1).expand(batch_size, tag_size)
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            
            if masked_cur_partition.dim() != 0:
                partition.masked_scatter_(mask_idx, masked_cur_partition)

        cur_values = partition + self.stop_transitions.unsqueeze(0)
        final_partition = torch.logsumexp(cur_values, 1)
        return final_partition

    def _viterbi_decode(self, feats, mask, transitions):
        """
        Args:
            feats: size=(batch_size, seq_len, self.target_size)
            mask: size=(batch_size, seq_len)

        Returns:
            decode_idx: (batch_size, seq_len), viterbi decode结果
            path_score: size=(batch_size, 1), 每个句子的得分
        """
        mask = mask.bool()
        batch_size, seq_len, tag_size = feats.size()
        
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()

        mask = mask.transpose(1, 0).contiguous()
        feats = feats.transpose(1, 0).contiguous()
        
        # record the position of the best score
        back_points = list()
        partition_history = list()

        # mask = 1 + (-1) * mask
        inv_mask = (1 - mask.long()).bool()
        
        partition = feats[0] + self.start_transitions.unsqueeze(0)
        partition_history.append(partition)
        # transitions = self.transitions.unsqueeze(0)
        transitions = transitions.unsqueeze(0)

        for idx in range(1, seq_len):
            cur_values = feats[idx].unsqueeze(-2) + partition.unsqueeze(-1) + transitions
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)

            cur_bp.masked_fill_(inv_mask[idx].view(batch_size, 1).bool(), 0)
            back_points.append(cur_bp)

        partition_history = torch.cat(partition_history).view(
            seq_len, batch_size, -1).transpose(1, 0).contiguous()

        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(
            partition_history, 1, last_position).view(batch_size, tag_size)

        last_values = last_partition + self.stop_transitions.unsqueeze(0)
        path_score, last_bp = torch.max(last_values, 1)

        pad_zero = torch.zeros(batch_size, tag_size).long().to(device)
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()

        pointer = last_bp
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points.scatter_(1, last_position, insert_last)

        back_points = back_points.transpose(1, 0).contiguous()

        decode_idx = torch.LongTensor(seq_len, batch_size).to(device)
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.view(-1).data
        # path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def _score_sentence(self, feats, mask, tags, transitions):
        """
        Args:
            scores: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            score:
        """
        mask = mask.bool()
        batch_size, seq_len, tag_size = feats.size()
        
        feat_score = (feats.gather(2, tags.unsqueeze(-1)).squeeze(-1) * mask.float()).sum(-1)

        # transitions = self.transitions

        tags_pairs = tags.unfold(1, 2, 1)
        indices = tags_pairs.permute(2, 0, 1).chunk(2)
        trans_score = (transitions[indices].squeeze(0) * mask[:,1:].float()).sum(-1)

        start_score = self.start_transitions[tags[:, 0]]

        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask-1).squeeze(1)
        stop_score = torch.gather(self.stop_transitions, 0, end_ids)

        gold_score = feat_score + start_score + trans_score + stop_score
        return gold_score

    def inference(self, feats, mask):
        transitions = self.build_transitions(self.label_vocab)
        path_score, best_path = self._viterbi_decode(feats, mask, transitions)
        return path_score, best_path

    def compute_loss(self, feats, mask, tags):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        transitions = self.build_transitions(self.label_vocab)
        forward_score = self._forward_alg(feats, mask, transitions)
        gold_score = self._score_sentence(feats, mask, tags, transitions)
        return forward_score - gold_score
