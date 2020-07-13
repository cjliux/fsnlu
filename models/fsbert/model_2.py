#coding: utf-8
"""
    @author: cjliux@gmail.com
"""
import os, sys
import copy
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from .modeling_2 import BertForTaskNLU
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
        with open(os.path.join(args.data_path, "dom2intents.json"), 'r', encoding='utf8') as fd:
            self.dom2intents = json.load(fd)
        with open(os.path.join(args.data_path, "dom2slots.json"), 'r', encoding='utf8') as fd:
            self.dom2slots = json.load(fd)

        dom_int_mask = {}
        for i_dom, dom in enumerate(self.intent_map._vocab):
            dom_int_mask[i_dom] = torch.ByteTensor([
                0 if i in self.dom2intents[dom] else 1 
                    for i in range(self.intent_map.n_words)]).cuda()
        self.dom_int_mask = dom_int_mask

        dom_slot_mask = {}
        for i_dom, dom in enumerate(self.domain_map._vocab):
            cand_labels = [self.label_vocab.word2index['O']]
            for sl in self.dom2slots[dom]:
                cand_labels.extend([
                    self.label_vocab.word2index['B-' + sl], 
                    self.label_vocab.word2index['I-' + sl]])
            dom_slot_mask[i_dom] = torch.LongTensor([
                0 if i in cand_labels else 1 for i in range(self.label_vocab.n_words)
            ] + [1, 1]).byte().cuda()
        self.dom_slot_mask = dom_slot_mask

        self.tokenizer = tokenizer

        # state_dict = torch.load(os.path.join(args.bert_dir, "pytorch_model.bin"))
        self.bert_nlu = BertForTaskNLU.from_pretrained(
            os.path.join(args.bert_dir, "pytorch_model.bin"),
            os.path.join(args.bert_dir, "bert_config.json"),
            domain_num=self.domain_map.n_words,
            intent_num=self.intent_map.n_words,
            label_num=self.label_vocab.n_words+2,
            max_seq_len=args.max_seq_length)
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

        self.crf_layer = CRF_sltk(self.label_vocab.n_words+2)
        # init_trans = build_init_crf_trans_bio(self.label_vocab)
        # self.crf_layer.init_weights(init_trans)

    def forward(self, padded_seqs, seq_lengths, dom_idxs, segids):
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

        # for i_sam, i_dom in enumerate(dom_idxs.tolist()):
        #     sl_logits[i_sam].masked_fill_(self.dom_label_mask[i_dom], -1e9)
        
        return { "dom_idxs": dom_idxs,
            "attn_mask": attn_mask, "dom_logits": dom_logits, 
            "int_logits": int_logits, "sl_logits": sl_logits}

    def compute_loss(self, dom_idx, int_idx, padded_y, fwd_dict):
        # inputs = batch["model_input"]
        # dom_idx, int_idx = inputs["dom_idx"].cuda(), inputs["int_idx"].cuda()
        loss = (5 * self.loss_fct(fwd_dict["dom_logits"], dom_idx) 
                + 3 * self.loss_fct(fwd_dict["int_logits"], int_idx))

        seq_len = fwd_dict["sl_logits"].size(1)

        # padded_y = inputs["padded_y"].cuda()
        loss += 2 * self.crf_layer.compute_loss(fwd_dict["sl_logits"][:,1:], 
            fwd_dict["attn_mask"][:,1:], padded_y[:,1:seq_len])
        return loss.mean()

    def predict(self, seq_lengths, fwd_dict):
        for i_sam, i_dom in enumerate(fwd_dict["dom_idxs"].tolist()):
            fwd_dict["int_logits"][i_sam].masked_fill_(self.dom_int_mask[i_dom], -1e9)
            fwd_dict["sl_logits"][i_sam].masked_fill_(self.dom_label_mask[i_dom], -1e9)
            
        dom_pred = torch.argmax(fwd_dict["dom_logits"], dim=-1).detach().cpu().numpy()
        int_pred = torch.argmax(fwd_dict["int_logits"], dim=-1).detach().cpu().numpy()
        _, crf_pred = self.crf_layer.inference(
            fwd_dict["sl_logits"][:,1:], fwd_dict["attn_mask"][:,1:])
        lbl_pred = [[self.label_vocab.word2index['O']] 
                        + crf_pred[i, :ln-1].data.tolist() 
                            for i, ln in enumerate(seq_lengths.tolist())]
        return dom_pred, int_pred, lbl_pred


def log_sum_exp(vec, m_size):
    """
    Args:
        vec: size=(batch_size, vanishing_dim, hidden_dim)
        m_size: hidden_dim

    Returns:
        size=(batch_size, hidden_dim)
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(
        torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


def build_init_crf_trans_bio(label_vocab, neg_inf=-1e9):
    # label vocab -> init_trans
    vocab = label_vocab.get_vocab()
    e_type_to_lbl = defaultdict(dict)
    i_o = None
    for i_lbl, lbl in enumerate(vocab):
        i_hyp = lbl.find('-') 
        if i_hyp != -1:
            e_type_to_lbl[lbl[i_hyp+1:]][lbl[:i_hyp]] = i_lbl
        else:
            i_o = i_lbl
    for e_type in e_type_to_lbl.keys():
        e_type_to_lbl[e_type]['O'] = i_o

    init_trans = {}
    for e_type1 in e_type_to_lbl.keys():
        for e_type2 in e_type_to_lbl.keys():
            if e_type1 != e_type2:
                for tag1 in ['B', 'I', 'O']:
                    if tag1 in e_type_to_lbl[e_type1].keys() and 'I' in e_type_to_lbl[e_type2].keys():
                        init_trans[(e_type_to_lbl[e_type2]['I'], e_type_to_lbl[e_type1][tag1])] = neg_inf  
            else:
                if 'O' in e_type_to_lbl[e_type1].keys() and 'I' in e_type_to_lbl[e_type2].keys():
                    init_trans[(e_type_to_lbl[e_type2]['I'], e_type_to_lbl[e_type1]['O'])] = neg_inf
    return init_trans


class CRF_sltk(nn.Module):

    def __init__(self, target_size, average_batch=False):
        """
        config:
            target_size: int, target size = dim_state_space + 2
            average_batch: bool, loss是否作平均, default is True
        """
        super().__init__()
        # self.label_vocab = label_vocab
        self.target_size = target_size
        self.average_batch = average_batch

        self.START_TAG_IDX, self.END_TAG_IDX = -2, -1
        # target_size = label_vocab.n_words
        init_transitions = torch.zeros(target_size, target_size)
        init_transitions[:, self.START_TAG_IDX] = -1e9
        init_transitions[self.END_TAG_IDX, :] = -1e9
        self.transitions = nn.Parameter(init_transitions.cuda())

    def init_weights(self, init_trans):
        """
        init_trans: dict((to:long, fr:long) -> val:float)
        """
        for (to, fr), val in init_trans.items():
            self.transitions.data[to, fr].fill_(val)
        self.valid_trans = (self.transitions != -1e9)

    def _forward_alg(self, feats, mask):
        """
        Do the forward algorithm to compute the partition function (batched).

        Args:
            feats: size=(batch_size, seq_len, self.target_size)
            mask: size=(batch_size, seq_len)

        Returns:
            xxx
        """
        mask = mask.byte()

        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        mask = mask.transpose(1, 0).contiguous()
        ins_num = batch_size * seq_len

        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)

            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            masked_cur_partition = cur_partition.masked_select(mask_idx)
            if masked_cur_partition.dim() != 0:
                mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
                partition.masked_scatter_(mask_idx, masked_cur_partition)

        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size) + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.END_TAG_IDX]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
        Args:
            feats: size=(batch_size, seq_len, self.target_size)
            mask: size=(batch_size, seq_len)

        Returns:
            decode_idx: (batch_size, seq_len), viterbi decode结果
            path_score: size=(batch_size, 1), 每个句子的得分
        """
        mask = mask.byte()

        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()

        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size

        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)
        # record the position of the best score
        back_points = list()
        partition_history = list()

        # mask = 1 + (-1) * mask
        mask = (1 - mask.long()).byte()
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()

        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        partition_history.append(partition)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition.unsqueeze(-1))

            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)

        partition_history = torch.cat(partition_history).view(
            seq_len, batch_size, -1).transpose(1, 0).contiguous()

        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(
            partition_history, 1, last_position).view(batch_size, tag_size, 1)

        last_values = last_partition.expand(batch_size, tag_size, tag_size) + \
            self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = torch.zeros(batch_size, tag_size).long().cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        pointer = last_bp[:, self.END_TAG_IDX]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()

        back_points.scatter_(1, last_position, insert_last)

        back_points = back_points.transpose(1, 0).contiguous()

        decode_idx = torch.LongTensor(seq_len, batch_size).cuda()
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.view(-1).data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def _score_sentence(self, scores, mask, tags):
        """
        Args:
            scores: size=(seq_len, batch_size, tag_size, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            score:
        """
        mask = mask.byte()

        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)

        new_tags = torch.LongTensor(batch_size, seq_len).cuda()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx-1] * tag_size + tags[:, idx]

        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask-1)

        end_energy = torch.gather(end_transition, 1, end_ids)

        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(
            seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        gold_score = tg_energy.sum() + end_energy.sum()

        return gold_score

    def inference(self, feats, mask):
        path_score, best_path = self._viterbi_decode(feats, mask)
        return path_score, best_path

    def compute_loss(self, feats, mask, tags):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        batch_size = feats.size(0)
        forward_score, scores = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        return forward_score - gold_score
