#coding: utf-8
"""
    @author: cjliux@gmail.com
    @elems: bert, mtl, feat_mask, crf, cdt
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
                    for i in range(self.intent_map.n_words)]).cuda()
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
            ]).byte().cuda()
        self.dom_label_mask = dom_label_mask

        self.tokenizer = tokenizer

        # self.bert_enc = BertModel.from_pretrained(args.bert_dir)
        self.bert_enc = BertModel.from_pretrained(
            pretrained_model_path=os.path.join(args.bert_dir, "pytorch_model.bin"),
            config_path=os.path.join(args.bert_dir, "bert_config.json"))
        # # curated bert
        # self.bert_enc.token_type_embeddings = nn.Embedding(
        #     2 + self.label_vocab.n_words, self.bert_enc.config.hidden_size)

        self.dropout = nn.Dropout(p = 0.1)

        self.domain_outputs = nn.Linear(
            self.bert_enc.config.hidden_size, self.domain_map.n_words)
        self.intent_outputs = nn.Linear(
            self.bert_enc.config.hidden_size, self.intent_map.n_words)
        
        ## prior slots
        self.sltype_outputs = nn.Linear(
            self.bert_enc.config.hidden_size, self.slots_map.n_words+1)
        self.bio_outputs = nn.Linear(self.bert_enc.config.hidden_size, 3)
        
        self.sltype_map = torch.LongTensor(
            [self.slots_map.word2index[lbl[2:]] + 1 if lbl != 'O' else 0
                    for lbl in self.label_vocab._vocab ]).cuda()
        m = {'O': 0, 'B': 1, 'I':2}
        self.bio_map = torch.LongTensor(
            [m[lbl[0]] for lbl in self.label_vocab._vocab]).cuda()

        self.sltype_emb = nn.Parameter(torch.randn(self.sltype_outputs.weight.size()))
        nn.init.xavier_normal_(self.sltype_emb)
        self.bio_emb = nn.Parameter(torch.randn(self.bio_outputs.weight.size()))
        nn.init.xavier_normal_(self.bio_emb)

        # ## postr slots
        # self.postr_sltype_outputs = nn.Linear(
        #     self.bert_enc.config.hidden_size, self.slots_map.n_words+1)
        # self.postr_bio_outputs = nn.Linear(self.bert_enc.config.hidden_size, 3)
        
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

        self.crf_layer = CRF(self.label_vocab)
        # init_trans = build_init_crf_trans_bio(self.label_vocab)
        # self.crf_layer.init_weights(init_trans)

    def get_proto_dict(self, padded_seqs, seq_lengths, segids, dom_idx,
                                int_idx, padded_y, padded_bin_y):
        batch_size, max_len = padded_seqs.size(0), seq_lengths.max().item()
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        attn_mask = (idxes.cuda() < seq_lengths.unsqueeze(1)).float()

        return { "input_ids": padded_seqs, "label_ids": padded_y,
                 "segids": segids, "attention_mask": attn_mask,
                 "dom_idx": dom_idx, "int_idx": int_idx }

    def get_label_repr(self, label_ids):
        sltype_repr = self.sltype_emb[self.sltype_map]
        bio_repr = self.bio_emb[self.bio_map]
        label_repr = sltype_repr + bio_repr
        return label_repr[label_ids]

    def get_one_hot_repr(self, label_ids, label_voc):
        return F.one_hot(label_ids, label_voc.n_words)

    def map_seq_feature(self, lin_sltype, lin_bio, seq_output):
        sltype_logits = lin_sltype(seq_output)
        bio_logits = lin_bio(seq_output)

        batch_size, seq_len, _ = sltype_logits.size()
        feats = (sltype_logits.index_select(-1, self.sltype_map)
                            + bio_logits.index_select(-1, self.bio_map))
        return feats

    def attention_on_supset(self, query, key, value, mask):
        scores_ = torch.matmul(query.unsqueeze(1), key.unsqueeze(0).transpose(-1,-2))
        scores_.masked_fill_((1 - mask.unsqueeze(1).unsqueeze(0)).byte(), -1e9)
        context = torch.matmul(F.softmax(scores_, -1), value.unsqueeze(0))
        return context # B1B2S1D

    def encode_with_proto(self, padded_seqs, seq_lengths, segids, proto_dict):
        batch_size = padded_seqs.size(0)
        if proto_dict is not None:
            assert batch_size == 1

        # seq_lengths
        max_len = seq_lengths.max().item()
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        attn_mask = (idxes.cuda() < seq_lengths.unsqueeze(1)).float()

        if padded_seqs.size(1) > max_len:
            padded_seqs = padded_seqs[:, :max_len]
            segids = segids[:, :max_len]
        
        # encode sup
        sup_seq, sup_cls = self.bert_enc(
            input_ids=proto_dict["input_ids"],
            token_type_ids=proto_dict["segids"],
            attention_mask=proto_dict["attention_mask"],
            output_all_encoded_layers=False) # BSD, BD
        sup_lblrepr = self.get_one_hot_repr(proto_dict["label_ids"], self.label_vocab) # BSD
        # k: sup_seq; v: sup_lblrepr

        # encode qry 
        seq_output, cls_output = self.bert_enc(
            input_ids=padded_seqs,
            token_type_ids=segids,
            attention_mask=attn_mask,
            output_all_encoded_layers=False)
        
        cls_output = self.dropout(cls_output)
        seq_output = self.dropout(seq_output)

        dom_logits = self.domain_outputs(cls_output)
        int_logits = self.intent_outputs(cls_output)

        feats = self.map_seq_feature(
            self.sltype_outputs, self.bio_outputs, seq_output)

        # cross attention
        sup_lblinfo = self.attention_on_supset(
            seq_output, sup_seq, sup_lblrepr.float(), proto_dict["attention_mask"]) # B1B2S1D
        sup_alpha = F.softmax(torch.matmul(cls_output, sup_cls.t()), 1)
        lbl_info = sup_alpha.unsqueeze(-1).unsqueeze(-1).expand_as(
                                    sup_lblinfo).mul(sup_lblinfo).sum(1)

        # postr_feats = self.map_seq_feature(
        #     self.sltype_outputs, self.bio_outputs, lblinfo)
        
        return {"attn_mask": attn_mask, "dom_logits": dom_logits, 
                "int_logits": int_logits, "feats": feats, "lbl_info": lbl_info,
                "proto_dict": proto_dict } 

    def encode_without_proto(self, padded_seqs, seq_lengths, segids):
        # seq_lengths
        max_len = seq_lengths.max().item()
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        attn_mask = (idxes.cuda() < seq_lengths.unsqueeze(1)).float()

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
        
        feats = self.map_seq_feature(
            self.sltype_outputs, self.bio_outputs, seq_output)

        return { "attn_mask": attn_mask, "dom_logits": dom_logits, 
                 "int_logits": int_logits, "sl_logits": feats}

    def forward(self, padded_seqs, seq_lengths, dom_idx, segids, proto_dict=None):
        if proto_dict is not None:
            return self.encode_with_proto(padded_seqs, seq_lengths, segids, proto_dict)
        else:
            return self.encode_without_proto(padded_seqs, seq_lengths, segids)

    def compute_loss(self, dom_idx, int_idx, padded_y, fwd_dict):
        # qry
        loss = (2 * self.loss_fct(fwd_dict["dom_logits"], dom_idx) 
                + 4 * self.loss_fct(fwd_dict["int_logits"], int_idx))

        seq_len = fwd_dict["sl_logits"].size(1)

        seq_logits, seq_mask = fwd_dict["sl_logits"][:,1:], fwd_dict["attn_mask"][:,1:]
        seq_label = padded_y[:,1:seq_len]

        log_slprob = F.log_softmax(seq_logits, -1)
        sl_loss = - log_slprob.gather(
                                2, seq_label.unsqueeze(-1)).squeeze(-1)
        loss = loss.sum() + sl_loss[seq_mask.byte()].sum()

        loss = loss + self.crf_layer.compute_loss(
                                log_slprob, seq_mask, seq_label).sum()
        return loss

    def compute_postr_loss(self, dom_idx, int_idx, padded_y, fwd_dict):
        # qry
        loss = (2 * self.loss_fct(fwd_dict["dom_logits"], dom_idx) 
                + 4 * self.loss_fct(fwd_dict["int_logits"], int_idx))

        seq_len = fwd_dict["feats"].size(1)

        seq_label = padded_y[:,1:seq_len]
        feats, seq_mask = fwd_dict["feats"][:,1:], fwd_dict["attn_mask"][:,1:]
        # seq_pfeats = fwd_dict["postr_feats"][:,1:]
        lbl_info = fwd_dict["lbl_info"][:,1:]

        log_slprob = torch.log((F.softmax(feats, -1) + lbl_info) / 2)
        sl_loss = - log_slprob.gather(
                                2, seq_label.unsqueeze(-1)).squeeze(-1)
        loss = loss.sum() + sl_loss[seq_mask.byte()].sum()

        # log_slprob = F.log_softmax(seq_pfeats, -1)
        log_slprob = torch.log((F.softmax(feats, -1) + lbl_info) / 2)
        loss = loss + self.crf_layer.compute_loss(
                                log_slprob, seq_mask, seq_label).sum()
        return loss

    def predict(self, seq_lengths, dom_idx, fwd_dict):
        for i_sam, i_dom in enumerate(dom_idx.tolist()):
            fwd_dict["int_logits"][i_sam].masked_fill_(self.dom_int_mask[i_dom], -1e9)
            fwd_dict["sl_logits"][i_sam].masked_fill_(self.dom_label_mask[i_dom], -1e9)

        dom_pred = torch.argmax(fwd_dict["dom_logits"], dim=-1).detach().cpu().numpy()
        int_pred = torch.argmax(fwd_dict["int_logits"], dim=-1).detach().cpu().numpy()
        log_slprob = F.log_softmax(fwd_dict["sl_logits"][:,1:], -1)
        _, crf_pred = self.crf_layer.inference(log_slprob, fwd_dict["attn_mask"][:,1:])
        lbl_pred = [[self.label_vocab.word2index['O']] 
                        + crf_pred[i, :ln-1].data.tolist() 
                            for i, ln in enumerate(seq_lengths.tolist())]
        return dom_pred, int_pred, lbl_pred

    def predict_postr(self, seq_lengths, dom_idx, fwd_dict):
        assert seq_lengths.size(0) == 1
        for i_sam, i_dom in enumerate(dom_idx.tolist()):
            fwd_dict["int_logits"][i_sam].masked_fill_(self.dom_int_mask[i_dom], -1e9)
            # fwd_dict["postr_feats"][i_sam].masked_fill_(self.dom_label_mask[i_dom], -1e9)
            fwd_dict["feats"][i_sam].masked_fill_(self.dom_label_mask[i_dom], -1e9)

        dom_pred = torch.argmax(fwd_dict["dom_logits"], dim=-1).detach().cpu().numpy()
        int_pred = torch.argmax(fwd_dict["int_logits"], dim=-1).detach().cpu().numpy()
        
        # log_slprob = F.log_softmax(fwd_dict["postr_feats"][:,1:], -1)
        log_slprob = torch.log(
            (F.softmax(fwd_dict["feats"][:,1:], -1) + fwd_dict["lbl_info"][:,1:]) / 2)
        _, crf_pred = self.crf_layer.inference(log_slprob, fwd_dict["attn_mask"][:,1:])
        lbl_pred = [[self.label_vocab.word2index['O']] 
                        + crf_pred[i, :ln-1].data.tolist() 
                            for i, ln in enumerate(seq_lengths.tolist())]
        return dom_pred, int_pred, lbl_pred


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

    def init_weights(self, init_trans):
        """
        init_trans: dict((to:long, fr:long) -> val:float)
        """
        for (to, fr), val in init_trans.items():
            self.transitions.data[to, fr].fill_(val)
        self.valid_trans = (self.transitions != -1e9)

    def build_transitions(self, label_vocab):
        trans = torch.zeros(self.num_tags, self.num_tags)
        m1 = {'O':0, 'B':1, 'I':2}
        m2 = {'O':0, 'B':3, 'I':4}
        types = [[[m1[li[0]], m2[lj[0]]] 
                if li != 'O' and li[2:] != lj[2:] 
                else [m1[li[0]], m1[lj[0]]] for lj in label_vocab._vocab] 
                                            for li in label_vocab._vocab]
        types = torch.LongTensor(types).cuda().permute(2,0,1)
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
        mask = mask.byte()
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
        mask = mask.byte()
        batch_size, seq_len, tag_size = feats.size()
        
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()

        mask = mask.transpose(1, 0).contiguous()
        feats = feats.transpose(1, 0).contiguous()
        
        # record the position of the best score
        back_points = list()
        partition_history = list()

        # mask = 1 + (-1) * mask
        inv_mask = (1 - mask.long()).byte()
        
        partition = feats[0] + self.start_transitions.unsqueeze(0)
        partition_history.append(partition)
        # transitions = self.transitions.unsqueeze(0)
        transitions = transitions.unsqueeze(0)

        for idx in range(1, seq_len):
            cur_values = feats[idx].unsqueeze(-2) + partition.unsqueeze(-1) + transitions
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)

            cur_bp.masked_fill_(inv_mask[idx].view(batch_size, 1), 0)
            back_points.append(cur_bp)

        partition_history = torch.cat(partition_history).view(
            seq_len, batch_size, -1).transpose(1, 0).contiguous()

        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(
            partition_history, 1, last_position).view(batch_size, tag_size)

        last_values = last_partition + self.stop_transitions.unsqueeze(0)
        path_score, last_bp = torch.max(last_values, 1)

        pad_zero = torch.zeros(batch_size, tag_size).long().cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()

        pointer = last_bp
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points.scatter_(1, last_position, insert_last)

        back_points = back_points.transpose(1, 0).contiguous()

        decode_idx = torch.LongTensor(seq_len, batch_size).cuda()
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
        mask = mask.byte()
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