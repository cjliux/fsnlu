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

        self.domain_outputs = nn.Linear(
            self.bert_enc.config.hidden_size, self.domain_map.n_words)
        self.intent_outputs = nn.Linear(
            self.bert_enc.config.hidden_size, self.intent_map.n_words)
        # self.slots_outputs = nn.Linear(
        #     self.bert_enc.config.hidden_size, self.label_vocab.n_words)
        self.sltype_outputs = nn.Linear(
            self.bert_enc.config.hidden_size, self.slots_map.n_words+1)
        self.bio_outputs = nn.Linear(self.bert_enc.config.hidden_size, 3)
        
        self.sltype_map = torch.LongTensor(
            [self.slots_map.word2index[lbl[2:]] + 1 if lbl != 'O' else 0
                    for lbl in self.label_vocab._vocab ]).cuda()
        m = {'O': 0, 'B': 1, 'I':2}
        self.bio_map = torch.LongTensor(
            [m[lbl[0]] for lbl in self.label_vocab._vocab]).cuda()

        self.dropout = nn.Dropout(p = 0.1)
        
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

        self.crf_layer = CRF(self.label_vocab)
        # init_trans = build_init_crf_trans_bio(self.label_vocab)
        # self.crf_layer.init_weights(init_trans)

    def encode_proto(self, padded_seqs, seq_lengths, segids, 
                                int_idx, padded_y, padded_bin_y):
        batch_size, max_len = padded_seqs.size(0), seq_lengths.max().item()
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        attn_mask = (idxes.cuda() < seq_lengths.unsqueeze(1)).float()

        seq_output, cls_output = self.bert_enc(
            input_ids=padded_seqs,
            token_type_ids=segids,
            attention_mask=attn_mask,
            output_all_encoded_layers=False)
        
        proto_values = []
        for i_sam in range(batch_size):
            sam_seq_feat, sam_cls_feat = seq_output[i_sam], cls_output[i_sam]
            label, bin_label = padded_y[i_sam], padded_bin_y[i_sam]
            
            prototype = {}

            B_list, I_list = (bin_label == 1), (bin_label == 2)
            nonzero_B = torch.nonzero(B_list)
            num_slotname = nonzero_B.size()[0]

            if num_slotname == 0:
                proto_values.append(prototype)
                continue

            for j in range(num_slotname):
                if num_slotname > 1 and j == 0:
                    prev_index = nonzero_B[j]
                    continue
                
                curr_index = nonzero_B[j]
                if not (num_slotname == 1 and j == 0):
                    nonzero_I = torch.nonzero(I_list[prev_index: curr_index])
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + prev_index).squeeze(1)
                        indices = torch.cat((prev_index, nonzero_I), dim=0)
                        hiddens_based_slotname = sam_seq_feat[indices]
                    else:
                        hiddens_based_slotname = sam_seq_feat[prev_index]

                    # get slot repr
                    slot_feat = torch.mean(hiddens_based_slotname, dim=0)
                    slot_feat /= slot_feat.norm(dim=-1, keepdim=True)
                    slot_name = self.label_vocab.index2word[label[prev_index].item()][2:]
                    assert len(slot_name) > 0
                    prototype[slot_name] = slot_feat

                if j == num_slotname - 1:
                    nonzero_I = torch.nonzero(I_list[curr_index:])
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + curr_index).squeeze(1)
                        indices = torch.cat((curr_index, nonzero_I), dim=0)
                        hiddens_based_slotname = sam_seq_feat[indices]
                    else:
                        hiddens_based_slotname = sam_seq_feat[curr_index]
                
                    slot_feat = torch.mean(hiddens_based_slotname, dim=0)
                    slot_feat /= slot_feat.norm(dim=-1, keepdim=True)
                    slot_name = self.label_vocab.index2word[label[curr_index].item()][2:]
                    assert len(slot_name) > 0
                    prototype[slot_name] = slot_feat
                else:
                    prev_index = curr_index
        
            proto_values.append(prototype)

        proto_key = cls_output
        return proto_key, proto_values

    def get_final_proto(self, proto_key, proto_values, qry_cls):
        """
            qry_sam_cls: T(1,D)
        """
        alpha = F.softmax(F.linear(qry_cls, proto_key).mean(0))
        
        ## hard restriction
        all_slotnames = set()
        for proto in proto_values:
            all_slotnames.update(proto.keys())

        proto_repr = {}
        for a, proto in zip(alpha, proto_values):
            for k in all_slotnames:
                sl_repr = (proto[k] if k in proto.keys() 
                    else self.sltype_outputs.weight[self.slots_map.word2index[k]+1])
                sl_repr = sl_repr * a

                if k not in proto_repr:
                    proto_repr[k] = sl_repr
                else:
                    proto_repr[k] = proto_repr[k] + sl_repr

        proto_mask, proto_mat = [0], [self.sltype_outputs.weight[0]]
        for k in self.slots_map._vocab:
            if k in proto_repr:
                proto_mask.append(0)
                proto_mat.append(proto_repr[k])
            else:
                proto_mask.append(1)
                proto_mat.append(self.sltype_outputs.weight[self.slots_map.word2index[k]+1])
        
        proto_mask = torch.ByteTensor(proto_mask).cuda()
        proto_mat = torch.stack(proto_mat)
        return proto_mask, proto_mat

    def forward(self, padded_seqs, seq_lengths, segids, proto_key, proto_values):
        batch_size = padded_seqs.size(0)
        assert batch_size == 1

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
        
        proto_mask, proto_mat = self.get_final_proto(proto_key, proto_values, cls_output)

        # sltype_logits = self.slots_type_outputs(seq_output)
        sltype_logits = F.linear(seq_output, proto_mat, bias=self.sltype_outputs.bias)
        bio_logits = self.bio_outputs(seq_output)

        batch_size, seq_len, _ = sltype_logits.size()
        feats = torch.zeros(batch_size, seq_len, self.label_vocab.n_words).cuda()
        feats = (sltype_logits.index_select(-1, self.sltype_map)
                            + bio_logits.index_select(-1, self.bio_map))
        # feats = sltype_logits.gather(-1, self.sltype_map) + bio_logits.gather(-1, self.bio_map)

        return {"attn_mask": attn_mask, "dom_logits": dom_logits, 
                "int_logits": int_logits, "sl_logits": feats,
                "proto_mask": proto_mask }

    def compute_loss(self, dom_idx, int_idx, padded_y, fwd_dict):
        loss = (2 * self.loss_fct(fwd_dict["dom_logits"], dom_idx) 
                + 4 * self.loss_fct(fwd_dict["int_logits"], int_idx))

        seq_len = fwd_dict["sl_logits"].size(1)

        seq_logits, seq_mask = fwd_dict["sl_logits"][:,1:], fwd_dict["attn_mask"][:,1:]
        seq_label = padded_y[:,1:seq_len]

        sl_loss = - F.log_softmax(seq_logits, -1).gather(
                                2, seq_label.unsqueeze(-1)).squeeze(-1)
        loss = loss.sum() + sl_loss[seq_mask.byte()].sum()

        loss = loss + self.crf_layer.compute_loss(
                                seq_logits, seq_mask, seq_label).sum()
        return loss

    def predict(self, seq_lengths, dom_idx, fwd_dict):
        assert seq_lengths.size(0) == 1
        proto_mask = fwd_dict["proto_mask"]
        proto_mask = proto_mask.index_select(-1, self.sltype_map)
        for i_sam, i_dom in enumerate(dom_idx.tolist()):
            fwd_dict["int_logits"][i_sam].masked_fill_(self.dom_int_mask[i_dom], -1e9)
            fwd_dict["sl_logits"][i_sam].masked_fill_(self.dom_label_mask[i_dom], -1e9)
            fwd_dict["sl_logits"][i_sam].masked_fill_(proto_mask, -1e9)

        dom_pred = torch.argmax(fwd_dict["dom_logits"], dim=-1).detach().cpu().numpy()
        int_pred = torch.argmax(fwd_dict["int_logits"], dim=-1).detach().cpu().numpy()
        _, crf_pred = self.crf_layer.inference(
            fwd_dict["sl_logits"][:,1:], fwd_dict["attn_mask"][:,1:])
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
