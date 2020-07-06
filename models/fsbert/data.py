#coding: utf-8
"""
    @author: cjliux@gmail.com
"""
import os
import re
import sys
import json
import random
import numpy as np
import logging
logger = logging.getLogger()
import copy
import torch
import torch.utils.data as thdata
from collections import defaultdict
import sklearn.feature_extraction.text as skltext
from scipy import sparse

from utils.vocab import Vocab


def segment_text_and_label_seq(item, tokenizer):
    labeled = "slots" in item
    text = item["text"]
    val_dic = {}
    if labeled:
        for sl, vals in item["slots"].items():
            if not isinstance(vals, list):
                vals = [vals]
            for v in vals:
                text = re.sub(v, " " + v + " ", text)
                val_dic[v] = sl

    token, label = [], []
    for seg in text.split():
        lseg = list(seg)
        token.extend(lseg)
        
        if labeled:
            if seg in val_dic:
                label.extend(['B-' + val_dic[seg]] 
                    + ['I-' + val_dic[seg]] * (len(lseg) - 1))
            else:
                label.extend(['O'] * len(lseg))
        else:
            label.extend(['O'] * len(lseg))
    return token, label


def preprocess_item(item, tokenizer):
    tokens, l2_list = segment_text_and_label_seq(item, tokenizer)
    item['token'], item['label'] = tokens, l2_list

    l1_list = []
    for l in l2_list:
        if "B" in l:
            l1_list.append("B")
        elif "I" in l:
            l1_list.append("I")
        else:
            l1_list.append("O")
    item["bin_label"] = l1_list

    return item


def read_data(filepath, tokenizer):
    with open(filepath, "r", encoding='utf8') as fd:
        json_data = json.load(fd)
    data = [preprocess_item(item, tokenizer) for item in json_data]
    return data


def binarize_data(data, tokenizer, 
        domain_map, intent_map, slots_map, 
        label_voc, bin_label_voc):
    new_data = []
    for item in data:
        item['token_ids'] = tokenizer.convert_token_to_ids(item["token"])
        item['blabel_ids'] = [bin_label_voc.word2index[tok] for tok in item["bin_label"]]
        item['label_ids'] = [label_voc.word2index[tok] for tok in item["label"]]
        
        item['dom_idx'] = domain_map.word2index[item['domain']]
        if "intent" in item:
            item['int_idx'] = intent_map.word2index[item['intent']]

        new_data.append(item)
    return new_data


def read_all_train_data(
        train_data_file, tokenizer,
        domain_map, intent_map, slots_map, 
        label_voc, bin_label_voc):
    logger.info("Loading and processing train data ...")

    train_data = read_data(train_data_file, tokenizer)
    train_dom_data = defaultdict(list)
    for item in train_data:
        train_dom_data[item['domain']].append(item)

    # binarize data
    data = {}
    for dom, dom_data in train_dom_data.items():
        data[dom] = binarize_data(dom_data, tokenizer,
            domain_map, intent_map, slots_map, label_voc, bin_label_voc)
    return data


def read_dev_support_and_query_data(
        dev_data_path, tokenizer,
        domain_map, intent_map, slots_map, 
        label_voc, bin_label_voc):
    logger.info("Loading and processing dev data ...")

    sup_dir = os.path.join(dev_data_path, "support")
    sup_files = [f for f in os.listdir(sup_dir) if f.startswith("support")]
    sup_dom_data, qry_dom_data = {}, {}
    for sup_file in sup_files:
        i_dom = sup_file[sup_file.index('_')+1:sup_file.rindex('.')]
        dom_data = read_data(
            os.path.join(sup_dir, sup_file), tokenizer)
        sup_dom_data[i_dom] = binarize_data(dom_data, tokenizer,
            domain_map, intent_map, slots_map, label_voc, bin_label_voc)

        dom_data = read_data(
            os.path.join(dev_data_path, "test", "test_{}.json".format(i_dom)),
            tokenizer)
        qry_dom_data[i_dom] = binarize_data(dom_data, tokenizer, 
            domain_map, intent_map, slots_map, label_voc, bin_label_voc)

    return sup_dom_data, qry_dom_data


def separate_data_to_support_and_query(dom_data, sup_size):
    return separate_data_to_support_and_query_v2(dom_data, sup_size)


def separate_data_to_support_and_query_v1(dom_data, sup_size):
    # sup_data, qry_data = [], []
    sup_data = dom_data[:sup_size]
    qry_data = dom_data[sup_size:]
    return sup_data, qry_data


def separate_data_to_support_and_query_v2(dom_data, sup_size):
    int_set, sl_set = set(), set()
    for item in dom_data:
        int_set.add(item["intent"])
        sl_set.update(item["slots"].keys())
    
    assert len(dom_data) > sup_size
    sup_data, qry_data = [], []
    
    data_pool = dom_data
    while True:
        cur_ints, cur_sls = set(), set()
        new_data_pool = []
        for i, item in enumerate(data_pool):
            if (len(sup_data) >= sup_size or
                    len(cur_ints) == len(int_set) and len(cur_sls) == len(sl_set)):
                new_data_pool.extend(data_pool[i:])
                break

            flag = 0 # overlap
            if item["intent"] not in cur_ints:
                flag = 1 # new intent
            else:
                for sl in item["slots"].keys():
                    if sl not in cur_sls:
                        flag = 2 # new slot
                        break

            if len(sup_data) < sup_size and flag > 0:
                sup_data.append(item)
                cur_ints.add(item["intent"])
                cur_sls.update(item["slots"].keys())
            else:
                new_data_pool.append(item) 

        if len(sup_data) >= sup_size:
            qry_data = new_data_pool
            break
        else:
            data_pool = new_data_pool
            int_set, sl_set = set(), set()
            for item in data_pool:
                int_set.add(item["intent"])
                sl_set.update(item["slots"].keys())

    assert len(sup_data) + len(qry_data) == len(dom_data)
    return sup_data, qry_data


def collect_support_instances(sup_data, qry_data, n_shots, is_same=False):
    """
        tfidf based support collection
    """
    tfidf_vec = skltext.TfidfVectorizer(
        ngram_range=(1,2), use_idf=True, smooth_idf=True,
        sublinear_tf = True)

    all_texts = [it['text'] for it in sup_data] + [it['text'] for it in qry_data]
    all_feats = tfidf_vec.fit_transform(all_texts).todense()
    sup_feats, qry_feats = all_feats[:len(sup_data)], all_feats[len(sup_data):]
    sup_feats, qry_feats = torch.Tensor(sup_feats).cuda(), torch.Tensor(qry_feats).cuda()
    sim_rank = torch.matmul(qry_feats, sup_feats.t()) / (
                        qry_feats.norm(p=2, dim=1).unsqueeze(1) 
                            * sup_feats.norm(p=2, dim=1).unsqueeze(0))
    if is_same:
        sim_rank.masked_fill_(torch.eye(sim_rank.size(0)).byte().cuda(), -1e6)
    sup_topk = torch.topk(sim_rank, n_shots, dim=1)[1]

    fs_data = []
    for item, sup_insts in zip(qry_data, sup_topk):
        if is_same:
            item = copy.deepcopy(item)
        item['support'] = [sup_data[i] for i in sup_insts]
        fs_data.append(item)
        
    return fs_data


class Dataset(thdata.Dataset):

    def __init__(self, qry_data):
        super().__init__()
        # self.sup_data = sup_data
        self.qry_data = qry_data
        # self.n_shots = n_shots

    def __len__(self):
        return len(self.qry_data)

    def __getitem__(self, index):
        # sup_set = (self.sup_data if self.n_shots is None 
        #     else random.sample(self.sup_data, self.n_shots))
        qry_inst = self.qry_data[index]
        return qry_inst


def collate_fn(batch, PAD_INDEX=0):
    batch_size = len(batch)

    new_batch = {k:[] for k in batch[0].keys()}
    for item in batch:
        for k in new_batch.keys():
            new_batch[k].append(item[k])
    batch = new_batch

    batch["model_input"] = {} # tensorized

    def pad_and_batch(in_seqs):
        lengths = [len(bs_x) for bs_x in in_seqs]
        max_lengths = max(lengths)
        padded_seqs = torch.LongTensor(batch_size, max_lengths).fill_(PAD_INDEX)
        for i, seq in enumerate(in_seqs):
            padded_seqs[i, :lengths[i]] = torch.LongTensor(seq)
        lengths = torch.LongTensor(lengths)
        return padded_seqs, lengths

    padded_seqs, lengths = pad_and_batch(batch["token_ids"])
    batch["model_input"]["padded_seqs"] = padded_seqs
    batch["model_input"]["seq_lengths"] = lengths

    padded_y, _ = pad_and_batch(batch["label_ids"])
    batch["model_input"]["padded_y"] = padded_y

    batch["model_input"]["dom_idx"] = torch.LongTensor(batch["dom_idx"])
    if "int_idx" in batch:
        batch["model_input"]["int_idx"] = torch.LongTensor(batch["int_idx"])

    # recursive
    if "support" in batch:
        new_supports = []
        for sup_batch in batch["support"]:
            new_supports.append(collate_fn(sup_batch))
        batch["support"] = new_supports

    return batch


def get_dataloader_for_fs_train(
        data_path, raw_data_path, eval_domains: list, 
        batch_size, n_shots, tokenizer):
    domain_map = Vocab.from_file(os.path.join(data_path, "domains.txt"))
    intent_map = Vocab.from_file(os.path.join(data_path, "intents.txt"))
    slots_map = Vocab.from_file(os.path.join(data_path, "slots.txt"))
    label_vocab = Vocab.from_file(os.path.join(data_path, "label_vocab.txt"))
    bin_label_vocab = Vocab.from_file(os.path.join(data_path, "bin_label_vocab.txt"))

    # train 
    all_train_data = read_all_train_data(
        os.path.join(raw_data_path, "source.json"), tokenizer,
        domain_map, intent_map, slots_map, label_vocab, bin_label_vocab)
    data = {k:v for k,v in all_train_data.items() if k not in eval_domains}
    
    ## wrap dataset
    fs_data = []
    for dom, dom_data in data.items():
        sup_data, qry_data = separate_data_to_support_and_query(dom_data, 2*int(n_shots))
        dom_data = collect_support_instances(sup_data, qry_data, int(n_shots))
        fs_data.extend(dom_data)

    dataloader = thdata.DataLoader(dataset=Dataset(fs_data), 
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


def get_dataloader_for_fs_eval(
        data_path, raw_data_path, eval_domains: list,
        batch_size, n_shots, tokenizer):
    domain_map = Vocab.from_file(os.path.join(data_path, "domains.txt"))
    intent_map = Vocab.from_file(os.path.join(data_path, "intents.txt"))
    slots_map = Vocab.from_file(os.path.join(data_path, "slots.txt"))
    label_vocab = Vocab.from_file(os.path.join(data_path, "label_vocab.txt"))
    bin_label_vocab = Vocab.from_file(os.path.join(data_path, "bin_label_vocab.txt"))

    # train 
    all_train_data = read_all_train_data(
        os.path.join(raw_data_path, "source.json"), tokenizer,
        domain_map, intent_map, slots_map, label_vocab, bin_label_vocab)
    data = {k:v for k,v in all_train_data.items() if k in eval_domains}

    # eval support & query
    fs_data = []
    for dom, dom_data in data.items():
        sup_data, qry_data = separate_data_to_support_and_query(dom_data, 2*int(n_shots))
        dom_data = collect_support_instances(sup_data, qry_data, int(n_shots))
        fs_data.extend(dom_data)

    dataloader = thdata.DataLoader(dataset=Dataset(fs_data), 
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader


def get_dataloader_for_fs_test(
        data_path, raw_data_path,
        batch_size, n_shots, tokenizer, sep_dom=False):
    domain_map = Vocab.from_file(os.path.join(data_path, "domains.txt"))
    intent_map = Vocab.from_file(os.path.join(data_path, "intents.txt"))
    slots_map = Vocab.from_file(os.path.join(data_path, "slots.txt"))
    label_vocab = Vocab.from_file(os.path.join(data_path, "label_vocab.txt"))
    bin_label_vocab = Vocab.from_file(os.path.join(data_path, "bin_label_vocab.txt"))

    ## dev support & query
    dev_sup_dom_data, dev_qry_dom_data = read_dev_support_and_query_data(
        os.path.join(raw_data_path, "dev"), tokenizer,
        domain_map, intent_map, slots_map, label_vocab, bin_label_vocab)

    if not sep_dom:
        fs_data = []
        for dom in dev_sup_dom_data.keys():
            dom_data = collect_support_instances(
                dev_sup_dom_data[dom], dev_qry_dom_data[dom], int(n_shots))
            fs_data.extend(dom_data)

        dataloader = thdata.DataLoader(dataset=Dataset(fs_data), 
            batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return dataloader
    else:
        dataloaders = {}
        for dom in dev_sup_dom_data.keys():
            dom_data = collect_support_instances(
                dev_sup_dom_data[dom], dev_qry_dom_data[dom], int(n_shots))
            
            dataloaders[dom] = thdata.DataLoader(dataset=Dataset(dom_data), 
                batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return dataloaders


if __name__=='__main__':
    pass
