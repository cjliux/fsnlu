#coding: utf-8
import os
import sys
import json
import numpy as np
import logging
logger = logging.getLogger()

import torch
import torch.utils.data as data

from utils.vocab import Vocab


def read_file(filepath, use_label_encoder=False, slot_list=None):
    utter_list, y1_list, y2_list = [], [], []
    if use_label_encoder:
        template_list = []
    with open(filepath, "r", encoding='utf8') as f:
        for i, line in enumerate(f):
            line = line.strip()  # text \t label
            splits = line.split("\t")
            tokens = splits[0].split()
            l2_list = splits[1].split()

            utter_list.append(tokens)
            y2_list.append(l2_list)

            l1_list = []
            for l in l2_list:
                if "B" in l:
                    l1_list.append("B")
                elif "I" in l:
                    l1_list.append("I")
                else:
                    l1_list.append("O")
            y1_list.append(l1_list)

            if use_label_encoder:
                """
                template_each_sample[0] is correct template
                template_each_sample[1] and template_each_sample[2] are incorrect template (replace correct slots with other slots)
                """
                template_each_sample = [[],[],[]]
                assert len(tokens) == len(l2_list)
                for token, l2 in zip(tokens, l2_list):
                    if "I" in l2: continue
                    if l2 == "O":
                        template_each_sample[0].append(token)
                        template_each_sample[1].append(token)
                        template_each_sample[2].append(token)
                    else:
                        # "B" in l2
                        slot_name = l2.split("-")[1]
                        template_each_sample[0].append(slot_name)
                        np.random.shuffle(slot_list)
                        idx = 0
                        for j in range(1, 3):  # j from 1 to 2
                            if slot_list[idx] != slot_name:
                                template_each_sample[j].append(slot_list[idx])
                            else:
                                idx = idx + 1
                                template_each_sample[j].append(slot_list[idx])
                            idx = idx + 1
                        
                assert len(template_each_sample[0]) == len(template_each_sample[1]) == len(template_each_sample[2])

                template_list.append(template_each_sample)

    if use_label_encoder:
        data_dict = {"utter": utter_list, "y1": y1_list, "y2": y2_list, "template_list": template_list}
    else:
        data_dict = {"utter": utter_list, "y1": y1_list, "y2": y2_list}
    
    return data_dict


def binarize_data(data, vocab, dom, use_label_encoder, domains, label_voc, simp_label_voc):
    if use_label_encoder:
        data_bin = {"utter": [], "y1": [], "y2": [], "domains": [], "template_list": []}
    else:
        data_bin = {"utter": [], "y1": [], "y2": [], "domains": []}
    assert len(data_bin["utter"]) == len(data_bin["y1"]) == len(data_bin["y2"])
    
    dom_idx = domains.word2index[dom]
    for utter_tokens, y1_list, y2_list in zip(data["utter"], data["y1"], data["y2"]):
        utter_bin, y1_bin, y2_bin = [], [], []
        # binarize utterence
        for token in utter_tokens:
            utter_bin.append(vocab.word2index[token])
        data_bin["utter"].append(utter_bin)
        # binarize y1
        for y1 in y1_list:
            y1_bin.append(simp_label_voc.word2index[y1])
        data_bin["y1"].append(y1_bin)
        # binarize y2
        for y2 in y2_list:
            y2_bin.append(label_voc.word2index[y2])
        data_bin["y2"].append(y2_bin)
        assert len(utter_bin) == len(y1_bin) == len(y2_bin)
        
        data_bin["domains"].append(dom_idx)
    
    if use_label_encoder:
        for template_each_sample in data["template_list"]:
            template_each_sample_bin = [[],[],[]]
            
            for tok1, tok2, tok3 in zip(template_each_sample[0], template_each_sample[1], template_each_sample[2]):
                template_each_sample_bin[0].append(vocab.word2index[tok1])
                template_each_sample_bin[1].append(vocab.word2index[tok2])
                template_each_sample_bin[2].append(vocab.word2index[tok3])
            data_bin["template_list"].append(template_each_sample_bin)

    return data_bin


def read_all_train_data_and_binarize(
        train_dom_data_path, slot_list, vocab, 
        domain_set, label_voc, simp_label_voc, use_label_encoder=False):
    logger.info("Loading and processing train data ...")

    # load data
    train_domain_data = {}
    # train_dom_data_path = os.path.join(data_path, "train_dom_data")
    train_domains = os.listdir(train_dom_data_path)
    for domain in train_domains:
        train_domain_data[domain] = read_file(
            os.path.join(train_dom_data_path, "{}/{}.txt".format(domain, domain)), 
            use_label_encoder, slot_list.get_vocab())
    
    # binarize data
    data = {}
    for domain, dom_data in train_domain_data.items():
        data[domain] = binarize_data(dom_data, vocab, domain, 
            use_label_encoder, domain_set, label_voc, simp_label_voc)

    return train_domain_data, data


def read_dev_support_and_query_data(
        dev_dom_data_path, slot_list, domain, vocab, 
        domain_set, label_voc, simp_label_voc, use_label_encoder=False):
    logger.info("Loading and processing dev data ...")

    raw_sup_data = read_file(
        os.path.join(dev_dom_data_path, "{}/support.txt".format(domain)),
        use_label_encoder, slot_list.get_vocab())
    sup_data = binarize_data(raw_sup_data, vocab, domain,
        use_label_encoder, domain_set, label_voc, simp_label_voc)

    raw_qry_data = read_file(
        os.path.join(dev_dom_data_path, "{}/query.txt".format(domain)),
        use_label_encoder, slot_list.get_vocab())
    qry_data = binarize_data(raw_qry_data, vocab, domain,
        use_label_encoder, domain_set, label_voc, simp_label_voc)

    # dev_data = { "support": sup_data, "query": qry_data }
    return raw_sup_data, sup_data, raw_qry_data, qry_data


class Dataset(data.Dataset):
    def __init__(self, X, y1, y2, domains, template_list=None, id_list=None):
        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.domains = domains
        self.template_list = template_list
        self.id_list = id_list

    def __getitem__(self, index):
        if self.template_list is not None:
            return self.X[index], self.y1[index], self.y2[index], self.domains[index], self.template_list[index]
        else:
            return self.X[index], self.y1[index], self.y2[index], self.domains[index]
    
    def __len__(self):
        return len(self.X)


def collate_fn_for_label_encoder(data, PAD_INDEX=0):
    X, y1, y2, domains, templates = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    domains = torch.LongTensor(domains)
    
    tem_lengths = [len(sample_tem[0]) for sample_tem in templates]
    max_tem_len = max(tem_lengths)
    padded_templates = torch.LongTensor(len(templates), 3, max_tem_len).fill_(PAD_INDEX)
    for j, sample_tem in enumerate(templates):
        length = tem_lengths[j]
        padded_templates[j, 0, :length] = torch.LongTensor(sample_tem[0])
        padded_templates[j, 1, :length] = torch.LongTensor(sample_tem[1])
        padded_templates[j, 2, :length] = torch.LongTensor(sample_tem[2])
    tem_lengths = torch.LongTensor(tem_lengths)
    
    return padded_seqs, lengths, y1, y2, domains, padded_templates, tem_lengths


def collate_fn(data, PAD_INDEX=0):
    X, y1, y2, domains = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    domains = torch.LongTensor(domains)
    
    return padded_seqs, lengths, y1, y2, domains


def get_dataloader(data_path, eval_domains: list, tgt_domains: list, 
        batch_size, use_label_encoder, n_shots):
    domain_set = Vocab.from_file(os.path.join(data_path, "domains.txt"))
    slot_list = Vocab.from_file(os.path.join(data_path, "slots.txt"))
    vocab = Vocab.from_file(os.path.join(data_path, "token_vocab.txt"))
    label_vocab = Vocab.from_file(os.path.join(data_path, "label_vocab.txt"))
    simp_label_vocab = Vocab.from_file(os.path.join(data_path, "bin_label_vocab.txt"))

    _, all_train_data = read_all_train_data_and_binarize(
        os.path.join(data_path, "train_dom_data"), 
        slot_list, vocab, domain_set, 
        label_vocab, simp_label_vocab, use_label_encoder)
    if use_label_encoder:
        train_data = {"utter": [], "y1": [], "y2": [], "domains": [], "template_list": []}
    else:
        train_data = {"utter": [], "y1": [], "y2": [], "domains": []}
    
    # train 
    for dm_name, dm_data in all_train_data.items():
        if dm_name not in eval_domains:
            train_data["utter"].extend(dm_data["utter"])
            train_data["y1"].extend(dm_data["y1"])
            train_data["y2"].extend(dm_data["y2"])
            train_data["domains"].extend(dm_data["domains"])

            if use_label_encoder:
                train_data["template_list"].extend(dm_data["template_list"])

    # eval support & query
    val_data = {"utter": [], "y1": [], "y2": [], "domains": []}
    if n_shots >= 1:
        n_shots = int(n_shots)
        for tgt_domain in eval_domains:
            train_data["utter"].extend(all_train_data[tgt_domain]["utter"][:n_shots])
            train_data["y1"].extend(all_train_data[tgt_domain]["y1"][:n_shots])
            train_data["y2"].extend(all_train_data[tgt_domain]["y2"][:n_shots])
            train_data["domains"].extend(all_train_data[tgt_domain]["domains"][:n_shots])

            if use_label_encoder:
                train_data["template_list"].extend(all_train_data[tgt_domain]["template_list"][:n_shots])

            val_data["utter"].extend(all_train_data[tgt_domain]["utter"][n_shots:])
            val_data["y1"].extend(all_train_data[tgt_domain]["y1"][n_shots:])
            val_data["y2"].extend(all_train_data[tgt_domain]["y2"][n_shots:])
            val_data["domains"].extend(all_train_data[tgt_domain]["domains"][n_shots:])
    else:
        for tgt_domain in eval_domains:
            n_sup = int(len(all_train_data[tgt_domain]["utter"]) * n_shots)
            train_data["utter"].extend(all_train_data[tgt_domain]["utter"][:n_sup])
            train_data["y1"].extend(all_train_data[tgt_domain]["y1"][:n_sup])
            train_data["y2"].extend(all_train_data[tgt_domain]["y2"][:n_sup])
            train_data["domains"].extend(all_train_data[tgt_domain]["domains"][:n_sup])

            if use_label_encoder:
                train_data["template_list"].extend(all_train_data[tgt_domain]["template_list"][:n_sup])

            val_data["utter"].extend(all_train_data[tgt_domain]["utter"][n_sup:])
            val_data["y1"].extend(all_train_data[tgt_domain]["y1"][n_sup:])
            val_data["y2"].extend(all_train_data[tgt_domain]["y2"][n_sup:])
            val_data["domains"].extend(all_train_data[tgt_domain]["domains"][n_sup:])

    ## dev support & query
    dev_sup_data, dev_qry_data = {}, {}
    if "all" in tgt_domains:
        tgt_domains = os.listdir(os.path.join(data_path, "dev_dom_data"))
    for tgt_domain in tgt_domains:
        _, sup_data, _, qry_data = read_dev_support_and_query_data(
            os.path.join(data_path, "dev_dom_data"), slot_list, tgt_domain, vocab,
            domain_set, label_vocab, simp_label_vocab, use_label_encoder)
        dev_sup_data[tgt_domain] = sup_data
        dev_qry_data[tgt_domain] = qry_data

    test_data = {"utter": [], "y1": [], "y2": [], "domains": []}
    for tgt_domain in tgt_domains:
        if n_shots >= 1:
            n_shots = int(n_shots)
            train_data["utter"].extend(dev_sup_data[tgt_domain]["utter"][:n_shots])
            train_data["y1"].extend(dev_sup_data[tgt_domain]["y1"][:n_shots])
            train_data["y2"].extend(dev_sup_data[tgt_domain]["y2"][:n_shots])
            train_data["domains"].extend(dev_sup_data[tgt_domain]["domains"][:n_shots])
        else:
            train_data["utter"].extend(dev_sup_data[tgt_domain]["utter"])
            train_data["y1"].extend(dev_sup_data[tgt_domain]["y1"])
            train_data["y2"].extend(dev_sup_data[tgt_domain]["y2"])
            train_data["domains"].extend(dev_sup_data[tgt_domain]["domains"])

        if use_label_encoder:
            train_data["template_list"].extend(dev_sup_data[tgt_domain]["template_list"][:n_shots])

        test_data["utter"].extend(dev_qry_data[tgt_domain]["utter"])
        test_data["y1"].extend(dev_qry_data[tgt_domain]["y1"])
        test_data["y2"].extend(dev_qry_data[tgt_domain]["y2"])
        test_data["domains"].extend(dev_qry_data[tgt_domain]["domains"])
    
    if use_label_encoder:
        dataset_tr = Dataset(train_data["utter"], train_data["y1"], train_data["y2"], train_data["domains"], train_data["template_list"])
    else:
        dataset_tr = Dataset(train_data["utter"], train_data["y1"], train_data["y2"], train_data["domains"])
    dataset_val = Dataset(val_data["utter"], val_data["y1"], val_data["y2"], val_data["domains"])
    dataset_test = Dataset(test_data["utter"], test_data["y1"], test_data["y2"], test_data["domains"])

    dataloader_tr = data.DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn_for_label_encoder if use_label_encoder else collate_fn)
    dataloader_val = data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn)
    dataloader_test = data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn)
    # dataloader_test = None

    return dataloader_tr, dataloader_val, dataloader_test
    

###########
# for test
###########
class TestDataset(data.Dataset):
    def __init__(self, X, domains, raw_domains, raw_text, id_list):
        self.X = X
        self.domains = domains
        self.raw_domains = raw_domains
        self.raw_text = raw_text
        self.id_list = id_list

    def __getitem__(self, index):
        # X, domains, raw_domains, raw_text, ids
        return self.X[index], self.domains[index], self.raw_domains[index], self.raw_text[index], self.id_list[index]
    
    def __len__(self):
        return len(self.X)


def read_file_for_test_only(filepath, use_label_encoder=False, slot_list=None):
    with open(filepath, "r", encoding='utf8') as f:
        data_json = json.load(f)
    
    utter_list = []
    raw_text_list, id_list = [], []
    for i, item in enumerate(data_json):
        line = item["text"].strip()  # text \t label
        tokens = list(line)
        utter_list.append(tokens)

        raw_text_list.append(item["text"])
        if "id" in item:
            id_list.append(item["id"])

    data_dict = {"utter": utter_list, "raw_text": raw_text_list, "id": id_list}
    return data_dict


def binarize_data_for_test_only(data, vocab, dom, use_label_encoder, domains, label_voc, simp_label_voc):
    # if use_label_encoder:
    #     data_bin = {"utter": [], "y1": [], "y2": [], "domains": [], "template_list": []}
    # else:
    #     data_bin = {"utter": [], "y1": [], "y2": [], "domains": []}
    data_bin = {"utter": [], "raw_text": [], 
        "domains": [], "raw_domains": [], "id": []}
    
    dom_idx = domains.word2index[dom]
    for utter_tokens, raw_text, uid in zip(data["utter"], data["raw_text"], data["id"]):
        utter_bin, y1_bin, y2_bin = [], [], []
        # binarize utterence
        for token in utter_tokens:
            utter_bin.append(vocab.word2index[token])
        data_bin["utter"].append(utter_bin)
        data_bin["raw_text"].append(raw_text)
        
        data_bin["domains"].append(dom_idx)
        data_bin["raw_domains"].append(dom)

        data_bin["id"].append(uid)

    return data_bin


def read_test_support_and_query_data(
        dev_dom_data_path, slot_list, domain, vocab, 
        domain_set, label_voc, simp_label_voc, use_label_encoder=False):
    logger.info("Loading and processing dev data ...")

    raw_sup_data = read_file_for_test_only(
        os.path.join(dev_dom_data_path, "{}/support.json".format(domain)),
        use_label_encoder, slot_list.get_vocab())
    sup_data = binarize_data_for_test_only(raw_sup_data, vocab, domain,
        use_label_encoder, domain_set, label_voc, simp_label_voc)

    raw_qry_data = read_file_for_test_only(
        os.path.join(dev_dom_data_path, "{}/query.json".format(domain)),
        use_label_encoder, slot_list.get_vocab())
    qry_data = binarize_data_for_test_only(raw_qry_data, vocab, domain,
        use_label_encoder, domain_set, label_voc, simp_label_voc)

    # dev_data = { "support": sup_data, "query": qry_data }
    return raw_sup_data, sup_data, raw_qry_data, qry_data


def collate_fn_for_test(data, PAD_INDEX=0):
    X, domains, raw_domains, raw_text, ids = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    domains = torch.LongTensor(domains)
    
    return padded_seqs, lengths, domains, raw_domains, raw_text, ids


def get_dataloader_for_test_only(data_path, tgt_domains: list, 
        batch_size, use_label_encoder, n_shots):
    domain_set = Vocab.from_file(os.path.join(data_path, "domains.txt"))
    slot_list = Vocab.from_file(os.path.join(data_path, "slots.txt"))
    vocab = Vocab.from_file(os.path.join(data_path, "token_vocab.txt"))
    label_vocab = Vocab.from_file(os.path.join(data_path, "label_vocab.txt"))
    simp_label_vocab = Vocab.from_file(os.path.join(data_path, "bin_label_vocab.txt"))

    if "all" in tgt_domains:
        tgt_domains = os.listdir(os.path.join(data_path, "dev_dom_data"))

    ## dev support & query
    dev_sup_data, dev_qry_data = {}, {}
    for tgt_domain in tgt_domains:
        _, sup_data, _, qry_data = read_test_support_and_query_data(
            os.path.join(data_path, "dev_dom_data"), slot_list, tgt_domain, vocab,
            domain_set, label_vocab, simp_label_vocab, use_label_encoder)
        dev_sup_data[tgt_domain] = sup_data
        dev_qry_data[tgt_domain] = qry_data

    # sup_data = {"utter": [], "y1": [], "y2": [], "domains": []}
    qry_data = {"utter": [], "raw_text": [], 
        "domains": [], "raw_domains": [], "id": []}
    for tgt_domain in tgt_domains:
        # sup_data["utter"].extend(dev_sup_data[tgt_domain]["utter"][:n_shots])
        # sup_data["y1"].extend(dev_sup_data[tgt_domain]["y1"][:n_shots])
        # sup_data["y2"].extend(dev_sup_data[tgt_domain]["y2"][:n_shots])
        # sup_data["domains"].extend(dev_sup_data[tgt_domain]["domains"][:n_shots])
        # if use_label_encoder:
        #     sup_data["template_list"].extend(dev_sup_data[tgt_domain]["template_list"][:n_shots])

        qry_data["utter"].extend(dev_qry_data[tgt_domain]["utter"])
        qry_data["raw_text"].extend(dev_qry_data[tgt_domain]["raw_text"])
        qry_data["domains"].extend(dev_qry_data[tgt_domain]["domains"])
        qry_data["raw_domains"].extend(dev_qry_data[tgt_domain]["raw_domains"])
        qry_data["id"].extend(dev_qry_data[tgt_domain]["id"])
    
    dataset_test = TestDataset(qry_data["utter"], 
        qry_data["domains"], qry_data["raw_domains"], qry_data["raw_text"], qry_data["id"])
    dataloader_test = data.DataLoader(dataset=dataset_test, 
            batch_size=batch_size, shuffle=False, collate_fn=collate_fn_for_test)
    return dev_qry_data, dataloader_test


if __name__=='__main__':
    pass
