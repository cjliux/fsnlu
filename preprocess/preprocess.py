#coding: utf-8
"""
    @author: cjliux@gmail.com
    borrowed heavily from https://github.com/zliucr/coach
"""
import os
import re
import sys
import json
import ipdb
# import pandas as pd
from collections import defaultdict, Counter


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, 
        default="../data/smp2020ecdt/smp2020ecdt_task1_v2")
    parser.add_argument("--save_path", type=str,
        default="./data")
    parser.add_argument("--key", type=str, default="default")
    return vars(parser.parse_args())


def segment_text_and_label_seq(json_data, labeled=True):
    new_json_data = []
    for item in json_data:
        text = item["text"]
        val_dic = {}
        if labeled:
            for sl, val in item["slots"].items():
                if not isinstance(val, list):
                    val = [val]
                for v in val:
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
        item['token'] = token
        item['label'] = label

        new_json_data.append(item)
    return new_json_data


def read_data(data_path):
    with open(os.path.join(data_path, "source.json"), 'r', encoding='utf8') as fd:
        train_json = json.load(fd)
    train_json = segment_text_and_label_seq(train_json)

    support_dir = os.path.join(data_path, "dev/support")
    test_dir = os.path.join(data_path, "dev/test")
    
    dev_support_dom_json, dev_test_dom_json = {}, {}
    for sup_file in os.listdir(support_dir):
        if not sup_file.startswith('support_'): 
            continue
        i_dom = sup_file[sup_file.rindex('_')+1:sup_file.rindex('.')]

        sup_file = os.path.join(support_dir, sup_file)
        with open(sup_file, "r", encoding="utf8") as fd:
            sup_data = json.load(fd)
        sup_data = segment_text_and_label_seq(sup_data)

        # ipdb.set_trace()
        tst_file = os.path.join(test_dir, "test_{}.json".format(i_dom))
        with open(tst_file, "r", encoding="utf8") as fd:
            tst_data = json.load(fd)
        tst_data = segment_text_and_label_seq(tst_data, labeled=False)

        dom = sup_data[0]['domain']
        dev_support_dom_json[dom] = sup_data
        dev_test_dom_json[dom] = tst_data

    return train_json, dev_support_dom_json, dev_test_dom_json


def get_vocab_and_labels(train_json, dev_support_dom_json, dev_test_dom_json):
    dom2slots = defaultdict(set)
    dom2intents, intent2slots = defaultdict(set), defaultdict(set)
    domint2slots = defaultdict(lambda:defaultdict(set))

    wfr, lfr = Counter(), Counter()
    for item in train_json:
        wfr += Counter(item['token'])
        lfr += Counter(item['label'])
        dom2slots[item['domain']].update(set(item['slots'].keys()))
        dom2intents[item['domain']].add(item['intent'])
        intent2slots[item['intent']].update(set(item['slots'].keys()))
        domint2slots[item['domain']][item['intent']].update(set(item['slots'].keys()))

    for dom in dev_support_dom_json:
        for item in dev_support_dom_json[dom]:
            wfr += Counter(item['token'])
            lfr += Counter(item['label'])
            dom2slots[item['domain']].update(set(item['slots'].keys()))
            dom2intents[item['domain']].add(item['intent'])
            intent2slots[item['intent']].update(set(item['slots'].keys()))
            domint2slots[item['domain']][item['intent']].update(set(item['slots'].keys()))
        
        for item in dev_test_dom_json[dom]:
            wfr += Counter(item['token'])

    # print(wfr)
    # print(lfr)

    dom2slots = {k:list(sorted(v)) for k, v in dom2slots.items()}
    slot2desc = {}
    for sls in dom2slots.values():
        for sl in sls:
            slot2desc[sl] = sl
    dom2intents = {k:list(sorted(v)) for k, v in dom2intents.items()}
    intent2slots = {k:list(sorted(v)) for k, v in intent2slots.items()}
    domint2slots = {k: {kk:list(sorted(vv)) for kk, vv in v.items()} for k, v in domint2slots.items()}

    token_vocab = list(sorted(wfr.keys(), key=lambda x: wfr[x], reverse=True))

    token_vocab = ["PAD", "UNK"] + list(sorted(slot2desc.keys())) + token_vocab
    label_vocab = sorted(lfr.keys(), key=lambda x: x[::-1])
    # print(token_vocab)
    # print(label_vocab)
    print(len(token_vocab), len(label_vocab))
    return token_vocab, label_vocab, { "dom2slots": dom2slots, 
        "slot2desc": slot2desc, "dom2intents": dom2intents, "intent2slots": intent2slots,
        "domint2slots": domint2slots}


def separate_train_domains(train_json):
    train_dom_json = defaultdict(list)
    for item in train_json:
        train_dom_json[item['domain']].append(item)
    return train_dom_json


def convert_to_support_and_test(train_json, n_shot=7):
    support = train_json[:n_shot]
    test = train_json[n_shot:]
    return support, test


def batch_support_and_test(support_set, test_set):
    fs_set = []
    for titem in test_set:
        fs_item = {}
        fs_item['support'] = support_set
        fs_item['test'] = titem
        fs_set.append(fs_item)
    return fs_set


def write_json_data(json_data, save_file):
    with open(save_file, 'w', encoding='utf8') as fd:
        json.dump(json_data, fd, indent=2, ensure_ascii=False)


def write_coach_format(json_data, save_file):
    with open(save_file, 'w', encoding='utf8') as fd:
        lines = []
        for item in json_data:
            lines.append(' '.join(item['token']) 
                        + '\t' + ' '.join(item['label']))
        fd.write('\n'.join(lines))


def write_vocab(vocab, save_file):
    with open(save_file, 'w', encoding='utf8') as fd:
        fd.write('\n'.join(vocab))


def gen_seen_and_unseen(train_dom_json):
    dom2slots = defaultdict(set)
    for dom, dom_data in train_dom_json.items():
        for item in dom_data:
            for sl, val in item["slots"].items():
                dom2slots[dom].add(sl)
    
    seen_dom, unseen_dom = defaultdict(set), {}
    for dom_i in dom2slots.keys():
        for dom_j in dom2slots.keys():
            if dom_j != dom_i:
                seen_dom[dom_i].update(dom2slots[dom_j])
        unseen_dom[dom_i] = dom2slots[dom_i].difference(seen_dom[dom_i])
    
    seen_dom_json, unseen_dom_json = defaultdict(list), defaultdict(list)
    for dom, dom_data in train_dom_json.items():
        seen_cnt, unseen_cnt = 0, 0
        for item in dom_data:
            is_unseen = False
            for sl in item["slots"].keys():
                if sl in unseen_dom[dom]:
                    is_unseen = True
                    break
            if is_unseen:
                unseen_dom_json[dom].append(item)
                unseen_cnt += 1
            else:
                seen_dom_json[dom].append(item)
                seen_cnt += 1
        print("domain: {}: seen: {}; unseen: {}; total: {}".format(
                    dom, seen_cnt, unseen_cnt, seen_cnt+unseen_cnt))
    return seen_dom_json, unseen_dom_json


def preprocess_and_save(args):
    train_json, dev_support_dom_json, dev_test_dom_json = read_data(args["data_path"])
    save_dir = os.path.join(args["save_path"], args["key"])

    # vocab
    token_vocab, label_vocab, mappings = get_vocab_and_labels(
            train_json, dev_support_dom_json, dev_test_dom_json)
    write_vocab(token_vocab, os.path.join(save_dir, "token_vocab.txt"))
    write_vocab(label_vocab, os.path.join(save_dir, "label_vocab.txt"))
    write_vocab(['O', 'B', 'I'], os.path.join(save_dir, "simp_label_vocab.txt"))
    for k, v in mappings.items():
        if k == 'slot2desc': continue
        with open(os.path.join(save_dir, "{}.json".format(k)), 'w', encoding='utf8') as fd:
            json.dump(v, fd, indent=2, ensure_ascii=False)
    write_vocab(list(sorted(mappings["dom2slots"].keys())), os.path.join(save_dir, "domains.txt"))
    write_vocab(list(sorted(mappings["intent2slots"].keys())), os.path.join(save_dir, "intents.txt"))
    write_vocab(list(sorted(mappings["slot2desc"].keys())), os.path.join(save_dir, "slots.txt"))
    
    # train
    train_dom_json = separate_train_domains(train_json)
    seen_dom_json, unseen_dom_json = gen_seen_and_unseen(train_dom_json)
    train_dom_data_dir = os.path.join(save_dir, "train_dom_data")
    # os.makedirs(train_dom_data_dir, exist_ok=True)
    for dom, dom_data in train_dom_json.items():
        sub_dir = os.path.join(train_dom_data_dir, dom)
        os.makedirs(sub_dir, exist_ok=True)
        write_json_data(dom_data, os.path.join(sub_dir, "data.json"))
        write_coach_format(dom_data, os.path.join(sub_dir, "{}.txt".format(dom)))

        # unseen & seen
        write_coach_format(seen_dom_json[dom], os.path.join(sub_dir, "seen_slots.txt"))
        write_coach_format(unseen_dom_json[dom], os.path.join(sub_dir, "unseen_slots.txt"))

        sup_data, tst_data = convert_to_support_and_test(dom_data)
        write_json_data(sup_data, os.path.join(sub_dir, "support.json"))
        write_coach_format(sup_data, os.path.join(sub_dir, "support.txt"))
        write_json_data(tst_data, os.path.join(sub_dir, "query.json"))
        write_coach_format(tst_data, os.path.join(sub_dir, "query.txt"))
    
    # dev
    dev_dom_data_dir = os.path.join(save_dir, "dev_dom_data")
    for dom in dev_support_dom_json.keys():
        sub_dir = os.path.join(dev_dom_data_dir, dom)
        os.makedirs(sub_dir, exist_ok=True)
        
        sup_data = dev_support_dom_json[dom]
        tst_data = dev_test_dom_json[dom]

        write_json_data(sup_data, os.path.join(sub_dir, "support.json"))
        write_coach_format(sup_data, os.path.join(sub_dir, "support.txt"))
        write_json_data(tst_data, os.path.join(sub_dir, "query.json"))
        write_coach_format(tst_data, os.path.join(sub_dir, "query.txt"))

if __name__ == "__main__":
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        args = parse_args()
        preprocess_and_save(args)
