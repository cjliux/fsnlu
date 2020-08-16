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
    parser.add_argument("--final_path", type=str, 
        default="../data/smp2020ecdt/test_support_v4")
    parser.add_argument("--save_path", type=str,
        default="./data")
    parser.add_argument("--key", type=str, default="default_v4")
    return vars(parser.parse_args())


def segment_text_and_label_seq(item, tokenizer=None):
    labeled = "slots" in item
    # text = ''.join(item["text"].split())
    text = item["text"]
    if labeled:
        sv_pairs = []
        for sl, val in item["slots"].items():
            def dfs_add(sl, val):
                nonlocal sv_pairs
                if isinstance(val, str):
                    sv_pairs.append((sl, val))
                elif isinstance(val, dict):
                    for k, v in val.items():
                        dfs_add(sl+ '_' + k, v)
                elif isinstance(val, list):
                    for v in val:
                        sv_pairs.append((sl, v))
                else:
                    raise Exception('unrecognized type')
            dfs_add(sl, val)

        sv_pairs = sorted(sv_pairs, key=lambda x: len(x[1]), reverse=True)

        def search_span(root, sv_pairs):
            found, index = False, None
            sl, start, end = None, None, None
            ist = None
            for ist in range(len(sv_pairs)):
                sl, val = sv_pairs[ist]
                regex = "\s*".join(list(''.join(val.split())))
                m = re.search(regex, root)
                if m is not None:
                    found = True
                    start, end = m.start(), m.end()
                    sv_pairs.pop(ist)
                    break

            if found:
                lhs, mid, rhs = root[:start], root[start:end], root[end:]
                node = {"mid": (mid, sl)}
                if len(lhs) > 0:
                    node["lhs"] = search_span(lhs, sv_pairs) 
                if len(rhs) > 0:
                    node["rhs"] = search_span(rhs, sv_pairs)
                return node
            else:
                return {"mid": (root, None)}

        sp_tree = search_span(text, sv_pairs)

        assert len(sv_pairs) == 0

        def merge_labels(sp_tree):
            token, label = [], []
            if "lhs" in sp_tree:
                sub_token, sub_label = merge_labels(sp_tree["lhs"])
                token.extend(sub_token)
                label.extend(sub_label)
            mid, sl = sp_tree['mid']
            # tok_mid = tokenizer.tokenize(mid)
            tok_mid = list(mid)
            token.extend(tok_mid)
            if sl is None:
                label.extend(['O'] * len(tok_mid))
            else:
                label.extend(['B-' + sl] + ['I-' + sl] * (len(tok_mid)-1))
            if "rhs" in sp_tree:
                sub_token, sub_label = merge_labels(sp_tree["rhs"])
                token.extend(sub_token)
                label.extend(sub_label)
            return token, label
        
        token, label = merge_labels(sp_tree)

        # check
        def dfs_check(sl, val, label):
            if isinstance(val, str):
                assert 'B-' + sl in label
            elif isinstance(val, dict):
                for k, v in val.items():
                    dfs_check(sl+ '_' + k, v, label)
            elif isinstance(val, list):
                for v in val:
                    dfs_check(sl, v, label)
            else:
                raise Exception('unrecognized type')
        for sl, val in item["slots"].items():
            dfs_check(sl, val, label)
    else:
        token = list(text)
        label = ['O'] * len(token)

    return token, label


def read_and_preprocess_data(json_file):
    with open(json_file, 'r', encoding='utf8') as fd:
        json_data = json.load(fd)

    new_json_data = []
    for item in json_data:
        tokens, label = segment_text_and_label_seq(item)
        item['token'] = tokens
        item['label'] = label
        new_json_data.append(item)
    return new_json_data


def read_data(data_path, final_path):
    train_json = read_and_preprocess_data(os.path.join(data_path, "source.json"))

    sup_dir = os.path.join(data_path, "dev/support")
    test_dir = os.path.join(data_path, "dev/test")
    dev_sup_dom_json, dev_test_dom_json = {}, {}
    for sup_file in os.listdir(sup_dir):
        if not sup_file.startswith('support_'): 
            continue
        i_dom = sup_file[sup_file.rindex('_')+1:sup_file.rindex('.')]

        sup_file = os.path.join(sup_dir, sup_file)
        sup_data = read_and_preprocess_data(sup_file)
        tst_file = os.path.join(test_dir, "test_{}.json".format(i_dom))
        tst_data = read_and_preprocess_data(tst_file)

        dom = sup_data[0]['domain']
        dev_sup_dom_json[dom] = sup_data
        dev_test_dom_json[dom] = tst_data

    fin_sup_dir = os.path.join(final_path, "support")
    # fin_test_dir = os.path.join(final_path, "test")
    fin_sup_dom_json = {}
    # fin_test_dom_json = {}
    for sup_file in os.listdir(fin_sup_dir):
        if not sup_file.startswith('support_'):
            continue
        i_dom = sup_file[sup_file.rindex('_')+1:sup_file.rindex('.')]

        sup_file = os.path.join(fin_sup_dir, sup_file)
        sup_data = read_and_preprocess_data(sup_file)
        # tst_file = os.path.join(test_dir, "test_{}.json".format(i_dom))
        # tst_data = read_and_preprocess_data(tst_file)

        dom = sup_data[0]['domain']
        fin_sup_dom_json[dom] = sup_data

    return (train_json, dev_sup_dom_json, 
                dev_test_dom_json, fin_sup_dom_json)


def get_vocab_and_labels(train_json, 
        dev_sup_dom_json, dev_test_dom_json,
        fin_sup_dom_json):
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

    for dom in dev_sup_dom_json:
        for item in dev_sup_dom_json[dom]:
            wfr += Counter(item['token'])
            lfr += Counter(item['label'])
            dom2slots[item['domain']].update(set(item['slots'].keys()))
            dom2intents[item['domain']].add(item['intent'])
            intent2slots[item['intent']].update(set(item['slots'].keys()))
            domint2slots[item['domain']][item['intent']].update(set(item['slots'].keys()))
        
        # for item in dev_test_dom_json[dom]:
        #     wfr += Counter(item['token'])

    for dom in fin_sup_dom_json:
        for item in fin_sup_dom_json[dom]:
            wfr += Counter(item['token'])
            lfr += Counter(item['label'])
            dom2slots[item['domain']].update(set(item['slots'].keys()))
            dom2intents[item['domain']].add(item['intent'])
            intent2slots[item['intent']].update(set(item['slots'].keys()))
            domint2slots[item['domain']][item['intent']].update(set(item['slots'].keys()))

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

    print(len(token_vocab))
    return token_vocab, { "dom2slots": dom2slots, 
        "slot2desc": slot2desc, "dom2intents": dom2intents, 
        "intent2slots": intent2slots, "domint2slots": domint2slots}

def write_json_data(json_data, save_file):
    with open(save_file, 'w', encoding='utf8') as fd:
        json.dump(json_data, fd, indent=2, ensure_ascii=False)


def write_vocab(vocab, save_file):
    with open(save_file, 'w', encoding='utf8') as fd:
        fd.write('\n'.join(vocab))


def preprocess_and_save(args):
    (train_json, dev_sup_dom_json, 
        dev_test_dom_json, fin_sup_dom_json
            ) = read_data(args["data_path"], args["final_path"])
    save_dir = os.path.join(args["save_path"], args["key"])
    os.makedirs(save_dir, exist_ok=True)

    # vocab
    token_vocab, mappings = get_vocab_and_labels(train_json, 
            dev_sup_dom_json, dev_test_dom_json, fin_sup_dom_json)
    # write_vocab(token_vocab, os.path.join(save_dir, "token_vocab.txt"))
    label_vocab = ['O']
    for sl in sorted(mappings["slot2desc"].keys()):
        label_vocab.extend(['B-' + sl, 'I-' + sl])
    write_vocab(label_vocab, os.path.join(save_dir, "label_vocab.txt"))
    write_vocab(['O', 'B', 'I'], os.path.join(save_dir, "bin_label_vocab.txt"))
    write_vocab(list(sorted(mappings["dom2slots"].keys())), os.path.join(save_dir, "domains.txt"))
    write_vocab(list(sorted(mappings["intent2slots"].keys())), os.path.join(save_dir, "intents.txt"))
    write_vocab(list(sorted(mappings["slot2desc"].keys())), os.path.join(save_dir, "slots.txt"))
    
    write_json_data(mappings["dom2slots"], os.path.join(save_dir, "dom2slots.json"))
    write_json_data(mappings["dom2intents"], os.path.join(save_dir, "dom2intents.json"))
    write_json_data(mappings["domint2slots"], os.path.join(save_dir, "domint2slots.json"))
    write_json_data(mappings["intent2slots"], os.path.join(save_dir, "intent2slots.json"))


if __name__ == "__main__":
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        args = parse_args()
        preprocess_and_save(args)
