#coding: utf-8
import os, sys
import json


def apply_domain_rule(domain, dom_data):
    return dom_data


def add_leak(sup_data, pred_data):
    leak_lut = {}
    for item in sup_data:
        leak_lut[item['text']] = item
    
    new_pred_data = []
    for item in pred_data:
        if item['text'] in leak_lut:
            leak_item = leak_lut[item['text']]
            item['intent'] = leak_item['intent']
            item['slots'] = leak_item['slots']
        new_pred_data.append(item)
    return new_pred_data


def main(args):
    sup_files = list(sorted([f for f in os.listdir(args.support_path) if f.startswith("support")]))
    pred_files = list(sorted([f for f in os.listdir(args.predict_path) if f.startswith("predict")]))

    os.makedirs(args.save_path, exist_ok=True)
    for fid, (sup_file, pred_file) in enumerate(zip(sup_files, pred_files)):
        with open(os.path.join(args.support_path, sup_file), 'r', encoding='utf8') as fd:
            sup_data = json.load(fd)
        with open(os.path.join(args.predict_path, pred_file), 'r', encoding='utf8') as fd:
            pred_data = json.load(fd)
        
        domain = sup_data[0]['domain']
        pred_data = apply_domain_rule(domain, pred_data)

        # final: ground truth lookup (strong)
        pred_data = add_leak(sup_data, pred_data)

        with open(os.path.join(args.save_path, pred_file), 'w', encoding='utf8') as fd:
            json.dump(pred_data, fd, ensure_ascii=False, indent=2)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--support_path", type=str, 
                    default="../data/smp2020ecdt/test_support_v4/support")
    parser.add_argument("--predict_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    return parser.parse_args()


if __name__=='__main__':
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        args = parse_args()
        main(args)

