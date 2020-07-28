#coding:utf-8
import os, json, sys
from collections import defaultdict

AVOID_ZERO = 0.000000001


def toSet(data):
    return set([data]) if not isinstance(data, list) else set(data)

def cal_sentence_acc(file_corr, file_pred):
    assert len(file_corr) == len(file_pred), "corr: {} - pred: {}".format(len(file_corr), len(file_pred))

    intent_acc_num = {}
    sent_acc_num = 0

    slot_f1_num = {}

    num = len(file_corr)
    corr, pred = {}, {}
    for item1, item2 in zip(file_corr, file_pred):
        corr[item1['id']] = item1
        pred[item2['id']] = item2

    for item_id in corr:
        if item_id not in pred or "intent" not in pred[item_id] or "slots" not in pred[item_id]:
            continue
        flag = 1
        # cal intent
        corr_intent = corr[item_id]['intent']
        if corr_intent not in intent_acc_num:
            intent_acc_num[corr_intent] = {'true': 0, 'all': 0}
        intent_acc_num[corr_intent]['all'] += 1
        if corr_intent != pred[item_id]['intent']:
            flag = 0
        else:
            intent_acc_num[corr_intent]['true'] += 1
        # cal slot
        # make sure slot keys are the same when flag = 1
        if len(pred[item_id]['slots']) != len(corr[item_id]['slots']):
            flag = 0
        # norm all slot values(str and list) to set
        corr_slots = {key: toSet(value) for key, value in corr[item_id]['slots'].items()}
        pred_slots = {key: toSet(value) for key, value in pred[item_id]['slots'].items()}
        # the variables for slot f1
        for key, value in corr_slots.items():
            if key not in slot_f1_num:
                slot_f1_num[key] = {'tp': 0, 'pred': 0, 'corr': 0}
            slot_f1_num[key]['corr'] += len(value)
        for key, value in pred_slots.items():
            if key not in slot_f1_num:
                slot_f1_num[key] = {'tp': 0, 'pred': 0, 'corr': 0}
            slot_f1_num[key]['pred'] += len(value)
        # check whether all slot key-value pairs are the same
        # and count the right slot `tp` number
        for key in corr_slots:
            if key not in pred_slots:  # do not have slot key
                flag = 0
            else:
                if corr_slots[key] != pred_slots[key]:
                    flag = 0
                else:
                    slot_f1_num[key]['tp'] += len(corr_slots[key])
        # if meet all strict conditions, the sementic error of this sample is 0 (means true)
        if flag:
            sent_acc_num += 1

    # cal sentence accuracy    
    sent_acc = float(sent_acc_num) / float(num)

    # cal intent accuracy
    macro_intent_acc = sum([float(item['true']) / float(item['all']) for item in intent_acc_num.values()]) / len(intent_acc_num)
    micro_intent_acc = float(sum([item['true'] for item in intent_acc_num.values()])) / float(sum([item['all'] for item in intent_acc_num.values()]))

    # cal slot f1
    macro_p = sum([float(item['tp']) / (float(item['pred'] + AVOID_ZERO)) for item in slot_f1_num.values()]) / len(slot_f1_num) + AVOID_ZERO
    macro_r = sum([float(item['tp']) / (float(item['corr'] + AVOID_ZERO)) for item in slot_f1_num.values()]) / len(slot_f1_num) + AVOID_ZERO
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r)
    micro_p = float(sum([item['tp'] for item in slot_f1_num.values()])) / (float(sum([item['pred'] for item in slot_f1_num.values()])) + AVOID_ZERO) + AVOID_ZERO
    micro_r = float(sum([item['tp'] for item in slot_f1_num.values()])) / (float(sum([item['corr'] for item in slot_f1_num.values()])) + AVOID_ZERO) + AVOID_ZERO
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)

    return sent_acc, macro_intent_acc, micro_intent_acc, macro_f1, micro_f1


def main(args):
    file_cnt = args.file_cnt
    sentence_acc_arr = []
    macro_intent_acc_arr = []
    micro_intent_acc_arr = []
    macro_slot_f1_arr = []
    micro_slot_f1_arr = []

    corr_files = list(sorted(os.listdir(args.correct_dir)))
    pred_files = list(sorted(os.listdir(args.predict_dir)))

    for file_id in range(file_cnt):
        with open(os.path.join(args.correct_dir, 
                corr_files[file_id]), 'r', encoding='utf-8') as f:
            file_corr = json.load(f)

        with open(os.path.join(args.predict_dir, 
                pred_files[file_id]), 'r', encoding='utf-8') as f:
            file_pred = json.load(f)

        sent_acc, macro_intent_acc, micro_intent_acc, macro_f1, micro_f1 = cal_sentence_acc(file_corr, file_pred)
        sentence_acc_arr.append(sent_acc)
        micro_intent_acc_arr.append(micro_intent_acc)
        macro_intent_acc_arr.append(macro_intent_acc)
        macro_slot_f1_arr.append(macro_f1)
        micro_slot_f1_arr.append(micro_f1)

        print('Domain %d sentence accuracy: %f \t intent accuracy: macro - %f | micro - %f \t slot f1: macro - %f | micro - %f ' % (
            file_id, sentence_acc_arr[-1], macro_intent_acc_arr[-1], micro_intent_acc_arr[-1], macro_slot_f1_arr[-1], micro_slot_f1_arr[-1]))

    sent_acc = float(sum(sentence_acc_arr)) / len(sentence_acc_arr)
    macro_intent_acc = float(sum(macro_intent_acc_arr)) / len(macro_intent_acc_arr)
    micro_intent_acc = float(sum(micro_intent_acc_arr)) / len(micro_intent_acc_arr)
    macro_slot_f1 = float(sum(macro_slot_f1_arr)) / len(macro_slot_f1_arr)
    micro_slot_f1 = float(sum(micro_slot_f1_arr)) / len(micro_slot_f1_arr)

    print('Avg sentence accuracy : %f ' % sent_acc)
    print('Micro Score - intent accuracy: %f \t slot f1: %f' % (micro_intent_acc, macro_slot_f1))
    print('Macro Score - intent accuracy: %f \t slot f1: %f' % (macro_intent_acc, micro_slot_f1))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--predict_dir", type=str, 
        default="../data/smp2020ecdt/smp2020ecdt_task1_v2/dev/predict")
    parser.add_argument("--correct_dir", type=str, 
        default="../data/smp2020ecdt/smp2020ecdt_task1_v2/dev/test.labeled")
    parser.add_argument("--file_cnt", type=int, default=5)

    return parser.parse_args()


if __name__ == '__main__':
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        args = parse_args()
        main(args)


    








