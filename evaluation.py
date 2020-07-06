#coding:utf-8
import os, json, sys
from collections import defaultdict


def cal_sentence_acc(file_corr, file_pred):
    assert len(file_corr) == len(file_pred)

    acc_num, iacc_num = 0, 0
    sacc_num, snum = 0, 0
    mistp, misfp, misfn = 0, 0, 0
    sasf1 = 0
    if1_cnt = defaultdict(lambda: defaultdict(int))

    num = len(file_corr)
    corr, pred = {}, {}
    for item1, item2 in zip(file_corr, file_pred):
        corr[item1['id']] = item1
        pred[item2['id']] = item2

    for item_id in corr:
        if (item_id not in pred 
                or "intent" not in pred[item_id] 
                or "slots" not in pred[item_id]):
            continue # skip bad item
        
        if corr[item_id]['intent'] == pred[item_id]['intent']:
            iacc_num += 1
            if1_cnt[corr[item_id]['intent']]['tp'] += 1
        else:
            if1_cnt[corr[item_id]['intent']]['fn'] += 1
            if1_cnt[corr[item_id]['intent']]['fp'] += 1

        corr_sv = set([sl + '|' + vs if isinstance(vs, str) 
            else sl + '|' + str('|'.join(vs)) for sl, vs in corr[item_id]["slots"].items()])
        pred_sv = set([sl + '|' + vs if isinstance(vs, str) 
            else sl + '|' + str('|'.join(vs)) for sl, vs in pred[item_id]["slots"].items()])
        sstp = len(corr_sv.intersection(pred_sv))
        ssfp = len(pred_sv.difference(corr_sv))
        ssfn = len(corr_sv.difference(pred_sv))
        sasf1 += (2 * float(sstp) / (2 * float(sstp) + float(ssfp) + float(ssfn))
            if len(corr_sv) + len(pred_sv) != 0 else 1)
        mistp += sstp
        misfp += ssfp
        misfn += ssfn

        flag = 1
        if corr[item_id]['intent'] != pred[item_id]['intent']:
            flag = 0

        for key in corr[item_id]['slots']:
            snum += 1

            if (key not in pred[item_id]['slots'] 
                    or corr[item_id]['slots'][key] != pred[item_id]['slots'][key]):
                flag = 0
            else:
                sacc_num += 1

        if flag:
            acc_num += 1

    if1_cnt = { k: 2 * float(c['tp']) / float(2 * c['tp'] + c['fp'] + c['fn'])
        for k, c in if1_cnt.items() }
    if1 = sum(if1_cnt.values()) / len(if1_cnt)

    scores = {
        "size": int(num),
        "int_acc": float(iacc_num) / float(num),
        "int_f1": if1,
        "slot_acc": float(sacc_num) / float(snum),
        "sam_slot_f1": sasf1 / float(num),
        "mic_slot_f1": 2 * float(mistp) / (2 * float(mistp) + float(misfp) + float(misfn)),
        "sent_acc": float(acc_num) / float(num),
    }
    return scores


def weighted_sum(score_list, weight_list):
    fin_score = 0
    sum_weight = 0
    for s, w in zip(score_list, weight_list):
        fin_score += s * w
        sum_weight += w
    fin_score /= sum_weight
    return fin_score


def main(args):
    # 需要把：预测文件夹命名predict放在dev/路径下,里面文件的下标和test中对应
    # 参数 1:正确的目录(以correct_id.json格式命名) 2:预测目录(以predict_id.json命名)
    # python evaluation.py dev/correct/ dev/predict/ 4
    file_cnt = args.file_cnt
    sentence_acc = []
    all_scores = defaultdict(list)

    corr_files = list(sorted(os.listdir(os.path.join(args.data_path, args.correct_dir))))
    pred_files = list(sorted(os.listdir(os.path.join(args.data_path, args.predict_dir))))

    for file_id in range(file_cnt):
        with open(os.path.join(args.data_path, args.correct_dir, 
                corr_files[file_id]), 'r', encoding='utf-8') as f:
            file_corr = json.load(f)

        with open(os.path.join(args.data_path, args.predict_dir, 
                pred_files[file_id]), 'r', encoding='utf-8') as f:
            file_pred = json.load(f)

        scores = cal_sentence_acc(file_corr, file_pred)
        sentence_acc.append(scores["sent_acc"])
        for k, v in scores.items():
            all_scores[k].append(v)

        buf = '[Domain %d] size: %d; ' % (file_id, scores['size'])
        for k, v in scores.items():
            if k == 'size': continue
            buf += '%s: %f; ' % (k, v)
        # print('[Domain %d] size: %d, int_acc: %f; slot_acc: %f; slot_f1: %f; sent_acc: %f' % (
        #     file_id, scores["size"],
        #     scores["int_acc"], scores["slot_acc"], scores["slot_f1"], scores["sent_acc"]))
        print(buf)

    for k in scores.keys():
        if k != 'size':
            print('Avg %s : %f; weighed : %f' % (k,
                sum(all_scores[k]) / file_cnt, 
                weighted_sum(all_scores[k], all_scores["size"])))
    # print('Avg sentence accuracy : %f; weighted: %f' % (
    #     sum(sentence_acc) / file_cnt,
    #     weighted_sum(all_scores["sent_acc"], all_scores["size"])))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, 
        default="../data/smp2020ecdt/smp2020ecdt_task1_v2/dev")

    parser.add_argument("--predict_dir", type=str, default="predict")
    parser.add_argument("--correct_dir", type=str, default="test.labeled")
    parser.add_argument("--file_cnt", type=int, default=5)

    return parser.parse_args()


if __name__ == '__main__':
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        args = parse_args()
        main(args)

