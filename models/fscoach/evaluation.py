# -*- coding -*-
# Copyright: WANG Hongru
# Time 2020 06 17 11 am
import json
from collections import OrderedDict

def cal_sentence_acc(file_corr, file_pred):
    assert len(file_corr) == len(file_pred)
    
    acc_num = 0
    all_num = len(file_corr)
    
    for item1, item2 in zip(file_corr, file_pred):
        if item1['text'] == item2['text']:
            if item1['intent'] == item2['intent']:
                for k, v in item1['slots'].items():
                    if k in item2['slots'] and item2['slots'][k] == v:
                        acc_num += 1
                        break
    
    return float(acc_num) / float(all_num)
    
        
def cal_intent_acc(file_corr, file_pred):
    assert len(file_corr) == len(file_pred)
    
    acc_num = 0
    all_num = len(file_corr)
    
    for item1, item2 in zip(file_corr, file_pred):
        if item1['text'] == item2['text']:
            if item1['intent'] == item2['intent']:
                acc_num += 1
                
    return float(acc_num) / float(all_num)

def cal_slot_acc(file_corr, file_pred):
    assert len(file_corr) == len(file_pred)
    
    acc_num = 0
    all_num = len(file_corr)
    
    for item1, item2 in zip(file_corr, file_pred):
        if item1['text'] == item2['text']:
            flag = 1
            for k, v in item1['slots'].items():
                if k not in item2['slots'] or item2['slots'][k] != v:
                    flag = 0
                    break
            if flag == 1:
                acc_num += 1
                
    return float(acc_num) / float(all_num)

def cal_slot_f1(file_corr, file_pred):
    assert len(file_corr) == len(file_pred)
    correct, p_denominator, r_denominator = 0, 0, 0
    for truth_dic, pred_dic in zip(file_corr, file_pred):
        r_denominator += len(truth_dic['slots'])
        p_denominator += len(pred_dic['slots'])
        for key, value in truth_dic['slots'].items():
            if key not in pred_dic['slots']:
                continue
            elif pred_dic['slots'][key] == truth_dic['slots'][key] and \
                truth_dic['domain'] == pred_dic['domain'] and \
                truth_dic['intent'] == pred_dic['intent']:
                correct += 1
    precision = float(correct) / p_denominator
    recall = float(correct) / r_denominator
    f1 = 2 * precision * recall / (precision + recall) * 1.0

    return f1
 

def eva_for_each():
    data_pred = []
    data_corr = []
    for i in range(5):
        test_file = DATA_PATH + 'few-shot/test/test_' + str(i) + '.json'
        predict_file = 'result' + '/predict_' + str(i) + '.json'
        
        test_data = json.load(open(test_file, encoding='utf-8'), object_pairs_hook=OrderedDict)
        predict_data = json.load(open(predict_file, encoding='utf-8'), object_pairs_hook=OrderedDict)
        
        data_pred.extend(predict_data)
        data_corr.extend(test_data)
    
    return data_pred, data_corr


if __name__ == "__main__":
    DATA_PATH = "data/v2/"
    
    """
    # the predict result file path
    pred =  "result/test_result.json"
    # the correct result file path
    corr = 'data/task1/test_eval.json'
    
    data_pred = json.load(open(pred, encoding='utf-8'), object_pairs_hook=OrderedDict)
    data_corr = json.load(open(corr, encoding='utf-8'), object_pairs_hook=OrderedDict)
    """
    # predict for each
    data_pred, data_corr = eva_for_each()
    
    print("==========Sentence Acc=====================")
    print(cal_sentence_acc(data_corr, data_pred))
    
    print("==========Intent Acc=======================")
    print(cal_intent_acc(data_corr, data_pred))
    
    print("==========Slot Acc=========================")
    print(cal_slot_acc(data_corr, data_pred))
    
    print("==========Slot F1==========================")
    print(cal_slot_f1(data_corr, data_pred))