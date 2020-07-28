# coding: utf-8
"""
    @author: WANG Hongru, cjliux@gmail.com
"""
import os, sys, json
import copy
import re


test_dom_map = {0: "temperature", 1: "wordFinding", 
            2: "joke", 3: "holiday", 4: "garbageClassify"}


def parse_args():
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/default_v2")
    return parser.parse_args()


def read_schema(data_path):
    with open(os.path.join(data_path, "domint2slots.json"), 
                                        'r', encoding='utf8') as fd:
        schm = json.load(fd)
    return schm


def ensemble(args, dirs, schm):
    for file in range(5):
        dom_schm = schm[test_dom_map[file]]

        result = []
        data = []  # init the file 
        
        for s_dir in dirs:
            data.append(json.load(
                open(s_dir + "/predict_" + str(file) + ".json", encoding="utf8")))
        for i in range(len(data[0])):
            exp = {}
            int_count, slot_count = {}, {}
            for j in range(len(data)):
                intent = data[j][i]["intent"]
                slots = data[j][i]["slots"]
                if intent not in int_count:
                    int_count[intent] = 1
                else:
                    int_count[intent] += 1
                
                for k, v in slots.items():
                    # if k not in slot_count:
                    #     slot_count[k] = {}
                    lv = [v] if not isinstance(v, list) else v
                    for vv in lv:
                        for m in re.finditer(vv, data[j][i]["text"]):
                            # slot_count[k][m.span()] = slot_count[k].get(m.span(), 0) + 1
                            key = (k, m.span())
                            slot_count[key] = slot_count.get(key, 0) + 1
                
            int_count = sorted(int_count.items(), key=lambda x: x[1], reverse=True)
            
            exp["domain"] = data[0][i]["domain"]
            exp["text"] = data[0][i]["text"]
            exp["id"] = data[0][i]["id"]
            exp["intent"] = int_count[0][0]
            slots, values = {}, []

            int_schm = set(dom_schm[exp["intent"]])
            
            # slot_count = sorted([ (k, v) for k, v in slot_count.items() if k in int_schm ], 
            #                                 key=lambda x: sum(x[1].values()), reverse=True)
            # new_slot_count = []
            # for k, v in slot_count:
            #     vv = list(sorted(v.items(), key=lambda x: x[1], reverse=True))
            #     new_slot_count.append((k, vv)) 
            
            # ens_pred = {}
            def overlap(lines, span): # simple implt
                found = False
                for line in lines:
                    if line[0] < span[1]:
                        if line[1] > span[0]:
                            found = True
                return found

            # def dfs(dep, lines): # recursive
            #     global ens_pred
            #     if dep < len(new_slot_count):
            #         k, v = new_slot_count[dep]
            #         found = False
            #         for kk, vv in v:
            #             if not overlap(lines, kk) and dfs(dep+1, lines + [kk]):
            #                 ens_pred[k] = kk
            #                 found = True
            #         return found
            #     else:
            #         return True

            # found = dfs(0, [])
            # assert found

            ens_pred, lines = {}, []
            slot_count = sorted(slot_count.items(), key=lambda x: x[1], reverse=True)
            for slsp, cnt in slot_count:
                if slsp[0] not in ens_pred.keys() and not overlap(lines, slsp[1]):
                    ens_pred[slsp[0]] = slsp[1]

            for sl, sp in ens_pred.items():
                slots[sl] = data[j][i]["text"][sp[0]:sp[1]]
            exp["slot"] = slots

            result.append(exp)
        
        # return result
        os.makedirs("ensemble", exist_ok=True)
        with open("ensemble/predict_"+str(file)+".json", 'w', encoding = 'utf-8') as fd:
            json.dump(result, fd, ensure_ascii = False, indent = 2)


if __name__ == "__main__":
    # dirs = ["data/v2/dev/result_5_5", "data/v2/dev/result_d1_dict", "data/v2/dev/result4"]
    dirs = ["pnbert_exp/pnbert/ecdt_pn_comb/predict", 
            "pnbert_exp/pnbert/ecdt_pn_comb_2_1/predict", 
            "pnbert_exp/pnbert/ecdt_comb/predict"]
    args = parse_args()
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        schm = read_schema(args.data_path)
        ensemble(args, dirs, schm)

