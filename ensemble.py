# coding: utf-8
"""
    @author: WANG Hongru, cjliux@gmail.com
"""
import os, sys, json
import copy
import re


# test_dom_map = {
#     0: "temperature", 
#     1: "wordFinding", 
#     2: "joke", 
#     3: "holiday", 
#     4: "garbageClassify"
# }


def parse_args():
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/default_v4")
    parser.add_argument("--save_path", type=str, default="./ensemble")
    parser.add_argument("--predict_dirs", nargs='+')
    return parser.parse_args()


def read_schema(data_path):
    with open(os.path.join(data_path, "domint2slots.json"), 
                                        'r', encoding='utf8') as fd:
        schm = json.load(fd)
    return schm


def ensemble(args, schm):
    print('ensembling:', args.predict_dirs)
    fname = [f for f in os.listdir(args.predict_dirs[0]) if f.endswith('.json')]

    for file in range(len(fname)):
        result = []
        data = []  # init the file 
        
        for s_dir in args.predict_dirs:
            rdata = json.load(open(s_dir + "/predict_" + str(file) + ".json", encoding="utf8"))
            # rdata = list(sorted(rdata, key=lambda x: x["id"]))
            data.append(rdata)
        
        dom_name = data[0][0]["domain"]
        dom_schm = schm[dom_name]

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
                        # for m in re.finditer(vv, data[j][i]["text"]):
                            # slot_count[k][m.span()] = slot_count[k].get(m.span(), 0) + 1
                            # key = (k, m.span())
                        key = (k, vv)
                        slot_count[key] = slot_count.get(key, 0) + 1
                
            int_count = sorted(int_count.items(), key=lambda x: x[1], reverse=True)
            
            exp["domain"] = data[0][i]["domain"]
            exp["text"] = data[0][i]["text"]
            exp["id"] = data[0][i]["id"]
            exp["intent"] = int_count[0][0]
            slots, values = {}, []

            try:
                int_schm = set(dom_schm[exp["intent"]])
            except:
                int_schm = set(dom_schm)

            # ens_pred = {}
            def sat_value_set(vv): # simple implt
                nonlocal slots
                text = copy.deepcopy(exp["text"])
                all_vals = sorted(list(slots.values()) + [vv], 
                            key=lambda x: len(x), reverse=True)
                for v in all_vals:
                    if not isinstance(v, list):
                        v = [v]
                    for vv in v:
                        i = text.find(vv)
                        if i == -1:
                            return False
                        else:
                            text = text.replace(vv, "XXX", 1)
                return True

            slot_count = sorted(slot_count.items(), key=lambda x: x[1], reverse=True)
            for slvv, cnt in slot_count:
                # if slvv[0] not in slots.keys() and sat_value_set(slvv[1]):
                if sat_value_set(slvv[1]):
                    if slvv[0] in slots.keys():
                        if not isinstance(slots[slvv[0]], list):
                            slots[slvv[0]] = [slots[slvv[0]]]
                        if slvv[1] not in slots[slvv[0]]:
                            slots[slvv[0]].append(slvv[1])
                    else:
                        slots[slvv[0]] = slvv[1]

            exp["slots"] = slots

            result.append(exp)

        # return result
        os.makedirs(args.save_path, exist_ok=True)
        with open(os.path.join(args.save_path, "predict_"+str(file)+".json"), 
                                                    'w', encoding = 'utf-8') as fd:
            json.dump(result, fd, ensure_ascii = False, indent = 2)


if __name__ == "__main__":
    args = parse_args()

    args.predict_dirs = [
        "ftbert_final_exp/ftbert_final/local/predict",
        "ftbert_final_nodict_exp/ftbert_final/local/predict",
    ]
    
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        schm = read_schema(args.data_path)
        ensemble(args, schm)

