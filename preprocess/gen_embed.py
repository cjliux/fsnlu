#coding: utf-8
"""
    @author: cjliux@gmail.com
    borrowed heavily from https://github.com/zliucr/coach
"""
import os
os.environ['HOME'] = './'
os.environ['EMBEDDINGS_ROOT'] = 'E:/WorkSpace/Research/HIT/dst/trade-dst/embeddings'
from embeddings import GloveEmbedding

import sys
import tqdm
import json
import gensim
import pickle
import numpy as np


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", 
        type=str, default="./data/default")
    parser.add_argument("--w2v_file", type=str,
        default="../resource/sgns.sogounews.bigram-char")
    return vars(parser.parse_args())


def read_vocab(voc_file):
    vocab = []
    with open(voc_file, 'r', encoding='utf8') as fd:
        for line in fd:
            line = line.strip()
            if len(line) > 0:
                vocab.append(line.strip())
    return vocab


def gen_word_embed_from_pretrained_and_save(vocab, w2v_file, slot2embs, save_file):
    model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file)
    final_embs = np.zeros((len(vocab), model.vector_size))

    hit = 0
    for i_w in tqdm.tqdm(range(len(vocab))):
        word = vocab[i_w]
        if len(word) > 1 and i_w >= 2:
            final_embs[i_w] = slot2embs[word]
            hit += 1
        else:
            word = word.lower()
            if word in model.vocab:            
                final_embs[i_w] = model[word]
                hit += 1
            
    print('hit rate: ', hit / len(vocab))

    np.save(save_file, final_embs)


def gen_slot_embed_for_each_dom_from_glove(dom2slots, slot2desc, save_file):
    ## 1. generate slot2embs
    slots = list(sorted(slot2desc.keys()))
    desps = [slot2desc[k] for k in slots]
    word2emb = {}
    # collect words
    for des in desps:
        splits = des.split()
        for word in splits:
            if word not in word2emb:
                word2emb[word] = []
    
    # load embeddings
    glove_emb = GloveEmbedding()

    # calculate slot embs
    slot2embs = {}
    for i, slot in enumerate(slots):
        word_list = slot2desc[slot].split()
        embs = np.zeros(300)
        for word in word_list:
            embs = embs + glove_emb.emb(word, default='zero')
        slot2embs[slot] = embs

    ## 2. generate slot2embs based on each domain
    slot_embs_based_on_each_domain = {}
    for domain, slot_names in dom2slots.items():
        slot_embs = np.zeros((len(slot_names), 300))
        for i, slot in enumerate(slot_names):
            embs = slot2embs[slot]
            slot_embs[i] = embs
        slot_embs_based_on_each_domain[domain] = slot_embs
    
    with open(save_file, "wb") as f:
        pickle.dump(slot_embs_based_on_each_domain, f)
    return slot2embs


def main(args):
    token_vocab = read_vocab(os.path.join(args["data_path"], "token_vocab.txt"))
    label_vocab = read_vocab(os.path.join(args["data_path"], "label_vocab.txt"))
    
    with open(os.path.join(args["data_path"], "dom2slots.json"), 'r', encoding='utf8') as fd:
        dom2slots = json.load(fd)
    with open(os.path.join(args["data_path"], "slot2desc.json"), 'r', encoding='utf8') as fd:
        slot2desc = json.load(fd)
    slot2embs = gen_slot_embed_for_each_dom_from_glove(dom2slots, slot2desc,
        os.path.join(args["data_path"], "slot_embs_based_on_each_domain.dict"))

    gen_word_embed_from_pretrained_and_save(
        token_vocab, args["w2v_file"], slot2embs,
        os.path.join(args["data_path"], "token_emb.npy"))


if __name__=='__main__':
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        args = parse_args()
        main(args)

