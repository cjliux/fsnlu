#encoding: utf-8
import os
#coding: utf-8
"""
    @author: cjliux@gmail.com
    borrowed heavily from https://github.com/zliucr/coach
"""
import sys
import copy


class Vocab(object):

    def __init__(self, vocab = None):
        super().__init__()
        self._vocab = vocab if vocab is not None else []
        self.index2word = dict(enumerate(self._vocab))
        self.word2index = {v:k for k,v in self.index2word.items()}
        self.n_words = len(self._vocab)

    def __getitem__(self, index):
        return self._vocab[index]

    def get_vocab(self):
        return copy.deepcopy(self._vocab)

    @classmethod
    def from_file(cls, file_name):
        with open(file_name, 'r', encoding='utf8') as fd:
            lines = [lin.strip() for lin in fd if len(lin.strip()) > 0]
            return cls(lines)


if __name__=='__main__':
    vocab = Vocab.from_file("./data/default/token_vocab.txt")
    print(vocab[:10])

