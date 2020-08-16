#coding: utf-8
import os, sys
import random
import torch
import numpy as np
from models.ftbert_final.config import add_parser_args
from models.ftbert_final.interface import do_predict


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42)

    parser = add_parser_args(parser)

    return parser.parse_args()


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    do_predict(args)


if __name__=="__main__":
    args = parse_args()
    main(args)
