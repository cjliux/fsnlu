#coding: utf-8
import os, sys
import torch
import random
import numpy as np

from models.pnbert.config import add_parser_args
from models.pnbert.interface import do_comb_train_no_eval


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
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    do_comb_train_no_eval(args)


if __name__=="__main__":
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        args = parse_args()
        main(args)
