#coding: utf-8
from models.fscoach.config import add_parser_args
from models.fscoach.fsinterface import train_model, eval_model


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser = add_parser_args(parser)

    return parser.parse_args()


def main(args):
    train_model(args)


if __name__=="__main__":
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        args = parse_args()
        main(args)
