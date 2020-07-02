#coding: utf-8
from models.fscoach.config import add_parser_args
from models.fscoach.fsinterface import eval_model, test_model


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser = add_parser_args(parser)

    return parser.parse_args()


def main(args):
    eval_model(args)
    test_model(args)


if __name__=="__main__":
    args = parse_args()
    main(args)
