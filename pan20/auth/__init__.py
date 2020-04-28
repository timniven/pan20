import json

from pan20 import util


# TODO: small dataset can load in memory, large one should use SQLite I think.


n_fandoms = 1600


def get_fandom_dict():
    return util.IxDict.load('data/auth/fd_ix_dict')


def load_small():
    with open('data/auth/train_small.jsonl') as f:
        X = [json.loads(x) for x in f.readlines()]
    with open('data/auth/train_small_truth.jsonl') as f:
        y = [json.loads(y) for y in f.readlines()]
    return X, y
