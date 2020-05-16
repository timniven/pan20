import json
import os
import random

from tqdm.notebook import tqdm

from pan20 import util


# TODO: small dataset can load in memory, large one should use SQLite I think.


n_fandoms = 1600


def get_auth_dict():
    path = 'data/auth/auth_ix_dict'
    if not os.path.exists(path):
        authors = set([])
        _, truth = load_small()
        with tqdm(total=len(truth)) as pbar:
            for y in truth:
                authors.update(y['authors'])
                pbar.update()
        ix_dict = util.IxDict(authors)
        ix_dict.save(path)
        return ix_dict
    else:
        return util.IxDict.load(path)


def get_fandom_dict():
    path = 'data/auth/fd_ix_dict'
    if not os.path.exists(path):
        fandoms = set([])
        X, _ = load_small()
        with tqdm(total=len(X)) as pbar:
            for x in X:
                fandoms.update(x['fandoms'])
                pbar.update()
        ix_dict = util.IxDict(fandoms)
        ix_dict.save(path)
        return ix_dict
    else:
        return util.IxDict.load(path)


def load_small():
    with open('data/auth/train_small.jsonl') as f:
        X = [json.loads(x) for x in f.readlines()]
    with open('data/auth/train_small_truth.jsonl') as f:
        y = [json.loads(y) for y in f.readlines()]
    return X, y


def small():
    # includes strain, dev, test split
    X, truth = load_small()

    random.seed(42)

    train_ixs = list(range(len(X)))
    dev_ixs = random.sample(train_ixs, 5000)
    for ix in dev_ixs:
        train_ixs.remove(ix)
    test_ixs = random.sample(train_ixs, 5000)
    for ix in test_ixs:
        train_ixs.remove(ix)

    random.shuffle(train_ixs)

    X_test = [X[ix] for ix in test_ixs]
    y_test = [truth[ix] for ix in test_ixs]
    X_dev = [X[ix] for ix in dev_ixs]
    y_dev = [truth[ix] for ix in dev_ixs]
    X_train = [X[ix] for ix in train_ixs]
    y_train = [truth[ix] for ix in train_ixs]

    return X_train, y_train, X_dev, y_dev, X_test, y_test
