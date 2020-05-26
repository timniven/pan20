import json
import os
import random

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn import utils

from pan20 import util
from pan20.util import text


# TODO: small dataset can load in memory, large one should use SQLite I think.


n_fandoms = 1600
n_x = 105202 / 2
n_docs = 105202


def ten_k_set():
    fp = 'tmp/ten_k.json'
    if os.path.exists(fp):
        with open(fp) as f:
            z = json.loads(f.read())
            train_ixs = z['train_ixs']
            dev_ixs = z['dev_ixs']
            test_ixs = z['test_ixs']
            y_train = np.array(z['y_train'])
            y_dev = np.array(z['y_dev'])
            y_test = np.array(z['y_test'])
    else:
        X, y = load_small()
        ixs = zip(
            list(range(len(y))),
            [int(x['same']) for x in y],
        )
        ixs = pd.DataFrame(ixs, columns=['i', 'label'])
        true_ixs = ixs[ixs.label == True]
        false_ixs = ixs[ixs.label == False]

        np.random.seed(42)

        train_true_ixs = true_ixs.sample(5000)
        true_ixs = true_ixs[~true_ixs.i.isin(train_true_ixs.i.unique())]
        train_false_ixs = false_ixs.sample(5000)
        false_ixs = false_ixs[~false_ixs.i.isin(train_false_ixs.i.unique())]
        train_ixs = pd.concat([train_true_ixs, train_false_ixs], axis=0)

        dev_true_ixs = true_ixs.sample(2500)
        true_ixs = true_ixs[~true_ixs.i.isin(dev_true_ixs.i.unique())]
        dev_false_ixs = false_ixs.sample(2500)
        false_ixs = false_ixs[~false_ixs.i.isin(dev_false_ixs.i.unique())]
        dev_ixs = pd.concat([dev_true_ixs, dev_false_ixs], axis=0)

        test_true_ixs = true_ixs.sample(2500)
        test_false_ixs = false_ixs.sample(2500)
        test_ixs = pd.concat([test_true_ixs, test_false_ixs], axis=0)

        train_ixs = utils.shuffle(train_ixs)
        dev_ixs = utils.shuffle(dev_ixs)
        test_ixs = utils.shuffle(test_ixs)

        train_ixs = list(train_ixs.i.values)
        dev_ixs = list(dev_ixs.i.values)
        test_ixs = list(test_ixs.i.values)
        y_train = train_ixs.label.values
        y_dev = dev_ixs.label.values
        y_test = test_ixs.label.values

        with open(fp, 'w+') as f:
            f.write(json.dumps({
                'train_ixs': [int(x) for x in train_ixs],
                'dev_ixs': [int(x) for x in dev_ixs],
                'test_ixs': [int(x) for x in test_ixs],
                'y_train': [int(y) for y in y_train],
                'y_dev': [int(y) for y in y_dev],
                'y_test': [int(y) for y in y_test],
            }))

    return train_ixs, dev_ixs, test_ixs, \
           y_train, y_dev, y_test


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


def small_docs():
    with open('data/auth/train_small.jsonl') as f:
        for x in f.readlines():
            x = json.loads(x)
            for i in range(2):
                doc = x['pair'][i]
                yield doc


def small_toks(n=1):
    with open('data/auth/train_small.jsonl') as f:
        for x in f.readlines():
            x = json.loads(x)
            for i in range(2):
                doc = x['pair'][i]
                toks = text.tokenize(doc, n)
                yield toks


def n_small():
    with open('data/auth/train_small.jsonl') as f:
        return sum(1 for l in f.readlines())


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
