import collections
import os

import numpy as np

from pan20 import fake, util
from pan20.util import text


glove_path = '/home/hanshan/dev/glove/glove.twitter.27B.200d.txt'


def word_dict():
    fp = 'tmp/fake_vocab.json'
    if os.path.exists(fp):
        return util.IxDict.load(fp)
    counts = collections.Counter()
    df = fake.load()
    for _, x in df.iterrows():
        counts.update(text.tokenize(x.tweet))
    wd = util.IxDict(counts.keys())
    wd.save(fp)
    return wd


def embeddings():
    fp = 'tmp/fake_embeds.npy'
    if os.path.exists(fp):
        return np.load(fp)
    wd = word_dict()
    embeddings = np.random.normal(size=(len(wd), 200)) \
        .astype('float32', copy=False)
    with open(glove_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            s = line.split()
            if len(s) > 301:  # a hack I have seemed to require for GloVe 840B
                s = [s[0]] + s[-200:]
                assert len(s) == 201
            if s[0] in wd.keys():
                embeddings[wd[s[0]], :] = np.asarray(s[1:])
    np.save(fp, embeddings)
    return embeddings


class Vectorize:

    def __init__(self):
        self.wd = word_dict()
        self.embeds = embeddings()

    def __call__(self, tweets):
        if isinstance(tweets, str):
            return self.one(tweets)
        elif isinstance(tweets, list):
            return self.many(tweets)
        else:
            raise ValueError

    def one(self, tweet, reduction=np.sum):
        toks = text.tokenize(tweet)
        tok_ixs = [self.wd[t] for t in toks]
        vecs = [self.embeds[ix, :] for ix in tok_ixs]
        vecs = [np.expand_dims(v, 0) for v in vecs]
        vecs = np.concatenate(vecs, axis=0)
        vecs = reduction(vecs, axis=0)
        return vecs

    def many(self, tweets):
        vecs = [self.one(t) for t in tweets]
        vecs = [np.expand_dims(v, 0) for v in vecs]
        vecs = np.concatenate(vecs, axis=0)
        return vecs
