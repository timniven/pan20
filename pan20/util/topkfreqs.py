"""Top-k frequent tokens from a dataset and scaling by entropy over docs."""
import collections
import json
import os

import numpy as np
from scipy import special
from tqdm.notebook import tqdm

from pan20 import auth, util
from pan20.util import text


def small_freqs(n):
    """Calculates frequencies over the small training set.

    Args:
      n: Integer, order of n-grams.
    """
    fp = f'tmp/small_freqs_{n}.json'
    if os.path.exists(fp):
        with open(fp) as f:
            return json.loads(f.read())
    else:
        docs = auth.small_docs()
        freqs = text.docs_to_freqs(docs, auth.n_docs, n)
        with open(fp, 'w+') as f:
            f.write(json.dumps(freqs))
        return freqs


def get_top_k(k, n):
    fp = f'tmp/small_top_{k}_{n}.json'
    if os.path.exists(fp):
        with open(fp) as f:
            return json.loads(f.read())
    else:
        freqs = small_freqs(n)
        sorted_freqs = reversed(sorted(freqs.values()))
        cutoff = -1
        taken = 1
        while taken <= k:
            cutoff = next(sorted_freqs)
            taken += 1
        topk = {w: f for w, f in freqs.items() if f >= cutoff}
        with open(fp, 'w+') as f:
            f.write(json.dumps(topk))
        return topk


class Vectorizer:

    def __init__(self, k, n):
        """Create a new Vectorizer.

        Args:
          k: Integer.
          n: Integer, degree of n-grams.
        """
        self.k = k
        self.top_k = get_top_k(k, n)
        self.word_dict = util.IxDict(self.top_k.keys())
        self.doc_ent = top_k_doc_ent(k, n)
        self.scaling = special.softmax(1 / self.doc_ent)

    def __call__(self, toks, entropy_scaling=True):
        """Vectorize a document.

        Args:
          toks: List of lists of tokens.
          entropy_scaling: Bool, whether to scale the frequencies by entropy
            over documents. Default is True.

        Returns:
          numpy.array of shape (k,).
        """
        vec = np.zeros((self.k,))
        doc_counts = collections.Counter(toks)
        for word_ix, word in self.word_dict.items():
            if word in doc_counts:
                vec[word_ix] = doc_counts[word]
        vec /= len(toks)
        if entropy_scaling:
            vec /= self.doc_ent
        return vec


def top_k_counts(k, n):
    fp = f'tmp/top_{k}_{n}_counts.npy'
    if os.path.exists(fp):
        return np.load(fp)
    else:
        top_k = get_top_k(k, n)
        word_dict = util.IxDict(top_k.keys())
        toks = auth.small_toks(n)
        counts = np.zeros((k, auth.n_docs))
        with tqdm(total=auth.n_docs, desc=f'Top {k} Counts') as pbar:
            for doc_ix, doc_toks in enumerate(toks):
                doc_counts = collections.Counter(doc_toks)
                for word_ix, word in word_dict.items():
                    if word in doc_counts:
                        counts[word_ix, doc_ix] = doc_counts[word]
                pbar.update()
        np.save(fp, counts)
        return counts


def top_k_doc_ent(k, n):
    fp = f'tmp/top_{k}_{n}_doc_ent.npy'
    if os.path.exists(fp):
        return np.load(fp)
    else:
        counts = top_k_counts(k, n)
        n_ = np.expand_dims(counts.sum(axis=1), 1)  # k * 1
        p = counts / n_
        p = p + 1e-16  # numerical stability
        h = util.entropy(p, axis=1)
        np.save(fp, h)
        return h


def top_k_Xy(k, n):
    folder = f'tmp/top_{k}_{n}_Xy'
    if not os.path.exists(folder):
        os.mkdir(folder)
    X_train_path = os.path.join(folder, 'X_train.npy')
    y_train_path = os.path.join(folder, 'y_train.npy')
    X_dev_path = os.path.join(folder, 'X_dev.npy')
    y_dev_path = os.path.join(folder, 'y_dev.npy')
    X_test_path = os.path.join(folder, 'X_test.npy')
    y_test_path = os.path.join(folder, 'y_test.npy')
    if os.path.exists(y_test_path):
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_dev = np.load(X_dev_path)
        y_dev = np.load(y_dev_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
    else:
        X, y = auth.load_small()
        train_ixs, dev_ixs, test_ixs, y_train, y_dev, y_test = auth.ten_k_set()
        vectorize = Vectorizer(k, n)

        def get_X(ixs, desc):
            vecs = []
            with tqdm(total=len(ixs), desc=desc) as pbar:
                for ix in ixs:
                    d0 = X[ix]['pair'][0]
                    d1 = X[ix]['pair'][1]
                    t0 = text.tokenize(d0, n)
                    t1 = text.tokenize(d1, n)
                    v0 = vectorize(t0)
                    v1 = vectorize(t1)
                    diff = np.abs(v0 - v1)
                    diff = np.expand_dims(diff, 0)
                    vecs.append(diff)
                    pbar.update()
            return np.concatenate(vecs, axis=0)

        X_train = get_X(train_ixs, 'Train')
        X_dev = get_X(dev_ixs, 'Dev')
        X_test = get_X(test_ixs, 'Test')

        np.save(X_train_path, X_train)
        np.save(y_train_path, y_train)
        np.save(X_dev_path, X_dev)
        np.save(y_dev_path, y_dev)
        np.save(X_test_path, X_test)
        np.save(y_test_path, y_test)

    return X_train, y_train, X_dev, y_dev, X_test, y_test
