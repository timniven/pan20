"""Custom lexicon learned via distant supervision from Twitter."""
import collections
import os

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from pan20 import util
from pan20.util import chis, text


deg_to_attr = {
    1: 'unigrams',
    2: 'bigrams',
    3: 'trigrams',
    4: 'quadgrams',
}


def str_to_list(s):
    if isinstance(s, list):
        return s
    s = s.replace('"', '')
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = s.replace(',', '')
    s = s.replace('\'', '')
    return s.split(' ')


def load_data():
    folder = 'data/fake/twt_emo'
    files = os.listdir(folder)
    months = list(sorted([f.split('_')[-1].split('.')[0] for f in files]))
    label_map = dict(zip(months, range(len(months))))

    def load_df(fn):
        fp = os.path.join(folder, fn)
        dfd = pd.read_csv(fp)
        if len(dfd[pd.isnull(dfd.tweet)]) > 0:
            dfd = dfd[~pd.isnull(dfd.tweet)]
        if 'trigrams' not in dfd.columns:
            emotion = 'anger' if 'angry' in fn else 'joy'
            month = fn.split('_')[-1].split('.')[0]
            date_label = label_map[month]
            dfd['emotion'] = emotion
            dfd['month'] = month
            dfd['date_label'] = date_label
            dfd['unigrams'] = dfd.tweet.apply(text.tokenize)
            dfd['bigrams'] = dfd.unigrams.apply(text.to_bigrams)
            dfd['trigrams'] = dfd.unigrams.apply(text.to_trigrams)
            dfd.to_csv(fp, index=False)
        dfd = dfd[~pd.isnull(dfd.unigrams)]
        dfd = dfd[~pd.isnull(dfd.bigrams)]
        dfd = dfd[~pd.isnull(dfd.trigrams)]
        dfd['unigrams'] = dfd.unigrams.apply(str_to_list)
        dfd['bigrams'] = dfd.unigrams.apply(str_to_list)
        dfd['trigrams'] = dfd.unigrams.apply(str_to_list)
        return dfd

    dfs = []
    with tqdm(total=len(files)) as pbar:
        for fn in files:
            pbar.set_description(fn)
            df = load_df(fn)
            dfs.append(df)
            pbar.update()

    df = pd.concat(dfs, axis=0)

    return df


def calculate_chi(df, degree=1):
    ca = collections.Counter()
    cj = collections.Counter()
    attr = deg_to_attr[degree]
    with tqdm(total=len(df)) as pbar:
        for _, x in df.iterrows():
            if x.emotion == 'anger':
                ca.update(x[attr])
            else:
                cj.update(x[attr])
            pbar.update()
    return chis.get_chis(ca, cj)


def calculate_entropy(df, degree=1):
    attr = deg_to_attr[degree]

    # get vocabulary
    V = set([])
    for _, x in df.iterrows():
        V.update(x[attr])
    V = util.IxDict(list(V))

    # calculate probabilities
    n_date_labels = len(df.date_label.unique())
    P = np.zeros((n_date_labels, len(V)))
    with tqdm(total=n_date_labels) as pbar:
        for date_label in range(n_date_labels):
            dfd = df[df.date_label == date_label]
            counts = collections.Counter()
            for _, x in dfd.iterrows():
                counts.update(x[attr])
            n = sum(counts.values())
            for token, count in counts.items():
                token_ix = V[token]
                P[date_label, token_ix] = counts[token] / n
            pbar.update()

    # calculate entropies
    h = util.entropy(np.clip(P, a_min=1e-16, a_max=1.), 0)

    # create a DataFrame
    dfe = []
    for ix, v in V.items():
        dfe.append({
            'token': v,
            'h': h[ix],
        })
    dfe = pd.DataFrame(dfe)

    return dfe


def join_chi_ent(dfc, dfe):
    dfc.set_index('token', inplace=True)
    dfe.set_index('token', inplace=True)
    return dfc.join(dfe, on='token')
