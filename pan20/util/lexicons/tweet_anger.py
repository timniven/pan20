"""Custom lexicon learned via distant supervision from Twitter."""
import collections
import json
import os

import emoji
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from pan20 import util
from pan20.util import chis, lexicons, text


"""
TODO:
- Consider higher order n-grams
- Consider contribution of lower to higher order n-grams in refining the list
- More averaging over random splits.
"""


deg_to_attr = {
    1: 'unigrams',
    2: 'bigrams',
    3: 'trigrams',
    4: 'quadgrams',
}
anger_emoji = [
    ':face_with_symbols_on_mouth:',
    ':face_with_steam_from_nose:',
    ':angry_face:',
    ':angry_face_with_horns:',
    ':anger_symbol:',
]
joy_emoji = [
    ':face_with_tears_of_joy:',
    ':cat_face_with_tears_of_joy:',
    ':beaming_face_with_smiling_eyes:',
    ':grinning_cat_face_with_smiling_eyes:',
    ':grinning_face_with_smiling_eyes:',
    ':kissing_face_with_smiling_eyes:',
    ':slightly_smiling_face:',
    ':smiling_cat_face_with_heart-eyes:',
    ':smiling_face:',
    ':smiling_face_with_halo:',
    ':smiling_face_with_3_hearts:',
    ':smiling_face_with_heart-eyes:',
    ':smiling_face_with_horns:',
    ':smiling_face_with_smiling_eyes:',
    ':smiling_face_with_sunglasses:',
]


class Lexicon(lexicons.LexicalCategoryDict):

    def __init__(self, file_path='data/anger/lex2.json'):
        super().__init__()
        stems = json.loads(open(file_path).read())
        self.word_dict = {}
        for s in stems:
            self.word_dict[s] = ['anger']
        self.cat_dict = {'anger': stems}


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
    seen = set([])

    def already_seen(tweet):
        tweet = ''.join(c for c in tweet if not c in emoji.UNICODE_EMOJI)
        tweet = tweet.strip()
        been_seen = tweet in seen
        seen.update(tweet)
        return been_seen

    def is_ambiguous(tweet):
        emos = [c for c in tweet if c in emoji.UNICODE_EMOJI]
        emos = [emoji.UNICODE_EMOJI[e] for e in emos]
        return all(e in anger_emoji for e in emos) \
               or all(e in joy_emoji for e in emos)

    def load_df(fn):
        fp = os.path.join(folder, fn)
        dfd = pd.read_csv(fp)
        if len(dfd[pd.isnull(dfd.tweet)]) > 0:
            dfd = dfd[~pd.isnull(dfd.tweet)]
        if True:
            emotion = 'anger' if 'angry' in fn else 'joy'
            month = fn.split('_')[-1].split('.')[0]
            date_label = label_map[month]

            dfd['ambiguous'] = dfd.tweet.apply(is_ambiguous)
            dfd['seen'] = dfd.tweet.apply(already_seen)
            dfd = dfd[dfd.seen == False]
            dfd = dfd[dfd.ambiguous == False]
            dfd.drop(columns=['seen', 'ambiguous'], inplace=True)

            dfd['emotion'] = emotion
            dfd['month'] = month
            dfd['date_label'] = date_label
            dfd['unigrams'] = dfd.tweet.apply(text.tokenize)
            dfd['bigrams'] = dfd.unigrams.apply(text.to_bigrams)
            dfd['trigrams'] = dfd.unigrams.apply(text.to_trigrams)

        dfd = dfd[~pd.isnull(dfd.unigrams)]
        dfd = dfd[~pd.isnull(dfd.bigrams)]
        dfd = dfd[~pd.isnull(dfd.trigrams)]
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


def join_chi_ent(dfc, dfh):
    dfc.set_index('token', inplace=True)
    dfh.set_index('token', inplace=True)
    dfr = dfc.join(dfh, on='token')
    dfr['s'] = dfr.h * dfr.chi
    dfr = dfr[dfr.c1 > dfr.c2].sort_values(by=['s'], ascending=False)
    return dfr


def get_top_k(df, k, degree):
    dfc = calculate_chi(df, degree=degree)
    dfh = calculate_entropy(df, degree=degree)
    dfr = join_chi_ent(dfc, dfh)
    return dfr.iloc[0:k]


def build_vocab(df, degree):
    vocab = set([])
    attr = deg_to_attr[degree]
    for _, x in df.iterrows():
        vocab.update(x[attr])
    vocab = util.IxDict(list(vocab))
    return vocab


def get_common_toks(df, degree):
    toks = None
    attr = deg_to_attr[degree]
    for period in df.date_label.unique():
        dfp = df[df.date_label == period]
        toksp = set([])
        for _, x in dfp.iterrows():
            toksp.update(x[attr])
        if toks is None:
            toks = toksp
        else:
            toks = toks.intersection(toksp)
    return toks


def chi_var(df, degree):
    common_toks = get_common_toks(df, degree=degree)
    periods = list(sorted(df.date_label.unique()))
    chi = []
    with tqdm(total=len(periods)) as pbar:
        for period in periods:
            dfp = df[df.date_label == period]
            chip = calculate_chi(dfp, degree=degree)
            chip = chip[chip.token.isin(common_toks)]
            chip['period'] = period
            chi.append(chip)
            pbar.update()
    return pd.concat(chi, axis=0)
