"""Custom features."""
import pandas as pd
from tqdm.notebook import tqdm

from pan20 import noble, util


def get_all(data, truth):
    feats = rt_freq(data)
    feats = feats.join(func_freq(data))
    feats = feats.join(rt_usr_func_freq(data))
    feats = feats.join(truth)
    return feats.reset_index()


def rt_freq(data):
    # data is DataFrame with cols: author, tweet, label
    data['is_retweet'] = data.tweet.apply(lambda x: x.startswith('RT '))
    feats = data.groupby(['author']).mean().reset_index()
    feats.rename(columns={'is_retweet': 'rt_freq'}, inplace=True)
    feats = feats[['author', 'rt_freq']]
    feats.set_index(keys=['author'], inplace=True)
    return feats


def retweet_mainstream_media_ratio(data):
    raise NotImplementedError


def func_freq(data):
    # data is DataFrame with cols: author, tweet, label
    noble_dict = noble.NobleDict()
    feats = {}
    with tqdm(total=len(data)) as pbar:
        for _, x in data.iterrows():
            if x.author not in feats:
                feats[x.author] = {'c': 0, 'n': 0}
            if util.is_retweet(x.tweet):
                pbar.update()
                continue
            tokens = util.tokenize(x.tweet, False)
            feats[x.author]['n'] += len(tokens)
            feats[x.author]['c'] += sum(1 for t in tokens if t in noble_dict)
            pbar.update()

    def get_freq(c, n):
        return c / n if n > 0 else 0.

    feats = [{'author': k, 'func_freq': get_freq(v['c'], v['n'])}
             for k, v in feats.items()]

    feats = pd.DataFrame(feats)
    feats.set_index(keys=['author'], inplace=True)

    return feats


def rt_usr_func_freq(data):
    # data is DataFrame with cols: author, tweet, label
    noble_dict = noble.NobleDict()
    feats = {}
    with tqdm(total=len(data)) as pbar:
        for _, x in data.iterrows():
            if x.author not in feats:
                feats[x.author] = {'c': 0, 'n': 0}
            if not util.is_user_retweet(x.tweet):
                pbar.update()
                continue
            tokens = util.tokenize(x.tweet, False)
            feats[x.author]['n'] += len(tokens)
            feats[x.author]['c'] += sum(1 for t in tokens if t in noble_dict)
            pbar.update()

    def get_freq(c, n):
        return c / n if n > 0 else 0.

    feats = [{'author': k, 'rt_usr_func_freq': get_freq(v['c'], v['n'])}
             for k, v in feats.items()]

    feats = pd.DataFrame(feats)
    feats.set_index(keys=['author'], inplace=True)

    return feats
