"""Custom features."""
import pandas as pd
from nltk import tokenize

from pan20.util import ctree, noble, sentiwordnet


def early_bird(data):
    # noble's function word categories
    cat_freqs = noble.CatFreqs()
    freqs = []
    for _, x in data.iterrows():
        freqs.append(cat_freqs(x.tweet))
    for cat in noble.cats:
        data[cat] = [f[cat] for f in freqs]
    data['function_words'] = [f['function_words'] for f in freqs]

    # constituency tree features
    # get_tree = ctree.GetTree()
    # trees = []
    # with tqdm(total=len(data)) as pbar:
    #     for _, x in data.iterrows():
    #         trees.append(get_tree(x.tweet))
    #         pbar.update()
    # data['avg_bf'] = [ctree.avg_branch_factor(t) for t in trees]
    # data['max_np_height'] = [ctree.max_const_height(t, 'NP') for t in trees]
    # data['max_vp_height'] = [ctree.max_const_height(t, 'VP') for t in trees]

    # sentiment features
    swn = sentiwordnet.SentiWordNet()
    data['toks'] = data.tweet.apply(lambda x: tokenize.word_tokenize(x))
    data['senti'] = data.toks.apply(swn.score)
    data['senti_neg'] = data.toks.apply(swn.score_neg)
    data['senti_pos'] = data.toks.apply(swn.score_pos)

    # normalization
    # data['avg_bf'] = normalize(data, 'avg_bf')
    # data['max_np_height'] = normalize(data, 'max_np_height')
    # data['max_vp_height'] = normalize(data, 'max_vp_height')
    data['senti'] = normalize(data, 'senti')
    data['senti_neg'] = normalize(data, 'senti_neg')
    data['senti_pos'] = normalize(data, 'senti_pos')

    # drop intermediate cols
    data = data.drop(columns=['author', 'tweet', 'toks'])

    return data


def normalize(df, col):
    return (df[col] - df[col].min())/(df[col].max() - df[col].min())







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
