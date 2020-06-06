import joblib

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import xgboost as xgb
from pan20.fake import run_pretrain

from pan20.util import ctree, text
from pan20.util.lexicons import noble, sentiwordnet, trust, tweet_anger


xgb_in_path = 'tmp/xgb_in.txt'


def predict(data):
    """Predict function for software submission.

    Args:
      data: pandas.DataFrame with one row per tweet.

    Returns:
      List of dicts like {author, pred}.
    """
    # build features
    X, authors = get_features(data)
    bert_features = run_pretrain.get_encoded(data, 'bert-large')
    roberta_features = run_pretrain.get_encoded(data, 'roberta-base')

    # load saved models
    svc = joblib.load('pan20/fake/svc.model')
    rf = joblib.load('pan20/fake/rf.model')
    nb = joblib.load('pan20/fake/nb.model')
    bert = joblib.load('pan20/fake/bert-large-sigmoid.model')
    roberta = joblib.load('pan20/fake/roberta-base.model')
    #bst = joblib.load('pan20/fake/bst.model')
    bst = joblib.load('pan20/fake/combined_bst.model')

    # get model predictions
    preds_svc = get_preds(svc, X)
    preds_rf = get_preds(rf, X)
    preds_nb = get_preds(nb, X)
    preds_bert = get_preds(bert, bert_features)
    preds_roberta = get_preds(roberta, roberta_features)

    # generate inputs to xgboost model
    to_txt(preds_svc=preds_svc, preds_rf=preds_rf, preds_nb=preds_nb,
           preds_bert=preds_bert, preds_roberta=preds_roberta)

    # get predictions
    dmatrix = xgb.DMatrix(xgb_in_path)
    preds = bst.predict(dmatrix)
    preds = [p > 0.5 for p in preds]

    # form into expected dictionary and return
    preds = [{'author': authors[i], 'pred': preds[i]}
             for i in range(len(preds))]

    return preds


def get_features(df):
    print('Getting features...')
    # tokenize
    df['toks'] = df.tweet.apply(text.tokenize)

    # noble: adverbs, impersonal_pronouns, personal_pronouns, function_words
    # ctree: avg_bf, max_np_height, max_vp_height
    # sentiwordnet: senti, senti_neg
    # other lexical: anger, distrust
    noble_dict = noble.NobleDict()
    get_tree = ctree.GetTree()
    swn = sentiwordnet.SentiWordNet()
    anger_dict = tweet_anger.Lexicon()
    distrust_dict = trust.Lexicon()

    # do it all in one pass over the data
    adverbs = []
    impersonal_pronouns = []
    personal_pronouns = []
    function_words = []
    avg_bf = []
    max_np_height = []
    max_vp_height = []
    senti = []
    senti_neg = []
    anger = []
    distrust = []

    with tqdm(total=len(df)) as pbar:
        for _, x in df.iterrows():
            # noble
            adverbs.append(noble_dict.freq(x.toks, 'adverbs'))
            impersonal_pronouns.append(
                noble_dict.freq(x.toks, 'impersonal_pronouns'))
            personal_pronouns.append(
                noble_dict.freq(x.toks, 'personal_pronouns'))
            function_words.append(
                noble_dict.freq(x.toks, 'function_words'))

            # ctree
            tree = get_tree(x.tweet)
            avg_bf.append(ctree.avg_branch_factor(tree))
            max_np_height.append(ctree.max_const_height(tree, 'NP'))
            max_vp_height.append(ctree.max_const_height(tree, 'VP'))

            # sentiwordnet
            senti.append(swn.score(x.toks))
            senti_neg.append(swn.score_neg(x.toks))

            # other lexical
            anger.append(anger_dict.cat_freq(x.toks, 'anger'))
            distrust.append(distrust_dict.cat_freq(x.toks, 'distrust'))

            pbar.update()

    df['adverbs'] = adverbs
    df['impersonal_pronouns'] = impersonal_pronouns
    df['personal_pronouns'] = personal_pronouns
    df['function_words'] = function_words
    df['avg_bf'] = avg_bf
    df['max_np_height'] = max_np_height
    df['max_vp_height'] = max_vp_height
    df['senti'] = senti
    df['senti_neg'] = senti_neg
    df['anger'] = anger
    df['distrust'] = distrust

    # group by author
    df.drop(columns=['tweet'], inplace=True)
    feats = df.groupby('author').mean().reset_index()
    authors = list(feats.author.values)
    feats = feats.loc[:, [c != 'author' for c in feats.columns]]
    X = feats.values

    # normalize
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, authors


def get_preds(clf, X):
    return list(np.exp(clf.predict_log_proba(X))[:, 1])


def to_txt(preds_svc, preds_rf, preds_nb, preds_bert, preds_roberta):
    with open(xgb_in_path, 'w+') as f:
        for i in range(len(preds_svc)):
            row = '0:%s 1:%s 2:%s\n' % (preds_svc[i], preds_rf[i], preds_nb[i], preds_bert[i], preds_roberta[i])
            f.write(row)


class Model:
    """Base class, defining required interface."""

    def predict(self, df):
        """Predict for the task.

        Args:
          df: pandas.DataFrame with one row per tweet.

        Returns:
          preds: pandas.DataFrame where each row has `author`, and `prob` (i.e.
            probability that author is a spreader.
        """
        raise NotImplementedError


class LexicalModel(Model):

    def __init__(self, clf):
        self.clf = clf

    def predict(self, df):
        X, authors = self.get_features(df)
        preds = self.get_preds(X)
        preds = pd.DataFrame({
            'author': authors,
            'probability': preds
        })
        return preds

    def get_preds(self, X):
        return list(np.exp(self.clf.predict_log_proba(X))[:, 1])


class LexicalSVM(LexicalModel):

    def __init__(self):
        clf = joblib.load('pan20/fake/svc.model')
        super().__init__(clf=clf)


class LexicalNB(LexicalModel):

    def __init__(self):
        clf = joblib.load('pan20/fake/nb.model')
        super().__init__(clf=clf)


class LexicalRF(LexicalModel):

    def __init__(self):
        clf = joblib.load('pan20/fake/rf.model')
        super().__init__(clf=clf)
