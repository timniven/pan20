"""Model for the competition."""
import joblib
import numpy as np
import xgboost as xgb

from pan20.util import text, topkfreqs


xgb_in_path = 'tmp/xgb_in_auth.txt'


vz1 = topkfreqs.Vectorizer(k=1024, n=1)
vz2 = topkfreqs.Vectorizer(k=2048, n=2)


def get_feats1(x):
    d0 = x['pair'][0]
    d1 = x['pair'][1]
    d0 = text.simple_tokenize(d0)
    d1 = text.simple_tokenize(d1)
    v0 = vz1(d0)
    v1 = vz1(d1)
    d = np.abs(v0 - v1)
    d = np.expand_dims(d, 0)
    return d


def get_feats2(x):
    d0 = x['pair'][0]
    d1 = x['pair'][1]
    d0 = text.simple_tokenize(d0, n=2)
    d1 = text.simple_tokenize(d1, n=2)
    v0 = vz2(d0)
    v1 = vz2(d1)
    d = np.abs(v0 - v1)
    d = np.expand_dims(d, 0)
    return d


def predict(data):
    """Predicts data for the software submission.

    Args:
      data: Generator of json objects each representing a data point to predict.

    Returns:
      List of dictionaries with predictions like {id: string, value: prob}.
    """
    ids = [x['id'] for x in data]

    # get feats
    X1 = np.concatenate([get_feats1(x) for x in data])
    X2 = np.concatenate([get_feats2(x) for x in data])

    # classifiers
    svc1 = joblib.load('pan20/auth/svc1.model')
    rf1 = joblib.load('pan20/auth/rf1.model')
    nb1 = joblib.load('pan20/auth/nb1.model')
    svc2 = joblib.load('pan20/auth/svc2.model')
    rf2 = joblib.load('pan20/auth/rf2.model')
    nb2 = joblib.load('pan20/auth/nb2.model')
    bst = joblib.load('pan20/auth/bst.model')

    # preds
    p_svc1 = get_preds(svc1, X1)
    p_rf1 = get_preds(rf1, X1)
    p_nb1 = get_preds(nb1, X1)
    p_svc2 = get_preds(svc2, X2)
    p_rf2 = get_preds(rf2, X2)
    p_nb2 = get_preds(nb2, X2)

    # to text
    to_txt(p_svc1, p_rf1, p_nb1, p_svc2, p_rf2, p_nb2)

    # xgboost
    dmatrix = xgb.DMatrix(xgb_in_path)
    preds = bst.predict(dmatrix)

    # form into expected dictionary and return
    preds = [{'id': ids[i], 'value': float(preds[i])}
             for i in range(len(preds))]

    return preds


def get_preds(clf, X):
    return list(np.exp(clf.predict_log_proba(X))[:, 1])


def to_txt(svc1, rf1, nb1, svc2, rf2, nb2):
    with open(xgb_in_path, 'w+') as f:
        for i in range(len(svc1)):
            row = f'0:%s 1:%s 2:%s 3:%s 4:%s 5:%s\n' \
                  % (svc1[i], rf1[i], nb1[i], svc2[i], rf2[i], nb2[i])
            f.write(row)
