import pickle

from sklearn.naive_bayes import GaussianNB


def early_bird():
    with open('clf', 'rb') as f:
        return pickle.loads(f.read())
