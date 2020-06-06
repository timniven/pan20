import pickle

from sklearn.naive_bayes import GaussianNB


def early_bird():
    with open('clf', 'rb') as f:
        return pickle.loads(f.read())


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


class LexicalSVM:

    def __init__(self):
        pass
