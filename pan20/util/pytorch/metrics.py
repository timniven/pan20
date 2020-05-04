"""Training metrics."""
from sklearn import metrics
import numpy as np


def get(name):
    if name == 'acc':
        return Accuracy()
    else:
        raise ValueError(f'Unexpected metric: {name}')


class Metric:
    """Abstract base class, defining the Metric interface."""

    def __init__(self, fn, abbr, criterion):
        """Create a new Metric.

        Args:
          fn: the metric function. It should accept two numpy arrays and return
            a scalar. The arguments are (y_true, y_pred). This is the sklearn
            metrics interface. Will be the callable for this class.
          abbr: string abbreviation for the metric name. E.g. `acc` for
            accuracy_score.
          criterion: a function on a set of metric scores that defines the best
            one. E.g. for accuracy np.max.
        """
        self.fn = fn
        self.abbr = abbr
        self.criterion = criterion

    def __call__(self, y_true, y_pred):
        if y_true.ndim == 1:
            y_true = np.expand_dims(y_true, 1)
        return self.fn(y_true, y_pred)

    def best(self, scores):
        # handle the case where scores are an item in dicts
        if isinstance(scores[0], dict):
            scores = [x[self.abbr] for x in scores]
        return self.criterion(scores)

    def is_best(self, score, scores):
        return score == self.best(scores)

    def is_better(self, score1, score0):
        """Determine if score1 improves on score0.

        Args:
          score1: Float.
          score0: Float.

        Returns:
          Bool.
        """
        return score1 == self.best([score0, score1])


class Accuracy(Metric):
    """Accuracy metric."""

    def __init__(self):
        super().__init__(metrics.accuracy_score, 'acc', np.max)
