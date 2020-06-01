"""Multi-Layer Perceptron for use with top-k n-gram features."""
from torch import nn
import numpy as np
import torch

from pan20.auth.models import base
from pan20.util.pytorch import config, training
from pan20.util import topkfreqs


class Config(config.Config):

    def __init__(self, k1, k2, k3, hidden_size, n_layers, dropout,
                 weight_decay):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.weight_decay = weight_decay


class MLP(base.BaseModel):

    def __init__(self, k1, k2, k3, hidden_size, n_layers, dropout,
                 weight_decay):
        """Create new MLP.

        Args:
          in_features: Integer.
          hidden_size: Integer.
          n_layers: Integer.
          dropout: Float.
          weight_decay: Float. L2 regularization.
        """
        super().__init__()
        in_features = k1 + k2 + k3
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.weight_decay = weight_decay
        self.layers = []
        for i in range(n_layers):
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.Linear(
                in_features=in_features if i == 0 else hidden_size,
                out_features=hidden_size
            ))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_size))
        self.layers = nn.Sequential(*self.layers)
        self.classify = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X, labels):
        """Calculate logits and loss.

        Args:
          X: Tensor of shape [batch, feats].
          labels: Tensor of shape [batch].

        Returns:
          loss: Tensor of shape [1].
          logits: Tensor of shape [batch].
        """
        feats = self.layers(X)
        logits = self.classify(feats)
        loss = self.loss(logits, labels)
        return loss, logits


class TopKFreqsBatch(training.ClassificationBatch):

    def __init__(self, X, y):
        super().__init__(labels=y)
        self.X = X


class CollateTopKNGrams:

    def __init__(self, k1=1024, k2=2048, k3=4096):
        super().__init__()
        self.vectorize_unigrams = topkfreqs.Vectorizer(k=k1, n=1)
        if k2 > 0:
            self.vectorize_bigrams = topkfreqs.Vectorizer(k=k2, n=2)
        if k3 > 0:
            self.vectorize_trigrams = topkfreqs.Vectorizer(k=k3, n=3)

    def __call__(self, items):
        X = [i[0] for i in items]
        d0s = [x['pair'][0] for x in X]
        d1s = [x['pair'][1] for x in X]
        v0s = self.get_vectors(d0s)
        v1s = self.get_vectors(d1s)
        X = torch.abs(v0s - v1s)

        y = torch.LongTensor([int(i[1]['same']) for i in items])

        return TopKFreqsBatch(X, y)

    def get_vectors(self, docs):
        vecs = []
        for doc in docs:
            vec_uni = self.vectorize_unigrams(doc, expand_axis=0)
            # vec_bi = self.vectorize_bigrams(doc, expand_axis=0)
            # vec_tri = self.vectorize_trigrams(doc, expand_axis=0)
            # vec = np.concatenate([
            #     vec_uni,
            #     # vec_bi,
            #     # vec_tri
            # ], axis=1)
            vecs.append(vec_uni)
        vecs = np.concatenate(vecs, axis=0)
        return torch.from_numpy(vecs).float()
