"""Multi-Layer Perceptron for use with top-k n-gram features."""
from torch import nn


class MLP(nn.Module):

    def __init__(self, in_features, hidden_size, n_layers, dropout):
        """Create new MLP.

        Args:
          in_features: Integer.
          hidden_size: Integer.
          n_layers: Integer.
          dropout: Float.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.layers = []
        for i in range(n_layers):
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.Linear(
                in_features=in_features if i == 0 else hidden_size,
                out_features=hidden_size
            ))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d)
        self.layers = nn.Sequential(*self.layers)
        self.classify = nn.Linear(hidden_size, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X, y):
        """Calculate logits and loss.

        Args:
          X: Tensor of shape [batch, in_features].
          y: Tensor of shape [batch].

        Returns:
          loss: Tensor of shape [1].
          logits: Tensor of shape [batch].
        """
        feats = self.layers(X)
        logits = self.classify(feats)
        loss = self.loss(logits, y)
        return loss, logits
