"""Base classes for Transformer models."""
from torch.nn import functional as F
from torch import nn
import torch


hidden = 768  # standard hidden unit size for all models


class TransformerModel(nn.Module):
    """Wraps a Transformer model, and returns layer outputs as a tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ixs):
        """Forward pass.

        Args:
          ixs: Tensor of shape [batch, seq_len].

        Returns:
          Tensor of shape [batch, n_layers, seq_len, hidden]
        """
        raise NotImplementedError('Implement for each Transformer model.')


class ExtractLayerFeats(nn.Module):
    """Extracts features from the sequence for each layer."""

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        """Forward pass.

        Args:
          hidden_states: Tensor of shape [batch, n_layers, seq_len, hidden].

        Returns:
          Tensor of shape [batch, n_layers, n_feats].
        """
        raise NotImplementedError('Subclasses to implement.')


class MeanStdMax(ExtractLayerFeats):
    """Concats mean, std, and max over seq dim for each layer."""

    def __init__(self):
        super().__init__()
        self.n_feats = hidden * 3

    def forward(self, hidden_states):
        means = hidden_states.sum(dim=2)
        stds = hidden_states.std(dim=2)
        maxs = hidden_states.max(dim=2).values
        return torch.cat([means, stds, maxs], dim=2)


class CombineLayers(nn.Module):
    """Abstract base class for module to combine layer representations."""

    def __init__(self):
        super().__init__()

    def forward(self, layer_feats):
        """Forward pass.

        Args:
          layer_feats: Tensor of shape [batch, n_layers, n_feats].

        Returns:
          Tensor of shape [batch, n_feats].
        """
        raise NotImplementedError('Subclasses to implement.')


class WeightedSum(CombineLayers):
    """Default implementation for combining layers: learned weighted sum."""

    def __init__(self, n_layers):
        super().__init__()
        self.layer_weights = nn.Parameter(
            torch.ones(1, n_layers + 1, 1) * (1 / (n_layers + 1)))

    def forward(self, layer_feats):
        """Forward pass.

        Args:
          layer_feats: Tensor of shape [batch, n_layers, n_feats].

        Returns:
          Tensor of shape [batch, n_feats].
        """
        layer_feats = layer_feats * self.layer_weights
        return layer_feats.sum(dim=1)


class SeqEnc(nn.Module):
    """Base class for a Transformer sequence encoder.

    Handles single sequences (i.e. not the comparison of two). Could be a
    sentence or a sequence of text.

    Components:
    - Transformer model: outputs hidden states for each item for each layer.
    - Layer feature extractor: extracts features for each layer, reducing the
      sequence to a vector.
    - Layer combiner: combines features for each layer into a final sequence
      representation.
    """

    def __init__(self, transformer, extract_feats, combine_layers):
        super().__init__()
        self.transformer = transformer
        self.extract_feats = extract_feats
        self.combine_layers = combine_layers

    def forward(self, ixs, fandoms):
        """Forward pass.

        Args:
          ixs: Tensor of shape [batch, max_seq_len].
          fandoms: Tensor of shape [batch, 2].

        Outputs:
          Tensor of shape [batch, n_feats], where n_feats is defined by the
            `layer_feats` module.
        """
        hidden_states = self.transformer(ixs)
        layer_feats = self.extract_feats(hidden_states)
        seq_rep = self.combine_layers(layer_feats)
        return seq_rep


class Classify(nn.Module):
    """Abstract base class for classifying two document representations."""

    def __init__(self):
        super().__init__()

    def forward(self, docs0, docs1):
        """Forward pass.

        Args:
          docs0, docs1: Tensor of shape [batch, n_feats].

        Returns:
          Tensor of shape [batch, 2].
        """
        raise NotImplementedError('Subclasses to implement.')


class LinearClassify(Classify):

    def __init__(self, n_feats, p_drop):
        super().__init__()
        self.classify = nn.Linear(n_feats * 2, 2)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, docs0, docs1):
        in_features = torch.cat([docs0, docs1], dim=1)
        in_features = self.dropout(in_features)
        return self.classify(in_features)


class ComparisonModel(nn.Module):
    """Comparison of reps of each document via classification layer."""

    def __init__(self, doc_enc, classify):
        super().__init__()
        self.doc_enc = doc_enc
        self.classify = classify
        self.loss = nn.CrossEntropyLoss()

    def forward(self, seqs0, seqs1, fandoms0, fandoms1, authors0, authors1,
                labels):
        # [batch, n_feats]
        docs0 = self.doc_enc(seqs0, fandoms0)
        docs1 = self.doc_enc(seqs1, fandoms1)

        # classification layer
        logits = self.classify(docs0, docs1)

        # calculate loss
        loss = self.loss(logits, labels)

        return loss, logits

    def optim_params(self):
        return self.parameters()
