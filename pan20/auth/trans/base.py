"""Base classes for Transformer models."""
from torch import nn
import torch

from pan20 import auth
from pan20.util.pytorch import layers


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
        feats = torch.cat([docs0, docs1], dim=1)
        feats = self.dropout(feats)
        return self.classify(feats)


class MouClassify(Classify):

    def __init__(self, n_feats, p_drop):
        super().__init__()
        self.classify = nn.Linear(n_feats * 4, 2)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, docs0, docs1):
        feats = torch.cat([docs0, docs1, docs0 * docs1, docs0 - docs1], dim=1)
        feats = self.dropout(feats)
        return self.classify(feats)


class BaseModel(nn.Module):

    def optim_params(self):
        no_decay = ['bias', 'gamma', 'beta']
        parameters = [
            {'params': [p for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.weight_decay},
            {'params': [p for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        return parameters


class BaseFrozenModel(nn.Module):
    raise NotImplementedError


class ComparisonModel(BaseModel):
    """Comparison of reps of each document via classification layer."""

    def __init__(self, doc_enc, classify, weight_decay):
        super().__init__()
        self.doc_enc = doc_enc
        self.classify = classify
        self.weight_decay = weight_decay
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


class FandomClassify(nn.Module):
    # TODO: the classify class patterns are rubbish.

    def __init__(self, n_feats, p_drop):
        super().__init__()
        self.linear = nn.Linear(n_feats, auth.n_fandoms)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, feats):
        feats = self.dropout(feats)
        return self.linear(feats)


class AdversarialFdComparisonModel(ComparisonModel):
    """Uses adversarial loss to discourage feats useful to classify fandom."""

    def __init__(self, doc_enc, classify_match, classify_fandom, weight_decay,
                 lambda_fd, lambda_grad):
        super().__init__(doc_enc, classify_match, weight_decay)
        self.doc_enc = doc_enc
        self.grad_rev = layers.GradientReversal(lambda_grad)
        self.classify_fandom = classify_fandom
        self.weight_decay = weight_decay
        self.loss = nn.CrossEntropyLoss()
        self.lambda_fd = lambda_fd

    def forward(self, seqs0, seqs1, fandoms0, fandoms1, authors0, authors1,
                labels):
        # [batch, n_feats]
        docs0 = self.doc_enc(seqs0, fandoms0)
        docs1 = self.doc_enc(seqs1, fandoms1)

        # combine docs and fandoms into a single tensor for prediction
        docs = torch.cat([docs0, docs1], dim=0)
        fandoms = torch.cat([fandoms0, fandoms1], dim=0).squeeze()

        # gradient reversal layer for fandom predictions
        docs = self.grad_rev(docs)

        # classification layer
        logits_match = self.classify(docs0, docs1)
        logits_fandom = self.classify_fandom(docs)

        # calculate loss
        loss_match = self.loss(logits_match, labels)
        loss_fandom = self.loss(logits_fandom, fandoms)

        # combine losses
        loss = loss_match + self.lambda_fd * loss_fandom

        return loss, logits_match
