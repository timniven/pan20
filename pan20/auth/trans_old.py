"""Transformer model(s).

Issues:
- Sentence tokenization difficult on some texts (nltk and spacy).
- Long documents (6,000 tokens).
Solution:
- Take the first k tokens of each document. Should be enough to get the style.
  Can also vary k and compare performance. Start with k=200.

Choice of RoBERTa motivated by its high performance, and because of many studies
of BERT that yield insight into its behaviour, which we can make use of here to
try and interpret what the model has learned.

Get started on DistilRoBERTa, though.
"""
import torch.nn.functional as F
from torch import nn
import torch
import transformers as tf


n_layers = {
    'roberta-base': 12,
    'distilroberta-base': 6,
}
hidden = 768  # number of units in hidden layers, same for all models



class RoBERTaSentEnc(nn.Module):
    """Encodes sentences with RoBERTa."""

    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.roberta = tf.RobertaModel.from_pretrained(model_name)
        self.layer_mixing_weights = nn.Parameter(
            torch.Tensor((n_layers[model_name], 1)))

    def forward(self, sents, lens):
        """Encode sentences.

        Args:
          sents: Tensor of shape [batch, max_sent_len].
          lens: Tensor of shape [batch, 1].

        Returns:
          Tensor of shape [batch, 1, hidden * 2].
        """
        # [batch, n_layers + 1, max_sent_len, hidden]
        hidden_states = self.get_hidden_states(sents)

        # [batch, n_layers + 1, hidden * 3]
        hidden_states = self.reduce_seqs(hidden_states, lens)

        # [batch, hidden * 2]
        hidden_states = self.combine_layers(hidden_states)

        # unsqueeze to allow concatenation along sents axis
        # [batch, 1, hidden * 2]
        hidden_states = hidden_states.unsqueeze(dim=1)

        return hidden_states

    def get_hidden_states(self, sents):
        # 13-tuple of shape[batch, max_sent_len, hidden], n_layers + 1(embeds)
        _, _, hidden_states = self.roberta(sents)

        # so we can first make a big tensor then reduce
        hidden_states = [x.unsqueeze(1) for x in hidden_states]
        # [batch, n_layers + 1, max_sent_len, hidden]
        return torch.cat(hidden_states, dim=1)

    def get_stds(self, hidden_states, means, lens):
        z = hidden_states - means
        z = z.pow(2)
        z = z / lens
        return torch.sqrt(z)

    def reduce_seqs(self, hidden_states, lens):
        # get means and stds over sequence dim
        # [batch, n_layers + 1, hidden]
        # NOTE: these mean and std ops should not count padding
        means = hidden_states.sum(dim=2) / lens
        stds = self.get_stds(hidden_states, means, lens)
        maxs = hidden_states.max(dim=2)

        # concatenate features
        # [batch, n_layers + 1, hidden * 3]
        return torch.cat([means, stds, maxs], dim=2)

    def combine_layers(self, hidden_states):
        # weighted sum of the layers
        # [batch, n_layers + 1, hidden * 3]
        hidden_states = hidden_states * self.layer_mixing_weights
        # [batch, hidden * 2]
        return hidden_states.sum(dim=1)


class DocEnc(nn.Module):
    """Basic doc encoder.

    Doc reps are formed by simple sum of sentence features.
    No conditioning on fandoms.
    """

    def __init__(self, model_name, n_feats, dim_fandom_emb):
        super().__init__()
        self.sent_enc = RoBERTaSentEnc(model_name)
        self.n_feats = n_feats
        self.dim_fandom_emb = dim_fandom_emb

    def forward(self, sents, lens, fandoms):
        """Encode a document represented as a sequence of sentences.

        Args:
          sents: Tensor of shape [batch, max_n_sents, max_seq_len].
          lens: Tensor of shape [batch, max_n_sents].
          fandoms: Tensor of shape [batch, dim_fandom_emb].

        Returns:
          Tensor of shape [].
        """
        # sentence encoding
        max_n_sents = sents.shape(1)
        # sent_enc returns a tensor of shape [batch, 1, n_feats]
        sents = [self.sent_enc(sents[ix], lens[ix])
                 for ix in range(max_n_sents)]
        # [batch, max_n_sents, n_feats]
        sents = torch.cat(sents, dim=1)

        # [batch, n_feats]
        return sents.sum(dim=1)


class DistilRoBERTaFirstKMeanStdMax(nn.Module):

    def __init__(self, k=300):
        super().__init__()








class DistilRoBERTaBasic(ComparisonModel):
    """"Basic DistilRoBERTa model.

    doc_enc: DistilRoBERTaFirstKMeanStdMax.
    classify: linear layer.
    """

    def __init__(self, k=300, n_feats=hidden*3):
        doc_enc = DistilRoBERTaFirstKMeanStdMax()
        classify = LinearClassify(n_feats)
        super().__init__(doc_enc, classify)
