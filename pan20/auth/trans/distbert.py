"""DistilBERT models."""
import torch
import transformers

from pan20.auth.trans import base
from pan20.auth import pytorch


class DistilBERT(base.TransformerModel):

    def __init__(self):
        config = transformers.DistilBertConfig\
            .from_pretrained('distilbert-base-cased')
        config.output_hidden_states = True
        model = transformers.DistilBertModel.from_pretrained(
            'distilbert-base-cased', config=config)
        super().__init__(model=model)

    def forward(self, ixs):
        # 13-tuple of shape[batch, max_sent_len, hidden], n_layers + 1
        _, hidden_states = self.model(ixs)
        # so we can first make a big tensor then reduce
        hidden_states = [x.unsqueeze(1) for x in hidden_states]
        # [batch, n_layers + 1, max_sent_len, hidden]
        return torch.cat(hidden_states, dim=1)


class FrozenDistilBERT(DistilBERT):

    def forward(self, ixs):
        with torch.no_grad():
            # 13-tuple of shape[batch, max_sent_len, hidden], n_layers + 1
            _, hidden_states = self.model(ixs)
            # so we can first make a big tensor then reduce
            hidden_states = [x.unsqueeze(1) for x in hidden_states]
            # [batch, n_layers + 1, max_sent_len, hidden]
            return torch.cat(hidden_states, dim=1)


class DistilBERTComparison1(base.ComparisonModel):

    def __init__(self, p_drop):
        transformer = DistilBERT()
        extract_feats = base.MeanStdMax()
        combine_layers = base.WeightedSum(n_layers=6)
        doc_enc = base.SeqEnc(transformer, extract_feats, combine_layers)
        classify = base.LinearClassify(
            n_feats=extract_feats.n_feats, p_drop=p_drop)
        super().__init__(doc_enc=doc_enc, classify=classify)


class FrozenDistilBERTComparison1(base.ComparisonModel):

    def __init__(self, p_drop):
        transformer = FrozenDistilBERT()
        extract_feats = base.MeanStdMax()
        combine_layers = base.WeightedSum(n_layers=6)
        doc_enc = base.SeqEnc(transformer, extract_feats, combine_layers)
        classify = base.LinearClassify(
            n_feats=extract_feats.n_feats, p_drop=p_drop)
        super().__init__(doc_enc=doc_enc, classify=classify)

    def optim_params(self):
        prefix = 'doc_enc.transformer.model.transformer.'
        params = [p for n, p in self.named_parameters()
                  if not n.startswith(prefix)]
        return params
        # TODO: deal with this
        # no_decay = ['bias', 'gamma', 'beta']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer
        #                 if not any(nd in n for nd in no_decay)],
        #      'weight_decay_rate': config.weight_decay},
        #     {'params': [p for n, p in param_optimizer
        #                 if any(nd in n for nd in no_decay)],
        #      'weight_decay_rate': 0.0}
        # ]


class CollateFirstK(pytorch.CollateFirstK):

    def __init__(self, k=300):
        tokenizer = transformers.DistilBertTokenizer \
            .from_pretrained('distilbert-base-cased')
        super().__init__(tokenizer=tokenizer, k=k)
