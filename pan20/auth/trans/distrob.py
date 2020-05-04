"""DistilRoBERTa models."""
import torch
import transformers

from pan20.auth.trans import base
from pan20.auth import pytorch


class DistilRoBERTa(base.TransformerModel):

    def __init__(self):
        config = transformers.RobertaConfig(output_hidden_states=True)
        model = transformers.RobertaModel.from_pretrained(
            'distilroberta-base', config=config)
        super().__init__(model=model)

    def forward(self, ixs):
        # 13-tuple of shape[batch, max_sent_len, hidden], n_layers + 1(embeds)
        _, _, hidden_states = self.roberta(ixs)
        # so we can first make a big tensor then reduce
        hidden_states = [x.unsqueeze(1) for x in hidden_states]
        # [batch, n_layers + 1, max_sent_len, hidden]
        return torch.cat(hidden_states, dim=1)


class DistilRoBERTaComparison1(base.ComparisonModel):

    def __init__(self):
        transformer = DistilRoBERTa()
        extract_feats = base.MeanStdMax()
        combine_layers = base.WeightedSum(n_layers=6)
        doc_enc = base.SeqEnc(transformer, extract_feats, combine_layers)
        classify = base.LinearClassify(n_feats=extract_feats.n_feats)
        super().__init__(doc_enc=doc_enc, classify=classify)


class CollateFirstK(pytorch.CollateFirstK):

    def __init__(self, k=300):
        tokenizer = transformers.RobertaTokenizer\
            .from_pretrained('distilroberta-base')
        super().__init__(tokenizer=tokenizer, k=k)
