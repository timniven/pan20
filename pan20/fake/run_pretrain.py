

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import cross_val_predict

class PanDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        text, label = self.data.iloc[idx, :].values
        label_tensors = torch.tensor(label)
        word_pieces = ["[CLS]"]
        tokens = self.tokenizer.tokenize(text)
        word_pieces += tokens + ["[SEP]"]

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensors = torch.tensor(ids)

        return tokens_tensors, label_tensors

def collate_fn(batch):

    tokens_tensors = [data[0] for data in batch]
    if batch[0][1] is not None:
        label_ids = torch.stack([label[1] for label in batch])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, masks_tensors, label_ids


def run_bert(df, p_model):
    '''Args:
            df: original dataframe of Pan profiling
            p_model: choice['bert-base', 'bert-large', 'roberta-base', 'roberta-large']
       Return:
            (u_vectors) encoded vectors by BERT
    '''
    print("Loading {} model".format(p_model))
    if p_model=='bert-base':
        from transformers import BertModel
        from transformers import BertTokenizer
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model= BertModel.from_pretrained('bert-base-uncased')
        
    if p_model=='bert-large':
        from transformers import BertModel
        from transformers import BertTokenizer
        
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model= BertModel.from_pretrained('bert-large-uncased')
        
    if p_model=='roberta-base':
        from transformers import RobertaModel
        from transformers import RobertaTokenizer

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model= RobertaModel.from_pretrained('roberta-base')
        
    if p_model=='roberta-large':
        from transformers import RobertaModel
        from transformers import RobertaTokenizer
        
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model= RobertaModel.from_pretrained('roberta-large')
        
    df_test = df.drop('author', 1)
    dataset = PanDataset(df_test, tokenizer)
    testloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print("Start encoding")
    u_vectors = []
    with torch.no_grad():
        for data in testloader:
            tokens_tensors, masks_tensors = data[:2]
            tokens_tensors = tokens_tensors.to(device)
            masks_tensors = masks_tensors.to(device)
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=None,
                            attention_mask=masks_tensors)

            u_vectors.append(outputs[0][:,0,:].cpu().squeeze().numpy())

            del tokens_tensors
            del masks_tensors

    return u_vectors
    
def get_vectors(u_vectors, mode):
    '''Args:
            u_vectors: encoded vectors by BERT
            mode: 'sum' or 'avg', the way to process all grouped vectors
       Return:
            (encoded_vectors) processed vectors
    '''
    print("Start transforming vectors")
    encoded_vectors = []
    for i in range(0, 30000, 100):
        if mode == 'sum':
            encoded_vectors.append(np.sum(u_vectors[i:i+100], axis=0))
        if mode == 'avg':
            encoded_vectors.append(np.average(u_vectors[i:i+100], axis=0))
 
    return encoded_vectors

def get_encoded_df(df, mode, p_model, classifier):
    '''Args:
            df: original dataframe of Pan profiling
            p_model: choice['bert-base', 'bert-large', 'roberta-base', 'roberta-large']
            mode: 'sum' or 'avg', the way to process all grouped vectors
            classifier: classification model
       Return:
            pandas.DataFrame where each row has `author`, and `prob` (i.e. probability that author is a spreader.)
    '''
    u_vectors = run_bert(df, p_model)
    encoded = get_vectors(u_vectors, mode='avg')
    preds = {
        'author': [],
        'label': []}
    for i in range(0, 30000, 100):
        preds['author'].append(df.author[i])
        preds['label'].append(df.label[i])
    preds = pd.DataFrame(preds)
    
    proba = cross_val_predict(classifier, encoded, preds['label'], cv=5, method='predict_proba')
    is_spreader = []
    for i in range(300):
        is_spreader.append(proba[i][1])
    preds = preds.assign(probability=is_spreader)
    
    return preds