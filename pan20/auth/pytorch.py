from torch.utils import data
import torch

from pan20 import auth
from pan20.util.pytorch import training


class Dataset(data.Dataset):

    def __init__(self, X, truth, char_lim=2500):
        self.X = X
        self.y = truth
        if char_lim:
            for x in self.X:
                for i in range(2):
                    x['pair'][i] = x['pair'][i][0:char_lim]

    def __getitem__(self, ix):
        return self.X[ix], self.y[ix]

    def __len__(self):
        return len(self.X)


class SmallDataset(data.Dataset):
    """Torch Dataset for the small training set.."""

    def __init__(self, char_lim=2500):
        """Initialize a torch Dataset for the small version.

        Args:
          char_lim: Int, all documents will be limited to the first `char_lim`
            characters. Default is 2500, which ensures 300 tokens by the
            RobertaTokenizer for all docs. Set to `None` to keep the full
            documents.
        """
        self.X, self.y = auth.load_small()
        if char_lim:
            for x in self.X:
                for i in range(2):
                    x['pair'][i] = x['pair'][i][0:char_lim]

    def __getitem__(self, ix):
        return self.X[ix], self.y[ix]

    def __len__(self):
        return len(self.X)


class Batch(training.ClassificationBatch):

    def __init__(self, seqs0, seqs1, fandoms0, fandoms1, authors0, authors1,
                 labels):
        super().__init__(labels)
        self.seqs0 = seqs0
        self.seqs1 = seqs1
        self.fandoms0 = fandoms0
        self.fandoms1 = fandoms1
        self.authors0 = authors0
        self.authors1 = authors1


class Collate:
    """Base collate function."""

    def __init__(self):
        self.auth_dict = auth.get_auth_dict()
        self.fandom_dict = auth.get_fandom_dict()

    def __call__(self, items):
        """Collate function.

        Args:
          items: List of tuples of (x, y), returned from DataSet.__getitem__.

        Returns:
          Batch object.
        """
        # split apart the tuples
        X = [i[0] for i in items]
        truth = [i[1] for i in items]
        y = [int(i[1]['same']) for i in items]

        # encode text seqs
        seqs0 = self.encode_seqs([x['pair'][0] for x in X])
        seqs1 = self.encode_seqs([x['pair'][1] for x in X])

        # lookup fandoms and cast to tensors
        fandoms0 = self.dict_lookup(
            items=[x['fandoms'][0] for x in X],
            ix_dict=self.fandom_dict)
        fandoms1 = self.dict_lookup(
            items=[x['fandoms'][1] for x in X],
            ix_dict=self.fandom_dict)

        # lookup authors and cast to tensors
        auths0 = self.dict_lookup(
            items=[x['authors'][0] for x in truth],
            ix_dict=self.auth_dict)
        auths1 = self.dict_lookup(
            items=[x['authors'][1] for x in truth],
            ix_dict=self.auth_dict)

        # cast labels as tensor
        labels = torch.LongTensor(y)

        # return a batch objects
        return Batch(seqs0, seqs1, fandoms0, fandoms1, auths0, auths1, labels)

    def dict_lookup(self, items, ix_dict):
        ixs = [ix_dict[x] for x in items]
        return torch.LongTensor(ixs).unsqueeze(dim=1)

    def encode_seqs(self, docs):
        raise NotImplementedError


class CollateFirstK(Collate):
    """Collate function for first k tokens of each doc."""

    def __init__(self, tokenizer, k=300):
        super().__init__()
        self.tokenizer = tokenizer
        self.k = k

    def encode_seqs(self, docs):
        docs = [self.encode_seq(doc) for doc in docs]
        return torch.cat(docs, dim=0)

    def encode_seq(self, doc):
        doc = self.tokenizer.encode(
            doc, max_length=self.k, pad_to_max_length=True)
        return torch.LongTensor(doc).unsqueeze(0)


def dataloaders_small(collate, train_batch_size, tune_batch_size):
    X_train, y_train, X_dev, y_dev, X_test, y_test = auth.small()
    train = Dataset(X_train, y_train)
    dev = Dataset(X_dev, y_dev)
    test = Dataset(X_test, y_test)
    train_loader = data.DataLoader(
        batch_size=train_batch_size,
        collate_fn=collate,
        dataset=train,
        shuffle=True)
    dev_loader = data.DataLoader(
        batch_size=tune_batch_size,
        collate_fn=collate,
        dataset=dev,
        shuffle=False)
    test_loader = data.DataLoader(
        batch_size=tune_batch_size,
        collate_fn=collate,
        dataset=test,
        shuffle=False)
    return train_loader, dev_loader, test_loader
