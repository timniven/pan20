from torch.utils import data

from pan20 import auth


class SmallDataset(data.Dataset):
    """Torch Dataset for the small training set.."""

    def __init__(self):
        self.X, self.y = auth.load_small()

    def __getitem__(self, ix):
        return self.X[ix], self.y[ix]

    def __len__(self):
        return len(self.X)


def collate(items):
    # items are what comes out of the Dataset.__getitem__
    # X to be a dict, y is what it is

    # guessing items are a list of tuples
    X = [i[0] for i in items]
    y = [i[1] for i in items]


