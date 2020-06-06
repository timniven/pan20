"""Putting data points in SQLite."""
from tqdm.notebook import tqdm

from pan20.util.sqldb import DBInterface, orm


class Add:

    def __init__(self):
        self.dbi = DBInterface()

    def __call__(self, text, id_, author_id, fandom_name):
        author = self.dbi.authors.add_or_get({'id': author_id})
        fandom = self.dbi.fandoms.add_or_get({'name': fandom_name})
        doc = orm.Doc(
            id=id_,
            text=text,
            author=author,
            fandom=fandom)
        self.dbi.docs.add(doc)

        self.dbi.commit()  # does the update


def add_all(X, Y, n):
    add = Add()
    id_set = set()
    with tqdm(total=n) as pbar:
        for i in range(n):
            x, y = X[i], Y[i]
            id_ = x['id']
            if id_ in id_set:
                pbar.update()
                continue
            id_set.update([id_])
            for j in range(2):
                add(
                    text=x['pair'][j],
                    id_=f'{id_}.{j}',
                    author_id=y['authors'][j],
                    fandom_name=x['fandoms'][j])
            pbar.update()
