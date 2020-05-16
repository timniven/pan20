"""For handling ouputs of predictions."""
import os


def save(preds):
    # preds are dicts like {author, pred}
    folder = 'preds/en'
    if not os.path.exists('preds'):
        os.mkdir('preds')
    if not os.path.exists(folder):
        os.mkdir(folder)
    for pred in preds:
        file_path = os.path.join(folder, '%s.xml' % pred['author'])
        with open(file_path, 'w+') as f:
            f.write(get_xml(pred))


def get_xml(pred):
    return '<author id="%s"\n    lang="en"\n    type="%s"\n/>' \
           % (pred['author'], pred['pred'])
