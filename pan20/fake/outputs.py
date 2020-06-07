"""For handling ouputs of predictions."""
import os


def save(preds, output_dir):
    # preds are dicts like {author, pred}
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    folder = os.path.join(output_dir, 'en')
    if not os.path.exists(folder):
        os.mkdir(folder)
    for pred in preds:
        file_path = os.path.join(folder, '%s.xml' % pred['author'])
        with open(file_path, 'w+') as f:
            f.write(get_xml(pred))


def get_xml(pred):
    return '<author id="%s" lang="en" type="%s"\n/>' \
           % (pred['author'], int(pred['pred']))
