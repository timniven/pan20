import os
from xml.etree import ElementTree as ET

import pandas as pd


# NOTE: use this one
def load(lang='en'):  # author, tweet, label
    data = load_data(lang)
    truth = {x['author']: x['label'] for x in load_truth(False, lang)}
    data['label'] = data.author.apply(lambda x: truth[x])
    data['lang'] = lang
    return data


def load_data(lang='en'):
    data = []
    for fname in os.listdir(f'data/fake/{lang}'):
        if fname == 'truth.txt':
            continue
        root = ET.parse(f'data/fake/{lang}/{fname}').getroot()
        docs = root.find('documents')
        for doc in docs.findall('document'):
            data.append({
                'author': fname.replace('.xml', ''),
                'tweet': doc.text,
            })
    return pd.DataFrame(data)


def load_truth(as_df=False, lang='en'):
    truth = []
    with open(f'data/fake/{lang}/truth.txt') as f:
        for line in f.readlines():
            line = line.strip()
            author, label = line.split(':::')
            truth.append({
                'author': author,
                'label': int(label),
            })

    if as_df:
        truth = pd.DataFrame(truth)
        truth.set_index(keys=['author'], inplace=True)
        return truth

    else:
        return truth
