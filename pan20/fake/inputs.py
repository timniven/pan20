"""Code for parsing incoming xml into a basic, common data format.

That format is a DataFrame with columns: author, tweet.
"""
from xml.etree import ElementTree as ET
import os

import pandas as pd


def parse(input_dir):
    """Parse an unpacked dataset into a Dataframe.

    Args:
      input_dir: String, path to the unpacked dataset.

    Returns:
      pandas.DataFrame.
    """
    en_path = os.path.join(input_dir, 'en')
    data = []
    for file_name in os.listdir(en_path):
        if file_name == 'truth.txt':
            continue
        file_path = os.path.join(en_path, file_name)
        root = ET.parse(file_path).getroot()
        docs = root.find('documents')
        for doc in docs.findall('document'):
            data.append({
                'author': file_name.replace('.xml', ''),
                'tweet': doc.text,
            })
    return pd.DataFrame(data)


def find_en():
    """Finds the folder where the en input data is.

    Not sure what to expect as folder names in the zip file.

    Returns:
      String.
    """
    base = 'tmp/unzipped_data'
    if 'en' in os.listdir(base):
        return os.path.join(base, 'en')
    # guess only one possible folder in between from structure of train data
    sub_folder = os.listdir(base)[0]
    sub_folder = os.path.join(base, sub_folder)
    en_path = os.path.join(sub_folder, 'en')
    if not os.path.exists(en_path):
        raise Exception('Where is the data? Not in %s' % en_path)
    return en_path
