"""WordNet Affect Emotion List."""
from . import LexicalCategoryDict


folder = '/home/hanshan/dev/lexicons/wordnet_affect_emotion_list'
categories = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']


class WordNetAffect(LexicalCategoryDict):

    def __init__(self):
        super().__init__()
        self.word_dict = {}
        self.cat_dict = {}
        for cat in categories:
            words = parse_file(cat)
            words += [self.stemmer.stem(w) for w in words]
            words = list(set(words))
            self.cat_dict[cat] = words
            for word in words:
                if word not in self.word_dict:
                    self.word_dict[word] = []
                self.word_dict[word].append(cat)


def parse_file(category):
    file_path = f'{folder}/{category}.txt'
    words = []
    with open(file_path) as f:
        for line in f.readlines():
            words += [w.strip() for w in line.split(' ')[1:]]
    return words
