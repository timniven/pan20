"""Bill Noble's function word list."""
from pan20 import util

import numpy as np
from nltk import tokenize


cats = [
    'adverbs',
    'articles',
    'auxiliary_verbs',
    'conjunctions',
    'impersonal_pronouns',
    'personal_pronouns',
    'prepositions',
    'quantifiers',
]
cat_ix_dict = util.IxDict(cats)


class NobleDict:

    def __init__(self):
        self.word_dict = {}
        self.cat_dict = {}

        for cat in cats:
            self.cat_dict[cat] = []
            with open(f'data/noble/{cat}.txt') as f:
                for line in f.readlines():
                    word = line.strip()
                    if word not in self.word_dict.keys():
                        self.word_dict[word] = []
                    self.word_dict[word].append(cat)
                    self.cat_dict[cat].append(word)

        self.word_set = set(self.word_dict.keys())

    def __contains__(self, word):
        return word in self.word_dict

    def cats_for(self, word):
        if word in self.word_dict:
            return self.word_dict[word]
        return []

    def is_in(self, word, cat):
        return word in self.cat_dict[cat]

    def words_in(self, cat):
        return self.cat_dict[cat]


class CatFreqs:

    def __init__(self, tokenize_fn=tokenize.word_tokenize):
        self.tokenize = tokenize_fn
        self.noble_dict = NobleDict()

    def __call__(self, doc):
        words = self.tokenize(doc)
        n = len(words)
        freqs = {cat: 0 for cat in cats}
        for word in words:
            for cat in self.noble_dict.cats_for(word):
                freqs[cat] += 1
        freqs['n'] = n
        freqs['function_words'] = 0
        for cat in cats:
            freqs['function_words'] += freqs[cat]
            freqs[cat] /= n
        freqs['function_words'] /= n
        return freqs
