"""Base classes and common code."""
import nltk


class LexicalCategoryDict:

    def __init__(self):
        self.word_dict = None
        self.cat_dict = None
        self.stemmer = nltk.stem.PorterStemmer()

    def __contains__(self, word):
        return word in self.word_dict

    @property
    def categories(self):
        return list(sorted(self.cat_dict.keys()))

    def cats_for(self, word):
        if word in self.word_dict:
            return self.word_dict[word]
        return []

    def cat_freq(self, doc, cat):
        n = len(doc)
        count = sum(1 for word in doc if self.is_in(cat, word))
        return count / n

    def cat_freqs(self, doc):
        n = len(doc)
        freqs = {cat: 0 for cat in self.categories}
        for word in doc:
            for cat in self.cats_for(word):
                freqs[cat] += 1
        freqs['n'] = n
        for cat, count in freqs.items():
            freqs[cat] = count / n
        return freqs

    def is_in(self, cat, word):
        word_stem = self.stemmer.stem(word)
        return any(w in self.cat_dict[cat] for w in [word, word_stem])

    @property
    def words(self):
        return list(sorted(self.word_dict.keys()))

    def words_in(self, cat):
        return self.cat_dict[cat]
