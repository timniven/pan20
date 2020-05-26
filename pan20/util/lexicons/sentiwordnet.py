"""SentiWordNet."""


file_path = '/home/hanshan/dev/lexicons/sentiwordnet/SentiWordNet_3.0.0.txt'
cols = ['POS', 'ID', 'PosScore', 'NegScore', 'SynsetTerms', 'Gloss']


class SentiWordNet:

    def __init__(self):
        self.senti_dict = {}
        with open(file_path) as f:
            for line in f.readlines():
                if line.startswith('#'):
                    continue
                line = dict(zip(cols, line.split('\t')))
                # NOTE: weird case we need to catch
                if line['SynsetTerms'] == '#':
                    continue
                # NOTE: not going to deal with multiple senses - just use first
                line['SynsetTerms'] = line['SynsetTerms'].split('#')[0]
                if line['SynsetTerms'] in self.senti_dict:
                    continue
                try:
                    self.senti_dict[line['SynsetTerms']] = {
                        'pos': float(line['PosScore']),
                        'neg': float(line['NegScore']),
                    }
                except Exception as e:
                    print(line)
                    raise e

    def __call__(self, word):
        if word not in self.senti_dict:
            return {'pos': 0., 'neg': 0.}
        return self.senti_dict[word]

    def __contains__(self, item):
        return item in self.senti_dict

    def __getitem__(self, item):
        return self(item)

    def neg(self, word):
        return self(word)['neg']

    def pos(self, word):
        return self(word)['pos']

    def words(self):
        return list(self.senti_dict.keys())

    # methods for scoring documents (already tokenized)

    def score(self, doc):
        score = 0.
        for word in doc:
            word = word.lower()
            score += self.pos(word)
            score -= self.neg(word)
        return score / len(doc)

    def score_neg(self, doc):
        score = 0.
        for word in doc:
            word = word.lower()
            score += self.neg(word)
        return score / len(doc)

    def score_pos(self, doc):
        score = 0.
        for word in doc:
            word = word.lower()
            score += self.pos(word)
        return score / len(doc)
