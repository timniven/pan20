"""Custom lexicon for trust-distrust."""
from nltk import stem

from pan20.util import lexicons


# this just taken from a thesaurus
distrust = [
    'mistrust', 'mistrusts', 'mistrusting',
    'distrust', 'distrusts', 'distrusting',
    'suspect', 'suspects', 'suspected', 'suspecting',
    'suspicion', 'suspicions', 'suspicious', 'suspiciousness',
    'fake', 'fakes', 'faked', 'faking', 'fakeness',
    'disbelieve', 'disbelieved', 'disbelieving', 'disbeliever', 'disbelievers',
    'unbelievable', 'unbelieved', 'unbelieving', 'unbeliever', 'unbelievers',
    'doubt', 'doubts', 'doubtful', 'doubted', 'doubting', 'doubter', 'doubters',
    'suppose', 'supposed', 'supposing', 'supposes',
    'so_called',
    'misgiving', 'misgivings',
    'fear', 'fears', 'fearing', 'feared',
    'qualm', 'qualms',
    'question', 'questions', 'questioning', 'questioned', 'questioner',
    'questioners',
    'inquire', 'inquires', 'inquired', 'inquirer', 'inquirers', 'inquiring',
    'skeptic', 'skeptics', 'skeptical',
    'wonder', 'wonders', 'wondering', 'wondered',
    # from the data
    'sucker', 'lying', 'scam', 'cheating', 'admit',
    'claims', 'twists', 'fraud', 'cheated', 'corruption', 'lies',
    'facts', 'truth', 'true', 'false', 'reveal', 'uncover',
    'cover-up', 'expose',
]


class Lexicon(lexicons.LexicalCategoryDict):

    def __init__(self):
        super().__init__()
        self.word_dict = {}
        ps = stem.PorterStemmer()
        stems = [ps.stem(w) for w in distrust]
        for s in stems:
            self.word_dict[s] = ['distrust']
        self.cat_dict = {'distrust': stems}

    def distrust_freq(self, toks):
        return self.cat_freq(toks, 'distrust')
