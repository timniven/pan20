"""Working with the lexical negation dict.

https://github.com/cltl/lexical-negation-dictionary
https://www.aclweb.org/anthology/W16-5007/
"""


file_path = '/home/hanshan/dev/lexicons/lexical_negation/all_annotations.txt'


class LexicalNegations:

    def __init__(self):
        self.pos_neg = {}
        self.pos = set([])
        self.neg = set([])
        with open(file_path) as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue  # header
                line = line.split('\t')
                pos = line[0]
                neg = line[1]
                self.pos_neg[pos] = neg
                self.neg.update([neg])
                self.pos.update([pos])
        self.neg_pos = {n: p for p, n in self.pos_neg.items()}
