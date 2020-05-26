import collections
import nltk
from tqdm.notebook import tqdm


def docs_to_freqs(docs, n_docs, n=1):
    """Convert documents into token frequencies.

    Args:
      docs: Generator.
      n_docs: Int.
      n: Int, order of n-grams.
    """
    toks = (tokenize(d, n) for d in docs)
    return toks_to_freqs(toks, n_docs)


def normalize(tok):
    if tok == "''":
        return '"'
    if tok == '``':
        return '"'
    return tok


def tokenize(text, n=1):
    """Tokenize text into n-grams.

    Args:
      text: String.
      n: Int, order of n-grams.
    """
    toks = nltk.word_tokenize(text)

    # strip and lower
    toks = [t.strip().lower() for t in toks]

    # normalize
    toks = [normalize(t) for t in toks]

    if n > 1:
        toks = ['_'.join(g) for g in nltk.ngrams(toks, n)]

    return toks


def toks_to_freqs(toks, n):
    freqs = collections.Counter()
    with tqdm(total=n) as pbar:
        for doc_toks in toks:
            freqs.update(doc_toks)
            pbar.update()
    n = sum(freqs.values())
    freqs = {t: c / n for t, c in freqs.items()}
    return freqs
