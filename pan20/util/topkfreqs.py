"""For getting top-k frequent tokens from a dataset."""
from nltk import tokenize


def top_k(docs, k=50, tokenize_fn=tokenize.word_tokenize):
    """Retrieve top-k frequent words.

    Args:
      docs: List of strings of documents.
      k:
      tokenize_fn: Function for tokenizing each document. Defaults to
        `nltk.tokenize.word_tokenize`.

    Returns:

    """
