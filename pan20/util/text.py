import collections
import emoji
import re
import string

import nltk
from nltk.util import ngrams
from nltk.tokenize import toktok
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


def pre_clean(text):
    text = normalize_punctuation(text)
    text = recover_swears(text)
    text = expand_contractions(text)
    return text


def normalize_punctuation(text):
    text = text.replace("''", '"')
    text = text.replace('``', '"')
    text = text.replace('’', "'")
    text = text.replace('‼', '!!')
    text = text.replace('⁉', '!?')
    text = text.replace('❗', '!')
    text = text.replace('❓', '?')
    return text


def recover_swears(text):
    text = text.replace('f**king', 'fucking')
    text = text.replace('f*cking', 'fucking')
    text = text.replace('fu*king', 'fucking')
    text = text.replace('f**k', 'fuck')
    text = text.replace('f*ck', 'fuck')
    text = text.replace('fu*k', 'fuck')
    text = text.replace('s**t', 'shit')
    text = text.replace('s*it', 'shit')
    text = text.replace('sh*t', 'shit')
    return text


def expand_contractions(text):
    text = text.lower()
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"i’m", "i am", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    return text


def post_clean(toks):
    toks = separate_repeat_emoticons(toks)
    toks = separate_emoticons_next_to_text(toks)
    toks = separate_repeat_punctuation(toks)
    toks = remove_unwanted_punctuation(toks)
    toks = remove_non_english(toks)
    toks = normalize_netspeak(toks)
    toks = remove_single_chars(toks)  # but not i - NOTE: removes emojis
    toks = normalize_growls(toks)
    return toks


def normalize_growls(toks):
    cleaned = []
    for t in toks:
        # TODO: regex
        if t in ['grrr', 'grrrr', 'grrrrr', 'grrrrrr']:
            cleaned.append('grr')
        else:
            cleaned.append(t)
    return cleaned


def remove_single_chars(toks):
    cleaned = []
    for t in toks:
        if len(t) == 1 and t != 'i':
            continue
        cleaned.append(t)
    return cleaned


def remove_non_english(toks):
    cleaned = []
    for t in toks:
        is_emoji = len(t) == 1 and t[0] in emoji.UNICODE_EMOJI
        if is_emoji:
            cleaned.append(t)
        is_punct = len(t) == 1 and t[0] in string.punctuation
        if is_punct:
            # NOTE: by now have filtered out unwanted, and separated
            cleaned.append(t)
        if all(c in string.ascii_lowercase for c in t):
            cleaned.append(t)
        # otherwise don't
    return cleaned


def separate_repeat_emoticons(toks):
    cleaned = []
    for t in toks:
        if len(t) > 1 and all(c in emoji.UNICODE_EMOJI for c in t):
            cleaned += list(t)
        else:
            cleaned.append(t)
    return cleaned


def separate_emoticons_next_to_text(toks):
    cleaned = []
    for t in toks:
        have_emoji = any(c in emoji.UNICODE_EMOJI for c in t)
        have_alpha = any(c in string.ascii_lowercase for c in t)
        if have_emoji and have_alpha:
            alpha = [c for c in t if c in string.ascii_lowercase]
            emoti = [c for c in t if c in emoji.UNICODE_EMOJI]
            cleaned.append(''.join(alpha))
            cleaned += emoti
        else:
            cleaned.append(t)
    return cleaned


def separate_repeat_punctuation(toks):
    cleaned = []
    for t in toks:
        if len(t) > 1 and all(c in string.punctuation for c in t):
            cleaned += list(t)
        else:
            cleaned.append(t)
    return cleaned


def remove_unwanted_punctuation(toks, wanted='!'):
    # assume repeaters already separated
    return [t for t in toks
            if t not in string.punctuation or t in wanted]


def normalize_netspeak(toks):
    def f(t):
        tok_map = {
            'u': ['you'],
            'ur': ['you',  'are'],
            'r': ['are'],
        }
        if t in tok_map:
            return tok_map[t]
        return [t]
    cleaned = []
    for t in toks:
        cleaned += f(t)
    return cleaned


def tokenize(text, n=1, lang='en'):
    """Tokenize text into n-grams.

    Args:
      text: String.
      n: Int, order of n-grams.
    """
    if lang == 'en':
        tokenizer = nltk.word_tokenize
    elif lang == 'es':
        tokenizer = toktok.ToktokTokenizer().tokenize
    else:
        raise ValueError(lang)

    # pre-tokenization hooks
    text = pre_clean(text)

    # tokenization
    toks = [t.strip() for t in tokenizer(text)]

    # post-tokenization cleaning
    toks = post_clean(toks)

    # higher order n-grams
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


def to_bigrams(unigrams):
    return ['_'.join(g) for g in ngrams(unigrams, 2)]


def to_trigrams(unigrams):
    return ['_'.join(g) for g in ngrams(unigrams, 3)]


def to_quadgrams(unigrams):
    return ['_'.join(g) for g in ngrams(unigrams, 4)]
