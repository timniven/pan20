"""General utilities."""
import json

# import nltk
# from gensim.parsing import preprocessing


class IxDict:

    def __init__(self, entities):
        self.entities = list(sorted(entities))
        self.ent_to_ix = dict(zip(self.entities, range(len(self.entities))))
        self.ix_to_ent = {v: k for k, v in self.ent_to_ix.items()}

    def __getitem__(self, item):
        try:
            if isinstance(item, str):
                return self.ent_to_ix[item]
            elif isinstance(item, int):
                return self.ix_to_ent[item]
            else:
                raise ValueError(type(item))
        except Exception as e:
            print(item)
            print(type(item))
            raise e

    def __len__(self):
        return len(self.entities)

    @classmethod
    def load(cls, file_name):
        with open(file_name, 'r') as f:
            entities = json.loads(f.read())
            return cls(entities)

    def save(self, file_name):
        with open(file_name, 'w+') as f:
            f.write(json.dumps(self.entities))


def get_retweets(tweets):
    return [x for x in tweets if is_retweet(x)]


def get_non_retweets(tweets):
    return [x for x in tweets if not is_retweet(x)]


def is_retweet(tweet):
    return tweet.startswith('RT ')


def is_user_retweet(tweet):
    if not tweet.startswith('RT '):
        return False
    return '#USER#' in tweet


# def tokenize(tweet, stem=False):
#     tweet = tweet.lower()
#     tweet = preprocessing.strip_punctuation(tweet)
#     if stem:
#         tweet = preprocessing.stem_text(tweet)
#     return nltk.word_tokenize(tweet)
