from topic_inference.utils import select
from . import Article, Namespace
import pandas as pd
from typing import List, Union
from copy import deepcopy
import random

SEED = 42


class ArticleCorpus(Namespace):

    def __init__(self, articles: Union[None, List[Article]] = None, **kwargs):
        super().__init__(**kwargs)
        self.articles = articles

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, item):
        return self.articles[item].__copy__()

    def __repr__(self):
        return '<{} ({}) #{}>'.format(
            self.__class__.__name__,
            hex(id(self)),
            len(self)
        )

    def __add__(self, other):
        corpus = deepcopy(self)
        corpus.articles += deepcopy(other.articles)
        return corpus

    def add(self, key, values):
        corpus = deepcopy(self)
        corpus.add_(key, values)
        return corpus

    def add_(self, key, values):
        if len(self) != len(values):
            raise ValueError("Lengths don't match.")
        articles = [a.update(**{key: v}) for a, v in zip(self.articles, values)]
        self.update(articles=articles)

    def apply(self, key, f):
        corpus = deepcopy(self)
        corpus.apply_(key, f)
        return corpus

    def apply_(self, key, f):
        articles = [o.update(**{key: f(o[key])}) for o in self.articles]
        self.update(articles=articles)

    def split(self, ratio=0.8, seed=SEED):
        if not (0 < ratio < 1):
            raise ValueError("Pick a `ratio` between 0 and 1 (non-inclusive).")

        n = max(1, min(len(self) - 1, round(len(self) * ratio)))
        indexes = list(range(len(self)))
        random.seed(seed)
        random.shuffle(indexes)
        left = sorted(indexes[:n])
        right = sorted(indexes[n:])

        return (
            self.__copy__().update(articles=[self.articles[i] for i in left]),
            self.__copy__().update(articles=[self.articles[i] for i in right])
        )

    def filter(self, key, f):
        corpus = deepcopy(self)
        corpus.filter_(key, f)
        return corpus

    def filter_(self, key, f):
        keep = [f(a[key]) for a in self.articles]
        articles = select(self.articles, keep)
        self.update(articles=articles)

    def shuffle(self, seed=SEED):
        corpus = deepcopy(self)
        corpus.shuffle_(seed)
        return corpus

    def shuffle_(self, seed=SEED):
        random.seed(seed)
        random.shuffle(self.articles)

    def texts(self, pattern="{} {}"):
        return [o.to_text(pattern=pattern) for o in self.articles]

    def labels(self):
        return [o.get('labels', []) for o in self.articles]

    def to_dataframe(self, columns=None):
        df = pd.DataFrame(o.to_dict() for o in self.articles)
        if columns:
            df = df[columns]
        return df
