from .mathdict import mathdict
from topic_inference.utils import flatten
from topic_inference.typing import *

from tqdm import tqdm

from itertools import combinations
from collections import Counter
from math import log


def sliding_window(sw):
    def _sliding_window(tokens):
        for doc in tokens:
            for i in range(max(len(doc) - sw + 1, 1)):
                yield doc[i: i + sw]

    return _sliding_window


def one2one(doc):
    return combinations(doc, 2)


def preceding(doc):
    return [(i, j) for j in range(1, len(doc)) for i in range(j)]


def boolean_document(tokens):
    for doc in tokens:
        yield set(doc)


def arithmetic_mean(x):
    return sum(x) / len(x)


METRICS = {
    'c_umass': {
        'segmentation': preceding,
        'prob_estimation': boolean_document,
        'measure': 'lc',
        'aggregation': arithmetic_mean
    },
    # c_npmi prob_estimation should be done with sliding_window(10) but our
    # data is out of order. our tokenizer leads to really sparse texts so this
    # should not impact the results
    'c_npmi': {
        'segmentation': preceding,
        'prob_estimation': boolean_document,
        'measure': 'npmi',
        'aggregation': arithmetic_mean
    }
}


def log2(x):
    return log(x, 2)


def joint_self_information(
        word1: str,
        word2: str,
        counts_pairs: Dict[Tuple[str, ...], int],
        n_tokens
) -> float:
    prob_pair = counts_pairs.get(tuple(sorted([word1, word2])), 1) / n_tokens
    return -log2(prob_pair)


def log_conditional_probability(
        word1: str,
        word2: str,
        counts_singles: Dict[str, int],
        counts_pairs: Dict[Tuple[str, ...], int],
        n_tokens: int,
) -> float:
    prob_pair = counts_pairs.get(tuple(sorted([word1, word2])), 1) / n_tokens
    prob_word2 = counts_singles.get(word2, 1) / n_tokens
    return log2(prob_pair / prob_word2)


def normalized_pointwise_mutual_information(
        word1: str,
        word2: str,
        counts_singles: Dict[str, int],
        counts_pairs: Dict[Tuple[str, ...], int],
        n_tokens: int,
) -> float:
    pmi_ = pointwise_mutual_information(
        word1, word2, counts_singles, counts_pairs, n_tokens)
    h_ = joint_self_information(word1, word2, counts_pairs, n_tokens)
    return pmi_ / h_


def pointwise_mutual_information(
        word1: str,
        word2: str,
        counts_singles: Dict[str, int],
        counts_pairs: Dict[Tuple[str, ...], int],
        n_tokens: int,
) -> float:
    prob_pair = counts_pairs.get(tuple(sorted([word1, word2])), 1) / n_tokens
    prob_word1 = counts_singles.get(word1, 1) / n_tokens
    prob_word2 = counts_singles.get(word2, 1) / n_tokens
    return log2(prob_pair / (prob_word1 * prob_word2))


class CoherenceDataset:  # holds a dataset

    def __init__(self, tokens: TokensList, metric: str):
        self.counts_singles = mathdict(int, Counter(flatten(tokens)))
        self.n_tokens = sum(self.counts_singles.values())

        metric = METRICS[metric]
        segmentation = metric['segmentation']
        prob_estimation = metric['prob_estimation']
        self.measure = metric['measure']
        self.aggregation = metric['aggregation']

        combinations_ = [set(map(tuple, map(sorted, segmentation(o))))
                         for o in tqdm(prob_estimation(tokens),
                                       desc="Generating word pairs")]

        self.counts_pairs = mathdict(int, {})
        for pairs in tqdm(combinations_, desc="Counting word pairs"):
            for words in pairs:
                self.counts_pairs[words] += 1

    @property
    def vocab(self):
        return sorted(self.counts_singles.keys())

    def lc(self, word1, word2):
        return log_conditional_probability(word1, word2, self.counts_singles,
                                           self.counts_pairs, self.n_tokens)

    def pmi(self, word1, word2):
        return pointwise_mutual_information(word1, word2, self.counts_singles,
                                            self.counts_pairs, self.n_tokens)

    def npmi(self, word1, word2):
        return normalized_pointwise_mutual_information(word1, word2,
                                                       self.counts_singles,
                                                       self.counts_pairs,
                                                       self.n_tokens)

    def metric(self, word1, word2):
        return getattr(self, self.measure)(word1, word2)


class Coherence:  # holds topics

    def __init__(self, topics: TokensList):
        self.topics = topics
        self.dataset: Optional[CoherenceDataset] = None

    def coherence(self, dataset: Optional[TokensList] = None,
                  metric: str = 'c_npmi', num_words: Optional[int] = None):
        if dataset is not None:
            self.dataset = CoherenceDataset(dataset, metric)
        coherence = [self.topic_coherence(i, num_words=num_words)
                     for i in tqdm(range(len(self.topics)),
                                   desc="Calculating coherence")]
        avg_coherence = sum(coherence) / max(len(coherence), 1)
        return avg_coherence, coherence

    def topic_coherence(self, topic_id: int, num_words: Optional[int] = None):
        if self.dataset is None:
            raise ValueError("Coherence model requires a dataset!")
        topic = self.topics[topic_id]
        if num_words is not None:
            topic = topic[:num_words]
        scores = [self.dataset.metric(word1, word2)
                  for word1, word2 in combinations(topic, 2)]

        return self.dataset.aggregation(scores)
