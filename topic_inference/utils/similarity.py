import numpy as np
from typing import Collection, Optional


def cosine_similarity(a: Collection[float],
                      b: Optional[Collection[float]] = None):
    if not isinstance(a, np.ndarray):
        a = np.array(a)

    if b is None:
        a /= np.linalg.norm(a, axis=1)[:, np.newaxis]
        return np.dot(a, a.T)

    if not isinstance(b, np.ndarray):
        b = np.array(b)

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pearson_correlation(a: Collection[float],
                        b: Optional[Collection[float]] = None):
    if b:
        a -= np.mean(a)
        b -= np.mean(b)
    else:
        a -= np.mean(a, axis=1, keepdims=True)
    return cosine_similarity(a, b)


cos_sim = cosine_similarity  # shortcut
corr = pearson_correlation  # shortcut


def cosine_similarity_set(a: Collection, b: Collection) -> float:
    a = set(a)
    b = set(b)
    return len(a.intersection(b)) / (len(a) * len(b)) ** (1 / 2)


def jaccard_similarity(a: Collection, b: Collection) -> float:
    a = set(a)
    b = set(b)
    length_a_and_b = len(a.intersection(b))
    return length_a_and_b / (len(a) + len(b) - length_a_and_b)


def overlap_coefficient(a: Collection, b: Collection) -> float:
    a = set(a)
    b = set(b)
    return len(a.intersection(b)) / min(len(a), len(b))
