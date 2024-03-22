from scipy import sparse
from itertools import chain, compress, islice
from topic_inference.typing import *


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def as_list(o: Any) -> list:
    """
    Return input as list considering type.
    Strings are returned as a list with a single string.
    """
    types = (str,)  # Expand as needed
    return [] if o is None else [o] if isinstance(o, types) else list(o)


def chunks(o: Iterable, chunksize: int) -> Iterable[list]:
    iterator = iter(o)
    piece = list(islice(iterator, chunksize))
    while piece:
        yield piece
        piece = list(islice(iterator, chunksize))


def dict_default(o: dict, default: dict) -> dict:
    """
    Add default values to a dict in case they do not exist.
    Useful with kwargs.
    """
    for k, v in default.items():
        o.update({k: o.get(k, v)})
    return o


def filter_by_list(collection: Collection, vocab: Collection) -> list:
    return [o for o in collection if o in vocab]


def flatten(o: Iterable[Iterable], as_iterable: bool = False) -> list:
    """Flattens an iterable of iterables by one level"""
    iterator = chain.from_iterable(o)
    if as_iterable is False:
        iterator = list(iterator)
    return iterator


def popget(o: dict, key: Hashable, default: Any = None) -> Any:
    """Pop value from dict or return default"""
    return o.pop(key) if key in o else o.get(key, default)


def remove_prefix(text: str, prefix: str) -> str:
    """If a string has a prefix, remove it"""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def select(o: Collection, keep: Collection[bool]) -> list:
    """Filter sized iterable using sized iterable of booleans"""
    if len(o) != len(keep):
        raise ValueError("Sizes do not match!")
    return list(compress(o, keep))


def sorted_by_list(obj, by_list, reverse=False):
    assert len(obj) == len(by_list)
    order = argsort(by_list)
    if reverse:
        order = reversed(order)
    return [obj[i] for i in order]


def sparsicity(data) -> float:
    """Returns the number of non-zeros of an array/matrix"""
    if sparse.issparse(data):
        data = data.todense()
    return (data > 0).sum() / data.size
