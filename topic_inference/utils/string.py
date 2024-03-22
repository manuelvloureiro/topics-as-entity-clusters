import string
from typing import List, Optional, Union


def allngrams(tokens: List[str], max_k: Optional[int] = None) -> List[
    List[str]]:
    grams = []
    for i in range(1, 1 + (max_k or len(tokens))):
        grams.extend(ngrams(tokens, i))
    return grams


def kshingles(text: str, k: int) -> List[str]:
    return [text[i: i + k] for i in range(len(text) - k + 1)]


def ngrams(tokens: List[str], n: int) -> List[List[str]]:
    return [tokens[i: i + n] for i in range(len(tokens) - n + 1)]


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))


def to_unicode(text: Union[str, bytes], encoding: str = 'utf8',
               errors: str = 'strict') -> str:
    # Copied from gensim.utils
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)
