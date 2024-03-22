from topic_inference.typing import *

import ahocorasick
from tqdm import tqdm


def to_ahocorasick(keys: Collection, values: Optional[Collection] = None,
                   desc: Optional[str] = None,
                   **kwargs) -> ahocorasick.Automaton:
    values = values or keys
    if len(keys) != len(values):
        raise ValueError(f"Keys ({len(keys)}) and values ({len(values)})"
                         " must have the same length.")
    automaton = ahocorasick.Automaton()
    iterator = zip(keys, values)
    if desc:
        iterator = tqdm(iterator, desc=desc, **kwargs)
    for k, v in iterator:
        automaton.add_word(str(k), v)
    automaton.make_automaton()
    return automaton
