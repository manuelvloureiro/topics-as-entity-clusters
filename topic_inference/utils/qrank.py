from . import to_ahocorasick

import pandas as pd

from pathlib import Path


def load_qrank(entities_add=None, n_entities=1_000_000, as_automaton=True):
    qrankpath = Path('data/corpora/wikidata/qrank.csv')
    qrank = list(pd.read_csv(qrankpath)['Entity'])[:n_entities]
    if entities_add:
        qrank = sorted(set(qrank + list(entities_add)))
    if as_automaton:
        qrank = to_ahocorasick(qrank, desc='Adding Qrank entities')
    return qrank
