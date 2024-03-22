import json
from pathlib import Path
import os
import logging
from topic_inference.utils import default_logger as logger
from typing import *

__config_path = Path(__file__).parent / 'config.json'
with open(__config_path, 'r') as f:
    config = json.load(f)

# os.environ['CUDA_VISIBLE_DEVICES'] = str(config['device'])

logger.parent.handlers.clear()  # Remove existing console logger

__fmt = '%(asctime)s:%(levelname)s > %(message)s'
__datefmt = '%Y-%m-%d %H:%M:%S'
logger_handler = logging.StreamHandler()  # Handler for the logger
logger_handler.setFormatter(logging.Formatter(__fmt, __datefmt))

logger.parent.addHandler(logger_handler)  # Add new logger


def set_device(device=config['device']):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    logger.info(f"Setting device: {device}")


def set_seed(seed: int):
    from torch import manual_seed
    from torch.cuda import manual_seed as cuda_manual_seed, manual_seed_all
    from numpy.random import seed as np_seed
    from random import seed as r_seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    manual_seed(seed)
    cuda_manual_seed(seed)
    manual_seed_all(seed)
    np_seed(seed)
    r_seed(seed)
    logger.info(f"Setting seed: {seed}")


def load_entitizer(device=config['device']):
    set_device(device)
    from topic_inference.tokenizers import WikidataMatcher
    from topic_inference.utils import readpickle
    entitizer: WikidataMatcher = readpickle(config['entitizer'])
    return entitizer


def get_model_statistics(id_: Any, texts: List[List[str]],
                         topics: List[List[str]], scores: List[List[float]],
                         num_words: int = 10,
                         save_path: Optional[Union[Path, str]] = None):
    from topic_inference.utils import writejsonl
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora import Dictionary
    from itertools import chain

    statistics = {'id': id_}

    # Topic Diversity
    logger.info("Computing topic diversity...")
    n = 25
    topics = [o[:n] for o in topics]
    idx_errors = [i for i, o in enumerate(topics) if len(o) != n]
    if idx_errors:
        msg = f"There are topics with less than {n} words: {idx_errors}"
        raise ValueError(msg)
    flatten = chain.from_iterable
    diversity = len(set(flatten(topics))) / len(list(flatten(topics)))
    logger.info(f"Topic diversity: {diversity:.4f}")
    statistics['diversity'] = round(diversity, 4)

    topics_ = [o[:num_words] for o in topics]
    dictionary = Dictionary(topics_ + texts)

    coherence_metrics = ['C_NPMI', 'C_UCI', 'U_Mass']
    for metric in coherence_metrics:
        logger.info(f"Computing {metric}...")
        model = CoherenceModel(topics=topics_, texts=texts,
                               dictionary=dictionary, coherence=metric.lower())
        score = model.get_coherence()
        logger.info(f"{metric}: {score:.4f}")
        statistics[metric.lower()] = round(score, 4)
        quality = score * diversity
        logger.info(f"Quality {metric}: {quality:.4f}")
        statistics[f'quality_{metric.lower()}'] = round(quality, 4)

    results = {'statistics': statistics, 'topics': topics, 'scores': scores}

    if save_path:
        writejsonl([results], path=save_path, append=True)

    return results
