from topic_inference.utils import readpickle, writepickle, \
    load_qrank, select, chunks, default_logger as logger
from topic_inference.models.utils import MINSCORE, NUMWORDS
from topic_inference.collections import mathdict
from topic_inference.typing import *
from . import AbstractTopicModel, SEED

import numpy as np
import faiss
from tqdm import tqdm

from abc import ABC
from pathlib import Path
import multiprocessing as mp

ENDSIGNAL = '<ENDSIGNAL>'
EMPTYMESSAGE = "<NO TOPIC FOUND>"
TOPTOPICS = 100
EMBEDDINGS = ['wikipedia', 'wikidata']


def mp_update_top_words(input_q: mp.Queue, output_q: mp.Queue,
                        num_topics: int, top_topics: int = TOPTOPICS):
    top_topics = min(num_topics, top_topics)
    topics = [mathdict(float, {}) for _ in range(num_topics)]

    while True:
        chunk = input_q.get()
        if chunk == ENDSIGNAL:
            break
        docs, probs_list = chunk
        for tokens, probs in zip(docs, probs_list):
            if tokens is None:
                continue
            tokens_freq = tokens.freq()

            for i, (topic, prob) in enumerate(probs.items()):
                if top_topics is not None and i >= top_topics:
                    break

                topic_update = tokens_freq * float(prob)
                topics[topic].add(topic_update, inplace=True)

    for i in range(len(topics)):
        topics[i] = topics[i].truncate(1000)

    output_q.put(topics)


def get_embeddings(entity_ids_list: Collection[str], knowledge_base: dict,
                   embedding_type: str, alpha: float, progress_bar: bool = False
                   ) -> Tuple[List[str], np.array]:
    assert alpha > 0

    entity_ids = []
    embeddings = []

    if progress_bar:
        iterator = tqdm(entity_ids_list, desc="Getting embeddings")
    else:
        iterator = entity_ids_list

    for entity_id in iterator:
        embedding = get_embedding(entity_id, knowledge_base=knowledge_base,
                                  embedding_type=embedding_type, alpha=alpha)
        if embedding is None:
            continue
        entity_ids.append(entity_id)
        embeddings.append(embedding)

    return entity_ids, np.array(embeddings, dtype=np.float32)


def get_embedding(entity_id: str, knowledge_base: dict, embedding_type: str,
                  alpha: float):
    entity = knowledge_base.get(entity_id, {})

    if embedding_type == 'both':
        key = f"both-{round(alpha, 6)}"
        embedding = entity.get('embedding', {}).get(key)
        if embedding is not None:
            return embedding
        # process the mixed embedding in case it doesn't exist
        embedding = {}
        for e_type in EMBEDDINGS:
            emb = entity.get('embedding', {}).get(e_type)
            if emb is None:
                return  # stop here and return None
            embedding[e_type] = np.array([emb], dtype=np.float32)
            # we assume that the embeddings are normalized
            # faiss.normalize_L2(embedding[e_type])

        weight = 1. / (1. + alpha)
        embedding[EMBEDDINGS[0]] *= (weight ** .5)
        embedding[EMBEDDINGS[1]] *= ((1 - weight) ** .5)

        embedding = np.concatenate([embedding[EMBEDDINGS[0]],
                                    embedding[EMBEDDINGS[1]]], axis=1)[0]
        entity['embedding'][key] = embedding
        return embedding
    else:
        # this can also return None if entity or embedding do not exist
        return entity.get('embedding', {}).get(embedding_type)


class KmeansFaissModel:

    def __init__(self, knowledge_base: Dict[str, dict], embedding_type: str,
                 alpha: float, topics: Optional[list] = None,
                 topics_labels: Optional[list] = None,
                 topics_embeddings: Optional[np.array] = None,
                 *args, **kwargs):
        if topics_labels:
            assert len(topics) == len(topics_labels)
        if topics_embeddings is not None:
            if topics is not None:
                assert len(topics) == len(topics_embeddings)

        self.knowledge_base = knowledge_base
        self.embedding_type = embedding_type
        self.alpha = alpha

        self.topics = topics or []
        for i, topic in enumerate(self.topics):
            self.topics[i] = mathdict(float, topic).freq(inplace=True)

        self.topics_labels = topics_labels or ([''] * self.num_topics)
        self.topics_embeddings = topics_embeddings

        self.index = faiss.IndexFlatL2(len(topics_embeddings[0]))
        self.index.add(topics_embeddings)

    def update_top_words(self, texts=None, top_entities=100_000, top_topics=100,
                         chunksize=10_000, n_jobs=36):
        if texts is None:
            is_original_dataset = True
            qrank = load_qrank(n_entities=top_entities, as_automaton=True)
            kb = {o: self.knowledge_base[o] for o in qrank
                  if o in self.knowledge_base}
            texts = kb.items()
        elif isinstance(texts, list) and isinstance(texts[0], str):
            is_original_dataset = False
        else:
            raise ValueError("The dataset must be a collection of texts")

        iterator = chunks(tqdm(texts, desc='Generating topics'),
                          chunksize=chunksize)

        input_q = mp.Queue()
        output_q = mp.Queue()

        num_topics = len(self.topics_embeddings)
        processes = [mp.Process(
            target=mp_update_top_words,
            args=(input_q, output_q, num_topics, top_topics))
            for _ in range(n_jobs)]

        for p in processes:
            p.daemon = True
            p.start()

        for chunk in iterator:
            if not is_original_dataset:
                chunk = {i: {'text_entities': o.split()}
                         for i, o in enumerate(chunk)}
            chunk = dict(chunk)
            for k, v in chunk.items():
                chunk[k] = mathdict(int, Counter(v.get('text_entities') or []))
            texts = list(chunk.values())
            probs_list = self.inference(texts)
            input_q.put((texts, probs_list))

        for _ in range(n_jobs):
            input_q.put(ENDSIGNAL)

        topics = [mathdict(float, {}) for _ in range(num_topics)]
        for _ in tqdm(range(n_jobs), desc="Reading processed chunks"):
            new_topics = output_q.get()
            for i in range(len(new_topics)):
                try:
                    topics[i].add(new_topics[i], inplace=True)
                    topics[i].add(self.topics[i] * 1e-12, inplace=True)
                    topics[i] = topics[i].truncate(1000)
                except Exception as e:
                    raise e

        for topic in tqdm(topics, desc="Calculating topics relative frequency"):
            topic.freq(inplace=True)

        self.topics = topics

    @property
    def num_topics(self):
        return len(self.topics)

    def _inference(self, entity_counts: mathdict) -> Optional[np.array]:

        # get the unique entities which are filtered with .get_embeddings
        unique_entities = set(entity_counts.keys())
        unique_entities, embeddings = self.get_embeddings(unique_entities)

        if len(unique_entities) == 0:
            return

        # remove the entities that do not have an embedding
        entity_frequencies = mathdict(int, {o: entity_counts[o]
                                            for o in unique_entities}).freq()

        entity_frequencies = np.array(tuple(entity_frequencies.values()),
                                      dtype=np.float32)

        return np.matmul(entity_frequencies, embeddings)

    def inference(self, tokens_list: Union[TokensList, List[mathdict]]
                  ) -> List[mathdict]:
        if isinstance(tokens_list[0], list):
            tokens_count = [mathdict(int, Counter(o)) for o in tokens_list]
        elif isinstance(tokens_list[0], mathdict):
            tokens_count = [o or mathdict(int, {}) for o in tokens_list]
        else:
            raise TypeError("Can't do inference on ", type(tokens_list[0]))

        embeddings = [self._inference(o) for o in tokens_count]

        output = [mathdict(float, {}) for _ in range(len(tokens_count))]
        keep = [i for i, o in enumerate(embeddings) if o is not None]

        if len(keep) == 0:
            return output  # this is the case when the documents are all empty

        # weighted average of the embeddings per document
        embeddings = np.array([o for o in embeddings if o is not None],
                              dtype=np.float32)

        distances, top_topics = self.index.search(embeddings, len(self.topics))

        # probabilities are the normalized inverted squared euclidean distances
        proximity = 1 / (distances ** 2)
        proximity /= proximity.sum(axis=1)[:, np.newaxis]

        probabilities = [mathdict(float, zip(top_topics[i], proximity[i]))
                         for i in range(len(embeddings))]

        # fix the ones that were removed because they have no valid input
        for k, p in zip(keep, probabilities):
            output[k] = p

        return output

    def get_embeddings(self, entity_id_list: Collection[str]
                       ) -> Tuple[List[str], np.array]:
        entity_id_list, embeddings = get_embeddings(
            entity_id_list,
            knowledge_base=self.knowledge_base,
            embedding_type=self.embedding_type,
            alpha=self.alpha)
        return entity_id_list, embeddings

    def get_topic(self, topic_id):
        return self.topics[topic_id]

    def get_topic_label(self, topic_id):
        return self.topics_labels[topic_id]

    def get_topic_words(self, topic_id, num_words=NUMWORDS,
                        min_score=MINSCORE):
        words_scores = self.topics[topic_id].gt(min_score).head(num_words)
        return [o for o, _ in words_scores]


class KmeansFaiss(AbstractTopicModel, ABC):
    """
    """

    def __init__(self, tokenizer: Optional[Tokenizer] = None,
                 token_filter: Optional[TokenFilter] = None,
                 lang: Optional[str] = None,
                 *args, **kwargs):
        super(KmeansFaiss, self).__init__(tokenizer=tokenizer,
                                          token_filter=token_filter,
                                          lang=lang)
        # self.embedding_type = 'wikidata'
        # self.alpha = 1.

        try:
            self.model = KmeansFaissModel(*args, **kwargs)
        except TypeError:
            pass

    def fit(self, tokens: TextsOrTokensList, num_topics: int,
            lang: Optional[str] = None, seed: int = SEED, *args, **kwargs):
        raise NotImplementedError("Use method .train")

    def train(self, tokenizer: Tokenizer, num_topics: int, max_iter: int = 300,
              entity_list: Optional[Collection[str]] = None, num_init: int = 1,
              embedding_type: Optional[str] = None, seed: int = 0,
              alpha: Optional[float] = None, num_words: int = NUMWORDS,
              use_gpu: bool = True, max_entities_train: Optional[int] = None,
              top_entities_to_topics: Optional[int] = None):

        options = list(EMBEDDINGS) + ['both']
        if embedding_type not in options:
            msg = f"Embedding type {embedding_type} is unknown. "
            f"Available options: {options}"
            raise ValueError(msg)

        assert alpha > 0

        entity_ids = set(tokenizer.knowledge_base.keys())
        if entity_list:
            entity_ids = entity_ids.intersection(entity_list)

        if max_entities_train:
            qrank = load_qrank(n_entities=max_entities_train,
                               as_automaton=False)
            entity_ids = entity_ids.intersection(qrank)

        entity_ids, entity_embeddings = get_embeddings(
            entity_ids_list=entity_ids,
            knowledge_base=tokenizer.knowledge_base,
            embedding_type=embedding_type,
            alpha=alpha,
            progress_bar=True
        )

        logger.info("Creating Kmeans instance...")
        kmeans = faiss.Kmeans(d=entity_embeddings.shape[1], k=num_topics,
                              niter=max_iter, nredo=num_init, gpu=use_gpu,
                              seed=seed)
        logger.info("Training Kmeans model...")
        kmeans.train(entity_embeddings.astype(np.float32))

        if top_entities_to_topics:
            qrank = load_qrank(n_entities=top_entities_to_topics,
                               as_automaton=True)
            entity_keep = np.array([o in qrank for o in entity_ids])
            entity_embeddings = entity_embeddings[entity_keep, :]
            entity_ids = select(entity_ids, entity_keep)
        top_words_index = faiss.IndexFlatL2(entity_embeddings.shape[1])
        top_words_index.add(entity_embeddings)
        tw_dist, tw_idx = top_words_index.search(kmeans.centroids, num_words)

        proximity = 1 / (tw_dist + 1e-6)
        proximity /= proximity.sum(axis=1)[:, np.newaxis]
        word_ids = [[entity_ids[o] for o in tw_idx[i]] for i in
                    range(num_topics)]
        topics = [mathdict(float, zip(word_ids[i], proximity[i]))
                  for i in range(num_topics)]

        self.model = KmeansFaissModel(knowledge_base=tokenizer.knowledge_base,
                                      topics=topics, topics_labels=None,
                                      embedding_type=embedding_type,
                                      alpha=alpha,
                                      topics_embeddings=kmeans.centroids)
        self.tokenizer = tokenizer

    def transform(self, tokens: TextsOrTokensList, lang: Optional[str] = None,
                  **kwargs) -> MultipleTopicPredictions:
        def max_(x):
            try:
                return max(x, key=x.get)
            except ValueError:
                return -1

        tokens = self.batch_preprocess(tokens, lang or self.lang, **kwargs)
        probs = self.model.inference(tokens)
        preds = [max_(o) for o in probs]

        return preds, probs

    def _topic(self, topic_id: int, num_words: int = 10) -> Topic:
        if topic_id < 0:
            return {'words': (), 'scores': ()}  # noqa
        if num_words <= 0:
            num_words = self.num_words
        m = mathdict(float, self.model.topics[topic_id]).head(num_words)
        words, weights = zip(*m) if m else ((), ())
        return {'words': words, 'scores': weights}

    def topic_label(self, topic_id: int) -> str:
        return self.model.topics_labels[topic_id]

    @property
    def num_topics(self) -> int:
        return self.model.num_topics

    def save(self, path: PathType, *args, **kwargs):
        path = Path(path)
        writepickle(self, path.with_suffix('.pkl'))

    def load(self, path: Union[str, Path], *args, **kwargs) -> 'KmeansFaiss':
        file = Path(path).with_suffix('.pkl')
        if not file.exists():
            raise FileNotFoundError(f"File `{str(file)}` not found!")
        obj = readpickle(file)
        self.__dict__ = obj.__dict__
        return self

    @classmethod
    def _from_excel(cls, tokenizer: Tokenizer, topics_list: List[mathdict],
                    topics_labels: List[str], topics_embed: Optional[np.array],
                    lang: str = 'en', embedding_type=None,
                    token_filter: Optional[TokenFilter] = None, **kwargs):

        tokencounter = KmeansFaiss(
            tokenizer=tokenizer,
            token_filter=token_filter,
            lang=lang,
        )

        tokencounter.model = KmeansFaissModel(
            knowledge_base=tokenizer.knowledge_base,
            embedding_type=embedding_type,
            topics=topics_list,
            topics_labels=topics_labels,
            topics_embeddings=topics_embed,
            max_topics=len(topics_list),
            **kwargs)

        return tokencounter
