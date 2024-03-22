import numpy as np

from .matcher import Matcher
from topic_inference.utils import as_list, chunks, cos_sim, flatten, \
    load_qrank, to_unicode, to_ahocorasick, writelines, default_logger as logger
from topic_inference.collections import ArticleCorpus, mathdict
from topic_inference.typing import *
from topic_inference.embedding_model import embedding_model

import pandas as pd
from pandas import errors
from ahocorasick import Automaton
from tqdm import tqdm

import json
from pathlib import Path
from copy import copy
import re
from functools import partial
import multiprocessing as mp

WIKIDATA = Path('data/corpora/wikidata/wikidata_summary.jsonl')
INTERIM = Path('data/interim/wikidatamatcher')

PROPERTIES = ['instance_of', 'facet_of', 'subclass_of']

TOKEN_MIN_LEN = 4
LABEL4MISSING = '<NO TRANSLATION>'

INSTANCES_THAT_MUST_HAVE_SPACE = set(
    ['Q5', 'Q95074', 'Q15632617', 'Q28020127'] +  # human
    ['Q215380', 'Q5741069'] +  # musical group
    ['Q47461344']  # book
)

LANGUAGES = tuple('en fr de ru pl tr es ml ar it'.split())

LANGUAGES2PREPROCESS = LANGUAGES  # 'ru pl tr ar'.split()
IGNORESUBWORDS = 'th'.split()

MAXQUEUESIZE = 100
CHUNKSIZE = 100
ENDSIGNAL = '<ENDSIGNAL>'

# all labels that do not start with punctuation and have at least a letter from
# the alphabet and some other character (thus 3 minimum characters)
label_pattern = re.compile(
    r"^[^!\"#$%&\'()*+,-.\\/:;<=>?@\[\]^_`{|}~].*[a-zA-Z].+$")


def automaton_add_keys(id_, row, lang, min_length, preprocessor, translations,
                       automaton):
    must_have_space = INSTANCES_THAT_MUST_HAVE_SPACE.intersection(
        row['properties'].get('instance_of', []))

    # if the entity has no label in the target language then add the english
    # label and aliases to the target language automaton
    if not row['labels'].get(lang) and row['labels'].get('en'):
        label = row['labels'].get('en')
        row['labels'][lang] = translations.get(label, label)
        row['aliases'][lang] = flatten([{o, translations.get(o, o)}
                                        for o in row['aliases'].get('en', [])])

        # add record of which languages have been fixed
        try:
            row['_fixed'] = row.get('_fixed', []) + [lang]
        except AttributeError as e:
            print("Issue with id:", id_)
            print("Language:", lang)
            print(row)
            raise e

    # returns if there is no label in either the original language or english
    label = row['labels'].get(lang)
    if not label:
        return automaton

    # run through all labels and their variations (surface forms)
    # if the label already exists in the automaton, append the new id
    aliases = row['aliases'].get(lang, [])
    key_list = [label] + aliases
    is_label_list = [True] + [False] * len(aliases)
    for key, is_label in zip(key_list, is_label_list):
        # 1) preprocess key (e.g.:lemmatization, lowercase, etc)
        key = (preprocessor or (lambda x: x))(key)

        # 2) check if key is valid
        if not is_key_valid(key, is_label, must_have_space, min_length) or \
                (must_have_space and ' ' not in key):
            continue

        # 3) add/append key to automaton
        if key in automaton:
            _, ids = automaton.get(key)
            ids.append(id_)
            automaton.add_word(key, (key, ids))
        else:
            automaton.add_word(key, (key, [id_]))

    return automaton


def automaton_patch_keys(patches, automaton, preprocessor, knowledge_base):
    for e, id_ in tqdm(patches.items(), desc='Patching labels'):
        cases = [e.lower(), e.upper(), e.title(), e[0].upper() + e[1:]]
        if preprocessor:
            cases.extend([preprocessor(o) for o in cases])
        if id_ in knowledge_base:
            for key in cases:
                automaton.add_word(key, (key, [id_]))
    return automaton


def automaton_remove_keys(labels_remove, preprocessor, lang, automaton):
    for e in tqdm(labels_remove, desc=f'Remove common words [{lang}]'):
        e = str(e)
        cases = [e.lower(), e.upper(), e.title(), e[0].upper() + e[1:]]
        if preprocessor:
            cases.extend([preprocessor(o) for o in cases])
        for w in cases:
            if w in automaton:
                automaton.remove_word(w)
    return automaton


def entity_description(entity: dict, lang: str = 'en') -> str:
    e_label = entity['labels'][lang]
    e_desc = entity['descriptions'].get(lang)
    if not e_desc:
        return e_label
    if e_label not in e_desc:
        e_desc = e_label + ', ' + e_desc
    return e_desc


def is_label_valid(label):
    # filter entities based on label
    return label and '(' not in label and ' - ' not in label \
           and label_pattern.match(label)


def is_key_valid(key, is_label, must_have_space, min_length):
    if not key or key.isnumeric():
        return False  # remove empty or numeric strings
    # conditions relevant for aliases only
    if not is_label and must_have_space:
        if ' ' not in key and not key.isupper():
            return False
    # must have a minimum length based on casing
    return (len(key) >= min_length or
            (key.isupper() and len(key) >= 3) or
            (key[0].isupper() and len(key) >= 4))


def load_knowledge_base(entities, sample_size=None):
    entities = to_ahocorasick(entities, desc='Adding entities')

    knowledge_base = dict()
    n_entities = sample_size if sample_size else len(entities)
    counter = tqdm(total=n_entities, desc='Entities found')
    for row in open(WIKIDATA, 'r'):
        # load the entity json
        try:
            row = json.loads(row)
        except json.JSONDecodeError:
            continue

        id_ = row.pop('id')
        if id_ in entities:
            knowledge_base[id_] = row
            counter.update()

        if counter.n >= n_entities:
            break  # when we find all the entities we can leave the cycle

    return knowledge_base


def load_knowledge_base_by_properties(properties_dict, include_properties=True):
    if not all([isinstance(o, (list, set)) for o in properties_dict.values()]):
        raise TypeError("The properties dict must have lists or sets as values")

    properties_dict = {k: set(v) for k, v in properties_dict.items()}
    entities = flatten(properties_dict.values())

    knowledge_base = dict()

    for row in tqdm(open(WIKIDATA, 'r'), desc='Total entities'):
        # load the entity json
        try:
            row = json.loads(row)
        except json.JSONDecodeError:
            continue

        id_ = row.pop('id')
        is_entity_relevant = False
        if id_ in entities and include_properties:
            is_entity_relevant = True
        else:
            for k, v in properties_dict.items():
                if v.intersection(row['properties'].get(k, [])):
                    is_entity_relevant = True
                    break
        if is_entity_relevant:
            knowledge_base[id_] = row

    return knowledge_base


def load_knowledge_base_edits(languages):
    df = pd.read_csv(INTERIM / 'entities_add.csv')
    entities_add = set(df['entity'])

    df = pd.read_csv(INTERIM / 'entities_remove.csv')
    entities_remove = set(df['entity'])

    labels_remove = {}
    df = pd.read_csv(INTERIM / 'labels_remove.csv')
    labels_remove['all'] = set(df['label'])
    for lang in languages:
        path = INTERIM / f'labels_remove_{lang}.csv'
        labels_remove[lang] = set()
        if path.exists():
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print('Could not process file:', path)
                raise e
            labels_remove[lang] = labels_remove[lang].union(df['label'])

    labels_patch = {}
    df = pd.read_csv(INTERIM / 'labels_patch.csv')
    labels_patch['all'] = dict(zip(df['alias'], df['id']))
    for lang in languages:
        path = INTERIM / f'labels_patch_{lang}.csv'
        labels_patch[lang] = dict()
        if path.exists():
            df = pd.read_csv(path)
            labels_patch[lang].update(dict(zip(df['alias'], df['id'])))

    translations = {}  # from english to key language
    for lang in languages:
        if lang == 'en':
            continue
        path = INTERIM / f'labels_pairs_{lang}.csv'
        translations[lang] = dict()
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except errors.ParserError as e:
            print(f'Error with file: {path}')
            raise e

        translations[lang].update(dict(zip(df['label'], df['translation'])))
        for k in labels_remove['en']:
            if k in translations[lang].keys():
                del translations[lang][k]
        inverted = {v: k for k, v in translations[lang].items()}
        for k in labels_remove[lang]:
            if k in inverted.keys():
                v = inverted[k]
                del translations[lang][v]
                del inverted[k]  # update the inverted dict

    return {
        'entities_add': entities_add,
        'entities_remove': entities_remove,
        'labels_remove': labels_remove,
        'labels_patch': labels_patch,
        'translations': translations
    }


def load_valid_properties(column='relevant'):
    valid_properties_ = {}
    for prop in PROPERTIES:
        df = pd.read_csv(INTERIM / f'{prop}_counts.csv')
        valid = df[column].fillna(0).astype(bool)
        valid_properties_[prop] = set(df['id'][valid])
    return valid_properties_


def parse_entity(entity: dict, data: dict):
    labels = entity['labels']
    properties = entity['properties']
    id_ = entity.pop('id')

    if not labels.get('en') or not entity['descriptions'].get('en'):
        return  # ignore entities missing an english label or description

    # ignore entities whose english label can be converted to a number
    try:
        float(labels['en'].replace(',', '.').rstrip('%'))
    except ValueError:
        pass
    else:
        return

    # check if entity should be added:
    if (data['entities'] and id_ not in data['entities']) \
            or id_ in data['edits']['entities_remove']:
        return

    # checks if the row should be processed based on its properties
    # as long as there's at least one match we process
    if id_ not in data['edits']['entities_add']:  # ignore property requirements
        if not flatten(v.intersection(properties.get(k, []))
                       for k, v in data['valid_properties'].items()):
            return

    # check if the entity has a valid english label
    label = labels.get('en')
    if not is_label_valid(label):
        return label

    # remove outstanding parenthesis inside labels and aliases
    entity['labels'] = {k: v.split(' (')[0]
                        for k, v in entity['labels'].items()}
    entity['aliases'] = {k: [o for o in v if '(' not in o]
                         for k, v in entity['aliases'].items()}

    return {id_: entity}


def mp_automaton(lang_queue: mp.Queue,
                 results_queue: mp.Queue,
                 lang: str,
                 data: dict):
    automaton = Automaton()

    fn = data['preprocessors'] \
        .get(lang) if lang in LANGUAGES2PREPROCESS else None
    translations = data['edits']['translations'].get(lang, {})

    while True:
        entities_chunk = lang_queue.get()
        if entities_chunk == ENDSIGNAL:
            break
        for id_, entity in entities_chunk.items():
            automaton = automaton_add_keys(id_, entity, lang,
                                           data['min_length'],
                                           fn, translations, automaton)

    labels_remove = data['edits']['labels_remove']
    labels_ = labels_remove['all'].union(labels_remove[lang])
    automaton = automaton_remove_keys(labels_, fn, lang, automaton)

    # automatons are not built because they must still be patched and that
    # requires a complete knowledge_base
    results_queue.put({lang: automaton})


def mp_load_file_to_queue(language_queues: Dict[str, mp.Queue],
                          filename: PathType, chunksize: int,
                          data: Dict,
                          stop_at: Optional[int] = None,
                          desc: Optional[str] = None,
                          ):
    def _process_row(json_as_text: str) -> dict:
        try:
            return json.loads(json_as_text)
        except json.JSONDecodeError:
            return {}

    def _process_chunk(chunk_: Collection) -> List[dict]:
        json_list = [_process_row(o) for o in chunk_]  # parse
        json_list = [o for o in json_list if o]  # remove parsing errors
        return json_list

    # reads the file by chunks
    file = tqdm(open(filename, 'r'), mininterval=0.1, desc=desc)
    filechunks = chunks(file, chunksize)
    filechunks = (_process_chunk(o) for o in filechunks)

    invalid_labels = []
    knowledge_base = {}

    # send a chunk (list of dicts) to queue
    for i, chunk in enumerate(filechunks):
        if stop_at and i >= (stop_at or 0) / chunksize + 1:
            break  # in case we want to process a sample this ends prematurely
        entities = {}
        for obj in chunk:
            entity = parse_entity(obj, data)
            if isinstance(entity, dict):
                entities.update(entity)
            elif isinstance(entity, str):
                # in this case the variable holds a string label
                invalid_labels.append(entity)

        knowledge_base.update(entities)
        if entities:  # ignore in case the chunk is empty
            for lang_q in language_queues.values():
                lang_q.put(entities)

    # terminate, send ENDSIGNAL to language queues and then stop
    for lang_q in language_queues.values():
        lang_q.put(ENDSIGNAL)

    return knowledge_base, invalid_labels


def mp_get_from_queue(q):
    while True:
        value = q.get()
        if value:
            return value


class WikidataMatcher(Tokenizer):

    def __init__(self, languages: Union[str, List[str]] = 'en',
                 preprocessors: Dict[str, Callable] = None,
                 use_qrank: bool = False, num_entities: int = 1_000_000,
                 threshold: float = 0.):
        self.use_qrank = use_qrank
        self.num_entities = int(num_entities)
        self.threshold = threshold

        self.languages = as_list(set(as_list(languages)).union(['en']))
        self.preprocessors = self._set_preprocessors(preprocessors)

        self.knowledge_base = {}
        self.matchers = None

        self.invalid_labels = []

    def __call__(self, obj: Union[ArticleCorpus, str, TextsList],
                 lang: str, return_labels: bool = False, extend: bool = False,
                 max_tokens: Optional[int] = None,
                 threshold: Optional[float] = None,
                 *args, **kwargs) -> Union[mathdict, List[mathdict]]:
        if isinstance(obj, ArticleCorpus):
            obj = [o.to_text() for o in obj]
        if isinstance(obj, str):
            return self.entitize(obj, lang=lang, max_tokens=max_tokens,
                                 extend=extend, threshold=threshold)
        else:
            return self.entitize_batch(obj, lang=lang, extend=extend,
                                       max_tokens=max_tokens,
                                       threshold=threshold)

    @classmethod
    def load_pickle(cls, path: PathType, threshold: float = None):
        from topic_inference.utils import readpickle

        # load existing pickle
        tokenizer: WikidataMatcher = readpickle(path)
        if threshold:
            tokenizer.threshold = threshold
        return tokenizer

    def fit(self, min_length=TOKEN_MIN_LEN, stop_at: Optional[int] = None,
            chunksize: int = CHUNKSIZE, maxqueuesize: int = MAXQUEUESIZE):
        """
        Build the tokenizer.

        Parameters
        ----------
        min_length : int, optional
            Defines the minimum number of characters of a valid text pattern.
        chunksize : int, optional
            Number of entities in a chunk.
        stop_at : int, optional
            Limits the number of processed chunks of entities.
            Useful for debugging.
        maxqueuesize : int, optional
            Change the maximum number of elements in the multiprocessing queue.
        """

        # temporary data to be processed
        edits = load_knowledge_base_edits(self.languages)
        entities = load_qrank(edits['entities_add'], self.num_entities) \
            if self.use_qrank else []

        data = {
            'edits': edits,
            'entities': entities,
            'min_length': min_length,
            'preprocessors': self.preprocessors,
            'valid_properties': load_valid_properties(),
        }

        # data to be kept
        self.invalid_labels = []
        self.knowledge_base = {}

        # multiprocessing
        language_queues = {o: mp.Queue(maxqueuesize) for o in self.languages}
        return_queue = mp.Queue()

        language_processes = {}
        for lang in self.languages:
            language_processes[lang] = mp.Process(
                target=mp_automaton,
                args=(
                    language_queues[lang],
                    return_queue,
                    lang,
                    data
                ),
                name='language_process_' + lang
            )
            language_processes[lang].daemon = True
            language_processes[lang].start()

        self.knowledge_base, self.invalid_labels = \
            mp_load_file_to_queue(
                language_queues,
                data=data,
                filename=WIKIDATA,
                chunksize=chunksize,
                stop_at=stop_at,
                desc='Loading rows'
            )

        if self.invalid_labels:
            print(f"There are {len(self.invalid_labels)} invalid labels.")

        automata = dict(flatten([mp_get_from_queue(return_queue).items()
                                 for _ in range(len(self.languages))]))

        # this block cannot be parallelized because it requires a complete
        # knowledge_base
        labels_patch = data['edits']['labels_patch']
        for lang in self.languages:
            fn = data['preprocessors'].get(lang) \
                if lang in LANGUAGES2PREPROCESS else None
            patches = {**labels_patch['all'], **labels_patch[lang]}
            automata[lang] = automaton_patch_keys(patches, automata[lang], fn,
                                                  self.knowledge_base)
            automata[lang].make_automaton()

        self.matchers = {
            lang: Matcher(v, remove_subwords=(lang not in IGNORESUBWORDS))
            for lang, v in automata.items()
        }

    def get_entities(self, labels, lang='en'):
        return [self.matchers[lang].automaton.get(o, (o, [])) for o in labels]

    def get_labels(self, entities, lang='en'):
        return [self.knowledge_base
                    .get(o, {'labels': {}})['labels']
                    .get(lang, LABEL4MISSING)
                for o in entities]

    def _set_preprocessors(self, preprocessors):
        preprocessors = preprocessors or {}

        if not isinstance(preprocessors, dict):
            raise TypeError("Variables `preprocessors` must be a dict!")

        non_callable = [k for k, v in preprocessors.items() if not callable(v)]
        if non_callable:
            raise ValueError(
                f"Preprocessors with keys {non_callable} are not callable!")

        self.preprocessors = preprocessors
        return preprocessors

    def tokenize(self, text: str, lang='en') -> List[str]:
        text = to_unicode(text).replace('\n', ' ')
        text = self.preprocessors.get(lang, lambda x: x)(text)
        return self.matchers.get(lang, self.matchers['en']).tokenize(text)

    def tokenize_batch(self, texts: Collection[str], lang='en') -> TokensList:
        texts = [to_unicode(o) for o in texts]
        texts = self.preprocessors.get(lang, lambda x: x).batch(texts)
        iterator = tqdm(texts, desc="Batch tokenizing")
        return [self.matchers[lang].tokenize(o) for o in iterator]

    def score_entities(self, text_embed: np.array, entities: Collection[str],
                       ref_score: float = 0., threshold: Optional[float] = None,
                       ) -> (mathdict, float):
        if threshold is not None:
            self.threshold = threshold

        kb = self.knowledge_base
        embeddings = {o: kb[o].get('embedding', {}).get('wikipedia')
                      for o in entities if o in self.knowledge_base}

        # remove entities and embeddings that do not have an embedding
        scores = mathdict(float, {
            entity: cos_sim(text_embed, embedding)
            for entity, embedding in embeddings.items()
            if embedding is not None
        })
        max_score = max(scores.values()) if scores else ref_score
        scores = scores.ge(self.threshold * max_score)

        return scores, max_score

    def embed_and_disambiguate(self, text: str, lang: str = 'en',
                               extend: bool = False,
                               max_tokens: Optional[int] = None,
                               threshold: Optional[float] = None
                               ) -> (np.array, mathdict):
        if threshold is not None:
            self.threshold = threshold
        if max_tokens:
            text = ' '.join(text.split()[:max_tokens])

        text_embed = embedding_model.encode(text)

        # tokenize and extract all entities
        labels = self.tokenize(text, lang=lang)
        if not labels:
            return text_embed, mathdict(int, {})
        all_entities = list(zip(*self.get_entities(labels, lang=lang)))[1]

        # generate scores for all unique entities
        unique_entities = list(set(flatten(all_entities)))
        scores, max_score = self.score_entities(text_embed, unique_entities)

        entities = [[i for i in o if scores.get(i)] for o in all_entities]
        entities = [max(o, key=lambda x: scores.get(x))
                    for o in entities if len(o)]

        if extend:
            extension = [self.knowledge_base[o]['properties'].values()
                         for o in entities]
            extension = flatten(set(flatten(o)) for o in extension)
            unique_extension = list(set(extension))
            scores, _ = self.score_entities(text_embed, unique_extension,
                                            ref_score=max_score)
            extension = [o for o in extension if scores.get(o)]
            entities.extend(extension)

        counter = mathdict(int, Counter(entities))

        return text_embed, counter

    def entitize(self, text: str, lang: str = 'en', extend: bool = False,
                 max_tokens: Optional[int] = None,
                 threshold: Optional[float] = None,
                 ) -> mathdict:
        _, counter = self.embed_and_disambiguate(
            text, lang, extend, max_tokens, threshold)
        return counter

    def entitize_batch(self, texts: Iterable[str], lang: str = 'en',
                       extend: bool = False, max_tokens: Optional[int] = None,
                       threshold: Optional[float] = None) -> List[mathdict]:
        f = partial(self.entitize, lang=lang, extend=extend,
                    max_tokens=max_tokens, threshold=threshold)
        iterator = tqdm(texts, desc="Batch entitizing")
        return [f(o) for o in iterator]

    def to_dictionary(self, lang='en'):
        from gensim.corpora import Dictionary
        return Dictionary(
            ([o] for o, _ in self.matchers[lang].automaton.values()))

    def to_edge(self, filename: PathType, add_text_entities=False):
        edges = []

        for id_, entity in tqdm(self.knowledge_base.items()):
            entities = set(flatten(entity['properties'].values()))
            if add_text_entities:
                entities = entities.union(entity.get('text_entities', []))
            new_edges = [f"{id_}\t{o}" for o in entities.difference([id_])]
            edges.extend(new_edges)

        writelines(edges, filename)

    def to_embedding(self, filename: PathType, embedding_type='wikipedia'):
        n_rows = len(self.knowledge_base)
        sample = next(iter(self.knowledge_base.values()))
        embed_size = len(sample['embedding'][embedding_type])

        def _generate_file():
            yield f"{n_rows} {embed_size}"
            for id_, entity in self.knowledge_base.items():
                embedding = entity['embedding'].get('wikipedia')
                yield id_ + ' ' + ' '.join(map("{:.8f}".format, embedding))

        writelines(_generate_file(), filename)

    def to_text(self, filename: PathType):
        def _as_text(entity):
            text_entities = entity.get('text_entities', [])
            return ' '.join(text_entities)

        texts = [_as_text(o) for o in tqdm(self.knowledge_base.values())]

        writelines(texts, filename)

    def translate_topic(self, topic, lang: str):
        topic = copy(topic)
        words = [str(o) for o in self.get_labels(topic['words'], lang=lang)]
        topic['words'] = tuple(words)
        return topic

    def invert_kb(self, lang: str):
        # it assumes that we want the smallest id value
        inverted_kb = {}
        for new_id, item in self.knowledge_base.items():
            label = item['labels'].get(lang)
            if label:
                old_id = inverted_kb.get(label)
                if not old_id or int(new_id[1:]) < int(old_id[1:]):
                    inverted_kb[label] = new_id
        return inverted_kb

    def accessible_entities(self, lang):
        # Some languages might not have labels to access all entities
        iterator = self.matchers[lang].automaton.items()
        return sorted(set(flatten(o for _, (_, o) in iterator)))

    @property
    def num_labels(self):
        return {o: len(self.matchers[o].automaton) for o in self.languages}

    @property
    def num_acessible_entities(self):
        return {o: len(self.accessible_entities(o)) for o in self.languages}

    @property
    def num_knowledge_base_entities(self):
        return len(self.knowledge_base)

    def update_embeddings(self, embeddings: Dict[str, np.array],
                          embedding_type: str):
        embedding_type = embedding_type.lower()
        existing_entities = set(embeddings).intersection(self.knowledge_base)
        missing_embeddings = set(embeddings).difference(existing_entities)

        for entity_id in existing_entities:
            if 'embedding' not in self.knowledge_base[entity_id]:
                self.knowledge_base[entity_id]['embedding'] = {}
            self.knowledge_base[entity_id]['embedding'].update(
                {embedding_type: embeddings[entity_id]})
        logger.warn(f'Missing embeddings for {len(missing_embeddings)} '
                    f'entities: {", ".join(missing_embeddings)}')

    def update_text_entities(self, text_entities: Dict[str, List[str]]):
        for entity_id in self.knowledge_base.keys():
            self.knowledge_base[entity_id]['text_entities'] = \
                text_entities.get(entity_id, [entity_id])
