from paper_experiments.utils import config, logger, load_entitizer

from topic_inference.embedding_model import embedding_model
from topic_inference.utils import flatten, chunks, writelines, getpickle

from datasets import load_dataset
from tqdm import tqdm

dataset_name = 'mlsum'
LANGUAGES = ['de', 'es', 'fr', 'ru', 'tu']
dataset = {o: load_dataset('mlsum', o, split='train') for o in LANGUAGES}
entitizer = load_entitizer()


def texts2embeddings(texts, chunksize=5000):
    iterator = chunks(tqdm(texts), chunksize)
    embeddings = []
    for chunk in tqdm(iterator, desc='Embeddings'):
        new_embeddings = embedding_model.encode(chunk).tolist()
        embeddings.extend(new_embeddings)
    return embeddings


def to_entities(id_, lang):
    text = texts[id_]
    embedding = embeddings[id_]
    intervals = entitizer.matchers[lang].get_intervals(text)
    intervals = {k: v[1] for k, v in intervals.items()}
    entities = set(flatten(intervals.values()))
    scores, _ = entitizer.score_entities(embedding, entities)
    filtered_paragraphs = \
        {k: {o: scores.get(o) for o in v if scores.get(o)}
         for k, v in intervals.items()}
    filtered_paragraphs = {
        k: max(v, key=v.get)
        for k, v in filtered_paragraphs.items()
        if len(v)
    }

    return list(filtered_paragraphs.values())


def preprocess_document(doc, lang):
    text = doc['title'] + '\n' + doc['text']
    return text, lang, doc['topic']


logger.info("Processing MLSUM")
docs = flatten((preprocess_document(o, lang) for o in dataset[lang])
               for lang in LANGUAGES)
texts, langs, topics = zip(*docs)

# fix turkish
langs = ['tr' if 'tu' else o for o in langs]

embeddings_path = config['assets'] + dataset_name + '_embeddings.pkl'
embeddings = getpickle(embeddings_path, texts2embeddings, texts)
entities = [to_entities(i, lang)
            for i, lang in tqdm(zip(range(len(embeddings)), langs),
                                desc="Preprocessing")]

pairs = [(text, ' '.join(e), t)
         for text, e, t in zip(texts, entities, topics) if len(e) >= 5]

filtered_texts, lines, filtered_topics = zip(*pairs)

filtered_texts = [o.replace('\n', ' ') for o in filtered_texts]

writelines(filtered_texts, config['datasets'][dataset_name]['texts'])
writelines(lines, config['datasets'][dataset_name]['entities'])
writelines(filtered_topics, config['datasets'][dataset_name]['texts'] + '_topics')

logger.info(f"Finished processing MLSUM. Number of lines: {len(lines)}")
