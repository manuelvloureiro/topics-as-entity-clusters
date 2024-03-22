from paper_experiments.utils import config, logger, load_entitizer

from topic_inference.embedding_model import embedding_model
from topic_inference.utils import flatten, chunks, writelines

from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('cc_news', split='train')

entitizer = load_entitizer()


def texts2embeddings(texts, chunksize=1000):
    iterator = chunks(tqdm(texts), chunksize)
    embeddings = []
    for chunk in tqdm(iterator):
        new_embeddings = embedding_model.encode(chunk).tolist()
        embeddings.extend(new_embeddings)
    return embeddings


def to_entities(id_):
    text = texts[id_]
    embedding = embeddings[id_]
    intervals = entitizer.matchers['en'].get_intervals(text)
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


logger.info("Processing CCNews")
texts = [title + '\n' + text for title, text in
         zip(dataset['title'], dataset['text'])]
embeddings = texts2embeddings(texts)
entities = [to_entities(i) for i in tqdm(range(len(embeddings)))]

pairs = [(t, ' '.join(e)) for t, e in zip(texts, entities) if len(e) >= 5]
filtered_texts, lines = zip(*pairs)
filtered_texts = [o.replace('\n', ' ') for o in filtered_texts]

writelines(filtered_texts, config['datasets']['ccnews']['texts'])
writelines(lines, config['datasets']['ccnews']['entities'])

logger.info(f"Finished processing CCNews. Number of lines: {len(lines)}")
