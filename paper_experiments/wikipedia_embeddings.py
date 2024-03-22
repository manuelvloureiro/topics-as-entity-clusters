from paper_experiments.utils import config, logger

from topic_inference.embedding_model import embedding_model
from topic_inference.utils import countlines, iteratejsonl, readpickle, \
    save_embeddings, chunks
from topic_inference.tokenizers.wikidatamatcher import entity_description

from tqdm import tqdm

from pathlib import Path
import argparse

NUM_WORDS = 300
CHUNKSIZE = 1000

entitizer = readpickle(config['entitizer'])


def get_texts(wikipedia):
    iterator = tqdm(iteratejsonl(wikipedia), desc="Embedding texts",
                    total=countlines(wikipedia))
    for wikipedia_page in iterator:
        entity_id = wikipedia_page.get('id')
        text = wikipedia_page.get('text')
        if entity_id and text:
            text = ' '.join(text.split()[:NUM_WORDS])
            yield entity_id, text


def get_descriptions(missing_entity_ids):
    for entity_id in tqdm(missing_entity_ids, desc="Embedding descriptions"):
        entity = entitizer.knowledge_base[entity_id]
        description = entity_description(entity, lang='en')
        yield entity_id, description


def process_chunk(embeddings, chunk):
    entity_ids, texts = zip(*chunk)
    chunk_embeddings = embedding_model.encode(texts)
    for i in range(len(entity_ids)):
        embeddings[entity_ids[i]] = chunk_embeddings[i]
    return embeddings


def get_wikipedia_embeddings(wikipedia):
    embeddings = {}

    for chunk in chunks(get_texts(wikipedia), 1000):
        embeddings = process_chunk(embeddings, chunk)

    missing_entity_ids = set(entitizer.knowledge_base.keys()) \
        .difference(embeddings.keys())

    for chunk in chunks(get_descriptions(missing_entity_ids), 1000):
        embeddings = process_chunk(embeddings, chunk)

    return embeddings


def wikipedia_embeddings(wikipedia, path):
    logger.info("Generating Wikipedia embeddings")
    embeddings = get_wikipedia_embeddings(wikipedia)
    logger.info("Saving Wikipedia embeddings")
    save_embeddings(embeddings, path)
    logger.info("Complete!")


def main(args):
    wikipedia_embeddings(wikipedia=args.wikipedia, path=args.embeddings)


def parse_args():
    wikipedia = Path(config.get('wikipedia', 'paper_experiments/assets/wikipedia.jsonl'))
    emb_path = Path(config.get('wikipedia_embeddings',
                               'paper_experiments/assets/wikipedia.emb'))

    parser = argparse.ArgumentParser("Generates Wikipedia embeddings file")
    parser.add_argument("--wikipedia", type=Path, default=wikipedia,
                        help="Scraped Wikipedia pages file")
    parser.add_argument("--embeddings", type=Path, default=emb_path,
                        help="Wikipedia embeddings file")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())
