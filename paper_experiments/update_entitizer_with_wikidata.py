from paper_experiments.utils import config, logger

from paper_experiments.scrape_wikipedia import WikiPage, entitizer
from topic_inference.utils import countlines, iteratejsonl, load_embeddings, \
    writepickle

from tqdm import tqdm

from pathlib import Path
import argparse

NUM_WORDS = 300


def update_entitizer_with_wikidata(wd_embeddings_path):
    logger.info(f"Loading {wd_embeddings_path}...")
    wd_embeddings = load_embeddings(wd_embeddings_path)
    entitizer.update_embeddings(wd_embeddings, 'wikidata')

    logger.info("Saving updated entitizer...")
    writepickle(entitizer, config['entitizer'])

    logger.info("Completed!")


def main(args):
    update_entitizer_with_wikidata(wd_embeddings_path=args.embeddings)


def parse_args():
    wd_embeddings_path = Path(config.get('wikidata_embeddings', 'paper_experiments/assets/wikidata.emb'))

    parser = argparse.ArgumentParser("")
    parser.add_argument("--embeddings", type=Path, default=wd_embeddings_path,
                        help="Wikidata embeddings file")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())
