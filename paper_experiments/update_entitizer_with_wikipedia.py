from paper_experiments.utils import config, logger

from paper_experiments.scrape_wikipedia import WikiPage, entitizer
from topic_inference.utils import countlines, iteratejsonl, load_embeddings, \
    writepickle

from tqdm import tqdm

from pathlib import Path
import argparse

NUM_WORDS = 300


def update_entitizer_with_wikipedia(wikipedia, wp_embeddings_path, wd_edge_path,
                                    text_entities_path):
    logger.info(f"Loading {wikipedia}...")
    wp_embeddings = load_embeddings(wp_embeddings_path)
    entitizer.update_embeddings(wp_embeddings, 'wikipedia')

    logger.info("Generating text entities...")
    iterator = tqdm(iteratejsonl(wikipedia), desc="Wikipedia pages",
                    total=countlines(wikipedia))
    text_entities = {}
    for wikipedia_page in map(WikiPage.from_dict, iterator):
        if not wikipedia_page.id:
            continue
        text_entities[wikipedia_page.id] = wikipedia_page.text_entities

    entitizer.update_text_entities(text_entities)

    logger.info(f"Exporting text entities to txt file...")
    entitizer.to_text(text_entities_path)
    logger.info(f"Saved text entities to: {text_entities}")

    logger.info(f"Exporting knowledge base to edges file...")
    entitizer.to_edge(wd_edge_path, add_text_entities=True)
    logger.info(f"Exported edge file to: {wd_edge_path}")

    logger.info(f"Exporting ")
    logger.info("Saving updated entitizer...")
    writepickle(entitizer, config['entitizer'])

    logger.info("Completed!")


def main(args):
    update_entitizer_with_wikipedia(wikipedia=args.wikipedia,
                                    wp_embeddings_path=args.embeddings,
                                    wd_edge_path=args.edges,
                                    text_entities_path=args.texts)


def parse_args():
    wikipedia = Path(config.get('wikipedia', 'paper_experiments/assets/wikipedia.jsonl'))
    wp_embeddings_path = Path(config.get('wikipedia_embeddings', 'paper_experiments/assets/wikipedia.emb'))
    wd_edge_path = Path(config.get('wikidata_edges', 'paper_experiments/assets/wikidata.edg'))
    text_entities_path = Path(config.get('text_entities', 'paper_experiments/assets/text_entities_lines.txt'))

    parser = argparse.ArgumentParser("")
    parser.add_argument("--wikipedia", type=Path, default=wikipedia,
                        help="Scraped Wikipedia pages file")
    parser.add_argument("--embeddings", type=Path,
                        default=wp_embeddings_path,
                        help="Wikipedia embeddings file")
    parser.add_argument("--edges", type=Path, default=wd_edge_path,
                        help="Wikidata edges file to output")
    parser.add_argument("--texts", type=Path, default=text_entities_path,
                        help="Text entities path")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())
