from paper_experiments.utils import config, logger

from topic_inference.tokenizers import WikidataMatcher
from topic_inference.lemmatizers import lemmatizers
from topic_inference.utils import writepickle

import argparse
from pathlib import Path


def main(args):
    logger.info(f"Building simple entitizer: {args.entitizer_path}")

    def build_entitizer():
        entitizer = WikidataMatcher(
            languages=config['languages'],
            preprocessors=lemmatizers,
            use_qrank=True,
            num_entities=args.num_entities,
            threshold=config.get('disambiguation_threshold', 0.5)
        )

        entitizer.fit()

        return entitizer

    entitizer = build_entitizer()
    writepickle(entitizer, args.entitizer_path)

    logger.info("Completed!")


def parse_args():
    parser = argparse.ArgumentParser("Build an entitizer without embeddings")

    ent_path = Path(config.get('entitizers', 'paper_experiments/assets/entitizer.pkl'))
    edg_path = Path(config.get('edges', 'paper_experiments/assets/knowledge_base.edg'))
    parser.add_argument("--entitizer-path", type=Path, default=ent_path,
                        help="Path to save entitizer")
    parser.add_argument("--edges-path", type=Path, default=edg_path,
                        help="Path to save knowledge base in edges format")
    parser.add_argument("--num-entities", type=int,
                        default=config.get('qrank_entities', 1_000_000),
                        help="Max. number of entities to the entizier using  "
                             "QRank")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    logger.info(f"Starting: {__file__}")
    main(parse_args())
