from paper_experiments.utils import config, logger, set_device, set_seed, \
    get_model_statistics

from topic_inference.models.kmeansfaiss import KmeansFaiss
from topic_inference.utils import readpickle, readlines, flatten, load_qrank

import argparse
import datetime

MODEL = "kmeansfaiss"


def main(args):
    if args.embedding_type in ['wikipedia', 'wikidata']:
        filename = '{}_{}_topics={}_seed={}_emb={}'.format(MODEL, args.dataset,
                                                           args.num_topics,
                                                           args.seed,
                                                           args.embedding_type)
    else:
        filename = '{}_{}_topics={}_seed={}_emb={}_alpha={}'.format(
            MODEL, args.dataset, args.num_topics, args.seed,
            args.embedding_type, args.alpha)
    pickle_path = f'paper_experiments/models/{filename}.pkl'
    stats_path = f'paper_experiments/stats/{filename}.jsonl'

    logger.info(f"Loading entitizer: {config['entitizer']}")
    entitizer = readpickle(config['entitizer'])

    model = KmeansFaiss()

    if args.dataset == 'wikipedia':
        qrank = load_qrank(n_entities=args.num_entities, as_automaton=True)
        entity_list = [o for o in qrank if o in entitizer.knowledge_base]
        kb = [entitizer.knowledge_base[o] for o in entity_list]
        texts = [' '.join(o.get('text_entities', [])) for o in kb
                 if o.get('text_entities')]
    else:
        texts = readlines(config['datasets'][args.dataset]['entities'])
        entity_list = set(flatten(o.split() for o in texts))
        logger.info(
            f"Sample entities (out of {len(entity_list)}: {list(entity_list)[:10]}")

    logger.info("Training...")
    start = datetime.datetime.now()
    model.train(
        entitizer,
        embedding_type=args.embedding_type,
        entity_list=entity_list,
        alpha=args.alpha,
        num_topics=args.num_topics,
        max_entities_train=args.num_entities,
        top_entities_to_topics=args.num_entities,
        use_gpu=bool(int(config['device']) >= 0),
        seed=int(args.seed)
    )
    end = datetime.datetime.now()
    msg = f"Training time: {end - start}"
    logger.info(msg)
    print(msg)

    logger.info("Updating topic top words...")
    model.model.update_top_words(
        texts=texts,
        top_entities=args.num_entities,
        chunksize=1_000,
        top_topics=1,
        n_jobs=10
    )

    topics = [list(o['words']) for o in model.topics(num_words=25).values()]
    scores = [list(o['scores']) for o in model.topics(num_words=25).values()]
    topics = [[e.upper() for e in o] for o in topics]
    scores = [[round(s, 6) for s in o] for o in scores]

    results = get_model_statistics(f"", [o.split() for o in texts],
                                   topics, scores, 10, stats_path)
    print(results['topics'])
    print(results['statistics'])

    logger.info(f"Saving pickle to {pickle_path}")
    model.save(pickle_path)
    logger.info(f"Saving completed!")


def parse_args():
    parser = argparse.ArgumentParser("Train a KMeans topic clustering model")

    parser.add_argument("--dataset", default='wikipedia', type=str,
                        help="Path of the dataset to train."
                             " Default is the entitizer")
    parser.add_argument("--embedding-type", default="wikidata",
                        help="Choose embedding type. Available: ['wikidata']")
    parser.add_argument("--num-topics", type=int,
                        default=config.get('num_topics', 1000),
                        help="Number of topics to train, default is 1000.")
    parser.add_argument("--num-entities", type=int,
                        default=config.get('qrank_entities', 1_000_000),
                        help="Max. number of entities to consider using QRank "
                             "(in case the tokenizer has more), default is 1M.")
    parser.add_argument("--alpha", type=float, default=1,
                        help="Alpha value, ratio between wikipedia and "
                             "wikidata embedding importance")
    parser.add_argument('--seed', type=int, default=config['device'],
                        help='Seed for deterministic results.')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device to use.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    logger.info(f"Starting: {__file__}")
    args_ = parse_args()
    set_device(args_.device)
    set_seed(int(args_.seed))
    main(args_)
