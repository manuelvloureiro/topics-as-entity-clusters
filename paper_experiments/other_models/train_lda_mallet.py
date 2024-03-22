from paper_experiments.utils import logger, get_model_statistics

from topic_inference.wrappers.ldamallet import LdaMallet
from topic_inference.utils import readlines, writepickle

from gensim.corpora import Dictionary

import time
import argparse

PATH_TO_MALLET_BINARY = 'resources/mallet-2.0.8/bin/mallet'
MODEL = 'lda'

parser = argparse.ArgumentParser('LDA Mallet')
parser.add_argument('--dataset', type=str, default='wikipedia',
                    help='Dataset name')
parser.add_argument('--num-topics', type=int, default=100,
                    help='The number of topics used in training.')
parser.add_argument('--num-epochs', type=int, default=1000,
                    help='Number of iterations')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for deterministic results.')

args = parser.parse_args()
filename = '{}_{}_topics={}_seed={}'.format(
    MODEL, args.dataset, args.num_topics, args.seed)
pickle_path = f'paper_experiments/other_models/models/{filename}.pkl'
stats_path = f'paper_experiments/stats/{filename}.jsonl'
dataset = 'paper_experiments/assets/{}_lines.txt'.format(args.dataset)
num_topics = args.num_topics
save_dir = 'paper_experiments/other_models/model'

tokens = [o.split() for o in readlines(dataset)]
dictionary = Dictionary(tokens)
bag_of_words = [dictionary.doc2bow(o) for o in tokens]

logger.info(f'Started training LDA Mallet for {num_topics} topics...')
print(f'Starting time {time.strftime("%Y-%m-%d %H:%M:%S")}')
model = LdaMallet(
    PATH_TO_MALLET_BINARY,
    num_topics=num_topics,
    corpus=bag_of_words,
    id2word=dictionary,
    iterations=args.num_epochs,
    random_seed=args.seed,
)
logger.info('Completed training.')
print(f'Ending time {time.strftime("%Y-%m-%d %H:%M:%S")}')

writepickle(model, pickle_path)

topics, scores = zip(*[list(zip(*model.show_topic(i, topn=25)))
                       for i in range(num_topics)])
topics = [[e.upper() for e in o] for o in topics]
scores = [[round(s, 6) for s in o] for o in scores]

get_model_statistics(args.num_epochs, tokens, topics, scores, 10, stats_path)
