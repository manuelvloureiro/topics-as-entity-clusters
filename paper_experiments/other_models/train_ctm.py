from paper_experiments.utils import config, load_entitizer, logger, set_seed, \
    get_model_statistics

from topic_inference.utils import flatten, iteratejsonl, getpickle, readlines, \
    writepickle

from contextualized_topic_models.models.ctm import CombinedTM, DataLoader
from contextualized_topic_models.utils.data_preparation import \
    TopicModelDataPreparation

from tqdm import tqdm

import datetime
import argparse
import os

MODEL = 'ctm250'

parser = argparse.ArgumentParser('Combined Topic Model')
parser.add_argument('--dataset', type=str, default='wikipedia',
                    help='Dataset name')
parser.add_argument('--num-topics', type=int, default=100,
                    help='The number of topics used in training.')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='Number of iterations')
parser.add_argument('--epochs-step', type=int, default=1,
                    help='Interval of epochs used to calculate metrics.')
parser.add_argument('--early-stop', type=int, default=5,
                    help='Number of epochs without improvement')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for deterministic results.')
parser.add_argument('--device', type=str, default='0',
                    help='GPU device to use.')
parser.add_argument('--learning-rate', type=str, default='2e-3',
                    help='Learning rate used in model training.')

args = parser.parse_args()

set_seed(args.seed)
filename = '{}_{}_topics={}_seed={}_lr={}'.format(
    MODEL, args.dataset, args.num_topics, args.seed, str(args.learning_rate))
pickle_path = f'paper_experiments/other_models/models/{filename}.pkl'
stats_path = f'paper_experiments/stats/{filename}.jsonl'
dataset = args.dataset
num_topics = args.num_topics
num_epochs = args.num_epochs
epochs_step = args.epochs_step
early_stop = args.early_stop
lr = float(args.learning_rate)
device = args.device
save_dir = 'paper_experiments/other_models/model'


os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


def get_wikipedia_training_dataset():
    entitizer = load_entitizer()
    kb = entitizer.knowledge_base

    tokens = {}
    for entity_id, entity in tqdm(kb.items(), desc="Get tokens"):
        if entity.get('text_entities'):
            tokens[entity_id] = entity['text_entities']

    texts = {}
    for entity in tqdm(iteratejsonl(config['wikipedia']),
                       desc="Get texts", total=len(tokens)):
        if entity.get('id') and entity.get('text'):
            texts[entity['id']] = entity['text']

    entity_id_list = set(texts.keys()).intersection(tokens.keys())
    texts, tokens = zip(
        *[(texts[o].strip(), tokens[o]) for o in entity_id_list])

    vocab = set(flatten(tokens))

    tp = TopicModelDataPreparation("paraphrase-distilroberta-base-v1")
    training_dataset = tp.fit(text_for_contextual=texts,
                              text_for_bow=[' '.join(o) for o in tokens])

    return {'training_dataset': training_dataset, 'n_vocab': len(vocab)}


def get_other_training_dataset():
    texts = readlines(config['datasets'][dataset]['texts'])
    entities_lines = readlines(config['datasets'][dataset]['entities'])
    vocab = set(flatten(o.split() for o in entities_lines))

    if dataset == 'mlsum':
        embedding_model_name = \
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    else:
        embedding_model_name = "paraphrase-distilroberta-base-v1"
    logger.info('Using embedding model:', embedding_model_name.split('/')[-1])
    tp = TopicModelDataPreparation(embedding_model_name)
    training_dataset = tp.fit(text_for_contextual=texts,
                              text_for_bow=entities_lines)

    return {'training_dataset': training_dataset, 'n_vocab': len(vocab)}


class CombinedTMExtended(CombinedTM):
    def fit(self, train_dataset, validation_dataset=None, save_dir=None,
            verbose=False, patience=5, delta=0, n_samples=20):
        # Print settings to output file
        if verbose:
            print("Settings: \n\
                   N Components: {}\n\
                   Topic Prior Mean: {}\n\
                   Topic Prior Variance: {}\n\
                   Model Type: {}\n\
                   Hidden Sizes: {}\n\
                   Activation: {}\n\
                   Dropout: {}\n\
                   Learn Priors: {}\n\
                   Learning Rate: {}\n\
                   Momentum: {}\n\
                   Reduce On Plateau: {}\n\
                   Save Dir: {}".format(
                self.n_components, 0.0, 1. - (1. / self.n_components),
                self.model_type, self.hidden_sizes, self.activation,
                self.dropout, self.learn_priors, self.lr, self.momentum,
                self.reduce_on_plateau, save_dir)
            )

        self.model_dir = save_dir
        self.train_data = train_dataset
        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_data_loader_workers)

        # init training variables
        samples_processed = 0

        # train loop
        pbar = tqdm(self.num_epochs, position=0, leave=True)
        best = -1
        epochs_without_improvement = 0
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss = self._train_epoch(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()
            pbar.update(1)

            # save last epoch
            self.best_components = self.model.beta
            desc = "Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tTime: {}"
            desc = desc.format(epoch + 1, self.num_epochs, samples_processed,
                               len(self.train_data) * self.num_epochs,
                               train_loss, e - s)
            pbar.set_description(desc)
            if (
                    epoch == 0 or
                    (epoch + 1) % epochs_step == 0 or
                    epoch == self.num_epochs - 1
            ):
                print(f'Completed epoch {epoch + 1}. Metrics follow.')
                if save_dir is not None:
                    self.save(save_dir)
                print(80 * '-')
                print(desc)
                print(80 * '-')
                score = \
                    self.get_score(epoch, config['datasets'][dataset]['entities'])
                if score > best:
                    best = score
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += epochs_step
                    if epochs_without_improvement >= early_stop:
                        print(f'Early stop: {epochs_without_improvement}"'
                              f'" >= {early_stop}')
                        break
            else:
                print(f'Completed epoch {epoch + 1}. No metrics.')
        pbar.close()

        # self.training_doc_topic_distributions = \
        #     self.get_doc_topic_distribution(train_dataset, n_samples)

    def get_topics(self, k=10):
        import torch
        from collections import defaultdict
        """
        Retrieve topic words.

        :param k: int, number of words to return per topic, default 10.
        """
        assert k <= self.bow_size, "k must be <= input size."
        component_dists = self.best_components
        sum_ = component_dists.sum(axis=1).unsqueeze(0).T
        component_dists = component_dists.div(sum_)
        topics = defaultdict(list)
        scores = defaultdict(list)
        for i in range(self.n_components):
            scores_, idxs = torch.topk(component_dists[i], k)
            scores[i] = scores_.cpu().detach().tolist()
            component_words = [self.train_data.idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            topics[i] = [o.upper() for o in component_words]
        return topics, scores

    def get_score(self, epoch, texts_path, num_words=10):
        topics, scores = self.get_topics(25)
        topics = [[e.upper() for e in o] for o in topics.values()]
        scores = [[round(s, 6) for s in o] for o in scores.values()]
        texts = [o.split() for o in readlines(texts_path)]
        results = get_model_statistics(epoch, texts, topics, scores, num_words,
                                       stats_path)
        print(results['topics'])
        print(results['statistics'])
        return results['statistics']['c_npmi']


if args.dataset == 'wikipedia':
    data_path = f'paper_experiments/other_models/models/ctm_data.pkl'
    data = getpickle(data_path, get_wikipedia_training_dataset)
else:
    data_path = f'paper_experiments/other_models/models/ctm_data_{args.dataset}.pkl'
    data = getpickle(data_path, get_other_training_dataset)

training_dataset = data['training_dataset']
n_vocab = data['n_vocab']

logger.info('Started training Contextualized Topic Model...')
ctm = CombinedTMExtended(bow_size=n_vocab, contextual_size=768, lr=lr,
                         n_components=num_topics, num_epochs=num_epochs)
ctm.fit(training_dataset, verbose=True, save_dir=save_dir)
logger.info('Completed training.')

writepickle(ctm, pickle_path)
