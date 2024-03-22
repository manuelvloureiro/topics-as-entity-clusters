from paper_experiments.utils import config, logger

from topic_inference.typing import *
from topic_inference.utils import readpickle, flatten
from topic_inference.tokenizers import WikidataMatcher

from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np

from pathlib import Path
import re
import argparse
import urllib3
import json
import requests
from html import unescape
import random

urllib3.disable_warnings()  # ignore InsecureRequestWarning (verify=False)

URL = 'https://en.wikipedia.org/wiki/{}'

EXCLUDED_URLS = [
    'File:',
    'Wikipedia:',
    'Category:',
    'Template:',
    'Help:',
    'Special:',
    'Template_talk:',
    'Talk:',
    'User:'
    'Portal:',
    'Main_Page',
    'List_of',
    '(identifier)'
]

UNK = '*'

# matches references inside wikipedia text
REFS = re.compile(r"\[[\d+, ]\]")  # noqa

entitizer: WikidataMatcher = readpickle(config['entitizer'])
entitizer.threshold = config.get('disambiguation_threshold', 0.5)
kb = entitizer.knowledge_base

# keys: enwiki text to be used to obtain wikipedia urls
# values: (wikidata id, english label) pairs. we include the label to allow for
# labels without a corresponding entity in the tokenizer
# the first pass adds
sorted_kb = sorted(kb.items(), key=lambda x: int(x[0][1:]), reverse=True)
enwiki2id = {v['labels'].get('en'): (k, v['labels'].get('en'))
             for k, v in sorted_kb if v['labels'].get('en')}
enwiki2id.update({v.get('enwiki').lower(): (k, v['labels'].get('en'))
                  for k, v in sorted_kb if v.get('enwiki')})
enwiki2id.update({v.get('enwiki').title(): (k, v['labels'].get('en'))
                  for k, v in sorted_kb if v.get('enwiki')})
enwiki2id.update({v.get('enwiki'): (k, v['labels'].get('en'))
                  for k, v in sorted_kb if v.get('enwiki')})
enwiki2id.update()

excel_options = {
    'strings_to_formulas': False,
    'strings_to_urls': False,
}


class WikiPage:

    def __init__(self, enwiki: Optional[str] = None, seed: int = 42):
        if enwiki:
            self.enwiki = enwiki.replace('_', ' ')
            self.id = enwiki2id.get(self.enwiki, [''])[0]
            soup = self.request_enwiki()
            self.page = str(soup)
            self.anchors = self._get_anchors(soup)
            self.text = self._get_text(soup)
            self._embedding = None
            self.seed = seed or 42

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.enwiki}>"

    @classmethod
    def from_dict(cls, wiki_dict: dict):
        self = cls.__new__(cls)
        self.__dict__.update(wiki_dict)
        if self.__dict__.get('anchors'):
            self.anchors = {eval(k): v for k, v in self.anchors.items()}
        return self

    def _get_anchors(self, soup: BeautifulSoup) -> Dict[Tuple[int, int], str]:
        anchors = {}
        current_index = 0

        for wiki_anchor in soup.find_all('a', href=True):
            href = wiki_anchor['href']
            if href.startswith('/wiki/'):
                if not any(exclusion in href for exclusion in EXCLUDED_URLS):
                    # get index
                    wiki_anchor = str(wiki_anchor)
                    start_index = self.page.find(wiki_anchor, current_index)
                    current_index = start_index + 1

                    # get entity
                    href = href.split('/wiki/')[-1]
                    href = href.split('#')[0]
                    href = href.replace('_', ' ')
                    href = href.split(' (')[0]

                    # add index, entity pair
                    entity_id = enwiki2id.get(href, (UNK, UNK))[0]
                    end_index = start_index + len(wiki_anchor) + 1
                    anchors[(start_index, end_index)] = entity_id

        # remove unknown entities
        for interval in list(anchors):
            if anchors[interval] == UNK:
                del anchors[interval]

        return anchors

    @staticmethod
    def _get_text(soup: BeautifulSoup) -> str:
        # this changes the span tags inplace
        for span_tag in soup.findAll('span'):
            span_tag.replace_with('')

        text = '\n'.join(o.text for o in list(soup.find_all('p')))
        text = REFS.sub('', text)
        text = unescape(text)
        return text

    def request_enwiki(self) -> BeautifulSoup:
        raw_html = requests.get(self.url, verify=False)
        soup = BeautifulSoup(raw_html.content, 'html.parser')
        return soup

    @property
    def text_entities(self) -> List[str]:
        # pairs: Dict[Tuple[int, int], str] = {}
        # pairs.update(self.filtered_anchors)
        # pairs.update(self.filtered_paragraphs)
        # text_entities_ = [pairs[o] for o in overlap(list(pairs.keys()))]

        # producing text_entities_ had to be altered from the code above to this
        # current iteration. the reason being that self.filtered_anchors used to
        # find text patterns across the whole page but that is 100x slower than
        # just running the code on the text, as such we assume the tokens to be
        # ordered first by the entities extracted from urls and then by the
        # entities extracted from the text
        text_entities_: List[str] = [self.id]
        text_entities_.extend(self.filtered_anchors.values())
        text_entities_.extend(self.filtered_paragraphs.values())

        entities_counts = Counter(text_entities_)

        counts_id = entities_counts.get(self.id, 0)
        max_counts = max(list(entities_counts.values()) + [0])
        random.seed(self.__dict__.get('seed', self.__dict__.get("seed", 42)))

        # in case the entity_id is not the most represented then we add new
        # copies in random positions
        for _ in range(max_counts - counts_id + 1):
            pos = random.randint(0, len(text_entities_))
            text_entities_.insert(pos, self.id)

        return text_entities_

    def to_dict(self):
        return {'enwiki': self.enwiki,
                'id': self.id,
                'text_entities': self.text_entities,
                'embedding': self.embedding.tolist(),
                'text': self.text
                }

    def to_raw_dict(self):
        return {'enwiki': self.enwiki,
                'id': enwiki2id.get(self.enwiki, [''])[0],
                'anchors': {str(k): v for k, v in self.anchors.items()},
                'text': self.text,
                'page': self.page
                }

    @property
    def filtered_anchors(self) -> Dict[Tuple[int, int], str]:
        entities = set(self.anchors.values())
        scores, _ = entitizer.score_entities(self.embedding, entities)
        filtered_anchors = {k: v for k, v in self.anchors.items()
                            if scores.get(v)}
        return filtered_anchors

    @property
    def filtered_paragraphs(self) -> Dict[Tuple[int, int], str]:
        paragraphs = entitizer.matchers['en'].get_intervals(self.text)
        paragraphs = {k: v[1] for k, v in paragraphs.items()}
        entities = set(flatten(paragraphs.values()))
        scores, _ = entitizer.score_entities(self.embedding, entities)
        filtered_paragraphs = \
            {k: {o: scores.get(o) for o in v if scores.get(o)}
             for k, v in paragraphs.items()}
        filtered_paragraphs = {
            k: max(v, key=v.get)
            for k, v in filtered_paragraphs.items()
            if len(v)
        }
        return filtered_paragraphs

    @property
    def embedding(self) -> np.array:
        if ('_embedding' not in self.__dict__.keys() or
                self.__dict__.get('_embedding') is None):
            text_embed, tokens = entitizer.embed_and_disambiguate(
                self.text, lang='en', max_tokens=300)
            self._embedding = text_embed
        return self._embedding

    @property
    def url(self) -> str:
        return URL.format(self.enwiki.replace(' ', '_'))


def scrape_wikipedia(path):
    entities = list(entitizer.knowledge_base.keys())

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = open(Path(path).with_suffix('.jsonl'), 'w')

    logger.info("Scraping Wikipedia pages")
    for e in tqdm(entities, desc="Wikipedia pages"):
        enwiki = entitizer.knowledge_base.get(e, {}).get('enwiki')
        if not enwiki:
            continue
        for i in range(10):
            try:
                wiki = WikiPage(enwiki)
            except Exception as e:
                logger.warn(f"Attempt {i + 1}:", e, 'Could not scrape:', enwiki)
            else:
                if wiki.id:
                    f.write(json.dumps(wiki.to_raw_dict()))
                    f.write('\n')
                break

    f.close()


def main(args):
    scrape_wikipedia(path=args.path)


def parse_args():
    path = Path(config.get('wikipedia', 'paper_experiments/assets/wikipedia.jsonl'))

    parser = argparse.ArgumentParser(
        "Scrapes Wikipedia pages that exist in an entitizer")
    parser.add_argument("--path", type=Path, default=path,
                        help="Scraped wikipedia pages file")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())
