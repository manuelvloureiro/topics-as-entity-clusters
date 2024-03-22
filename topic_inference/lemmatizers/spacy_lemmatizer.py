import spacy
from spacy.tokens import Doc, Token
from tqdm import tqdm
import re

import warnings

ENDINGS = ["'"]

acronym_simplifier = re.compile(r'(?<!\w)([A-Z])\.')


def add_whitespace(token: Token):
    if token.whitespace_:
        return True
    return any(token.text.endswith(e) for e in ENDINGS)


def getter_token_fix_case(token):
    if token.text.isupper():
        return token.lemma_.upper()
    elif token.text.istitle():
        return token.lemma_.title()
    else:
        return token.lemma_


def getter_lemmatized_doc(doc):
    return ''.join(o._.truelemma.strip() + add_whitespace(o) * ' ' for o in doc)


def getter_pos_idx(token):
    return (token.idx, token.idx + len(token)), token.pos_


def getter_pos_doc(doc):
    return dict(o._.pos for o in doc)


Token.set_extension("truelemma", getter=getter_token_fix_case)
Token.set_extension("pos", getter=getter_pos_idx)
Doc.set_extension("truelemma", getter=getter_lemmatized_doc)
Doc.set_extension("pos", getter=getter_pos_doc)


class SpacyLemmatizer:
    models = {
        'en': 'en_core_web_sm',
        'fr': 'fr_core_news_sm',
        'de': 'de_core_news_sm',
        'pl': 'pl_core_news_sm',
        'ru': 'ru_core_news_sm',
        'es': 'es_core_news_sm',
        'pt': 'pt_core_news_sm',
        'it': 'it_core_news_sm',
    }

    def __init__(self, lang='en'):
        if lang not in self.models.keys():
            raise NotImplementedError(f"Language `{lang}` is not implemented.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.nlp = spacy.load(self.models[lang], disable=["parser", "ner"])

    def __call__(self, text, with_pos=False, *args, **kwargs):
        doc = self.nlp(text)
        if with_pos:
            return doc._.truelemma, doc._.pos
        return acronym_simplifier.sub(r'\1', doc._.truelemma)

    def batch(self, texts):
        iterator = tqdm(self.nlp.pipe(texts), total=len(texts),
                        desc="Lemmatizing texts", )
        return [o._.truelemma for o in iterator]  # noqa
