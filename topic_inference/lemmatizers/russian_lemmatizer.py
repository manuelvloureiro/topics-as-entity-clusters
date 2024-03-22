import pymorphy2
from tqdm import tqdm

import re
from string import punctuation

punct = re.compile('([' + re.escape(punctuation) + '])')
acronym_simplifier = re.compile(r'(?<!\w)([A-Z])\.')


def fixcase(lemmatized_token: str, original_token: str):
    if original_token.isupper():
        return original_token  # ignore lemmatization
    elif original_token.istitle():
        return lemmatized_token.title()
    else:
        return lemmatized_token


class RussianLemmatizer:
    replacements = {
        "'": " '",
        "’": " ’",
    }

    def __init__(self):
        self.lemmatizer = pymorphy2.MorphAnalyzer()

    def __call__(self, text, *args, **kwargs):
        tokens = self.replace(text).split()
        raw_lemmas = [self.lemmatizer.parse(o) for o in tokens]
        truelemma = [o[0].normal_form for o in raw_lemmas]
        truelemma = [fixcase(x, y) for x, y in zip(truelemma, tokens)]
        return acronym_simplifier.sub(r'\1', ' '.join(truelemma))

    def batch(self, texts):
        iterator = tqdm(texts, total=len(texts), desc="Lemmatizing texts")
        return [self(o) for o in iterator]

    def replace(self, text):
        for k, v in self.replacements.items():
            text = text.replace(k, v)
        text = punct.sub(r' \1', text)
        return text
