from topic_inference.utils import setlogginglevel
import zeyrek
from tqdm import tqdm

import re
import logging

ablative_pattern = re.compile(r'(d|t)(a|e)n$')
plural_pattern = re.compile(r'l(a|e)r$')
acronym_simplifier = re.compile(r'(?<!\w)([A-Z])\.')


def postprocess(word):
    if word.islower():
        word = ablative_pattern.sub('', word)  # remove ablatives
        word = plural_pattern.sub('', word)  # remove plurals
    return word


class TurkishLemmatizer:
    replacements = {
        "'": " '",
        "’": " ’",
    }

    def __init__(self):
        self.lemmatizer = zeyrek.MorphAnalyzer()

    @setlogginglevel(logging.getLogger(), 'ERROR')
    def __call__(self, text, *args, **kwargs):

        raw_lemmas = self.lemmatizer.lemmatize(self.replace(text))
        truelemma = []
        for word, lemmas in raw_lemmas:

            if word.isupper():
                truelemma.append(word)
            elif word.istitle() and word.lower() in [o.lower() for o in lemmas]:
                truelemma.append(word)
            else:
                title_lemmas = [o for o in lemmas if o.istitle()]
                if word.istitle() and title_lemmas:
                    truelemma.append(max(title_lemmas, key=len))
                elif word.istitle() and not title_lemmas:
                    truelemma.append(str(max(lemmas, key=len)).title())
                else:
                    truelemma.append(max(lemmas, key=len))

        # postprocess
        truelemma = [postprocess(o) for o in truelemma]

        return acronym_simplifier.sub(r'\1', ' '.join(truelemma))

    def batch(self, texts):
        iterator = tqdm(texts, total=len(texts), desc="Lemmatizing texts")
        return [self(o) for o in iterator]

    def replace(self, text):
        for k, v in self.replacements.items():
            text = text.replace(k, v)
        return text
