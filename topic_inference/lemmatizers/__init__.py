from .spacy_lemmatizer import SpacyLemmatizer
from .russian_lemmatizer import RussianLemmatizer
from .turkish_lemmatizer import TurkishLemmatizer

lemmatizers = {
    'en': SpacyLemmatizer('en'),
    'fr': SpacyLemmatizer('fr'),
    'de': SpacyLemmatizer('de'),
    'es': SpacyLemmatizer('es'),
    'ru': RussianLemmatizer(),
    'tr': TurkishLemmatizer(),
}
