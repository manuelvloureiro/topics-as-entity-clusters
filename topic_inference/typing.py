from typing import *
from typing_extensions import TypedDict
from abc import ABCMeta, abstractmethod
from pathlib import Path

PathType = Union[str, Path]
PickleType = Any  # validation must be done with duck typing
JsonType = Union[Dict[str, Any],
                 List[Union[dict, Any]]]  # might need to be revised

TextsList = List[str]
TokensList = List[List[str]]
TextsOrTokensList = Union[TextsList, TokensList]

TopicProbabilities = Dict[int, float]
TopicPrediction = Tuple[int, TopicProbabilities]
MultipleTopicPredictions = Tuple[List[int], List[TopicProbabilities]]


class Topic(TypedDict):
    words: Union[Tuple[str], Tuple[()]]
    scores: Union[Tuple[float], Tuple[()]]


TopicDict = Dict[int, Topic]

TokenFilter = Callable[[TextsOrTokensList], TextsOrTokensList]


class Tokenizer(metaclass=ABCMeta):
    knowledge_base = None

    @abstractmethod
    def __call__(self, obj, lang, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def invert_kb(self, lang: str) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def get_labels(self, entities: Iterable[str], lang: str = 'en') -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def translate_topic(self, topic: Topic, lang: str) -> Topic:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_knowledge_base_entities(self):
        raise NotImplementedError()
