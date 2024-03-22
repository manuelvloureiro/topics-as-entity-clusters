from topic_inference.utils import dataframe_to_markdown, topic_words, flatten, default_logger as logger
from topic_inference.models.utils import read_topics_excel, MINSCORE, NUMWORDS
from topic_inference.collections import ArticleCorpus, mathdict, Coherence
from topic_inference.typing import *

import pandas as pd
import numpy as np
from tqdm import tqdm

from abc import ABCMeta, abstractmethod
from functools import partial
from pathlib import Path
import json

METRICS = {'c_npmi': 'npmi'}

SEED = 42


class AbstractTopicModel(metaclass=ABCMeta):
    """Abstract base class that defines a common API for topic modeling training
     and inference. Each subclass should encapsulate a different topic modeling
     implementation/package.
    """

    def __init__(self, tokenizer: Optional[Tokenizer] = None,
                 token_filter: Optional[TokenFilter] = None,
                 lang: Optional[str] = None):
        """

        Parameters
        ----------
        tokenizer : Tokenizer, optional
            Object that processes a text or a list of texts returning lists of
            tokens.
        token_filter : TokenFilter, optional
            Function to remove predefined tokens from one or more tokenized
            texts.
        lang : str, optional
            Default language.

        """
        self.tokenizer = tokenizer
        self.token_filter = token_filter
        self.lang = lang or 'en'
        self.model = None
        self._coherence = None
        self._topics = None
        self._topics_num_words = None

    def __call__(self, obj: Union[ArticleCorpus, TextsOrTokensList],
                 lang: Optional[str] = None, *args, **kwargs):
        if isinstance(obj, ArticleCorpus):
            obj = [o.to_text() for o in obj]
        return self.transform(obj, lang, **kwargs)

    def batch_preprocess(self, tokens: TextsOrTokensList,
                         lang: Optional[str] = None, use_pos: bool = False,
                         *args, **kwargs) -> TokensList:
        """
        Processes a list of texts into a tokenized representation.

        Parameters
        ----------
        tokens : TextsOrTokensList
            A list of texts to be tokenized or tokenized texts.
        lang : str, optional
            Language used in tokenization.
        use_pos: bool
            If True, part-of-speech tagging is used to improve the tokenizer
        args : Any
            Not used, kept for subclassing.
        kwargs : Any
            Not used, kept for subclassing.

        Returns
        -------
        TokensList
            A list of tokenized texts.

        """

        # tokenize in case `tokens` is a list of texts
        if isinstance(tokens[0], str):
            lang = lang or self.__dict__.get('lang', 'en')
            desc = 'Tokenizing texts'
            iterable = tokens if len(tokens) == 1 else tqdm(tokens,
                                                            desc=desc)
            tokens = [self.tokenizer(o, lang, use_pos=use_pos)
                      for o in iterable]

        if self.token_filter:
            tokens = self.token_filter(tokens)

        tokens = [t if len(t) else ['<EMPTY>'] for t in tokens]

        return tokens

    @property
    def weights_df(self) -> pd.DataFrame:
        """
        Generate weights matrix as a DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the topic word weights, for all .

        """
        word_weights = self.topics(num_words=-1)
        table = [dict(zip(word_weights[o]['words'], word_weights[0]['scores']))
                 for o in range(self.num_topics)]
        df = pd.DataFrame(table)
        df = df[sorted(df.columns)]
        return df

    @abstractmethod
    def fit(self, tokens: TextsOrTokensList, num_topics: int,
            lang: Optional[str] = None, seed: int = SEED, *args, **kwargs):
        """
       Abstract method. Override it to train a model.

        Parameters
        ----------
        tokens : TextsOrTokensList
            A list of texts to be tokenized or tokenized texts.
        num_topics : int
            Number of top topics to be added to output.
        lang : str, optional
            Language used in tokenization.
            If None, the default language is used.
        seed : int
            A random seed to allow reproducibility.
        args : Any
            Not used, kept for subclassing.
        kwargs : Any
            Not used, kept for subclassing.

        """
        pass

    @abstractmethod
    def transform(self, tokens: TextsOrTokensList,
                  lang: Optional[str] = None,
                  **kwargs) -> MultipleTopicPredictions:
        """
        Abstract method. Override it to transform a list of documents.

        Parameters
        ----------
        tokens : TextsOrTokensList
            A list of texts to be tokenized or tokenized texts.
        lang : str, optional
            Language used in tokenization.
            If None, the default language is used.

        Returns
        -------
        MultipleTopicPredictions
            A tuple containing the topic ids and dictionaries containing the
            probabilties for all topics.

        """
        pass

    def predict(self, text: Union[str, TokensList],
                lang: Optional[str] = None,
                **kwargs) -> TopicPrediction:
        """
        Predicts the topic probabilities for a single text.

        Parameters
        ----------
        text : str or TokensList
            A text to be tokenized or a tokenized text.
        lang : str, optional
            Language used in tokenization.
            If None, the default language is used.

        Returns
        -------
        TopicPrediction
            A tuple containing the top topic id and a dictionary containing the
            probabilities for all topics.

        """
        pred, probs = self.transform([text], lang, **kwargs)
        return pred[0], probs[0]

    def coherence(self, tokens: TextsOrTokensList, metric: str = 'c_npmi',
                  lang: Optional[str] = None, num_words: int = 20) -> dict:
        """

        Parameters
        ----------
        tokens : TextsOrTokensList
            A list of texts to be tokenized or tokenized texts.
        metric : str
            Coherence metric to be calculated.
            Can be one of the following: ['c_v', 'c_npmi', 'c_uci', 'u_mass'].
        lang : str, optional
            Language used in tokenization.
            If None, the default language is used.
        num_words : int
            Number of words used to score coherence.

        Returns
        -------
        dict
            A dictionary containing the coherence and the coherence per topic.

        """
        if metric not in METRICS:
            raise RuntimeError("Unrecognized metric:", metric)

        tokens = self.batch_preprocess(tokens, lang)

        topics = [list(o['words']) for o in self.topics(num_words).values()]
        coherence = Coherence(topics)

        avg_coherence, coherence_per_topic = coherence \
            .coherence(tokens, metric=METRICS[metric], num_words=num_words)
        return {
            'coherence': avg_coherence,
            'coherence_per_topic': coherence_per_topic
        }

    def diversity(self, num_words=25):
        top_words = flatten(
            o['words'] for o in self.topics(num_words=num_words).values())
        return len(set(top_words)) / len(top_words)

    @abstractmethod
    def _topic(self, topic_id: int, num_words: int = 10) -> Topic:
        """
        Abstract method. Override it to return an untranslated topic.

        Parameters
        ----------
        topic_id : int
            Topic identifier.
        num_words : int
            Number of returned top words representing the topic.

        Returns
        -------
        Topic
            A dictionary containing the top words and respective scores
            representing a topic.

        """
        pass

    def topic(self, topic_id: int, num_words: int = 25,
              lang: Optional[str] = None) -> Topic:
        """
        Get the top words and respective scores for a given topic.

        Parameters
        ----------
        topic_id : int
            Topic identifier.
        num_words : int
            Number of returned top words representing the topic.
        lang : str, optional
            Language used in tokenization.
            If None, the default language is used.

        Returns
        -------
        Topic
            A dictionary containing the top words and respective scores
             representing a topic.

        """
        if self._topics and self._topics_num_words == num_words:
            topic = self._topics[topic_id]
        else:
            topic = self._topic(topic_id, num_words=num_words)
        if lang:
            topic = self.tokenizer.translate_topic(topic, lang)
        return topic

    def topics(self, num_words: int = 20) -> TopicDict:
        """
        Returns the top words per topics.

        Parameters
        ----------
        num_words : int
            Number of returned top words representing the topic.

        Returns
        -------
        TopicDict
            A dictionary containing all topic ids as keys and top words as
            values.

        """
        # saves topics to variable to avoid running multiple times
        if self._topics_num_words == num_words:
            topics = self._topics
        else:
            topics = {i: self._topic(i, num_words)
                      for i in range(self.num_topics)}
            self._topics_num_words = num_words
            self._topics = topics
        return topics

    @abstractmethod
    def topic_label(self, topic_id: int) -> str:
        pass

    def get_top_topics(self, tokens: Union[str, TextsOrTokensList],
                       num_topics: int = 3, num_words: int = 10,
                       lang: Optional[str] = None) -> List[Dict[str, list]]:
        """
        Get the top words and respective scores for a specified number of top
        topics, for a list of texts or tokenized texts.

        Parameters
        ----------
        tokens : str or TextsOrTokensList
            A text, a list of texts to be tokenized or tokenized texts.
        num_topics : int
            Number of top topics to be added to output.
        num_words : int
            Number of top words per topic to be added to output.
        lang : str, optional
            Language to convert the top words.
            If None, Wikidata IDs are returned.

        Returns
        -------
        List[Dict[str, list]]
            A list containing dicts with top topic ids, top words per topic id
            and the respective scores. Each dict refers to a particular text.

        """

        # when `tokens` is a single text or a single list of tokens (akin to a
        # single text) we need to wrap it in a list to use it with .transform()
        if (
                isinstance(tokens, str)
                or (isinstance(tokens, list) and ' ' not in tokens[0])
        ):
            tokens = [tokens]

        _, probs = self.transform(tokens, lang=lang)

        results = []
        for p in probs:
            topic_ids, scores = zip(*(mathdict(float, p).head(num_topics)))
            topic_words_list = tuple([
                self.topic(o, num_words=num_words, lang=lang)['words']
                for o in topic_ids
            ])
            results.append({'topics': topic_ids, 'scores': scores,
                            'words': topic_words_list})

        return results

    @property
    @abstractmethod
    def num_topics(self) -> int:
        """
        Abstract property. Override it to return the number of topics
        of a trained model.

        Returns
        -------
        int
            Number of topics of a trained model.

        """
        pass

    @property
    def num_words(self) -> int:
        """
        Get the number of words that compose the vocabulary of the training
        dataset.

        Returns
        -------
        int
            Size of the vocabulary.

        """
        return self.tokenizer.num_knowledge_base_entities

    @property
    def perplexity(self) -> float:
        """
        Override it to return perplexity.

        Returns
        -------
        float
            Perplexity value.

        """
        raise NotImplementedError()

    def corpus_to_dataframe(self, corpus: ArticleCorpus,
                            num_words: int = 10,
                            num_topics: int = 3, lang: Optional[str] = None,
                            lang_topics: Optional[str] = None,
                            lang_corpus: Optional[str] = None, **kwargs):
        """
        Processes an ArticleCorpus and presents the results as a DataFrame.

        Parameters
        ----------
        corpus : ArticleCorpus
            A corpus of news articles.
        num_words : int
            Number of top words per topic to be added to the DataFrame.
        num_topics : int
            Number of top topics to be added to the DataFrame.
        lang : str, optional
            Used to set both 'lang_topics' and 'lang_corpus'.
        lang_topics : str, optional
            Language to convert the top words.
            If none, it uses 'lang'. If 'lang' is none, returns Wikidata.
        lang_corpus: str, optional
            Language to use to tokenize a corpus.
            If none, it uses 'lang'. If 'lang' is none, it uses english.
        kwargs : Any
            Key word arguments propagated to
             :meth:`~topic_inference.utils.topic_words`.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the article identification and top topics.

        """
        format_fn = partial(topic_words, **kwargs)
        lang_topics = lang_topics or lang
        lang_corpus = lang_corpus or lang or 'en'
        to_translate = lang_topics
        articles = corpus.articles.copy()

        # get scores
        tokens = self.batch_preprocess(
            [o.to_text(pattern=(5 * "{0} " + "{1}"))
             for o in corpus], lang=lang_corpus)
        pred, probs = self.transform(tokens, lang=lang_corpus)

        topics = self.topics(num_words=num_words)

        # update corpus with top topics
        for a_idx in range(len(articles)):
            top_topics = list(
                mathdict(float, probs[a_idx]).head(num_topics))
            for t_idx, (topic_id, topic_score) in enumerate(top_topics):
                topic = topics[topic_id]
                article_update_dict = {
                    f"topic_{t_idx + 1}": f"{topic_id} ({topic_score:.3f})",
                    f"topic_words_{t_idx + 1}": format_fn(topic)
                }

                if to_translate:
                    article_update_dict[f"topic_translated_{t_idx + 1}"] = \
                        format_fn(self.tokenizer.translate_topic(topic,
                                                                 lang_topics))

                articles[a_idx] = articles[a_idx].update(
                    **article_update_dict)

        df = ArticleCorpus(articles=articles).to_dataframe()

        # generate relevant columns and then pick the existing
        topic_columns = flatten([f"topic_{j + 1}", f"topic_words_{j + 1}",
                                 f"topic_translated_{j + 1}"]
                                for j in range(num_topics))
        topic_columns = [o for o in topic_columns if o in df.columns]
        df = df[['ref', 'title'] + topic_columns]
        df.columns = [o.replace('_', ' ').title() for o in df.columns]
        return df

    def corpus_to_excel(self, corpus: ArticleCorpus, path: PathType,
                        num_words: int = 10, num_topics: int = 3,
                        weights: bool = True, sep: str = ' | ',
                        lang: Optional[str] = None, **kwargs):
        """
        Processes an ArticleCorpus, presents the results as a DataFrame and
        saves the results in an excel file.

        Parameters
        ----------
        corpus : ArticleCorpus
            A corpus of news articles.
        path : PathType
            Path to save the instance.
        num_words : int
            Number of top words per topic to be added to the dataframe.
        num_topics : int
            Number of top topics to be added to the dataframe.
        weights : bool
            If True, saves the top words weights.
        sep : str
            Pattern to separate top words.
        lang : str, optional
            Language to convert the top words.
            If none, Wikidata IDs are returned.
        kwargs : Any
            Key word arguments propagated to
            :meth:`~topic_inference.utils.topic_words`.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the article identification and top topics.

        """
        df = self.corpus_to_dataframe(corpus, num_words=num_words,
                                      weights=weights,
                                      num_topics=num_topics, sep=sep,
                                      lang=lang,
                                      **kwargs)
        options = {'strings_to_formulas': False, 'strings_to_urls': False}
        writer = pd.ExcelWriter(path, engine='xlsxwriter', options=options)
        df.to_excel(writer, index=False)
        writer.save()

    def corpus_to_markdown(self, corpus: ArticleCorpus,
                           path: Optional[PathType] = None,
                           num_words: int = 10,
                           num_topics: int = 3, weights: bool = True,
                           sep: str = ' <br> ', lang: Optional[str] = None,
                           lang_topics: Optional[str] = None,
                           lang_corpus: Optional[str] = None,
                           verbose: bool = True, **kwargs):
        """
        Processes an ArticleCorpus, presents the results as a DataFrame and
        saves the results in a markdown file.

        Parameters
        ----------
        corpus : ArticleCorpus
            A corpus of news articles.
        path : PathType, optional
            Path to save the instance.
        num_words : int
            Number of top words per topic to be added to the dataframe.
        num_topics : int
            Number of top topics to be added to the dataframe.
        weights : bool
            If True, saves the top words weights.
        sep : str
            Pattern to separate top words.
        lang : str, optional
            Language to convert the top words.
            If none, Wikidata IDs are returned.
        lang_topics : str, optional
            Language used to represent topics.
            If none, it returns IDs.
        lang_corpus : str, optional
            Language to use to tokenize a corpus.
            If none, it uses 'lang'. If 'lang' is also none, it uses english.
        verbose : bool
            If True, the results DataFrame is printed to console in markdown
            format.
        kwargs : Any
            Key word arguments propagated to
            :meth:`~topic_inference.utils.topic_words`.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the article identification and top topics.

        """
        df = self.corpus_to_dataframe(corpus, num_words=num_words,
                                      weights=weights,
                                      num_topics=num_topics, sep=sep,
                                      lang=lang,
                                      lang_topics=lang_topics,
                                      lang_corpus=lang_corpus,
                                      **kwargs)
        dataframe_to_markdown(df, path=path, verbose=verbose)

    def to_dataframe(self, num_words: int = 10,
                     lang: Optional[str] = None,
                     min_score: Optional[float] = MINSCORE) -> pd.DataFrame:
        """
        Lists the top words of each topic as a DataFrame.

        Parameters
        ----------
        num_words : int
            Number of top words per topic to be added to the DataFrame.
        lang : str, optional
            Language to convert the top words.
            If None, Wikidata IDs are returned.
        min_score : float, optional
            Minimum percentage to consider a word to be part of a topic.
            If None, all words are part of the topic.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the top words per topic.

        """

        topics = self.topics(num_words)
        output = []
        for topic_id, topic in tqdm(topics.items(),
                                    desc="Building DataFrame"):
            row = {'Topic No.': topic_id,
                   'Topic Label': self.topic_label(topic_id)}
            entities = topic['words']
            words = self.tokenizer.get_labels(entities, lang=lang or 'en')
            scores = topic['scores']
            for i, (e, w, s) in enumerate(zip(entities, words, scores)):
                if not min_score or float(s) > min_score:
                    row.update({f'Id {i + 1}': e, f'Word {i + 1}': w,
                                f'Score {i + 1}': s})
            output.append(row)

        return pd.DataFrame(output)

    @classmethod
    @abstractmethod
    def _from_excel(cls, tokenizer: Tokenizer, topics_list: List[mathdict],
                    topics_labels: List[str],
                    topics_embed: Optional[np.array],
                    lang: str = 'en',
                    token_filter: Optional[TokenFilter] = None, **kwargs):
        pass

    @classmethod
    def from_excel(cls, path: PathType, tokenizer: Tokenizer,
                   lang: str = 'en',
                   token_filter: Optional[TokenFilter] = None,
                   max_column: int = NUMWORDS, min_score: float = MINSCORE,
                   min_words_per_topic: int = 1, min_word_count: int = 1,
                   **kwargs):

        meta, df = read_topics_excel(path, max_column=max_column)
        logger.info("Loaded topics from xlsx as a DataFrame")
        meta = [dict(o) for _, o in meta.iterrows()]

        # read the tokens (either ids or words) and scores
        uses_ids = 'Id 1' in df.columns

        if not uses_ids:
            msg = "Excel file must have IDs"
            logger.error(msg)
            raise NotImplementedError(msg)

        tokens = df[[o for o in df.columns if 'Id' in o]].astype(str)
        scores = df[[o for o in df.columns if 'Score' in o]]
        embeddings = df[[o for o in df.columns if 'Embedding' in o]]

        num_topics = tokens.shape[0]
        # sanity check
        if tokens.shape[1] != scores.shape[1]:
            msg = "Number of tokens and scores should be the same."
            logger.error(msg)
            raise ValueError(msg)

        logger.info(
            "Converting all material topics into a list of mathdicts")
        # fill the matrix when both word and material score exist
        word_count = mathdict(int, {})

        for i in tqdm(range(num_topics), desc="Finding vocabulary"):
            word_score_pairs = zip(tokens.iloc[i, :], scores.iloc[i, :])
            word_score_pairs = mathdict(float, word_score_pairs).gt(
                min_score)

            if len(word_score_pairs) < min_words_per_topic:
                continue

            for w in word_score_pairs.keys():
                word_count[w] += 1

        topics_list = []
        topics_labels = []
        topics_embed = []

        for i in tqdm(range(num_topics), desc="Adding topics"):
            word_score_pairs = zip(tokens.iloc[i, :], scores.iloc[i, :])
            word_score_pairs = mathdict(float, word_score_pairs).gt(
                min_score)

            if len(word_score_pairs) < min_words_per_topic:
                continue

            topics_list.append(word_score_pairs)
            topics_labels.append(meta[i].get('Topic Label', ''))
            topics_embed.append(embeddings.iloc[i, :])

        if len(topics_embed):
            topics_embed = np.array(topics_embed)
        else:
            topics_embed = None

        return cls._from_excel(tokenizer=tokenizer, topics_list=topics_list,
                               topics_labels=topics_labels,
                               topics_embed=topics_embed,
                               lang=lang, token_filter=token_filter,
                               **kwargs)

    def to_excel(self, path: Optional[PathType], num_words: int = 10,
                 lang: Optional[str] = None,
                 min_score: Optional[float] = MINSCORE) -> pd.DataFrame:
        """
        Saves the top words of each topic as an excel file.

        Parameters
        ----------
        path: PathType, optional
            Path to save the topics' top words in excel format.
            If None, a dataframe is returned instead of saved to file.
        num_words : int
            Number of top words per topic to be added to the DataFrame.
        lang : str, optional
            Language to convert the top words.
            If None, Wikidata IDs are returned.
        min_score : float, optional
            Minimum percentage to consider a word to be part of a topic.
            If None, all words are part of the topic.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the top words per topic.

        """
        df = self.to_dataframe(num_words=num_words, lang=lang,
                               min_score=min_score)
        if path:
            options = {'strings_to_formulas': False,
                       'strings_to_urls': False}
            writer = pd.ExcelWriter(path.with_suffix('.xlsx'),
                                    engine='xlsxwriter', options=options)
            df.to_excel(writer, index=False)
            writer.save()

        return df

    def to_json(self, path: Optional[PathType] = None, num_words: int = 500,
                lang: Optional[str] = None,
                min_score: Optional[float] = MINSCORE,
                edit_suffix: bool = True) -> str:
        """
        Dumps the top words of each topic as a json. If a `path` is defined then
        the json is saved to file, else it is returned.

        Parameters
        ----------
        path: PathType, optional
            Path to save the topics' top words in json format.
            If None, the dump is returned instead of saved to file.
        num_words : int
            Number of top words per topic to be added to output.
        lang : str, optional
            Language to convert the top words.
            If None, Wikidata IDs are returned.
        min_score : float, optional
            Minimum percentage to consider a word to be part of a topic.
            If None, all words are part of the topic.
        edit_suffix: bool
            If True, the path is edited to include the number of topics and
             topic words and language.

        Returns
        -------
        str
            A json dump of the top words of each topic.

        """
        df = self.to_dataframe(num_words=num_words, lang=lang,
                               min_score=min_score)

        to_translate = bool(lang)

        def translate_topic(topic):
            return self.tokenizer.get_labels(topic, lang=lang)

        f = translate_topic if to_translate else lambda x: x

        def process_topic(topic):
            return {
                "TopicID": str(topic['Topic No.']).zfill(6),
                "FeatureWords":
                    f([topic[f'Id {i}'] for i in range(1, num_words + 1)])
            }

        topics = [process_topic(topic) for _, topic in
                  tqdm(df.iterrows(), total=self.num_topics,
                       desc="Processing topics")]
        json_ = json.dumps(topics)

        if path:
            if edit_suffix:
                suffix = '_'.join(['topics', str(len(topics)), 'words',
                                   str(num_words), str(lang or 'wikidata')])
                path = Path(path).with_suffix('.' + suffix + '.json')
            with open(path, 'w') as f:
                f.write(json_)

        return json_

    def to_markdown(self, path: Optional[PathType] = None,
                    num_words: int = 10, sep: str = ' <br> ',
                    weights: bool = True, verbose: bool = True,
                    lang: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Saves the top words of each topic as a markdown file.

        Parameters
        ----------
        path: PathType
            Path to save the topics' top words in markdown format.
        num_words : int
            Number of top words per topic to be added to output.
        sep : str
            Separator to use between top words.
        weights : bool
            If True, weights are added to top words inside parenthesis.
        verbose : bool
            If True, the topics DataFrame is printed to console in markdown
             format.
        lang : str, optional
            Optional parameter to add a column is added with the canonical label
             of the Wikidata item in the target language.
        kwargs : Any
            Key word arguments propagated to
             :meth:`~topic_inference.utils.topic_words`.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the top words per topic.

        """
        if not (path or verbose):
            raise ValueError(
                "Either `path` or `verbose` must have a value.")

        format_fn = partial(topic_words, sep=sep, weights=weights, **kwargs)
        to_translate = bool(lang)

        topics = self.topics(num_words)
        output = [{'Topic No.': i, 'Words': format_fn(topic)}
                  for i, topic in topics.items()]

        if to_translate:
            for t_idx in range(len(topics)):
                t = self.tokenizer.translate_topic(topics[t_idx], lang)
                output[t_idx]['Translated'] = format_fn(t)

        df = pd.DataFrame(output)
        dataframe_to_markdown(df, path=path, verbose=verbose)

        return df

    @abstractmethod
    def save(self, path: PathType, *args, **kwargs):
        """
        Abstract method. Override it to save a trained instance.

        Parameters
        ----------
        path : PathType
            Path to save the instance.
        args : Any
            Not used, kept for subclassing.
        kwargs : Any
            Not used, kept for subclassing.

        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path, *args, **kwargs) -> Any:
        """
        Abstract class method. Override it to load a trained instance.

        Parameters
        ----------
        path : PathType
            Path to load the instance.
        args : Any
            Not used, kept for subclassing.
        kwargs : Any
            Not used, kept for subclassing.

        Returns
        -------
        Any
            Returns a subclass instance.

        """
        pass
