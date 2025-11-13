"""
BM25-based reranker implementation.
"""

import asyncio
import re
from collections.abc import Callable

from rank_bm25 import BM25Okapi

from ..configuration.reranker_conf import BM25RerankerConf
from .reranker import Reranker


def get_tokenizer(name: str, language: str) -> Callable[[str], list[str]]:
    if name == "default":
        from nltk import word_tokenize
        from nltk.corpus import stopwords

        stop_words = stopwords.words(language)

        def _default_tokenize(text: str) -> list[str]:
            """
            Preprocess the input text
            by removing non-alphanumeric characters,
            converting to lowercase,
            word-tokenizing,
            and removing stop words.

            Args:
                text (str): The input text to preprocess.

            Returns:
                list[str]: A list of tokens for use in BM25 scoring.
            """
            alphanumeric_text = re.sub(r"\W+", " ", text)
            lower_text = alphanumeric_text.lower()
            words = word_tokenize(lower_text, language)
            tokens = [word for word in words if word and word not in stop_words]
            return tokens

        return _default_tokenize
    elif name == "simple":
        return lambda text: re.sub(r"\W+", " ", text).lower().split()
    else:
        raise ValueError(f"Unknown tokenizer: {name}")


class BM25Reranker(Reranker):
    """
    Reranker that uses the BM25 algorithm to score candidates
    based on their relevance to the query.
    """

    def __init__(self, params: BM25RerankerConf):
        """
        Initialize a BM25Reranker with the provided parameters.

        Args:
            params (BM25RerankerParams):
                Parameters for the BM25Reranker.
        """
        super().__init__()

        self._k1 = params.k1
        self._b = params.b
        self._epsilon = params.epsilon
        self._tokenize = get_tokenizer(params.tokenize, params.language)

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        tokenized_query_future = asyncio.to_thread(self._tokenize, query)
        tokenized_candidates_future = asyncio.to_thread(
            self._tokenize_multiple, candidates
        )

        tokenized_query = await tokenized_query_future
        tokenized_candidates = await tokenized_candidates_future

        if not any(tokenized_candidates):
            # There are no tokens in the corpus.
            return [0.0 for _ in candidates]

        # There is at least one token in the corpus.
        bm25 = BM25Okapi(
            tokenized_candidates,
            k1=self._k1,
            b=self._b,
            epsilon=self._epsilon,
        )

        scores = [float(score) for score in bm25.get_scores(tokenized_query)]

        return scores

    def _tokenize_multiple(self, corpus: list[str]) -> list[list[str]]:
        return [self._tokenize(document) for document in corpus]
