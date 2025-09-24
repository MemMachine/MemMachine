"""
BM25-based reranker implementation.
"""

import re
from typing import Any

from nltk import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi

from .reranker import Reranker


class BM25Reranker(Reranker):
    """
    Reranker that uses the BM25 algorithm to score candidates
    based on their relevance to the query.
    """

    def __init__(self, config: dict[str, Any] = {}):
        """
        Initialize a BM25Reranker with the provided configuration.

        Args:
            config (dict[str, Any], optional):
                Configuration dictionary containing:
                - languages (str | list[str], optional):
                  Language(s) for stop words (default: "english").
                  Languages must be supported by NLTK stopwords corpus.
        """
        super().__init__()

        languages = config.get("languages", "english")
        try:
            self._stop_words = stopwords.words(languages)
        except Exception as e:
            raise ValueError(
                f"Unsupported language(s) provided: {languages}"
            ) from e

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        candidates_tokens = [
            self._preprocess_text(candidate) for candidate in candidates
        ]

        if not any(candidates_tokens):
            # There are no tokens in the corpus.
            return [0.0 for _ in candidates]

        # There is at least one token in the corpus.
        bm25 = BM25Okapi(candidates_tokens)

        scores = [
            float(score)
            for score in bm25.get_scores(self._preprocess_text(query))
        ]

        return scores

    def _preprocess_text(self, text: str) -> list[str]:
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
        words = word_tokenize(lower_text)
        tokens = [
            word for word in words if word and word not in self._stop_words
        ]
        return tokens
