"""Common utility functions."""

import asyncio
import functools
import math
import re
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AbstractAsyncContextManager
from typing import Any, ParamSpec, TypeVar

import numpy as np
from nltk import sent_tokenize

from .data_types import SimilarityMetric

T = TypeVar("T")
P = ParamSpec("P")


async def async_with[T](
    async_context_manager: AbstractAsyncContextManager,
    awaitable: Awaitable[T],
) -> T:
    """
    Use an async context manager while awaiting a coroutine.

    Args:
        async_context_manager (AbstractAsyncContextManager):
            The async context manager to use.
        awaitable (Awaitable):
            The awaitable to execute within the context.

    Returns:
        Any:
            The result of the awaitable.

    """
    async with async_context_manager:
        return await awaitable


def async_locked[**P, T](func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """
    Ensure that a coroutine function is executed with a shared lock.

    The lock is shared across all invocations of the decorated coroutine function.
    """
    lock = asyncio.Lock()

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        async with lock:
            return await func(*args, **kwargs)

    return wrapper


def chunk_text(text: str, max_length: int) -> list[str]:
    """
    Chunk text into partitions not exceeding max_length.

    Args:
        text (str): The input text to chunk.
        max_length (int): The maximum length of each chunk.

    Returns:
        list[str]: A list of text chunks.

    """
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")

    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


def chunk_text_balanced(text: str, max_length: int) -> list[str]:
    """
    Chunk text into balanced partitions not exceeding max_length.

    Args:
        text (str): The input text to chunk.
        max_length (int): The maximum length of each chunk.

    Returns:
        list[str]: A list of text chunks.

    """
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")

    if len(text) == 0:
        return []

    num_chunks = math.ceil(len(text) / max_length)
    chunk_size = math.ceil(len(text) / num_chunks)

    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def extract_sentences(text: str) -> set[str]:
    """
    Extract unique sentences from the input text.

    Args:
        text (str): The input text from which to extract sentences.

    Returns:
        set[str]: A set of unique sentences.

    """
    partitions = {
        partition
        for line in text.strip().splitlines()
        for partition in sent_tokenize(line.strip())
    }

    sentences = {
        sentence
        for partition in partitions
        for sentence in re.findall(
            r".*?(?:[?!\.\uff1f\uff01\u3002]*[?!\uff1f\uff01\u3002][?!\.\uff1f\uff01\u3002]*)+|.+$",
            partition,
        )
        if any(c.isalnum() for c in sentence)
    }

    return sentences


def unflatten_like[T](
    flat_list: list[T],
    template_list: list[list[Any]],
) -> list[list[T]]:
    """
    Unflatten a flat list into a nested list structure based on a template.

    Args:
        flat_list (list): The flat list to unflatten.
        template_list (list): The template nested list structure.

    Returns:
        list: The unflattened nested list.

    """
    if not all(isinstance(template, list) for template in template_list):
        raise TypeError("All elements in template_list must be lists.")

    unflattened_list = []
    current_index = 0

    for template in template_list:
        unflattened_list.append(
            flat_list[current_index : current_index + len(template)]
        )
        current_index += len(template)

    if current_index != len(flat_list):
        raise ValueError("flat_list cannot be unflattened to match template_list.")

    return unflattened_list


def cluster_texts(
    texts: Iterable[str],
    max_num_texts_per_cluster: int,
    max_total_length_per_cluster: int,
) -> list[list[str]]:
    """
    Cluster texts based on maximum number of texts and total length of texts per cluster.

    Args:
        texts (Iterable[str]): The input texts to cluster.
        max_num_texts_per_cluster (int): The maximum number of texts per cluster.
        max_total_length_per_cluster (int): The maximum total length of texts per cluster.

    Returns:
        list[list[str]]: A list of text clusters.

    """
    if max_num_texts_per_cluster <= 0:
        raise ValueError("max_num_texts_per_cluster must be greater than 0")
    if max_total_length_per_cluster <= 0:
        raise ValueError("max_total_length_per_cluster must be greater than 0")

    clusters: list[list[str]] = []
    current_cluster: list[str] = []
    current_length = 0

    for text in texts:
        text_length = len(text)
        if text_length > max_total_length_per_cluster:
            raise ValueError(
                f"Text length {text_length} exceeds max_total_length_per_cluster {max_total_length_per_cluster}"
            )

        if (
            len(current_cluster) >= max_num_texts_per_cluster
            or current_length + text_length > max_total_length_per_cluster
        ):
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [text]
            current_length = text_length
        else:
            current_cluster.append(text)
            current_length += text_length

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def compute_similarity(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    similarity_metric: SimilarityMetric | None = None,
) -> list[float]:
    """
    Compute similarity scores between a query embedding and candidate embeddings.

    Args:
        query_embedding (list[float]): The embedding of the query.
        candidate_embeddings (list[list[float]]): A list of candidate embeddings to compare against.
        similarity_metric (SimilarityMetric | None): The similarity metric to use (default: None).

    Returns:
        list[float]: A list of similarity scores for each candidate embedding.

    """
    if not candidate_embeddings:
        return []

    query_embedding_np = np.array(query_embedding)
    candidate_embeddings_np = np.array(candidate_embeddings)

    match similarity_metric:
        case SimilarityMetric.COSINE:
            magnitude_products = np.linalg.norm(
                candidate_embeddings_np,
                axis=-1,
            ) * np.linalg.norm(query_embedding_np)
            magnitude_products[magnitude_products == 0] = float("inf")

            scores = (
                np.dot(candidate_embeddings_np, query_embedding_np) / magnitude_products
            )
        case SimilarityMetric.DOT:
            scores = np.dot(candidate_embeddings_np, query_embedding_np)
        case SimilarityMetric.EUCLIDEAN:
            scores = np.linalg.norm(
                candidate_embeddings_np - query_embedding_np,
                axis=-1,
            )
        case SimilarityMetric.MANHATTAN:
            scores = np.sum(
                np.abs(candidate_embeddings_np - query_embedding_np),
                axis=-1,
            )
        case _:
            # Default to cosine similarity.
            magnitude_products = np.linalg.norm(
                candidate_embeddings_np,
                axis=-1,
            ) * np.linalg.norm(query_embedding_np)
            magnitude_products[magnitude_products == 0] = float("inf")

            scores = (
                np.dot(candidate_embeddings_np, query_embedding_np) / magnitude_products
            )

    return scores.astype(float).tolist()


def similarity_gt(
    similarity_metric: SimilarityMetric | None = None,
) -> Callable[[float, float], bool]:
    """
    Return a comparator for similarity scores.

    The returned function `gt(a, b)` returns True when score `a`
    represents greater similarity than score `b` under the given metric.

    For COSINE and DOT, higher scores mean greater similarity (`a > b`).
    For EUCLIDEAN and MANHATTAN, lower scores mean greater similarity (`a < b`).

    Args:
        similarity_metric (SimilarityMetric | None):
            The similarity metric.
            If None, defaults to greater-than comparison (`a > b`)
            (default: None).

    Returns:
        Callable[[float, float], bool]: A comparator for similarity scores.

    """
    import operator

    match similarity_metric:
        case SimilarityMetric.EUCLIDEAN | SimilarityMetric.MANHATTAN:
            return operator.lt
        case _:
            return operator.gt
