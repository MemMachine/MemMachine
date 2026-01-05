"""Common utility functions."""

import asyncio
import functools
import math
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AbstractAsyncContextManager
from typing import Any, ParamSpec, TypeVar

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
