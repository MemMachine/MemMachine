"""
Common utility functions.
"""

import asyncio
import functools
from collections.abc import Awaitable
from contextlib import AbstractAsyncContextManager
from typing import Any


async def async_with(
    async_context_manager: AbstractAsyncContextManager,
    awaitable: Awaitable,
) -> Any:
    """
    Helper function to use an async context manager with an awaitable.

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


def async_locked(func):
    """
    Decorator to ensure that a coroutine function is executed with a lock.
    The lock is shared across all invocations of the decorated coroutine function.
    """
    lock = asyncio.Lock()

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        async with lock:
            return await func(*args, **kwargs)

    return wrapper


def get_default_headers(bearer_token: str) -> dict[str, str]:
            """
            Generate default headers for API requests based on configuration.
            
            Args:
                bearer_token: the bearer token to use for authorization.
                
            Returns:
                dict[str, str]: Dictionary of default headers
            """
            headers = {}
            if bearer_token:
                headers["Authorization"] = f"Bearer {bearer_token}"
            return headers