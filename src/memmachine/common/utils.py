"""
Common utility functions.
"""

import asyncio
import functools
from collections.abc import Awaitable, Mapping
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


def extract_metrics_labels_from_isolations(
    isolations: Mapping[str, bool | int | float | str],
    default_user_id: str = "",
) -> dict[str, str]:
    """Extract individual metrics labels from an isolations dictionary.

    For metrics tracking, we need single user_id and agent_id values, not sets.
    This function extracts the appropriate IDs for attributing LLM token usage.

    Args:
        isolations: A dictionary containing group_id, session_id,
                   producer, produced_for, and other isolation keys.
        default_user_id: The default user_id to use if not found in isolations.

    Returns:
        A dictionary with keys: user_id, agent_id, group_id, session_id.
        Each value is a single string identifier.

    Example:
        >>> isolations = {
        ...     "group_id": "team-a",
        ...     "session_id": "conv-123",
        ...     "producer": "user-1",
        ...     "produced_for": "agent-1"
        ... }
        >>> labels = extract_metrics_labels_from_isolations(isolations)
        >>> labels["user_id"]
        'user-1'
        >>> labels["agent_id"]
        'agent-1'
    """
    user_id = str(isolations.get("producer", default_user_id))
    agent_id = str(isolations.get("agent_id", ""))
    group_id = str(isolations.get("group_id", ""))
    session_id = str(isolations.get("session_id", ""))

    return {
        "user_id": user_id,
        "agent_id": agent_id,
        "group_id": group_id,
        "session_id": session_id,
    }
