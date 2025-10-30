"""
Common utility functions.
"""

import asyncio
import functools
from collections.abc import Awaitable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from memmachine.common.data_types import SessionData

if TYPE_CHECKING:
    from memmachine.episodic_memory.data_types import MemoryContext


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


def isolations_to_session_data(
    isolations: Mapping[str, bool | int | float | str] | None = None,
    default_user_id: str = "",
) -> SessionData:
    """Convert isolations to session data."""
    if isolations is None:
        return SessionData(
            group_id="",
            user_id=[default_user_id],
            session_id="",
            agent_id=[""],
        )

    return SessionData(
        group_id=str(isolations.get("group_id", "")),
        user_id=[str(isolations.get("producer", default_user_id))],
        session_id=str(isolations.get("session_id", "")),
        agent_id=[str(isolations.get("agent_id", ""))],
    )


def memory_context_to_session_data(
    memory_context: "MemoryContext",
) -> SessionData:
    """Convert memory context to session data."""
    from memmachine.episodic_memory.data_types import MemoryContext

    # Type check to ensure the parameter is a MemoryContext
    if not isinstance(memory_context, MemoryContext):
        raise TypeError(f"Expected MemoryContext, got {type(memory_context)}")

    return SessionData(
        group_id=memory_context.group_id,
        user_id=list(memory_context.user_id),
        session_id=memory_context.session_id,
        agent_id=list(memory_context.agent_id),
    )
