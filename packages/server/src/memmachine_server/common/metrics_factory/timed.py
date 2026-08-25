"""A decorator that times one store operation.

Every store that owns an OperationTracker wants the same wrapper, so it lives
here beside the tracker rather than being copied into each of them.
"""

import functools
from collections.abc import Awaitable, Callable
from typing import Concatenate, ParamSpec, Protocol, TypeVar

from .operation_tracker import OperationTracker

_P = ParamSpec("_P")
_R = TypeVar("_R")


class HasTracker(Protocol):
    """What timed() needs of the instance whose method it decorates.

    Binding the type variable to this rather than leaving it free is what lets
    a type checker see that self._tracker exists; an unbound variable cannot
    carry the attribute.
    """

    _tracker: OperationTracker


# The decorated callables are always methods; _S is the bound instance.
_S = TypeVar("_S", bound=HasTracker)


def timed(
    operation: str,
) -> Callable[
    [Callable[Concatenate[_S, _P], Awaitable[_R]]],
    Callable[Concatenate[_S, _P], Awaitable[_R]],
]:
    """Record how long one store operation takes.

    Applied as a decorator so the method bodies are untouched: instrumentation
    must not reshape the code it measures.
    """

    def decorator(
        func: Callable[Concatenate[_S, _P], Awaitable[_R]],
    ) -> Callable[Concatenate[_S, _P], Awaitable[_R]]:
        @functools.wraps(func)
        async def wrapper(self: _S, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            async with self._tracker(operation):
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator
