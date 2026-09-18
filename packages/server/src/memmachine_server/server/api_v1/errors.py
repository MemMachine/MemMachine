"""The v1 error contract: one envelope, a closed set of codes.

Every v1 response that is not a success carries
`{"error": {"code", "message"}}`, and `code` is one of `ErrorCode`. A
failure with no code of its own is `internal`, whose traceback is logged
and never answered with.
"""

import logging
from collections.abc import Callable, Coroutine
from typing import Any, Literal, override

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

from memmachine_server.episodic_memory.event_memory.event_memory_store import (
    EventMemoryStoreEventAlreadyStoredError,
)

from .tenant_event_memories import ComponentNotEnabledError, TenantNotFoundError

logger = logging.getLogger(__name__)

ErrorCode = Literal[
    "tenant_not_found",
    "component_not_enabled",
    "event_exists",
    "invalid_request",
    "internal",
]
"""Every code a v1 route answers with."""


class ErrorBody(BaseModel):
    """What went wrong, under a code a client can branch on."""

    code: ErrorCode = Field(description="The kind of failure")
    message: str = Field(description="What failed, in words")


class ErrorResponse(BaseModel):
    """The body of every v1 response that is not a success."""

    error: ErrorBody = Field(description="What went wrong")


class ErrorEnvelopeRoute(APIRoute):
    """A route that answers every failure with the v1 error envelope.

    The mapping lives here rather than in an application-wide handler, so
    it holds for the v1 routes alone and no other router's errors change
    shape.
    """

    @override
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        # The handler the superclass builds validates the request before it
        # calls the endpoint, so a validation failure is raised inside it.
        handle = super().get_route_handler()

        async def answer(request: Request) -> Response:
            try:
                return await handle(request)
            except TenantNotFoundError as error:
                return _envelope(404, "tenant_not_found", str(error))
            except ComponentNotEnabledError as error:
                return _envelope(404, "component_not_enabled", str(error))
            except EventMemoryStoreEventAlreadyStoredError as error:
                # The batch named events the tenant already holds and was
                # rejected whole; the message names them.
                return _envelope(409, "event_exists", str(error))
            except RequestValidationError as error:
                return _envelope(422, "invalid_request", _validation_message(error))
            except (ValueError, LookupError) as error:
                return _envelope(422, "invalid_request", str(error))
            except Exception:
                logger.exception(
                    "Unhandled error answering %s %s",
                    request.method,
                    request.url.path,
                )
                return _envelope(500, "internal", "Internal server error")

        return answer


def _envelope(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
    """The error response carrying `code` and `message`."""
    body = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _validation_message(error: RequestValidationError) -> str:
    """Each failing location and what it wants, joined into one line."""
    failures: list[str] = []
    for failure in error.errors():
        location = ".".join(
            str(part) for part in failure.get("loc", ()) if part != "body"
        )
        reason = str(failure.get("msg", "invalid value"))
        failures.append(f"{location}: {reason}" if location else reason)
    return "; ".join(failures) if failures else "Invalid request"
