"""REST API exception classes."""

import logging
import traceback

from fastapi.exceptions import HTTPException, RequestValidationError

from memmachine.common.api.spec import InvalidNameError, RestErrorModel
from memmachine.common.errors import (
    InvalidArgumentError,
    ResourceNotReadyError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)

logger = logging.getLogger(__name__)


class RestError(HTTPException):
    """
    Exception with a structured RestErrorModel as the 'detail'.

    Inherits from HTTPException, which dynamically resolves to:
    - FastAPI's HTTPException in server environments (when FastAPI is available)
    - A lightweight fallback Exception in client-only environments (when FastAPI is not installed)

    This design allows RestError to work in both server and client contexts without
    requiring FastAPI as a dependency for client packages. In server environments,
    RestError behaves as a standard FastAPI HTTPException and can be raised in
    FastAPI route handlers. In client environments, it provides the same interface
    but without the FastAPI dependency overhead.
    """

    def __init__(
        self,
        code: int,
        message: str,
        ex: Exception | None = None,
    ) -> None:
        """Initialize RestError with structured error details."""
        self.payload: RestErrorModel | None = None
        if ex is not None:
            if isinstance(ex, RequestValidationError):
                trace = ""
                message = self.format_validation_error_message(ex)
            elif self.is_known_error(ex):
                trace = ""
            else:
                trace = "".join(
                    traceback.format_exception(
                        type(ex),
                        ex,
                        ex.__traceback__,
                    )
                ).strip()

            self.payload = RestErrorModel(
                code=code,
                message=message,
                exception=type(ex).__name__,
                internal_error=str(ex),
                trace=trace,
            )

        # Call HTTPException with structured detail
        if self.payload is not None:
            logger.warning(
                "exception handling request, code %d, message: %s, payload: %s",
                code,
                message,
                self.payload,
            )
            super().__init__(status_code=code, detail=self.payload.model_dump())
        else:
            logger.info("error handling request, code %d, message: %s", code, message)
            super().__init__(status_code=code, detail=message)

    @staticmethod
    def is_known_error(ex: Exception) -> bool:
        known_errors = [
            SessionAlreadyExistsError,
            SessionNotFoundError,
            InvalidNameError,
            InvalidArgumentError,
            ResourceNotReadyError,
        ]
        return any(isinstance(ex, err) for err in known_errors)

    @staticmethod
    def format_validation_error_message(exc: RequestValidationError) -> str:
        parts: list[str] = []

        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
            msg = err.get("msg", "Invalid value")

            if loc:
                parts.append(f"{loc}: {msg}")
            else:
                parts.append(msg)

        if not parts:
            return "Invalid request payload"

        if len(parts) == 1:
            return f"Invalid request payload: {parts[0]}"

        return "Invalid request payload:\n- " + "\n- ".join(parts)
