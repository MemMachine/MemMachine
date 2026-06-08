"""
Ambient per-request context.

Holds request-scoped values that deep, generic code needs
without threading them through every call signature.
"""

from contextvars import ContextVar, Token

from babel import Locale

# Client locale assumed when a request carries none (or an unrecognized one).
DEFAULT_LOCALE = Locale("en", "US")

_request_locale: ContextVar[Locale] = ContextVar(
    "request_locale", default=DEFAULT_LOCALE
)


def get_request_locale() -> Locale:
    """Return the current request's locale."""
    return _request_locale.get()


def set_request_locale(locale: Locale) -> Token[Locale]:
    """Set the current request's locale; returns a token for `reset_request_locale`."""
    return _request_locale.set(locale)


def reset_request_locale(token: Token[Locale]) -> None:
    """Restore the locale to its value before the matching `set_request_locale`."""
    _request_locale.reset(token)
