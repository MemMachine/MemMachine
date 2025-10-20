"""
Authentication and token management for the MemMachine Extension Server.
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from ..core.database import get_db_session, get_db_transaction
from ..models import Token
from ..settings import settings

logger = logging.getLogger(__name__)


# Token management functions
def issue_token(subject: str, ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
    """
    Issue a new token for a user.

    Args:
        subject: The subject/username identifier for the token.
        ttl_seconds: Time-to-live in seconds. If None, uses default TTL.

    Returns:
        Dictionary with token, expires_at, and subject.
    """
    now = datetime.now()
    ttl = ttl_seconds if ttl_seconds is not None else settings.token_ttl_seconds
    expires_at = now + timedelta(seconds=ttl)
    token_str = secrets.token_urlsafe(32)

    try:
        with get_db_transaction() as db:
            token = Token(
                token=token_str,
                subject=subject,
                issued_at=now,
                expires_at=expires_at,
            )
            db.add(token)
            logger.debug(f"Issued token for subject: {subject}")
    except Exception as e:
        logger.error(f"Failed to issue token: {e}")
        raise

    return {
        "token": token_str,
        "expires_at": expires_at.isoformat(),
        "subject": subject,
    }


def validate_token(token: Optional[str]) -> bool:
    """
    Validate if a token exists and is not expired.

    Args:
        token: The token string to validate.

    Returns:
        True if token is valid and not expired, False otherwise.
    """
    if not token:
        return False

    try:
        with get_db_session() as db:
            token_obj = db.query(Token).filter(Token.token == token).first()

            if not token_obj:
                return False

            now = datetime.now()
            if token_obj.expires_at < now:
                # Token expired, but don't delete it here (handled by cron task)
                logger.debug("Token expired")
                return False

            return True
    except Exception as e:
        logger.error(f"Failed to validate token: {e}")
        return False


def revoke_token(token: Optional[str]) -> bool:
    """
    Revoke a token by removing it from the store.

    Args:
        token: The token string to revoke.

    Returns:
        True if token was found and removed, False otherwise.
    """
    if not token:
        return False

    try:
        with get_db_transaction() as db:
            token_obj = db.query(Token).filter(Token.token == token).first()
            if token_obj:
                db.delete(token_obj)
                logger.debug("Revoked token")
                return True
            return False
    except Exception as e:
        logger.error(f"Failed to revoke token: {e}")
        return False


def cleanup_expired_tokens() -> int:
    """
    Remove all expired tokens from the store.

    Returns:
        Number of tokens removed.
    """
    now = datetime.now()

    try:
        with get_db_transaction() as db:
            expired_tokens = db.query(Token).filter(Token.expires_at < now).all()
            removed = len(expired_tokens)

            for token in expired_tokens:
                db.delete(token)

            if removed:
                logger.info(f"Token cleanup removed {removed} expired tokens")
            return removed
    except Exception as e:
        logger.error(f"Failed to cleanup expired tokens: {e}")
        return 0


# User management functions moved to user.py

def get_token_info(token: str) -> Optional[Dict[str, Any]]:
    """
    Get token information including expiration time.

    Args:
        token: Token string.

    Returns:
        Dictionary with token and expiration info, or None if token is invalid.
    """
    try:
        with get_db_session() as db:
            token_obj = db.query(Token).filter(Token.token == token).first()
            if not token_obj:
                return None

            # Check if token is expired
            if token_obj.expires_at < datetime.now():
                # Token expired, but don't delete it here (handled by cron task)
                return None

            return {
                "token": token_obj.token,
                "expires_at": token_obj.expires_at.isoformat(),
                "subject": token_obj.subject,
            }
    except Exception as e:
        logger.error(f"Failed to get token info: {e}")
        return None
