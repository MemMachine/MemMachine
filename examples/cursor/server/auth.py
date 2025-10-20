"""
Authentication and token management for the Cursor MCP Server.
"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, Optional

from fastapi.security import HTTPBearer

from .config import TOKEN_TTL_SECONDS
from .database import get_db_transaction, init_db
from .models import Token, User

# Security scheme
security = HTTPBearer()


# Password hashing utilities
def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256 with a salt.

    Args:
        password: Plain text password

    Returns:
        Hashed password in format: salt$hash
    """
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${pwd_hash}"


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hashed password.

    Args:
        password: Plain text password to verify
        hashed_password: Hashed password in format: salt$hash

    Returns:
        True if password matches, False otherwise
    """
    try:
        salt, pwd_hash = hashed_password.split("$")
        computed_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        return computed_hash == pwd_hash
    except Exception:
        return False


class InMemoryTokenStore:
    """Simple in-memory token store with expiration."""

    def __init__(self, default_ttl_seconds: int = TOKEN_TTL_SECONDS):
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._default_ttl_seconds = default_ttl_seconds
        self.logger = logging.getLogger(f"{__name__}.InMemoryTokenStore")

    def issue_token(
        self, subject: str, ttl_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        now = datetime.now()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        expires_at = now + timedelta(seconds=ttl)
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._tokens[token] = {
                "subject": subject,
                "issued_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
            }
        return {
            "token": token,
            "expires_at": expires_at.isoformat(),
            "subject": subject,
        }

    def validate_token(self, token: Optional[str]) -> bool:
        if not token:
            return False
        with self._lock:
            data = self._tokens.get(token)
            if not data:
                return False
            try:
                if datetime.fromisoformat(data["expires_at"]) < datetime.now():
                    # expired, remove
                    self._tokens.pop(token, None)
                    return False
            except Exception:
                # if parsing fails, revoke token defensively
                self._tokens.pop(token, None)
                return False
            return True

    def revoke_token(self, token: Optional[str]) -> bool:
        if not token:
            return False
        with self._lock:
            return self._tokens.pop(token, None) is not None

    def get_token_subject(self, token: Optional[str]) -> Optional[str]:
        """Get the subject (username) associated with a token."""
        if not token:
            return None
        with self._lock:
            data = self._tokens.get(token)
            if not data:
                return None
            try:
                # Check if token is still valid (not expired)
                if datetime.fromisoformat(data["expires_at"]) < datetime.now():
                    # expired, remove
                    self._tokens.pop(token, None)
                    return None
                return data.get("subject")
            except Exception:
                # if parsing fails, revoke token defensively
                self._tokens.pop(token, None)
                return None

    def cleanup_expired(self) -> int:
        removed = 0
        now = datetime.now()
        with self._lock:
            to_delete = [
                t
                for t, d in self._tokens.items()
                if datetime.fromisoformat(d.get("expires_at", "1970-01-01T00:00:00"))
                < now
            ]
            for t in to_delete:
                self._tokens.pop(t, None)
                removed += 1
        if removed:
            self.logger.info(f"Token cleanup removed {removed} expired tokens")
        return removed


class TokenStore:
    """SQLAlchemy-based token store with expiration."""

    def __init__(self, default_ttl_seconds: int = TOKEN_TTL_SECONDS):
        """
        Initialize the SQLAlchemy token store.

        Args:
            default_ttl_seconds: Default time-to-live for tokens in seconds.
        """
        self._default_ttl_seconds = default_ttl_seconds
        self._lock = Lock()
        self.logger = logging.getLogger(f"{__name__}.TokenStore")


    def issue_token(
        self, subject: str, ttl_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Issue a new token for a user.

        Args:
            subject: The subject/username identifier for the token.
            ttl_seconds: Time-to-live in seconds. If None, uses default TTL.

        Returns:
            Dictionary with token, expires_at, and subject.
        """
        now = datetime.now()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        expires_at = now + timedelta(seconds=ttl)
        token_str = secrets.token_urlsafe(32)

        try:
            with self._lock, get_db_transaction() as db:
                token = Token(
                    token=token_str,
                    subject=subject,
                    issued_at=now,
                    expires_at=expires_at,
                )
                db.add(token)
                self.logger.debug(f"Issued token for subject: {subject}")
        except Exception as e:
            self.logger.error(f"Failed to issue token: {e}")
            raise

        return {
            "token": token_str,
            "expires_at": expires_at.isoformat(),
            "subject": subject,
        }

    def validate_token(self, token: Optional[str]) -> bool:
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
            with self._lock, get_db_transaction() as db:
                token_obj = db.query(Token).filter(Token.token == token).first()

                if not token_obj:
                    return False

                now = datetime.now()
                if token_obj.expires_at < now:
                    # Token expired, remove it
                    db.delete(token_obj)
                    self.logger.debug("Removed expired token")
                    return False

                return True
        except Exception as e:
            self.logger.error(f"Failed to validate token: {e}")
            return False

    def revoke_token(self, token: Optional[str]) -> bool:
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
            with self._lock, get_db_transaction() as db:
                token_obj = db.query(Token).filter(Token.token == token).first()
                if token_obj:
                    db.delete(token_obj)
                    self.logger.debug("Revoked token")
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Failed to revoke token: {e}")
            return False

    def cleanup_expired(self) -> int:
        """
        Remove all expired tokens from the store.

        Returns:
            Number of tokens removed.
        """
        now = datetime.now()

        try:
            with self._lock, get_db_transaction() as db:
                expired_tokens = db.query(Token).filter(Token.expires_at < now).all()
                removed = len(expired_tokens)

                for token in expired_tokens:
                    db.delete(token)

                if removed:
                    self.logger.info(f"Token cleanup removed {removed} expired tokens")
                return removed
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired tokens: {e}")
            return 0

    def get_token_count(self) -> int:
        """
        Get the total number of tokens in the store (including expired ones).

        Returns:
            Number of tokens.
        """
        try:
            # Use get_db_transaction for consistency, even though this is read-only
            # The transaction will be auto-rolled back since we don't make changes
            with self._lock, get_db_transaction() as db:
                count = db.query(Token).count()
                return count
        except Exception as e:
            self.logger.error(f"Failed to get token count: {e}")
            return 0

    def close(self):
        """Close the database connection and perform final cleanup."""
        with self._lock:
            self.cleanup_expired()
            self.logger.info("TokenStore closed")

    # User management methods
    def create_user(
        self, username: str, password: str, email: Optional[str] = None
    ) -> Optional[User]:
        """
        Create a new user.

        Args:
            username: Username for the new user.
            password: Plain text password (will be hashed).
            email: Optional email address.

        Returns:
            Created User object or None if creation failed.
        """
        try:
            with get_db_transaction() as db:
                # Check if username already exists
                existing_user = db.query(User).filter(User.username == username).first()
                if existing_user:
                    self.logger.warning(f"User '{username}' already exists")
                    return None

                # Check if email already exists (if provided)
                if email:
                    existing_email = db.query(User).filter(User.email == email).first()
                    if existing_email:
                        self.logger.warning(f"Email '{email}' already exists")
                        return None

                # Create new user
                hashed_pwd = hash_password(password)
                new_user = User(
                    username=username,
                    email=email,
                    hashed_password=hashed_pwd,
                    is_active=True,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                db.add(new_user)
                db.flush()  # Flush to get the ID without committing
                self.logger.info(f"Created new user: {username}")
                return new_user
        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user by username and password.

        Args:
            username: Username to authenticate.
            password: Plain text password to verify.

        Returns:
            User object if authentication successful, None otherwise.
        """
        try:
            with get_db_transaction() as db:
                user = db.query(User).filter(User.username == username).first()
                if not user:
                    self.logger.debug(f"User not found: {username}")
                    return None

                if not user.is_active:
                    self.logger.debug(f"User is inactive: {username}")
                    return None

                if not verify_password(password, user.hashed_password):
                    self.logger.debug(f"Invalid password for user: {username}")
                    return None

                self.logger.debug(f"User authenticated: {username}")
                return user
        except Exception as e:
            self.logger.error(f"Failed to authenticate user: {e}")
            return None

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Get a user by ID.

        Args:
            user_id: User ID to look up.

        Returns:
            User object or None if not found.
        """
        try:
            with get_db_transaction() as db:
                user = db.query(User).filter(User.id == user_id).first()
                return user
        except Exception as e:
            self.logger.error(f"Failed to get user by ID: {e}")
            return None

    def get_user_by_token(self, token: str) -> Optional[User]:
        """
        Get a user associated with a token.

        Args:
            token: Token string.

        Returns:
            User object or None if token is invalid or expired.
        """
        try:
            with get_db_transaction() as db:
                token_obj = db.query(Token).filter(Token.token == token).first()
                if not token_obj:
                    return None

                # Check if token is expired
                if token_obj.expires_at < datetime.now():
                    db.delete(token_obj)
                    return None

                # Get the associated user by username (subject)
                user = db.query(User).filter(User.username == token_obj.subject).first()
                return user
        except Exception as e:
            self.logger.error(f"Failed to get user by token: {e}")
            return None

    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get token information including expiration time.

        Args:
            token: Token string.

        Returns:
            Dictionary with token and expiration info, or None if token is invalid.
        """
        try:
            with get_db_transaction() as db:
                token_obj = db.query(Token).filter(Token.token == token).first()
                if not token_obj:
                    return None

                # Check if token is expired
                if token_obj.expires_at < datetime.now():
                    db.delete(token_obj)
                    return None

                return {
                    "token": token_obj.token,
                    "expires_at": token_obj.expires_at.isoformat(),
                    "subject": token_obj.subject,
                }
        except Exception as e:
            self.logger.error(f"Failed to get token info: {e}")
            return None
