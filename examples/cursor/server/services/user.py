"""
User management functions for the MemMachine Extension MCP Server.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..auth.password import hash_password, verify_password
from ..core.database import get_db_session, get_db_transaction
from ..models import Token, User

# Security scheme
security = HTTPBearer()

logger = logging.getLogger(__name__)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """Dependency to get the current authenticated user from request state."""
    token = credentials.credentials

    try:
        user = get_user_by_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return user
    except Exception as e:
        logger.error(f"Failed to get user by token: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")



def get_user_by_token(token: str) -> Optional[User]:
    """
    Get a user associated with a token.

    Args:
        token: Token string.

    Returns:
        User object or None if token is invalid or expired.
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

            # Get the associated user by username (subject)
            user = db.query(User).filter(User.username == token_obj.subject).first()
            return user
    except Exception as e:
        logger.error(f"Failed to get user by token: {e}")
        return None


def create_user(username: str, password: str, email: Optional[str] = None) -> Optional[User]:
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
                logger.warning(f"User '{username}' already exists")
                return None

            # Check if email already exists (if provided)
            if email:
                existing_email = db.query(User).filter(User.email == email).first()
                if existing_email:
                    logger.warning(f"Email '{email}' already exists")
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
            logger.info(f"Created new user: {username}")
            return new_user
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        return None


def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user by username and password.

    Args:
        username: Username to authenticate.
        password: Plain text password to verify.

    Returns:
        User object if authentication successful, None otherwise.
    """
    try:
        with get_db_session() as db:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                logger.debug(f"User not found: {username}")
                return None

            if not verify_password(password, user.hashed_password):
                logger.debug(f"Invalid password for user: {username}")
                return None

            if not user.is_active:
                logger.debug(f"User is inactive: {username}")
                return None

            logger.debug(f"User authenticated: {username}")
            return user
    except Exception as e:
        logger.error(f"Failed to authenticate user: {e}")
        return None
