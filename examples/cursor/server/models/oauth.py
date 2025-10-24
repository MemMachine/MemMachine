"""
OAuth models
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String

from ..core.database import Base


class OAuthClient(Base):
    """OAuth client model"""

    __tablename__ = "oauth_clients"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    client_id = Column(String, unique=True, nullable=False, index=True)
    client_secret = Column(String, nullable=False)
    redirect_uri = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )


class AuthorizationCode(Base):
    """Authorization code model"""

    __tablename__ = "authorization_codes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    code = Column(String, unique=True, nullable=False, index=True)
    client_id = Column(String, nullable=False, index=True)
    redirect_uri = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False, nullable=False)
    code_challenge = Column(String, nullable=True)
    code_challenge_method = Column(String, nullable=True)
    resource = Column(String, nullable=True)
    scope = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )


class AccessToken(Base):
    """Access token model"""

    __tablename__ = "access_tokens"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    token = Column(String, unique=True, nullable=False, index=True)
    client_id = Column(String, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    scope = Column(String, nullable=False, default="mcp:read mcp:write")
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )
