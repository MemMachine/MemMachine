"""
Data models for the Cursor MCP Server.
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, String

from .database import Base


# SQLAlchemy Models
class User(Base):
    """SQLAlchemy model for users."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=True, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )


class Token(Base):
    """SQLAlchemy model for authentication tokens."""

    __tablename__ = "tokens"

    token = Column(String, primary_key=True, index=True)
    subject = Column(String, nullable=False, index=True)  # Username or user identifier
    issued_at = Column(DateTime, nullable=False, default=datetime.now)
    expires_at = Column(DateTime, nullable=False, index=True)

    __table_args__ = (
        Index("idx_expires_at", "expires_at"),
        Index("idx_subject", "subject"),
    )
