
"""
Token model
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Index, String

from ..core.database import Base


class Token(Base):
    """Token model"""

    __tablename__ = "tokens"

    token = Column(String, primary_key=True, index=True)
    subject = Column(String, nullable=False, index=True)  # Username or user identifier
    issued_at = Column(DateTime, nullable=False, default=datetime.now)
    expires_at = Column(DateTime, nullable=False, index=True)

