from sqlalchemy import (
    Column, Integer, String, DateTime, func,
    Boolean, ForeignKey, UniqueConstraint, Text, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
import enum
from .database import Base


class ModelProvider(str, enum.Enum):
    """Supported AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class User(Base):
    """User model with encrypted API keys and email verification"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # ========================================================================
    # EMAIL VERIFICATION FIELDS (NEW)
    # ========================================================================
    email_verified = Column(Boolean, default=False, nullable=False)
    verification_token = Column(String(255), nullable=True, index=True)
    verification_token_expires = Column(DateTime(timezone=True), nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    # ========================================================================
    
    # Multi-provider API keys (all encrypted)
    encrypted_openai_key = Column(Text, nullable=True)
    encrypted_anthropic_key = Column(Text, nullable=True)
    encrypted_google_key = Column(Text, nullable=True)
    encrypted_tavily_key = Column(Text, nullable=True)
    
    # Default provider preference
    default_provider = Column(SQLEnum(ModelProvider), default=ModelProvider.OPENAI)
    
    # Location data from geolocation
    location_city = Column(String(255), nullable=True)
    location_state = Column(String(255), nullable=True)
    location_country = Column(String(255), nullable=True)
    location_latitude = Column(String(50), nullable=True)
    location_longitude = Column(String(50), nullable=True)
    location_timezone = Column(String(100), nullable=True)
    location_formatted = Column(Text, nullable=True)
    location_updated_at = Column(DateTime(timezone=True), nullable=True)
    
    name = Column(String(255), nullable=True)
    occupation = Column(String(255), nullable=True)
    preferences = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    threads = relationship("Thread", back_populates="owner", cascade="all, delete-orphan")
    owned_projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")


class Project(Base):
    """Project for organizing collaborative threads"""
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    owner = relationship("User", back_populates="owned_projects")
    members = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan")
    threads = relationship("Thread", back_populates="project", cascade="all, delete-orphan")


class ProjectMember(Base):
    """Project membership with role-based access"""
    __tablename__ = "project_members"
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    role = Column(String, default="member")  # owner, admin, member, viewer
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (UniqueConstraint("project_id", "user_id", name="uq_project_member"),)
    
    # Relationships
    project = relationship("Project", back_populates="members")


class Thread(Base):
    """Conversation thread with multi-provider model support"""
    __tablename__ = "threads"
    
    id = Column(Integer, primary_key=True)
    title = Column(String, default="New chat")
    owner_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True)
    
    # Memory and session management
    session_id = Column(String, unique=True, index=True, nullable=False)
    group_scope = Column(String, nullable=False, index=True)
    
    # Model configuration
    active_model = Column(String, default="gpt-4o-mini")
    active_provider = Column(SQLEnum(ModelProvider), default=ModelProvider.OPENAI)
    temperature = Column(String, default="1.0")
    system_prompt = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_message_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    owner = relationship("User", back_populates="threads")
    project = relationship("Project", back_populates="threads")
    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    """Individual message in a conversation thread"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, ForeignKey("threads.id", ondelete="CASCADE"), index=True, nullable=False)
    
    # Message content
    sender = Column(String, nullable=False)
    content = Column(Text, nullable=True)
    type = Column(String, default="text")
    user_email = Column(String, nullable=True)
    # File attachment metadata
    filename = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    file_type = Column(String, nullable=True)
    
    # Model metadata
    model_used = Column(String, default="gpt-4o-mini")
    provider_used = Column(SQLEnum(ModelProvider), default=ModelProvider.OPENAI)
    
    # Token usage tracking
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    thread = relationship("Thread", back_populates="messages")