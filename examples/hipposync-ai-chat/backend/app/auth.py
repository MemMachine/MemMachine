"""
Enhanced auth.py with multi-provider API key management + EMAIL VERIFICATION.
Supports OpenAI, Anthropic Claude, Google Gemini, and Tavily.
✅ UPDATED FOR MEMMACHINE V2
✅ NEW: Email verification with token
✅ NEW: Verification email sending
✅ NEW: Resend verification endpoint
"""
from datetime import datetime, timedelta, timezone  
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import jwt
from typing import Optional, Dict, Any
from uuid import uuid4
import secrets
import logging

from .config import settings
from .database import SessionLocal, Base, engine
from .models import User, Thread
from .schemas import UserCreate, UserOut, Token
from .utils.encryption import (
    encrypt_api_key, decrypt_api_key,
    validate_openai_api_key, validate_anthropic_api_key,
    validate_google_api_key, validate_tavily_api_key
)
from .utils.email_service import email_service

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Create tables
Base.metadata.create_all(bind=engine)


# ============================================================================
# DATABASE & PASSWORD UTILITIES
# ============================================================================

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_password_hash(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def generate_verification_token() -> str:
    """Generate secure random verification token"""
    return secrets.token_urlsafe(32)


# ============================================================================
# JWT TOKEN MANAGEMENT
# ============================================================================

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.now(tz=timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm="HS256")
    return encoded_jwt


def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        user_id: int = int(payload.get("sub"))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    return user


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def get_user_api_keys(user: User) -> Dict[str, Optional[str]]:
    """
    Get all decrypted API keys for user.
    Returns dict with provider names as keys.
    
    Handles both old single-key and new multi-provider formats for backward compatibility.
    """
    keys = {}
    
    # OpenAI - check both new and old field names
    if hasattr(user, 'encrypted_openai_key') and user.encrypted_openai_key:
        try:
            keys['openai'] = decrypt_api_key(user.encrypted_openai_key)
        except Exception as e:
            logger.error(f"Failed to decrypt OpenAI key: {e}")
            keys['openai'] = None
    elif hasattr(user, 'encrypted_api_key') and user.encrypted_api_key:
        # Backward compatibility with old single key
        try:
            keys['openai'] = decrypt_api_key(user.encrypted_api_key)
        except Exception as e:
            logger.error(f"Failed to decrypt API key: {e}")
            keys['openai'] = None
    else:
        keys['openai'] = None
    
    # Anthropic
    if hasattr(user, 'encrypted_anthropic_key') and user.encrypted_anthropic_key:
        try:
            keys['anthropic'] = decrypt_api_key(user.encrypted_anthropic_key)
        except Exception as e:
            logger.error(f"Failed to decrypt Anthropic key: {e}")
            keys['anthropic'] = None
    else:
        keys['anthropic'] = None
    
    # Google
    if hasattr(user, 'encrypted_google_key') and user.encrypted_google_key:
        try:
            keys['google'] = decrypt_api_key(user.encrypted_google_key)
        except Exception as e:
            logger.error(f"Failed to decrypt Google key: {e}")
            keys['google'] = None
    else:
        keys['google'] = None
    
    # Tavily
    if hasattr(user, 'encrypted_tavily_key') and user.encrypted_tavily_key:
        try:
            keys['tavily'] = decrypt_api_key(user.encrypted_tavily_key)
        except Exception as e:
            logger.error(f"Failed to decrypt Tavily key: {e}")
            keys['tavily'] = None
    else:
        keys['tavily'] = None
    
    return keys


def get_user_api_key(user: User, provider: str = 'openai') -> Optional[str]:
    """
    Get decrypted API key for specific provider.
    Backward compatible with old single-key system.
    
    Args:
        user: User object
        provider: Provider name ('openai', 'anthropic', 'google', 'tavily')
    
    Returns:
        Decrypted API key or None
    """
    keys = get_user_api_keys(user)
    return keys.get(provider.lower())


def has_any_provider_key(user: User) -> bool:
    """Check if user has at least one provider API key configured"""
    # Check new format
    if hasattr(user, 'encrypted_openai_key'):
        return any([
            user.encrypted_openai_key,
            getattr(user, 'encrypted_anthropic_key', None),
            getattr(user, 'encrypted_google_key', None)
        ])
    # Check old format (backward compatibility)
    return bool(getattr(user, 'encrypted_api_key', None))


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@router.post("/signup", response_model=UserOut)
def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register new user with EMAIL VERIFICATION.
    API keys can be added later in settings via the just-in-time flow.
    Accepts optional location data from geolocation.
    
    Workflow:
    1. Create user account (email_verified=False)
    2. Generate verification token (expires in 24 hours)
    3. Send verification email
    4. Return user data
    """
    # Check if user exists
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Generate verification token (expires in 24 hours)
    verification_token = generate_verification_token()
    token_expires = datetime.now(tz=timezone.utc) + timedelta(hours=24)
    
    user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        name=user_data.name,
        occupation=user_data.occupation,
        location_city=user_data.location_city,
        location_state=user_data.location_state,
        location_country=user_data.location_country,
        location_latitude=user_data.location_latitude,
        location_longitude=user_data.location_longitude,
        location_timezone=user_data.location_timezone,
        location_formatted=user_data.location_formatted,
        location_updated_at=datetime.utcnow() if user_data.location_city else None,
        # Email verification fields
        email_verified=False,
        verification_token=verification_token,
        verification_token_expires=token_expires,
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    logger.info(f"✅ New user registered: {user.email} (ID: {user.id})")
    
    # Send verification email
    try:
        email_sent = email_service.send_verification_email(
            to_email=user.email,
            verification_token=verification_token,
            user_name=user.name
        )
        
        if email_sent:
            logger.info(f"📧 Verification email sent to {user.email}")
        else:
            logger.warning(f"⚠️ Failed to send verification email to {user.email}")
    except Exception as e:
        logger.error(f"❌ Error sending verification email: {e}")
        # Don't fail signup if email fails
    
    return UserOut(
        id=user.id,
        email=user.email,
        email_verified=False,
        has_openai_key=False,
        has_anthropic_key=False,
        has_google_key=False,
        has_tavily_key=False,
    )


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login with email and password.
    User can add API keys later via just-in-time flow.
    
    ✅ REQUIRES EMAIL VERIFICATION
    """
    user = db.query(User).filter(User.email == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=400,
            detail="Incorrect email or password"
        )
    
    # ✅ CHECK EMAIL VERIFICATION
    if not getattr(user, 'email_verified', False):
        raise HTTPException(
            status_code=403,
            detail="Please verify your email before logging in. Check your inbox for the verification link."
        )
    
    token = create_access_token({"sub": str(user.id)})
    
    logger.info(f"✅ User logged in: {user.email} (verified: {getattr(user, 'email_verified', False)})")
    
    return {"access_token": token, "token_type": "bearer"}


@router.get("/auth/verify-email")
def verify_email(token: str, db: Session = Depends(get_db)):
    """
    Verify user email with token
    
    Args:
        token: Verification token from email link
    
    Returns:
        Success message or error
    """
    # Find user with this token
    user = db.query(User).filter(User.verification_token == token).first()
    
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Invalid verification token. Please request a new verification email."
        )
    
    # Check if already verified
    if user.email_verified:
        return {
            "status": "already_verified",
            "message": "Email already verified. You can login now!"
        }
    
    # Check if token expired
    # Handle both timezone-aware and timezone-naive datetimes
    now = datetime.now(tz=timezone.utc)
    expires = user.verification_token_expires
    
    # If expires is naive, make it aware (assume UTC)
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    
    if expires < now:
        raise HTTPException(
            status_code=400,
            detail="Verification token expired. Please request a new verification email."
        )
    
    # Verify email
    user.email_verified = True
    user.verified_at = datetime.now(tz=timezone.utc)
    user.verification_token = None  # Clear token
    user.verification_token_expires = None
    
    db.commit()
    
    logger.info(f"✅ Email verified for user: {user.email}")
    
    return {
        "status": "success",
        "message": "Email verified successfully! You can now login and use all features.",
        "email": user.email
    }


@router.post("/resend-verification")
def resend_verification(email: str, db: Session = Depends(get_db)):
    """
    Resend verification email
    
    Args:
        email: User's email address
    
    Returns:
        Success message
    """
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        # Don't reveal if email exists for security
        return {
            "status": "sent",
            "message": "If this email is registered, a verification link has been sent."
        }
    
    if user.email_verified:
        return {
            "status": "already_verified",
            "message": "Email already verified. You can login now!"
        }
    
    # Generate new token
    verification_token = generate_verification_token()
    token_expires = datetime.now(tz=timezone.utc) + timedelta(hours=24)
    
    user.verification_token = verification_token
    user.verification_token_expires = token_expires
    
    db.commit()
    
    # Send email
    try:
        email_sent = email_service.send_verification_email(
            to_email=user.email,
            verification_token=verification_token,
            user_name=user.name
        )
        
        if email_sent:
            logger.info(f"📧 Verification email resent to {user.email}")
        else:
            logger.warning(f"⚠️ Failed to resend verification email to {user.email}")
    except Exception as e:
        logger.error(f"❌ Error resending verification email: {e}")
    
    return {
        "status": "sent",
        "message": "Verification email sent. Please check your inbox."
    }


@router.get("/me", response_model=UserOut)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserOut(
        id=current_user.id,
        email=current_user.email,
        email_verified=getattr(current_user, 'email_verified', False),
        has_openai_key=bool(
            getattr(current_user, 'encrypted_openai_key', None) or 
            getattr(current_user, 'encrypted_api_key', None)
        ),
        has_anthropic_key=bool(getattr(current_user, 'encrypted_anthropic_key', None)),
        has_google_key=bool(getattr(current_user, 'encrypted_google_key', None)),
        has_tavily_key=bool(getattr(current_user, 'encrypted_tavily_key', None))
    )


# ============================================================================
# API KEY MANAGEMENT ENDPOINTS
# ============================================================================

@router.put("/api-keys")
def update_api_keys(
    payload: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user's API keys.
    Can update any combination of keys.
    Accepts: openai_api_key, anthropic_api_key, google_api_key, tavily_api_key
    """
    updated = False
    
    # Update OpenAI key
    if 'openai_api_key' in payload and payload['openai_api_key']:
        if not validate_openai_api_key(payload['openai_api_key']):
            raise HTTPException(
                status_code=400,
                detail="Invalid OpenAI API key format. Key should start with 'sk-'"
            )
        # Use new field if available, fall back to old field
        if hasattr(current_user, 'encrypted_openai_key'):
            current_user.encrypted_openai_key = encrypt_api_key(payload['openai_api_key'])
        else:
            current_user.encrypted_api_key = encrypt_api_key(payload['openai_api_key'])
        updated = True
    
    # Update Anthropic key
    if 'anthropic_api_key' in payload and payload['anthropic_api_key']:
        if not validate_anthropic_api_key(payload['anthropic_api_key']):
            raise HTTPException(
                status_code=400,
                detail="Invalid Anthropic API key format. Key should start with 'sk-ant-'"
            )
        if hasattr(current_user, 'encrypted_anthropic_key'):
            current_user.encrypted_anthropic_key = encrypt_api_key(payload['anthropic_api_key'])
            updated = True
    
    # Update Google key
    if 'google_api_key' in payload and payload['google_api_key']:
        if not validate_google_api_key(payload['google_api_key']):
            raise HTTPException(
                status_code=400,
                detail="Invalid Google API key format. Key should start with 'AIza'"
            )
        if hasattr(current_user, 'encrypted_google_key'):
            current_user.encrypted_google_key = encrypt_api_key(payload['google_api_key'])
            updated = True
    
    # Update Tavily key
    if 'tavily_api_key' in payload and payload['tavily_api_key']:
        if not validate_tavily_api_key(payload['tavily_api_key']):
            raise HTTPException(
                status_code=400,
                detail="Invalid Tavily API key format. Key should start with 'tvly-'"
            )
        if hasattr(current_user, 'encrypted_tavily_key'):
            current_user.encrypted_tavily_key = encrypt_api_key(payload['tavily_api_key'])
            updated = True
    
    if not updated:
        raise HTTPException(status_code=400, detail="No valid API keys provided to update")
    
    db.commit()
    db.refresh(current_user)
    
    return {
        "id": current_user.id,
        "email": current_user.email,
        "email_verified": getattr(current_user, 'email_verified', False),
        "has_openai_key": bool(
            getattr(current_user, 'encrypted_openai_key', None) or 
            getattr(current_user, 'encrypted_api_key', None)
        ),
        "has_anthropic_key": bool(getattr(current_user, 'encrypted_anthropic_key', None)),
        "has_google_key": bool(getattr(current_user, 'encrypted_google_key', None)),
        "has_tavily_key": bool(getattr(current_user, 'encrypted_tavily_key', None))
    }


@router.put("/api-key", response_model=UserOut)
def update_single_api_key(
    openai_api_key: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user's OpenAI API key.
    Kept for backward compatibility.
    """
    if not validate_openai_api_key(openai_api_key):
        raise HTTPException(
            status_code=400,
            detail="Invalid OpenAI API key format. Key should start with 'sk-'"
        )
    
    encrypted_key = encrypt_api_key(openai_api_key)
    
    # Use new field if available, fall back to old field
    if hasattr(current_user, 'encrypted_openai_key'):
        current_user.encrypted_openai_key = encrypted_key
    else:
        current_user.encrypted_api_key = encrypted_key
    
    db.commit()
    db.refresh(current_user)
    
    return UserOut(
        id=current_user.id,
        email=current_user.email,
        email_verified=getattr(current_user, 'email_verified', False),
        has_openai_key=True,
        has_anthropic_key=bool(getattr(current_user, 'encrypted_anthropic_key', None)),
        has_google_key=bool(getattr(current_user, 'encrypted_google_key', None)),
        has_tavily_key=bool(getattr(current_user, 'encrypted_tavily_key', None))
    )


@router.delete("/api-key/{provider}")
def delete_api_key(
    provider: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove specific provider API key"""
    provider = provider.lower()
    
    if provider == 'openai':
        if hasattr(current_user, 'encrypted_openai_key'):
            current_user.encrypted_openai_key = None
        else:
            current_user.encrypted_api_key = None
    elif provider == 'anthropic':
        if hasattr(current_user, 'encrypted_anthropic_key'):
            current_user.encrypted_anthropic_key = None
    elif provider == 'google':
        if hasattr(current_user, 'encrypted_google_key'):
            current_user.encrypted_google_key = None
    elif provider == 'tavily':
        if hasattr(current_user, 'encrypted_tavily_key'):
            current_user.encrypted_tavily_key = None
    else:
        raise HTTPException(status_code=400, detail="Invalid provider")
    
    db.commit()
    
    return {"status": f"{provider.title()} API key removed"}


# ============================================================================
# MEMORY/HISTORY ENDPOINTS - UPDATED FOR V2
# ============================================================================

from . import memmachine_client

@router.get("/history")
def history(user: User = Depends(get_current_user)):
    """
    Returns a lightweight view of user's remembered items (semantic + episodic).
    Frontend can render this as a sidebar or preload list.
    
    ✅ UPDATED FOR MEMMACHINE V2:
    - Uses new search_memories function
    - Searches across ALL personal threads
    - Returns both semantic and episodic memories
    """
    try:
        # Search user's personal memories (across all threads)
        results = memmachine_client.search_memories(
            user_id=user.id,
            query="recent documents conversations topics",
            project_id=None,  # Personal scope
            thread_id=None,  # All threads (cross-thread search)
            limit=20,
            search_semantic=True,
            search_episodic=True
        )
        
        # Format for frontend
        history_items = []
        
        # Add semantic memories (documents, preferences)
        for mem in results.get("semantic_results", []):
            history_items.append({
                "type": "semantic",
                "content": mem.get("profile_content", "")[:200],
                "metadata": mem.get("metadata", {}),
                "score": mem.get("score", 0)
            })
        
        # Add episodic memories (conversations)
        for mem in results.get("episodic_results", []):
            history_items.append({
                "type": "episodic",
                "content": mem.get("episode_content", "")[:200],
                "metadata": mem.get("metadata", {}),
                "score": mem.get("score", 0)
            })
        
        logger.info(f"📋 History retrieved: {len(history_items)} items for user {user.id}")
        
        return {
            "history": history_items,
            "total": len(history_items),
            "semantic_count": len(results.get("semantic_results", [])),
            "episodic_count": len(results.get("episodic_results", []))
        }
        
    except Exception as e:
        logger.error(f"❌ History retrieval failed: {e}")
        return {
            "history": [],
            "total": 0,
            "error": str(e)
        }