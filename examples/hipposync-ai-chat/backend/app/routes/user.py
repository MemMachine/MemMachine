"""
User settings, API keys, and location endpoints

"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import logging
from datetime import datetime

from ..auth import get_current_user, get_db
from ..models import User
from ..schemas import ApiKeysCheck, ApiKeysSave, UserSettings, LocationData
from ..utils.encryption import (
    encrypt_api_key,
    validate_openai_api_key,
    validate_anthropic_api_key,
    validate_google_api_key,
    validate_tavily_api_key,
)

router = APIRouter(prefix="/user", tags=["user"])
logger = logging.getLogger(__name__)


@router.get("/api-keys", response_model=ApiKeysCheck)
def check_api_keys(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Check which API keys the user has configured.
    Returns boolean for each provider.
    Used by frontend to show "Add API Keys" prompt.
    """
    return ApiKeysCheck(
        has_openai=bool(user.encrypted_openai_key),
        has_anthropic=bool(user.encrypted_anthropic_key),
        has_google=bool(user.encrypted_google_key),
        has_tavily=bool(user.encrypted_tavily_key),
    )


@router.post("/api-keys")
def save_api_keys(
    keys: ApiKeysSave,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Save encrypted API keys for the user.
    Validates format before saving.
    """
    saved = []
    errors = []
    
    # Validate and save OpenAI key
    if keys.openai_api_key:
        if validate_openai_api_key(keys.openai_api_key):
            user.encrypted_openai_key = encrypt_api_key(keys.openai_api_key)
            saved.append("openai")
        else:
            errors.append("Invalid OpenAI key format (should start with 'sk-')")
    
    # Validate and save Anthropic key
    if keys.anthropic_api_key:
        if validate_anthropic_api_key(keys.anthropic_api_key):
            user.encrypted_anthropic_key = encrypt_api_key(keys.anthropic_api_key)
            saved.append("anthropic")
        else:
            errors.append("Invalid Anthropic key format (should start with 'sk-ant-')")
    
    # Validate and save Google key
    if keys.google_api_key:
        if validate_google_api_key(keys.google_api_key):
            user.encrypted_google_key = encrypt_api_key(keys.google_api_key)
            saved.append("google")
        else:
            errors.append("Invalid Google key format (should start with 'AIza')")
    
    # Validate and save Tavily key
    if keys.tavily_api_key:
        if validate_tavily_api_key(keys.tavily_api_key):
            user.encrypted_tavily_key = encrypt_api_key(keys.tavily_api_key)
            saved.append("tavily")
        else:
            errors.append("Invalid Tavily key format (should start with 'tvly-')")
    
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))
    
    db.commit()
    logger.info(f"✅ Saved {len(saved)} API keys for user {user.email}")
    
    return {
        "success": True,
        "saved": saved,
        "message": f"Saved {len(saved)} API key(s)"
    }


@router.post("/settings")
def save_settings(
    settings: UserSettings,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Save user profile settings (name, occupation, preferences).
    """
    updated = []
    
    if settings.name:
        user.name = settings.name
        updated.append("name")
    
    if settings.occupation:
        user.occupation = settings.occupation
        updated.append("occupation")
    
    if settings.preferences:
        user.preferences = settings.preferences
        updated.append("preferences")
    
    if updated:
        db.commit()
        logger.info(f"✅ Updated {', '.join(updated)} for user {user.email}")
    
    return {
        "success": True,
        "message": f"Updated {len(updated)} setting(s)"
    }


@router.post("/location")
def save_location(
    location: LocationData,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Save user location from browser geolocation.
    Called automatically when user allows location access.
    """
    user.location_city = location.city
    user.location_state = location.state
    user.location_country = location.country
    user.location_latitude = location.latitude
    user.location_longitude = location.longitude
    user.location_timezone = location.timezone
    user.location_formatted = location.formatted
    user.location_updated_at = datetime.utcnow()
    
    db.commit()
    logger.info(f"📍 Updated location for {user.email}: {location.formatted}")
    
    return {
        "success": True,
        "location": location.formatted,
        "timezone": location.timezone
    }


@router.get("/settings")
def get_settings(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's current settings.
    """
    return {
        "name": user.name,
        "occupation": user.occupation,
        "preferences": user.preferences,
        "location": {
            "city": user.location_city,
            "state": user.location_state,
            "country": user.location_country,
            "formatted": user.location_formatted,
            "timezone": user.location_timezone,
        } if user.location_city else None
    }