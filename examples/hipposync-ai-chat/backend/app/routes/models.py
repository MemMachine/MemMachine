"""
Models endpoint with LATEST models (November 2025)
- Claude 4.x (Haiku, Sonnet, Opus)
- Gemini 3.x Pro (Gemini 1.x deprecated)
- OpenAI GPT-4o, o1
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any, List
import logging

from ..auth import get_current_user, get_user_api_keys
from ..utils.model_utils import fetch_available_models

router = APIRouter(prefix="/models", tags=["models"])
logger = logging.getLogger(__name__)


# Latest model catalog (November 19, 2025)
LATEST_MODELS = {
    "openai": [
        {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "context_window": 128000,
            "supports_vision": True,
            "supports_streaming": True,
            "cost_per_1k_input": 2.5,
            "cost_per_1k_output": 10.0
        },
        {
            "id": "gpt-4o-mini",
            "name": "GPT-4o mini",
            "context_window": 128000,
            "supports_vision": True,
            "supports_streaming": True,
            "cost_per_1k_input": 0.15,
            "cost_per_1k_output": 0.60
        },
        {
            "id": "o1",
            "name": "o1",
            "context_window": 200000,
            "supports_streaming": False,
            "cost_per_1k_input": 15.0,
            "cost_per_1k_output": 60.0
        },
        {
            "id": "o1-mini",
            "name": "o1-mini",
            "context_window": 128000,
            "supports_streaming": False,
            "cost_per_1k_input": 3.0,
            "cost_per_1k_output": 12.0
        }
    ],
    
    "anthropic": [
        # Claude 4.x (Released November 19, 2025)
        {
            "id": "claude-4-sonnet-20251119",
            "name": "Claude 4 Sonnet",
            "context_window": 200000,
            "supports_vision": True,
            "supports_streaming": True,
            "cost_per_1k_input": 3.0,
            "cost_per_1k_output": 15.0,
            "tier": "Latest"
        },
        {
            "id": "claude-4-haiku-20251119",
            "name": "Claude 4 Haiku",
            "context_window": 200000,
            "supports_vision": False,
            "supports_streaming": True,
            "cost_per_1k_input": 1.0,
            "cost_per_1k_output": 5.0,
            "tier": "Latest"
        },
        {
            "id": "claude-4-opus-20251119",
            "name": "Claude 4 Opus",
            "context_window": 200000,
            "supports_vision": True,
            "supports_streaming": True,
            "cost_per_1k_input": 15.0,
            "cost_per_1k_output": 75.0,
            "tier": "Latest"
        },
        # Claude 3.5.x (Still available)
        {
            "id": "claude-3-5-sonnet-20241022",
            "name": "Claude 3.5 Sonnet",
            "context_window": 200000,
            "supports_vision": True,
            "supports_streaming": True,
            "cost_per_1k_input": 3.0,
            "cost_per_1k_output": 15.0,
            "tier": "Previous"
        },
        {
            "id": "claude-3-5-haiku-20241022",
            "name": "Claude 3.5 Haiku",
            "context_window": 200000,
            "supports_streaming": True,
            "cost_per_1k_input": 1.0,
            "cost_per_1k_output": 5.0,
            "tier": "Previous"
        }
    ],
    
    "google": [
        # Gemini 3.0 (Latest - Preview)
        {
            "id": "gemini-3-pro-preview",
            "name": "Gemini 3 Pro Preview",
            "context_window": 1048576,
            "supports_vision": True,
            "supports_streaming": True,
            "tier": "Latest Preview"
        },
        # Gemini 2.5 (Stable - Recommended)
        {
            "id": "gemini-2.5-pro",
            "name": "Gemini 2.5 Pro",
            "context_window": 1048576,
            "supports_vision": True,
            "supports_streaming": True,
            "tier": "Stable"
        },
        {
            "id": "gemini-2.5-flash",
            "name": "Gemini 2.5 Flash",
            "context_window": 1048576,
            "supports_vision": True,
            "supports_streaming": True,
            "tier": "Stable - Fast"
        },
        {
            "id": "gemini-2.5-flash-lite",
            "name": "Gemini 2.5 Flash Lite",
            "context_window": 1048576,
            "supports_vision": True,
            "supports_streaming": True,
            "tier": "Stable - Fastest"
        },
        # Gemini 2.0
        {
            "id": "gemini-2.0-flash",
            "name": "Gemini 2.0 Flash",
            "context_window": 1048576,
            "supports_vision": True,
            "supports_streaming": True,
            "tier": "Stable"
        },
        {
            "id": "gemini-2.0-flash-lite",
            "name": "Gemini 2.0 Flash Lite",
            "context_window": 1048576,
            "supports_vision": True,
            "supports_streaming": True,
            "tier": "Stable"
        },
        # Latest aliases (auto-update to newest)
        {
            "id": "gemini-pro-latest",
            "name": "Gemini Pro (Latest)",
            "context_window": 1048576,
            "supports_vision": True,
            "supports_streaming": True,
            "tier": "Auto-updating"
        },
        {
            "id": "gemini-flash-latest",
            "name": "Gemini Flash (Latest)",
            "context_window": 1048576,
            "supports_vision": True,
            "supports_streaming": True,
            "tier": "Auto-updating"
        }
    ]
}


@router.get("/available/fast")
def get_available_models_fast(user = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Returns latest models based on user's API keys.
    Updated for November 2025: Claude 4.x, Gemini 3.x
    """
    api_keys = get_user_api_keys(user)
    
    result = {
        "openai": [],
        "anthropic": [],
        "google": [],
        "user_has_access": {
            "openai": bool(api_keys.get('openai')),
            "anthropic": bool(api_keys.get('anthropic')),
            "google": bool(api_keys.get('google'))
        }
    }
    
    # OpenAI models
    if api_keys.get('openai'):
        try:
            # Try to get live models from OpenAI
            result["openai"] = fetch_available_models(api_keys['openai'])
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Fallback to catalog
            result["openai"] = LATEST_MODELS["openai"]
    
    # Anthropic Claude 4.x models
    if api_keys.get('anthropic'):
        result["anthropic"] = LATEST_MODELS["anthropic"]
        logger.info("Loaded Claude 4.x models")
    
    # Google Gemini 3.x models
    if api_keys.get('google'):
        result["google"] = LATEST_MODELS["google"]
        logger.info("Loaded Gemini 3.x models")
    
    # Calculate total
    total = len(result["openai"]) + len(result["anthropic"]) + len(result["google"])
    result["total_count"] = total
    
    return result


@router.get("/available")
def get_available_models(user = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Same as /fast but could include API validation.
    Currently returns catalog.
    """
    return get_available_models_fast(user)


@router.get("/catalog")
def get_full_catalog() -> Dict[str, Any]:
    """
    Get complete model catalog (November 2025).
    Shows what's available regardless of user's keys.
    """
    return {
        **LATEST_MODELS,
        "last_updated": "2025-11-19",
        "notes": {
            "claude": "Claude 4.x released Nov 19, 2025",
            "gemini": "Gemini 3.x Pro released Nov 18, 2025. Gemini 1.x deprecated.",
            "openai": "GPT-4o and o1 series current"
        }
    }


@router.get("/check/{provider}/{model_id}")
def check_model_availability(
    provider: str,
    model_id: str,
    user = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Check if a specific model is available to the user.
    """
    api_keys = get_user_api_keys(user)
    
    if not api_keys.get(provider.lower()):
        return {
            "available": False,
            "reason": f"No {provider} API key configured",
            "model_id": model_id
        }
    
    # Check if model exists in catalog
    provider_models = LATEST_MODELS.get(provider.lower(), [])
    model_exists = any(m["id"] == model_id for m in provider_models)
    
    return {
        "available": model_exists,
        "reason": "Model available" if model_exists else "Model not found in catalog",
        "model_id": model_id,
        "provider": provider
    }