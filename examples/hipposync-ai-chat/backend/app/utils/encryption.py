"""
Enhanced encryption utility for securely storing user API keys.
Supports multiple AI providers: OpenAI, Anthropic, Google, Tavily.
Uses Fernet symmetric encryption from cryptography library.
"""
from cryptography.fernet import Fernet
from ..config import settings
import base64
import hashlib
import re


def get_encryption_key() -> bytes:
    """
    Generate encryption key from SECRET_KEY.
    This ensures consistent encryption/decryption across app restarts.
    """
    # Use SECRET_KEY from settings to derive encryption key
    key_material = settings.secret_key.encode()
    # Create a proper 32-byte key for Fernet
    key = hashlib.sha256(key_material).digest()
    # Fernet requires base64-encoded key
    return base64.urlsafe_b64encode(key)


def encrypt_api_key(api_key: str) -> str:
    """
    Encrypt an API key for secure storage.
    
    Args:
        api_key: Plain text API key
    
    Returns:
        Encrypted API key as string
    """
    if not api_key:
        return None
    
    try:
        fernet = Fernet(get_encryption_key())
        encrypted = fernet.encrypt(api_key.encode())
        return encrypted.decode()
    except Exception as e:
        print(f"Encryption error: {e}")
        raise ValueError("Failed to encrypt API key")


def decrypt_api_key(encrypted_api_key: str) -> str:
    """
    Decrypt an API key for use.
    
    Args:
        encrypted_api_key: Encrypted API key string
    
    Returns:
        Plain text API key
    """
    if not encrypted_api_key:
        return None
    
    try:
        fernet = Fernet(get_encryption_key())
        decrypted = fernet.decrypt(encrypted_api_key.encode())
        return decrypted.decode()
    except Exception as e:
        print(f"Decryption error: {e}")
        raise ValueError("Failed to decrypt API key")


# ============================================================================
# OPENAI API KEY VALIDATION
# ============================================================================

def validate_openai_api_key(api_key: str) -> bool:
    """
    Validate that an API key looks like a valid OpenAI key.
    
    Format: sk-[48+ alphanumeric characters]
    Example: sk-proj-abc123...
    
    Args:
        api_key: API key to validate
    
    Returns:
        True if key format is valid
    """
    if not api_key:
        return False
    
    # OpenAI keys start with "sk-" or "sk-proj-"
    if not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
        return False
    
    # Minimum length check
    if len(api_key) < 20:
        return False
    
    # Basic format validation
    # OpenAI keys are alphanumeric with hyphens
    pattern = r'^sk-([a-zA-Z0-9\-_])+$'
    if not re.match(pattern, api_key):
        return False
    
    return True


# ============================================================================
# ANTHROPIC (CLAUDE) API KEY VALIDATION
# ============================================================================

def validate_anthropic_api_key(api_key: str) -> bool:
    """
    Validate that an API key looks like a valid Anthropic Claude key.
    
    Format: sk-ant-[base58-style characters]
    Example: sk-ant-api03-abc123...
    
    Args:
        api_key: API key to validate
    
    Returns:
        True if key format is valid
    """
    if not api_key:
        return False
    
    # Anthropic keys start with "sk-ant-"
    if not api_key.startswith("sk-ant-"):
        return False
    
    # Minimum length check
    if len(api_key) < 20:
        return False
    
    # Basic format validation
    # Anthropic keys are alphanumeric with hyphens
    pattern = r'^sk-ant-([a-zA-Z0-9\-_])+$'
    if not re.match(pattern, api_key):
        return False
    
    return True


# ============================================================================
# GOOGLE (GEMINI) API KEY VALIDATION
# ============================================================================

def validate_google_api_key(api_key: str) -> bool:
    """
    Validate that an API key looks like a valid Google AI key.
    
    Format: AIza[alphanumeric characters] (39 chars total)
    Example: AIzaSyABC123...
    
    Args:
        api_key: API key to validate
    
    Returns:
        True if key format is valid
    """
    if not api_key:
        return False
    
    # Google AI keys typically start with "AIza"
    if not api_key.startswith("AIza"):
        return False
    
    # Standard length is 39 characters
    if len(api_key) != 39:
        # Allow some flexibility (38-42 chars)
        if len(api_key) < 35 or len(api_key) > 45:
            return False
    
    # Basic format validation
    # Google keys are alphanumeric with dashes and underscores
    pattern = r'^AIza([a-zA-Z0-9\-_])+$'
    if not re.match(pattern, api_key):
        return False
    
    return True


# ============================================================================
# TAVILY API KEY VALIDATION
# ============================================================================

def validate_tavily_api_key(api_key: str) -> bool:
    """
    Validate that an API key looks like a valid Tavily search key.
    
    Format: tvly-[alphanumeric characters]
    Example: tvly-abc123...
    
    Args:
        api_key: API key to validate
    
    Returns:
        True if key format is valid
    """
    if not api_key:
        return False
    
    # Tavily keys start with "tvly-"
    if not api_key.startswith("tvly-"):
        return False
    
    # Minimum length check
    if len(api_key) < 20:
        return False
    
    # Basic format validation
    # Tavily keys are alphanumeric with hyphens
    pattern = r'^tvly-([a-zA-Z0-9\-_])+$'
    if not re.match(pattern, api_key):
        return False
    
    return True


# ============================================================================
# PROVIDER DETECTION
# ============================================================================

def detect_api_key_provider(api_key: str) -> str:
    """
    Detect which provider an API key belongs to based on format.
    
    Args:
        api_key: API key to analyze
    
    Returns:
        Provider name: 'openai', 'anthropic', 'google', 'tavily', or 'unknown'
    """
    if not api_key:
        return 'unknown'
    
    if api_key.startswith('sk-ant-'):
        return 'anthropic'
    elif api_key.startswith('sk-'):
        return 'openai'
    elif api_key.startswith('AIza'):
        return 'google'
    elif api_key.startswith('tvly-'):
        return 'tavily'
    else:
        return 'unknown'


# ============================================================================
# BULK VALIDATION
# ============================================================================

def validate_api_keys(keys: dict) -> dict:
    """
    Validate multiple API keys at once.
    
    Args:
        keys: Dict with provider names as keys and API keys as values
              Example: {'openai': 'sk-...', 'anthropic': 'sk-ant-...'}
    
    Returns:
        Dict with provider names as keys and validation results as values
        Example: {'openai': True, 'anthropic': False}
    """
    results = {}
    
    validators = {
        'openai': validate_openai_api_key,
        'anthropic': validate_anthropic_api_key,
        'google': validate_google_api_key,
        'tavily': validate_tavily_api_key,
    }
    
    for provider, api_key in keys.items():
        validator = validators.get(provider.lower())
        if validator:
            results[provider] = validator(api_key) if api_key else False
        else:
            results[provider] = False
    
    return results


# ============================================================================
# KEY MASKING FOR DISPLAY
# ============================================================================

def mask_api_key(api_key: str, show_chars: int = 4) -> str:
    """
    Mask an API key for safe display.
    Shows only first few and last few characters.
    
    Args:
        api_key: API key to mask
        show_chars: Number of characters to show at start and end
    
    Returns:
        Masked key string
        Example: sk-proj-abc...xyz
    """
    if not api_key:
        return ""
    
    if len(api_key) <= show_chars * 2:
        return api_key
    
    prefix = api_key[:show_chars]
    suffix = api_key[-show_chars:]
    
    # Detect provider for appropriate prefix
    if api_key.startswith('sk-ant-'):
        return f"sk-ant-...{suffix}"
    elif api_key.startswith('sk-proj-'):
        return f"sk-proj-...{suffix}"
    elif api_key.startswith('sk-'):
        return f"sk-...{suffix}"
    elif api_key.startswith('AIza'):
        return f"AIza...{suffix}"
    elif api_key.startswith('tvly-'):
        return f"tvly-...{suffix}"
    else:
        return f"{prefix}...{suffix}"