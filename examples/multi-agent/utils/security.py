"""
Security utilities for input validation and sanitization.
Critical for open source security.
"""
import re
from typing import Optional, Tuple


def sanitize_user_id(user_id: str, max_length: int = 50) -> str:
    """
    Sanitize user ID to prevent injection attacks.
    
    Only allows alphanumeric characters, hyphens, and underscores.
    Enforces length limits to prevent DoS attacks.
    
    Args:
        user_id: Raw user ID from input
        max_length: Maximum allowed length (default: 50)
    
    Returns:
        Sanitized user ID safe for use in headers, URLs, and database keys
    
    Example:
        >>> sanitize_user_id("user@123!")
        'user123'
        >>> sanitize_user_id("my-user_id")
        'my-user_id'
    """
    if not isinstance(user_id, str):
        user_id = str(user_id)
    
    # Remove any non-alphanumeric characters except hyphen and underscore
    sanitized = re.sub(r'[^a-z0-9_-]', '', user_id.lower())
    
    # Enforce length limit
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Ensure it's not empty (fallback to default)
    if not sanitized or sanitized.strip() == "":
        sanitized = "default_user"
    
    # Prevent leading/trailing special characters
    sanitized = sanitized.strip('-_')
    
    # Final fallback
    if not sanitized:
        sanitized = "default_user"
    
    return sanitized


def validate_user_id(user_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user ID format and return sanitized version.
    
    Args:
        user_id: User ID to validate
    
    Returns:
        Tuple of (is_valid, sanitized_user_id)
        If invalid, sanitized_user_id will be None
    """
    if not user_id or not isinstance(user_id, str):
        return False, None
    
    sanitized = sanitize_user_id(user_id)
    
    # Check if sanitization changed the input significantly (indicating malicious input)
    if len(user_id) > 100:  # Suspiciously long
        return False, None
    
    # Must start with alphanumeric
    if not re.match(r'^[a-z0-9]', sanitized):
        return False, None
    
    return True, sanitized

