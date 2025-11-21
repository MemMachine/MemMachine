"""
Password hashing and verification
"""

import hashlib
import secrets

def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256 with a salt.

    Args:
        password: Plain text password

    Returns:
        Hashed password in format: salt$hash
    """
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${pwd_hash}"


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hashed password.

    Args:
        password: Plain text password to verify
        hashed_password: Hashed password in format: salt$hash

    Returns:
        True if password matches, False otherwise
    """
    try:
        salt, pwd_hash = hashed_password.split("$")
        computed_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        return computed_hash == pwd_hash
    except Exception:
        return False
