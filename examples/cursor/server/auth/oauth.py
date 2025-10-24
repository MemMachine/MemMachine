"""OAuth utility functions for authentication and authorization."""
import base64
import hashlib
import secrets


def hash_secret(secret: str) -> str:
    """Simple hash function for demo purposes."""
    return hashlib.sha256(secret.encode()).hexdigest()


def generate_client_id() -> str:
    """Generate a unique client ID."""
    return f"client_{secrets.token_urlsafe(16)}"


def generate_client_secret() -> str:
    """Generate a secure client secret."""
    return secrets.token_urlsafe(32)


def generate_authorization_code() -> str:
    """Generate a secure authorization code."""
    return secrets.token_urlsafe(32)


def generate_code_challenge(code_verifier: str) -> str:
    """Generate PKCE code challenge from code verifier."""
    # Create SHA256 hash of the code verifier
    digest = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    # Base64url encode the hash
    challenge = base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    return challenge


def verify_code_challenge(code_verifier: str, code_challenge: str) -> bool:
    """Verify PKCE code challenge."""
    expected_challenge = generate_code_challenge(code_verifier)
    return expected_challenge == code_challenge


def generate_access_token() -> str:
    """Generate a secure access token (legacy method for backward compatibility)."""
    return secrets.token_urlsafe(32)
