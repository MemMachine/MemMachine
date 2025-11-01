"""OAuth implementation using FastAPI security components."""

from fastapi import APIRouter

from ..settings import settings

well_known_router = APIRouter(tags=["well-known"])


@well_known_router.get("/.well-known/oauth-protected-resource")
async def get_protected_resource_metadata():
    """
    OAuth 2.0 Protected Resource Metadata endpoint (RFC9728).

    This endpoint allows MCP clients to discover authorization servers
    and other metadata about the protected resource.
    """
    return {
        "resource": f"{settings.oauth_domain}/mcp",
        "authorization_servers": [f"{settings.oauth_domain}/oauth"],
        "scopes_supported": ["mcp:read", "mcp:write"],
    }


@well_known_router.get("/.well-known/oauth-authorization-server")
async def get_authorization_server_metadata():
    """
    OAuth 2.0 Authorization Server Metadata endpoint (RFC8414).

    This endpoint provides metadata about the authorization server
    including supported features and endpoints.
    """
    return {
        "issuer": f"{settings.oauth_domain}/oauth",
        "authorization_endpoint": f"{settings.oauth_domain}/oauth/authorize",
        "token_endpoint": f"{settings.oauth_domain}/oauth/token",
        "registration_endpoint": f"{settings.oauth_domain}/oauth/register",
        "scopes_supported": ["mcp:read", "mcp:write"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],  # PKCE support
    }


@well_known_router.get("/.well-known/openid-configuration")
async def get_openid_configuration():
    """
    OpenID Connect Discovery endpoint.

    Required by ChatGPT for OAuth 2.1 integration. Must include:
    - authorization_endpoint
    - token_endpoint
    - jwks_uri
    - registration_endpoint
    """
    return {
        "issuer": f"{settings.oauth_domain}/oauth",
        "authorization_endpoint": f"{settings.oauth_domain}/oauth/authorize",
        "token_endpoint": f"{settings.oauth_domain}/oauth/token",
        "registration_endpoint": f"{settings.oauth_domain}/oauth/register",
        "jwks_uri": f"{settings.oauth_domain}/.well-known/jwks.json",
        "scopes_supported": ["mcp:read", "mcp:write"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
    }


@well_known_router.get("/.well-known/jwks.json")
async def get_jwks():
    """
    JSON Web Key Set (JWKS) endpoint.

    Required by ChatGPT for token verification. In production, this should
    contain the public keys for verifying JWT tokens.
    """
    # For demo purposes, we'll return a simple JWKS
    # In production, you should generate proper RSA keys
    return {
        "keys": [
            {
                "kty": "oct",
                "kid": "demo-key",
                "use": "sig",
                "alg": "HS256",
                "k": "your-secret-key-change-in-production",  # Base64 encoded
            }
        ]
    }
