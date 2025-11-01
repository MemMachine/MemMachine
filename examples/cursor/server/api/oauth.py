"""OAuth implementation using FastAPI security components."""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ..auth.jwt import (
    create_access_token,
)
from ..auth.oauth import (
    generate_authorization_code,
    generate_client_id,
    generate_client_secret,
    hash_secret,
    verify_code_challenge,
)
from ..core.database import get_db
from ..models.oauth import (
    AccessToken,
    AuthorizationCode,
    OAuthClient,
)
from ..schemas.oauth import (
    ClientRegistrationRequest,
    ClientRegistrationResponse,
    TokenRequestForm,
    TokenResponse,
)
from ..settings import settings

# ChatGPT OAuth flow
#    │
#    ├── 1. GET /.well-known/oauth-protected-resource ───────► Resource Server (MCP)
#    │
#    ├── 2. GET /.well-known/openid-configuration ───────────► Auth Server
#    │
#    ├── 3. POST /register (Dynamic Client Registration) ────► Auth Server
#    │
#    ├── 4. Redirect to /authorize (PKCE Login) ─────────────► Auth Server
#    │
#    ├── 5. POST /token (Exchange code→token) ───────────────► Auth Server
#    │
#    └── 6. Authorization: Bearer <token> ───────────────────► MCP API


# OAuth2 scheme for token validation
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="oauth/token")

oauth_router = APIRouter()


@oauth_router.post("/register", response_model=ClientRegistrationResponse)
async def register_client(
    request: ClientRegistrationRequest, db: Session = Depends(get_db)
):
    """
    Register a new OAuth client.

    This endpoint allows applications to register and obtain client credentials for OAuth2 authentication.

    **Example Request:**
    ```json
    {
        "redirect_uri": "http://localhost:3000/callback"
    }
    ```

    **Example Response:**
    ```json
    {
        "client_id": "client_xxxxx",
        "client_secret": "secret_xxxxx",
        "redirect_uri": "http://localhost:3000/callback"
    }
    ```

    **Security Note:** Store the client_secret securely as it's only returned once.
    """
    client_id = generate_client_id()
    client_secret = generate_client_secret()

    try:
        client = OAuthClient(
            client_id=client_id,
            client_secret=hash_secret(client_secret),
            redirect_uri=request.redirect_uri,
            created_at=datetime.now(),
        )

        db.add(client)
        db.commit()
        db.refresh(client)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register OAuth client",
        )

    return ClientRegistrationResponse(
        client_id=client_id,
        client_secret=client_secret,  # Return unhashed secret only once
        redirect_uri=request.redirect_uri,
    )


@oauth_router.get("/authorize")
async def authorize(
    client_id: str = Query(..., description="Client ID obtained from registration"),
    redirect_uri: str = Query(
        ..., description="Redirect URI registered with the client"
    ),
    response_type: str = Query(
        ..., description="Must be 'code' for authorization code flow"
    ),
    state: Optional[str] = Query(
        None, description="Optional state parameter for CSRF protection"
    ),
    code_challenge: Optional[str] = Query(
        None, description="PKCE code challenge (S256)"
    ),
    code_challenge_method: Optional[str] = Query(
        None, description="PKCE code challenge method (S256)"
    ),
    resource: Optional[str] = Query(
        None,
        description=f"Resource parameter (RFC 8707) - must be {settings.oauth_domain}/mcp",
    ),
    scope: Optional[str] = Query("mcp:read mcp:write", description="Requested scopes"),
    db: Session = Depends(get_db),
):
    """
    OAuth2 Authorization Endpoint.

    This endpoint initiates the OAuth2 authorization code flow. In a real application,
    this would show a user consent screen. For this demo, we auto-approve and redirect
    with an authorization code.

    **Parameters:**
    - `client_id`: The client identifier obtained from registration
    - `redirect_uri`: Must match the URI registered with the client
    - `response_type`: Must be "code" for authorization code flow
    - `state`: Optional parameter for CSRF protection
    - `code_challenge`: PKCE code challenge (recommended for security)
    - `code_challenge_method`: Must be "S256" for SHA256
    - `resource`: Resource identifier (RFC 8707) - must be "{settings.oauth_domain}/mcp"
    - `scope`: Requested permissions (default: "mcp:read mcp:write")

    **Example URL:**
    ```
    {settings.oauth_domain}/oauth/authorize?client_id=client_xxxxx&redirect_uri=http://localhost:3000/callback&response_type=code&state=random_state&code_challenge=challenge&code_challenge_method=S256&resource={settings.oauth_domain}/mcp
    ```

    **Response:** Redirects to the redirect_uri with authorization code:
    ```
    http://localhost:3000/callback?code=AUTHORIZATION_CODE&state=random_state
    """
    # Validate response_type
    if response_type != "code":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported response_type. Must be 'code'",
        )

    # Validate client
    client = db.query(OAuthClient).filter(OAuthClient.client_id == client_id).first()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid client_id",
        )

    # Validate redirect_uri
    if client.redirect_uri != redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid redirect_uri",
        )

    # Validate PKCE parameters if provided
    if code_challenge and code_challenge_method != "S256":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported code challenge method. Only S256 is supported.",
        )

    # Validate resource parameter (RFC 8707)
    if resource and resource != f"{settings.oauth_domain}/mcp":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid resource parameter. Must be {settings.oauth_domain}/mcp",
        )

    # Generate authorization code
    try:
        code = generate_authorization_code()
        auth_code = AuthorizationCode(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            expires_at=datetime.now() + timedelta(minutes=10),
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method or "S256"
            if code_challenge
            else None,
            resource=resource,
            scope=scope,
        )

        db.add(auth_code)
        db.commit()
        db.refresh(auth_code)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create authorization code",
        )

    # Build redirect URL
    redirect_url = f"{redirect_uri}?code={code}"
    if state:
        redirect_url += f"&state={state}"

    return RedirectResponse(url=redirect_url)


@oauth_router.post("/token", response_model=TokenResponse)
async def token(form_data: TokenRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    OAuth token endpoint.

    Exchange an authorization code for an access token.
    """
    # Validate grant_type
    if form_data.grant_type != "authorization_code":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported grant_type. Must be 'authorization_code'",
        )

    # Validate client credentials
    client = (
        db.query(OAuthClient)
        .filter(OAuthClient.client_id == form_data.client_id)
        .first()
    )
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
        )

    if client.client_secret != hash_secret(form_data.client_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
        )

    # Validate authorization code
    auth_code = (
        db.query(AuthorizationCode)
        .filter(AuthorizationCode.code == form_data.code)
        .first()
    )
    if not auth_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid authorization code",
        )

    # Check if code is expired
    if datetime.now() > auth_code.expires_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code expired",
        )

    # Check if code was already used
    if auth_code.used:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code already used",
        )

    # Validate code belongs to client
    if auth_code.client_id != form_data.client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code does not belong to this client",
        )

    # Validate redirect_uri
    if auth_code.redirect_uri != form_data.redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid redirect_uri",
        )

    # Validate PKCE if the authorization request included PKCE
    if auth_code.code_challenge:
        if not form_data.code_verifier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="code_verifier is required for PKCE",
            )

        if not verify_code_challenge(form_data.code_verifier, auth_code.code_challenge):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid code_verifier",
            )

    # Validate resource parameter (RFC 8707)
    if form_data.resource and form_data.resource != f"{settings.oauth_domain}/mcp":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid resource parameter. Must be {settings.oauth_domain}/mcp",
        )

    try:
        # Mark code as used
        auth_code.used = True
        db.commit()

        # Generate JWT access token
        token_data = {
            "client_id": form_data.client_id,
            "scope": "mcp:read mcp:write",
            "sub": form_data.client_id,  # Subject (client ID)
        }
        expires_delta = timedelta(minutes=settings.jwt_access_token_expire_hours)
        token_value = create_access_token(data=token_data, expires_delta=expires_delta)

        # Store access token in database
        access_token = AccessToken(
            token=token_value,
            client_id=form_data.client_id,
            expires_at=datetime.now() + expires_delta,
            scope="mcp:read mcp:write",
        )
        db.add(access_token)
        db.commit()
        db.refresh(access_token)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create access token",
        )

    return TokenResponse(
        access_token=token_value,
        expires_in=settings.jwt_access_token_expire_hours * 60,  # Convert to seconds
        scope=access_token.scope,
    )
