"""
OAuth schemas
"""
from typing import Optional

from fastapi import Form
from pydantic import BaseModel


class ClientRegistrationRequest(BaseModel):
    """Request model for client registration."""
    redirect_uri: str


class ClientRegistrationResponse(BaseModel):
    """Response model for client registration."""
    client_id: str
    client_secret: str
    redirect_uri: str


# class TokenRequest(BaseModel):
#     """Request model for token exchange."""
#     grant_type: str
#     code: str
#     client_id: str
#     client_secret: str
#     redirect_uri: str
#     code_verifier: Optional[str] = None
#     resource: Optional[str] = None


class TokenRequestForm:
    """Form-based request model for OAuth2 token endpoint (RFC 6749 compliant)."""

    def __init__(
        self,
        grant_type: str = Form(..., description="OAuth2 grant type"),
        code: str = Form(..., description="Authorization code"),
        client_id: str = Form(..., description="Client identifier"),
        client_secret: str = Form(..., description="Client secret"),
        redirect_uri: str = Form(..., description="Redirect URI"),
        code_verifier: Optional[str] = Form(None, description="PKCE code verifier"),
        resource: Optional[str] = Form(None, description="Resource identifier (RFC 8707)"),
    ):
        self.grant_type = grant_type
        self.code = code
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.code_verifier = code_verifier
        self.resource = resource


class TokenResponse(BaseModel):
    """Response model for token exchange."""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    scope: str

