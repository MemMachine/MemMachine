"""
Authentication API routes.
"""

import logging

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials

from ..auth.token import get_token_info, issue_token, revoke_token
from ..models import User
from ..schemas import (
    LoginResponseData,
    RegistrationResponseData,
    ResponseSchema,
    UserInfo,
    UserInfoResponseData,
    UserLogin,
    UserRegistration,
)
from ..services.user import create_user, get_current_user, security
from ..settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/register")
async def register(
    registration: UserRegistration,
) -> ResponseSchema:
    """Register a new user."""
    # Validate the invitation code
    if registration.invitationCode != "MemVerge!666":
        return ResponseSchema.error_response(message="Invalid invitation code")

    user = create_user(
        username=registration.username,
        password=registration.password,
        email=registration.email,
    )

    if not user:
        return ResponseSchema.error_response(
            message="User registration failed. Username or email may already exist."
        )

    # Issue token for the newly registered user
    issued = issue_token(subject=user.username, ttl_seconds=settings.token_ttl_seconds)

    registration_data = RegistrationResponseData(
        user=UserInfo(
            id=user.id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            created_at=user.created_at.isoformat(),
            updated_at=user.updated_at.isoformat(),
        ),
        token=issued["token"],
        expires_at=issued["expires_at"],
    )
    return ResponseSchema.success_response(
        data=registration_data.model_dump(), message="User registered successfully"
    )


@router.post("/login")
async def login(credentials: UserLogin) -> ResponseSchema:
    """Login with username/password to get a bearer token."""
    from ..services.user import authenticate_user

    # First try database authentication
    user = authenticate_user(credentials.username, credentials.password)

    if user:
        # Database user authenticated
        issued = issue_token(
            subject=user.username, ttl_seconds=settings.token_ttl_seconds
        )
        login_data = LoginResponseData(
            user=UserInfo(
                id=user.id,
                username=user.username,
                email=user.email,
                is_active=user.is_active,
                created_at=user.created_at.isoformat(),
                updated_at=user.updated_at.isoformat(),
            ),
            token=issued["token"],
            expires_at=issued["expires_at"],
        )
        return ResponseSchema.success_response(
            data=login_data.model_dump(), message="Login successful"
        )

    return ResponseSchema.error_response(message="Invalid credentials")


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> ResponseSchema:
    """Logout by revoking the provided token."""
    # Get token from credentials
    provided_token = credentials.credentials
    if not provided_token:
        return ResponseSchema.error_response(message="Missing token")
    revoked = revoke_token(provided_token)
    if not revoked:
        return ResponseSchema.error_response(
            message="Token not found or already revoked"
        )
    return ResponseSchema.success_response(message="Logout successful")


@router.get("/me")
async def get_current_user_info(
    request: Request,
    current_user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> ResponseSchema:
    """Get current authenticated user information."""
    try:
        # Get token from credentials
        provided_token = credentials.credentials

        # Get token information including expiration
        token_info = get_token_info(provided_token)

        # This should not happen since middleware validates the token
        if not token_info:
            return ResponseSchema.error_response(
                message="Unable to retrieve token information"
            )

        user_info_data = UserInfoResponseData(
            user=UserInfo(
                id=current_user.id,
                username=current_user.username,
                email=current_user.email,
                is_active=current_user.is_active,
                created_at=current_user.created_at.isoformat(),
                updated_at=current_user.updated_at.isoformat(),
            ),
            token=token_info["token"],
            expires_at=token_info["expires_at"],
        )
        return ResponseSchema.success_response(
            data=user_info_data.model_dump(),
            message="User information retrieved successfully",
        )
    except Exception as e:
        return ResponseSchema.error_response(message=f"Internal error: {str(e)}")
