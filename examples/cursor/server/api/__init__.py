"""
API routes
"""

from .auth import router as auth_router
from .memory import router as memory_router
from .misc import router as misc_router
from .oauth import oauth_router
from .well_known import well_known_router

__all__ = ["auth_router", "memory_router", "misc_router", "well_known_router", "oauth_router"]