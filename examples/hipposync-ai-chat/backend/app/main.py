"""
 Added user routes for API keys, settings, and location management
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .auth import router as auth_router
from app.routes.chat import router as chat_router
from .routes.threads import router as threads_router
from .routes.projects import router as projects_router
from .routes.models import router as models_router
from app.routes import user 

app = FastAPI(title="MemoryChat")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[*settings.cors_origins] if isinstance(settings.cors_origins, list) else [settings.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router)
app.include_router(models_router)
app.include_router(projects_router)
app.include_router(threads_router)
app.include_router(chat_router)
app.include_router(user.router, prefix="/api")  

@app.get("/healthz")
def healthz():
    return {"status": "ok"}