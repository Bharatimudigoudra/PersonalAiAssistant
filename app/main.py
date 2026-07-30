"""
Application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import PersonalAIException
from app.core.handlers import application_exception_handler
from app.core.logging import logger
from app.core.middleware import RequestLoggingMiddleware


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Offline Personal AI Assistant",
)

# ---------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------

app.add_exception_handler(
    PersonalAIException,
    application_exception_handler,
)

# ---------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Restrict this in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    RequestLoggingMiddleware,
)

# ---------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------

app.include_router(api_router)

# ---------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    """
    Application startup.
    """

    logger.info("=" * 60)
    logger.info("Starting {}", settings.APP_NAME)
    logger.info("Version : {}", settings.APP_VERSION)
    logger.info("Environment : {}", settings.DEBUG)
    logger.info("Model : {}", settings.MODEL_NAME)
    logger.info("Ollama : {}", settings.OLLAMA_BASE_URL)
    logger.info("Host : {}:{}", settings.HOST, settings.PORT)
    logger.info("=" * 60)


# ---------------------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------------------


@app.get(
    "/",
    tags=["Root"],
)
def root() -> dict:
    """
    Health endpoint.
    """

    return {
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "Running",
    }