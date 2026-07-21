from fastapi import FastAPI

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


# Register global exception handlers
app.add_exception_handler(
    PersonalAIException,
    application_exception_handler,
)

app.add_middleware(
    RequestLoggingMiddleware
)

# Register API routes
app.include_router(api_router)


@app.on_event("startup")
async def startup_event() -> None:
    """
    Startup event.
    """

    logger.info("=" * 60)
    logger.info("Starting {}", settings.APP_NAME)
    logger.info("Version : {}", settings.APP_VERSION)
    logger.info("Model   : {}", settings.MODEL_NAME)
    logger.info("Ollama  : {}", settings.OLLAMA_BASE_URL)
    logger.info("=" * 60)


@app.get(
    "/",
    tags=["Root"],
)
def root():
    """
    Root endpoint.
    """

    return {
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "Running",
    }