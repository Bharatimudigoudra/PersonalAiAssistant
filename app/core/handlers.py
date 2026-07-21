"""
Global exception handlers.
"""

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import PersonalAIException


async def application_exception_handler(
    request: Request,
    exc: PersonalAIException,
):
    """
    Handle application exceptions.
    """

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
            },
        },
    )