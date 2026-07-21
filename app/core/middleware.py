"""
Application middleware.
"""

import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every incoming request.
    """

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):

        request_id = str(uuid.uuid4())[:8]

        start = time.perf_counter()

        logger.info("=" * 60)
        logger.info("Incoming Request")
        logger.info("Request ID : {}", request_id)
        logger.info("Method     : {}", request.method)
        logger.info("Path       : {}", request.url.path)
        logger.info("=" * 60)

        response = await call_next(request)

        elapsed = time.perf_counter() - start

        logger.info("Completed Request")
        logger.info("Request ID : {}", request_id)
        logger.info("Status     : {}", response.status_code)
        logger.info("Time       : %.2f sec", elapsed)
        logger.info("=" * 60)

        return response