"""
Exception handling module for the DevDox AI Portal API.

This module contains all application-wide exception handlers, including custom handlers
(e.g. for domain-specific exceptions) and the fallback handler for unhandled errors.

To keep the main application setup clean, all exception handlers defined here are registered
via the centralized `register.py` module. That module exposes a `register_exception_handlers(app)`
function which imports this file and binds each handler to the FastAPI app instance.

If you are adding a new custom exception and want it globally handled:
1. Define its handler function here.
2. Import and register it inside `register_exception_handlers()` in `register.py`.

This ensures a clean separation of concerns:
- `handlers.py` = defines the logic of how to respond to errors
- `register.py` = plugs that logic into the FastAPI lifecycle
"""

import logging

from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse

import app.exceptions.exception_constants
from app.config import settings
from app.exceptions.custom_exceptions import DevDoxAPIException
from app.utils.api_response import APIResponse

logger = logging.getLogger(__name__)

generic_exception_handler_status_code = status.HTTP_503_SERVICE_UNAVAILABLE


def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all uncaught exceptions with standardized error response.

    Args:
        request: The incoming request that triggered the exception
        exc: The uncaught exception

    Returns:
        Standardized error response dictionary
    """

    path = request.url.path
    method = getattr(request, "method", None) or request.scope.get("method") or "HTTP"
    exc_type = type(exc).__name__
    status_code = generic_exception_handler_status_code

    logger.exception(
        "[UNHANDLED_EXCEPTION] %s occurred | Path: %s | Method: %s | Status: %s",
        exc_type,
        path,
        method,
        status_code,
    )

    debug_payload = None
    if settings.API_ENV in ["development", "test"]:
        debug_payload = {"exception": type(exc).__name__, "str": str(exc)}

    return APIResponse.error(
        message=app.exceptions.exception_constants.SERVICE_UNAVAILABLE,
        status_code=status_code,
        debug=debug_payload,
        error_type=exc_type,
    )


def devdox_base_exception_handler(
    request: Request, exc: DevDoxAPIException
) -> JSONResponse:
    """
    Handle DevDoxAPIException with structured logging and response.

    Args:
        request: The incoming request that triggered the exception
        exc: The DevDoxAPIException instance

    Returns:
        Standardized error response dictionary with optional debug info
    """
    path = request.url.path
    method = getattr(request, "method", None) or request.scope.get("method") or "HTTP"
    exc_error_type = exc.error_type

    log_parts = [
        f"[{exc_error_type}] {exc.log_message}",
        f"Path: {path}",
        f"Method: {method}",
        f"Http Status: {exc.http_status}",
    ]

    if exc.internal_context:
        log_parts.append(f"Context: {exc.internal_context}")

    # Combine all parts into the final message
    log_message = " | ".join(log_parts)

    # Log with traceback if root_exception is present
    if exc.log_level == "error":
        logger.error(log_message, exc_info=exc.root_exception or exc)
    elif exc.log_level == "exception":
        logger.exception(log_message, exc_info=exc.root_exception or exc)
    else:
        logger.warning(log_message)

    debug_payload = None
    if settings.API_ENV in ["development", "test"]:
        debug_payload = {"exception": type(exc).__name__, "str": str(exc)}

    return APIResponse.error(
        message=exc.user_message,
        status_code=exc.http_status,
        details=exc.public_context,
        debug=debug_payload,
        error_type=exc_error_type,
    )
