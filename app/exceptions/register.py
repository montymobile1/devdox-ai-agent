"""
Exception registration module for the DevDox AI Portal API.

This module is responsible for binding all exception handlers defined in `handlers.py`
to the FastAPI application instance.

Use the `register_exception_handlers(app)` function during app initialization
to ensure that global and custom exception handlers are properly wired in.

This centralizes error handling configuration, keeping the main app setup clean
and ensuring that all exceptions are processed through the defined logic in `handlers.py`.

Example usage in your main application file:
    from app.exceptions.register import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)
"""

from fastapi import FastAPI

from app.exceptions.custom_exceptions import DevDoxAPIException
from app.exceptions.handlers import (
    devdox_base_exception_handler,
    generic_exception_handler,
)


def register_exception_handlers(app: FastAPI):
    """
    Register all exception handlers with the FastAPI application.

    Args:
        app: The FastAPI application instance to register handlers with.
    """

    app.add_exception_handler(Exception, generic_exception_handler)
    app.add_exception_handler(DevDoxAPIException, devdox_base_exception_handler)
