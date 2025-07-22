"""
Custom exception base class for the DevDox AI Portal API.

This module defines `DevDoxAPIException`, the base exception from which all
application-specific errors should inherit.

Key responsibilities:
- Clean separation of user-facing messages vs. developer/debugging logs
- Standard structure for error codes, contexts, and HTTP status mapping
- Centralized support for structured API responses and log formatting

To create a new custom exception:
1. Subclass `DevDoxAPIException`
2. Optionally override `http_status`
3. Pass `user_message`, `log_message`, `error_type`, etc.

Example:
    class RepoConflictError(DevDoxAPIException):
        http_status = 409
        def __init__(self, repo_id: str):
            super().__init__(
                user_message="Repository conflict detected.",
                log_message=f"Conflict when syncing repo ID: {repo_id}",
                error_type="REPO_CONFLICT",
                internal_context={"repo_id": repo_id},
            )

These exceptions are globally handled and logged by the `devdox_exception_handler`
defined in `handlers.py`, and registered into the FastAPI app via `register.py`.
"""

from typing import Any, Dict, Optional

from starlette import status

from app.exceptions.exception_constants import (
    AUTH_FAILED,
    GENERIC_BAD_REQUEST,
    GENERIC_RESOURCE_NOT_FOUND,
)


class DevDoxAPIException(Exception):
    """
    Base exception class for DevDox AI Portal API.

    This exception is designed to cleanly separate:
    - What is returned to the **user**
    - What is logged for **developers**

    example custom exception:
    ```python
            class RepositorySyncFailed(DevDoxAPIException):
                http_status = 502

                def __init__(self, repo_id: str, provider: str, cause: Exception):
                    super().__init__(
                        user_message="Repository sync failed. Please try again later.",
                        log_message=f"Sync failed for repo `{repo_id}` on provider `{provider}`",
                        error_type="REPO_SYNC_FAILED",
                        public_context=None,
                        internal_context={"repo_id": repo_id, "provider": provider},
                        root_exception=cause,
                    )
    ```
    """

    http_status = (
        status.HTTP_500_INTERNAL_SERVER_ERROR
    )  # Subclasses may override this default

    def __init__(
        self,
        *,
        user_message: str,
        log_message: Optional[str] = None,
        error_type: Optional[str] = None,
        public_context: Optional[Dict[str, Any]] = None,
        internal_context: Optional[Dict[str, Any]] = None,
        root_exception: Optional[Exception] = None,
        http_status_override: Optional[int] = None,
        log_level: Optional[str] = None,
    ):
        """
        Args:
                user_message: Safe, generic message returned to the user.
                public_context: Optional context to return in the API response (e.g., {"quota": "exceeded"}).

                log_message: Detailed internal message for logs/debugging.
                internal_context: Optional context to include in logs only (e.g., {"repo_id": 123}).

                error_type: Optional machine-readable code.
                root_exception: Original exception (for traceback chaining).
                http_status_override: Override the default HTTP status for this exception.
        """
        super().__init__(user_message)

        self.user_message = user_message
        self.log_message = log_message or user_message
        self.error_type = error_type or self.__class__.__name__.upper()
        self.public_context = public_context or {}
        self.internal_context = internal_context or {}
        self.root_exception = root_exception
        self.http_status = http_status_override or self.http_status
        self.log_level = log_level.lower() if log_level else "warning"

    def __str__(self):
        return f"[{self.error_type}] {self.user_message}"


class UnauthorizedAccess(DevDoxAPIException):
    http_status = status.HTTP_401_UNAUTHORIZED

    def __init__(self, reason=AUTH_FAILED, log_message=None, log_level=None):
        super().__init__(
            user_message=reason, log_message=log_message, log_level=log_level
        )


class BadRequest(DevDoxAPIException):
    http_status = status.HTTP_400_BAD_REQUEST

    def __init__(self, reason=GENERIC_BAD_REQUEST, log_message: Optional[str] = None):
        super().__init__(user_message=reason, log_message=log_message)


class ResourceNotFound(DevDoxAPIException):
    http_status = status.HTTP_404_NOT_FOUND

    def __init__(self, reason=GENERIC_RESOURCE_NOT_FOUND):
        super().__init__(user_message=reason)
