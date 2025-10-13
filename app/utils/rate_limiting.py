"""
Rate limiting utilities for DevDox AI Agent API.

Provides per-user rate limiting for authenticated requests based on api key.
Falls back to IP-based limiting for unauthenticated requests.
"""

from slowapi import Limiter
from fastapi import Request
import hashlib
import logging

logger = logging.getLogger(__name__)

# Rate limit presets for different endpoint types
class RateLimits:
    """Common rate limit configurations"""

    # Standard endpoints
    STANDARD = "100/minute"

    ANALYSIS = "10/minute"
    LOAD_TEST = "5/minute"

    # Quick operations
    HEALTH_CHECK = "1000/minute"


def get_client_ip(request: Request) -> str:
    """Get client IP, considering proxy headers safely."""

    # If behind trusted proxy, use X-Forwarded-For (first IP)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Take first IP (original client), validate it's an IP
        client_ip = forwarded_for.split(",")[0].strip()

        # Validate it's a real IP address
        import ipaddress
        try:
            ipaddress.ip_address(client_ip)
            return client_ip
        except ValueError:
            pass  # Fall through to direct connection IP

    # Fall back to direct connection
    return request.client.host if request.client else "unknown"

def get_user_identifier(request: Request) -> str:
    """
    Extract unique identifier for rate limiting.

    Priority:
    1. API key (if present)
    3. Client IP address (fallback)

    Args:
        request: FastAPI request object

    Returns:
        Unique identifier string for rate limiting
    """


    # Try to get API key from headers
    api_key = request.headers.get("api-key") or request.headers.get("API-KEY")
    if api_key:
        # Use first 16 chars of API key for identification
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        return f"apikey:{api_key_hash}"


    # Fall back to IP address
    client_ip = get_client_ip(request)
    logger.debug(f"Rate limiting by IP: {client_ip}")
    return f"ip:{client_ip}"


# Initialize rate limiter with user-based key function
limiter = Limiter(
    key_func=get_user_identifier,
    default_limits=[RateLimits.STANDARD],  # Default: 100 requests per minute
    enabled=True,
    headers_enabled=True,  # Add rate limit headers to responses
)

