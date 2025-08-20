"""
Authentication utility for the DevDox AI Agent.
"""

import hashlib
import logging
import secrets
import string
from dataclasses import dataclass
from typing import  Annotated, Dict, Optional, Protocol

from fastapi import Depends, Request, HTTPException
from fastapi.security import HTTPBearer, APIKeyHeader
from pydantic import BaseModel, ConfigDict

from app.exceptions.custom_exceptions import UnauthorizedAccess
from app.exceptions.exception_constants import (
    NO_API_KEY_PROVIDED,
    INVALID_API_KEY,
    API_KEY_IS_INACTIVE

)

from app.repositories.api_key import TortoiseApiKeyStore
from models_src.repositories.user import TortoiseUserStore


http_bearer_security_schema = HTTPBearer(auto_error=False)

logger = logging.getLogger(__name__)


class UserClaims(BaseModel):
    sub: str
    email: Optional[str] = None
    name: Optional[str] = None

    model_config = ConfigDict(extra="ignore")

api_key_header = APIKeyHeader(
    name="API-KEY",
    description="API Key for authentication"
)

@dataclass
class APIKeyManagerReturn:
    plain: str
    hashed: str
    masked: str


class IUserAuthenticator(Protocol):
    async def authenticate(self, request: Request, api_key: str) -> UserClaims: ...


class APIKeyAuthenticator(IUserAuthenticator):
    """API key based authentication"""

    def __init__(self, api_key_store: TortoiseApiKeyStore):
        self.api_key_store = api_key_store
        self.user_store = TortoiseUserStore()

    async def authenticate(self, request: Request, api_key: str) -> UserClaims:
        """
        Authenticate user using API key

        Args:
            request: FastAPI request object
            token: The API key  from the Authorization header

        Returns:
            UserClaims: User claims if authentication successful

        Raises:
            UnauthorizedAccess: If authentication fails
        """
        if not api_key:
            raise UnauthorizedAccess(
                log_message=NO_API_KEY_PROVIDED
            )

        # Hash the provided API key
        hashed_key = APIKeyManager.hash_key(api_key)

        try:
            # Query the database for the API key
            api_key_record = await self.api_key_store.get_api_key_by_hash(hashed_key)


            if not api_key_record:
                raise UnauthorizedAccess(
                    log_message=INVALID_API_KEY
                )

            # Check if API key is active (assuming you have this field)
            if hasattr(api_key_record, 'is_active') and not api_key_record.is_active:
                raise UnauthorizedAccess(
                    log_message=API_KEY_IS_INACTIVE
                )
            user_record = await self.user_store.find_by_user_id(user_id=api_key_record.user_id)
            if not user_record:
                raise UnauthorizedAccess(
                    log_message="User not found"
                )
            # Get user information from the API key record
            # You might need to join with user table or have user info in the API key record
            user_claims = UserClaims(
                sub=api_key_record.user_id,
                email=getattr(user_record, 'email', None),
                name=getattr(user_record, 'name', None)
            )

            logger.info(f"Successfully authenticated user {user_claims.sub} with API key {api_key_record.masked_api_key}")
            return user_claims

        except Exception as e:
            if isinstance(e, UnauthorizedAccess):
                raise

            logger.error(f"Error during API key authentication: {str(e)}")
            raise UnauthorizedAccess(
                log_message=f"Authentication failed: {str(e)}"
            )


class APIKeyManager:
    DEFAULT_MAX_KEY_LENGTH = 32
    DEFAULT_PREFIX = "dvd_"

    def __init__(self, api_key_store: TortoiseApiKeyStore):
        self.api_key_store = api_key_store

    @staticmethod
    def hash_key(unhashed_api_key: str) -> str:
        return hashlib.sha256(unhashed_api_key.encode("utf-8")).hexdigest()

    @staticmethod
    def __generate_plain_key(
        prefix: str = DEFAULT_PREFIX, length: int = DEFAULT_MAX_KEY_LENGTH
    ) -> str:
        chars = string.ascii_letters + string.digits
        random_part = "".join(
            secrets.choice(chars) for _ in range(length - len(prefix))
        )
        return prefix + random_part

    async def generate_unique_api_key(
        self,
        prefix: str = DEFAULT_PREFIX,
        length: int = DEFAULT_MAX_KEY_LENGTH,
    ) -> Optional[APIKeyManagerReturn]:

        plain_key = self.__generate_plain_key(prefix=prefix, length=length)
        hashed_key = self.hash_key(plain_key)

        # Query existing hashes in DB
        key_exists = await self.api_key_store.query_for_existing_hashes(hashed_key)

        if key_exists:
            return None



        return APIKeyManagerReturn(
            plain=plain_key, hashed=hashed_key
        )


class PostApiKeyService:
    def __init__(
        self,
        api_key_store: TortoiseApiKeyStore,
        api_key_manager: APIKeyManager,
    ):
        self.api_key_store = api_key_store
        self.api_key_manager = api_key_manager

    @classmethod
    def with_dependency(
        cls,
        api_key_store: Annotated[TortoiseApiKeyStore, Depends()],
    ) -> "PostApiKeyService":

        api_key_manager = APIKeyManager(api_key_store=api_key_store)

        return cls(
            api_key_store=api_key_store,
            api_key_manager=api_key_manager,
        )


def get_user_authenticator_dependency(
    api_key_store: Annotated[TortoiseApiKeyStore, Depends()]
) -> IUserAuthenticator:
    """
    Returns the appropriate authenticator based on configuration.
    Defaults to API key authentication.
    """
    # For now, defaulting to API key authentication

    return APIKeyAuthenticator(api_key_store)


async def get_authenticated_user(
    request: Request,
    api_key_header: Annotated[str, Depends(api_key_header)],
    authenticator: IUserAuthenticator = Depends(get_user_authenticator_dependency),
) -> UserClaims:
    """
    Authenticate user using API key from API-KEY header
    """
    if not api_key_header:
        raise UnauthorizedAccess(
            reason=INVALID_API_KEY,
            log_message="Missing or empty API-KEY header"
        )

    try:
        # Pass the API key directly to the authenticator
        return await authenticator.authenticate(request, api_key_header)
    except Exception as e:
        raise UnauthorizedAccess(
            reason=INVALID_API_KEY,
            log_message=f"Authentication failed: {str(e)}"
        )





class MCPUserContextManager:
    def __init__(self):
        self._api_key_to_user: Dict[str, UserClaims] = {}
        self._session_to_user: Dict[str, UserClaims] = {}

    def store_user_for_api_key(self, api_key: str, user_claims: UserClaims):
        """Store user context by API key"""
        self._api_key_to_user[api_key] = user_claims
        logger.info(f"Stored user context: {user_claims.sub}")

    def get_most_recent_user(self) -> Optional[UserClaims]:
        """Get the most recently stored user"""
        if self._api_key_to_user:
            return list(self._api_key_to_user.values())[-1]
        return None

mcp_context = MCPUserContextManager()


def mcp_auth_interceptor(
        request: Request,
        user_claims: UserClaims = Depends(get_authenticated_user)
) -> UserClaims:
    """Store user context when MCP authenticates"""
    api_key = request.headers.get("api-key") or request.headers.get("API-KEY")
    if api_key:
        mcp_context.store_user_for_api_key(api_key, user_claims)

    logger.info(f"MCP intercepted auth for user: {user_claims.sub}")
    return user_claims


async def get_mcp_aware_user_context(request: Request) -> UserClaims:
    """Get user context for both MCP and regular requests"""
    host = request.headers.get("host", "")
    api_key_header = request.headers.get("api-key") or request.headers.get("API-KEY")
    if "apiserver" in host:
        # MCP internal request - get stored user context
        user = mcp_context.get_most_recent_user()
        if user:
            logger.info(f"Using stored user context for MCP: {user.sub}")
            return user
        else:
            raise HTTPException(status_code=500, detail="No user context found")
    else:
        # Regular API request
        return await get_authenticated_user(request, api_key_header)