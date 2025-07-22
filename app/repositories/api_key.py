import uuid
from abc import abstractmethod
from typing import Any, List, Protocol, Optional

from models import APIKEY



class IApiKeyStore(Protocol):

    @abstractmethod
    async def query_for_existing_hashes(self, hash_key: str) -> bool: ...


    @abstractmethod
    async def set_inactive_by_user_id_and_api_key_id(
        self, user_id, api_key_id
    ) -> int: ...

    @abstractmethod
    async def get_all_api_keys(self, user_id) -> List[Any]: ...


class TortoiseApiKeyStore(IApiKeyStore):

    def __init__(self):
        """
        Have to add this as an empty __init__ to override it, because when using it with Depends(),
        FastAPI dependency mechanism will automatically assume its
        ```
        def __init__(self, *args, **kwargs):
            pass
        ```
        Causing unneeded behavior.
        """
        pass

    async def get_api_key_by_hash(self, hashed_key: str) -> Optional[Any]:
        """
        Retrieve API key record by its hashed value

        Args:
            hashed_key: The SHA256 hash of the API key

        Returns:
            API key record if found, None otherwise
        """
        return (
            await APIKEY.filter(api_key=hashed_key ,is_active=True)
            .first()
        )
        pass

    async def query_for_existing_hashes(self, hash_key: str) -> bool:

        if not hash_key or not hash_key.strip():
            return False

        return await APIKEY.filter(api_key=hash_key).exists()

    async def set_inactive_by_user_id_and_api_key_id(
        self, user_id: str, api_key_id: uuid.UUID
    ) -> int:
        if (not user_id or not user_id.strip()) or not api_key_id:
            return -1

        return await APIKEY.filter(
            user_id=user_id, id=api_key_id, is_active=True
        ).update(is_active=False)

    async def get_all_api_keys(self, user_id: str) -> List[APIKEY]:

        if not user_id or not user_id.strip():
            return []

        return (
            await APIKEY.filter(user_id=user_id, is_active=True)
            .order_by("-created_at")
            .all()
        )
