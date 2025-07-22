from abc import abstractmethod
from typing import Any, Protocol

from models import User


class IUserStore(Protocol):

    @abstractmethod
    async def get_by_user_id(self, user_id: str) -> Any | None: ...


class TortoiseUserStore(IUserStore):

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

    async def get_by_user_id(self, user_id: str) -> User | None:
        if not user_id or not user_id.strip():
            return None

        return await User.filter(user_id=user_id).first()
