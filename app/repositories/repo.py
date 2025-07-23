from abc import abstractmethod
from typing import Any, List, Protocol

from models import Repo


class IRepoStore(Protocol):
    @abstractmethod
    async def get_all_by_user(
        self, user_id: str, offset: int, limit: int
    ) -> List[Any]: ...

    @abstractmethod
    async def count_by_user(self, user_id: str) -> int: ...

    @abstractmethod
    async def create_new_repo(self, repo_model: Any) -> Any: ...

    @abstractmethod
    async def find_repo_by_user_relative_path(self, user_id: str, relative_path: str) -> Repo: ...

class TortoiseRepoStore(IRepoStore):

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

    async def get_all_by_user(
        self, user_id: str, offset: int, limit: int
    ) -> List[Repo]:
        return (
            await Repo.filter(user_id=user_id)
            .order_by("-created_at")
            .offset(offset)
            .limit(limit)
            .all()
        )


    async def count_by_user(self, user_id: str) -> int:
        return await Repo.filter(user_id=user_id).count()

    async def create_new_repo(self, repo_model: Repo) -> Repo:
        await repo_model.save(force_create=True)
        await repo_model.refresh_from_db()
        return repo_model

    async def find_repo_by_user_relative_path(
        self, user_id: str, relative_path: str
    ) -> Repo:
        return await Repo.filter(user_id=user_id, relative_path=relative_path).first()