import logging
from uuid import UUID
from abc import abstractmethod
from tortoise.connection import connections
from typing import Dict, List, Protocol, Union
from models.code_chunks import CodeChunks

class ICodeChunksStore(Protocol):
    @abstractmethod
    async def get_user_repo_chunks(
        self, user_id: Union[str, UUID] , repo_id: Union[str, UUID],question:str, limit: int = None
    ) -> List[Dict]: ...

    @abstractmethod
    async def similarity_search(self, embedding:str, user_id : Union[str, UUID] , repo_id: Union[str, UUID],  limit: int = None,  threshold: float = 0.8,): ...

    @abstractmethod
    async def get_repo_file_chunks(self, user_id : Union[str, UUID] , repo_id: Union[str, UUID],  file_name:str): ...

class TortoiseCodeChunksStore(ICodeChunksStore):

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


    async def get_user_repo_chunks(
            self,user_id: Union[str, UUID] , repo_id: Union[str, UUID], query_embedding: List[float], limit: int = None
    ) -> List[Dict]:
        if not repo_id:
            return []

        # Convert the embedding list to a proper format for the database query
        embedding_str = f"[{','.join(map(str, query_embedding[0]))}]"
        return await  self.similarity_search(embedding_str, user_id, repo_id,user_id, limit)


    async def similarity_search(self, embedding:str, user_id : Union[str, UUID] , repo_id: Union[str, UUID],  limit: int = None,  threshold: float = 0.8,):
        """Perform semantic similarity search"""
        try:
            query = """
                    SELECT *, 1 - (embedding <=> $1) AS score
                    FROM code_chunks
                    ORDER BY score DESC LIMIT 5 \
                    """
            connection = connections.get("default")

            result = await connection.execute_query_dict(query, [embedding])


            return result
        except Exception as e:
            logging.error(f"Similarity search failed: {e}")
            return []  # Return empty list on error


    async def get_repo_file_chunks(self,  user_id : Union[str, UUID] , repo_id: Union[str, UUID],  file_name:str="readme"):
        """Return chunks of a specific file"""
        try:
            result = await CodeChunks.filter( file_name__icontains=file_name,  user_id=user_id, repo_id=repo_id).order_by("-created_at").values("content")


            return result
        except Exception as e:

            return []  # Return empty list on error


