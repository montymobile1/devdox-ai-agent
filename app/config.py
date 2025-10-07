"""
Configuration settings for the DevDox AI Portal API.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, ClassVar

from pydantic_settings import BaseSettings
from app.infrastructure.supabase_queue import SupabaseQueue

class GitHosting(str, Enum):
    GITLAB = "gitlab"
    GITHUB = "github"


class Settings(BaseSettings):
    """Application settings."""

    # API configuration
    API_ENV: Literal["development", "staging", "production", "test", "local"] = (
        "development"
    )
    API_DEBUG: bool = True
    SECRET_KEY: str = "testtesttesttesttesttesttesttest"  # Only for local/testing
    LOG_LEVEL:str = "INFO"
    # SUPABASE VAULT
    SUPABASE_VAULT_ENABLED: bool = True
    # SUPABASE settings
    SUPABASE_URL: str = "https://your-project.supabase.co"
    SUPABASE_SECRET_KEY: str = "test-supabase-key"

    SUPABASE_REST_API: bool = True
    VAULT_KEYS: str = ""

    SUPABASE_HOST: str = "https://localhost"
    SUPABASE_USER: str = "postgres"
    SUPABASE_PASSWORD: str = "test"
    SUPABASE_QUEUE_PORT:int = 5432
    SUPABASE_PORT: int = 6543
    SUPABASE_DB_NAME: str = "postgres"

    DB_MIN_CONNECTIONS: int = 5
    DB_MAX_CONNECTIONS: int = 25

    DB_CONNECT_TIMEOUT: float = 30.0
    DB_COMMAND_TIMEOUT: float = 30.0
    DB_STATEMENT_TIMEOUT: int = 30000  # milliseconds
    DB_IDLE_SESSION_TIMEOUT: int = 300000  # 5 minutes in milliseconds

    # Connection lifecycle settings
    DB_MAX_CACHED_STATEMENT_LIFETIME: int = 300  # 5 minutes
    DB_MAX_CACHEABLE_STATEMENT_SIZE: int = 1024 * 15  # 15KB

    DB_SCHEMA:str="public"


    VECTOR_SIZE:int = 768

    QUEUE_POLLING_INTERVAL_SECONDS: int = 10

    TOGETHER_API_KEY: str = "test-clerk-key"

    COHERE_API_KEY:str = "test-cohere-key"

    MAX_QUESTIONS:int = 5

    LOG_DIR:str = "/app/logs"
    BASE_DIR: ClassVar[str] = "app/repos"

    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 9002

    # Version
    VERSION: str = "0.1.1"

    class Config:
        """Pydantic config class."""
        env_file = str(Path(__file__).resolve().parent / "instance" / ".env")
        case_sensitive = True
        git_hosting: Optional[GitHosting] = None
        extra = "ignore"


# Initialize settings instance
settings = Settings()

# Initialize Supabase queue
supabase_queue = SupabaseQueue(
    host=settings.SUPABASE_HOST,
    port=settings.SUPABASE_QUEUE_PORT,
    user=settings.SUPABASE_USER,
    password=settings.SUPABASE_PASSWORD,
    db_name=settings.SUPABASE_DB_NAME,
    table_name="testing"
)


def get_database_config() -> Dict[str, Any]:
    """
     Returns the appropriate database configuration based on available credentials.
    Uses REST API connection when SUPABASE_REST_API is True, otherwise uses direct PostgreSQL.
    """
    base_credentials = {
        "min_size": settings.DB_MIN_CONNECTIONS,
        "max_size": settings.DB_MAX_CONNECTIONS,
        "ssl": "require",
        "statement_cache_size":0,
        "server_settings": {
            "search_path": settings.DB_SCHEMA,
            "statement_timeout": str(settings.DB_STATEMENT_TIMEOUT),  # Prevent runaway queries
            "idle_in_transaction_session_timeout": str(settings.DB_IDLE_SESSION_TIMEOUT),
            "application_name": "devdox-ai-agent"
        },

        "timeout": settings.DB_CONNECT_TIMEOUT,  # Connection establishment timeout (seconds)
        "command_timeout": settings.DB_COMMAND_TIMEOUT,  # Query execution timeout (seconds)
        "max_inactive_connection_lifetime": 3600,  # 1 hour - recycle old connections
        "max_queries": 50000,  # Queries before connection recycling

    }
    # Check if developer wants to use RESTAPI
    if settings.SUPABASE_REST_API:


        # Extract database connection info from Supabase URL
        # Supabase URL format: https://your-project.supabase.co
        if not settings.SUPABASE_URL.startswith(
            "https://"
        ) or not settings.SUPABASE_URL.endswith(".supabase.co"):
            raise ValueError(f"Invalid Supabase URL format: {settings.SUPABASE_URL}")

        project_id = settings.SUPABASE_URL.replace("https://", "").replace(
            ".supabase.co", ""
        )
        if not project_id:
            raise ValueError("Unable to extract project ID from Supabase URL")
        pooler_host = f"db.{project_id}.supabase.co"

        credentials = {
            **base_credentials,
            "host": pooler_host,  # Use project_id directly as host
            "port": settings.SUPABASE_PORT,
            "user": "postgres",
            "password": settings.SUPABASE_SECRET_KEY,
            "database": "postgres",

        }

    # Method 2: Supabase postgress sql
    else:
        credentials = {
            **base_credentials,
            "host": settings.SUPABASE_HOST,
            "port": settings.SUPABASE_PORT,
            "user": settings.SUPABASE_USER,
            "password": settings.SUPABASE_PASSWORD,
            "database": settings.SUPABASE_DB_NAME,
        }

    return {"engine": "tortoise.backends.asyncpg", "credentials": credentials}


def get_tortoise_config():
    db_config = get_database_config()
    # Add server_settings to the credentials

    return {
        "connections": {"default": db_config},
        "apps": {
            "models": {
                "models": [
                    "models_src.models",
                    "aerich.models",  # Required for aerich migrations
                ],
                "default_connection": "default",
            }
        },
        "use_tz": False,
        "timezone": "UTC",
    }


TORTOISE_ORM = get_tortoise_config()
