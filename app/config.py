"""
Configuration settings for the DevDox AI Portal API.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic_settings import BaseSettings


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
    SUPABASE_PORT: int = 5432
    SUPABASE_DB_NAME: str = "postgres"

    DB_MIN_CONNECTIONS: int = 1
    DB_MAX_CONNECTIONS: int = 10

    TOGETHER_API_KEY: str = "test-clerk-key"

    COHERE_API_KEY:str = "test-cohere-key"

    LOG_DIR:str = "/app/logs"

    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Version
    VERSION: str = "0.1.1"

    class Config:
        """Pydantic config class."""

        env_file = "app/instance/.env"
        case_sensitive = True
        git_hosting: Optional[GitHosting] = None
        extra = "ignore"


# Initialize settings instance
settings = Settings()


def get_database_config() -> Dict[str, Any]:
    """
     Returns the appropriate database configuration based on available credentials.
    Uses REST API connection when SUPABASE_REST_API is True, otherwise uses direct PostgreSQL.
    """
    base_credentials = {
        "minsize": settings.DB_MIN_CONNECTIONS,
        "maxsize": settings.DB_MAX_CONNECTIONS,
        "ssl": "require",
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
        credentials = {
            **base_credentials,
            "host": project_id,  # Use project_id directly as host
            "port": 5432,
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
                    "models",
                    "aerich.models",  # Required for aerich migrations
                ],
                "default_connection": "default",
            }
        },
        "use_tz": False,
        "timezone": "UTC",
    }


TORTOISE_ORM = get_tortoise_config()
