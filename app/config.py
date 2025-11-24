"""
Configuration settings for the DevDox AI Portal API.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, ClassVar

from models_src import MongoConfig
from pydantic import EmailStr, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.infrastructure.supabase_queue import SupabaseQueue

class GitHosting(str, Enum):
    GITLAB = "gitlab"
    GITHUB = "github"


class MailSettings(BaseSettings):
    
    MAIL_SEND_TIMEOUT_MIN: ClassVar[int] = 20

    MAIL_CHARSET :str = "utf-8"
    MAIL_DEFAULT_SENDER_ENCODING:str = "utf-8"
    MAIL_USERNAME: str = Field(
        ...,
        description="SMTP username. Some providers require it separately, others just use MAIL_FROM.",
    )
    MAIL_PASSWORD: str = Field(
        ...,
        description="Password or app-specific key for authenticating to the SMTP server.",
    )
    MAIL_FROM: EmailStr = Field(
        ...,
        description="Default sender email address (appears in the 'From' header).",
    )
    MAIL_FROM_NAME: str | None = Field(
        default=None,
        description="Friendly name for the sender (appears alongside MAIL_FROM).",
    )
    MAIL_PORT: int = Field(
        default=587,
        description="Port for SMTP server. Usually 587 for STARTTLS, 465 for SSL/TLS, 25 as legacy.",
    )
    MAIL_SERVER: str = Field(
        ...,
        description="SMTP server hostname or IP address (e.g., smtp.gmail.com).",
    )
    MAIL_STARTTLS: bool = Field(
        default=True,
        description="Use STARTTLS (opportunistic TLS upgrade). Set false if server doesn’t support it.",
    )
    MAIL_SSL_TLS: bool = Field(
        default=False,
        description="Use direct SSL/TLS connection (usually on port 465).",
    )
    MAIL_USE_CREDENTIALS: bool = Field(
        default=True,
        description="Whether to authenticate with username/password. Set False for open relays (rare).",
    )
    MAIL_VALIDATE_CERTS: bool = Field(
        default=True,
        description="Validate SMTP server's TLS/SSL certificate. Set False only for self-signed certs.",
    )
    
    MAIL_SUPPRESS_SEND: bool = Field(
        default=False,
        description="If True, suppresses actual sending (emails are 'mocked'). Useful in testing.",
    )
    MAIL_DEBUG: int = Field(
        default=0,
        description="Debug output level for SMTP interactions. 0 = silent, 1+ = verbose.",
    )
    
    MAIL_AUDIT_RECIPIENTS: List[EmailStr] = Field(
        default_factory=list,
        description=(
            "Addresses to receive audit/ops/compliance notifications (can be internal or external). "
        ),
    )
    
    MAIL_SEND_TIMEOUT: int | None = Field(
        default=60,
        ge=MAIL_SEND_TIMEOUT_MIN,
        description=(
            "Max seconds to wait for the SMTP send to complete. If omitted, defaults to 60s. "
            "Set to None to disable the timeout entirely (use with caution). "
            "Values below 20s are rejected to avoid flaky timeouts under normal network operation."
        ),
    )
    
    MAIL_TEMPLATES_PARENT_DIR: Path | None = Field(
        default=None,
        description=(
            "Absolute or relative path to a *parent* directory that contains an 'email/' "
            "subfolder with Jinja templates. The effective template folder passed to the mail "
            "engine is <MAIL_TEMPLATES_PARENT_DIR>/email. If the value is empty, 'none', or "
            "'null', template rendering is disabled and only raw bodies may be used. The path "
            "is expanded (supports ~) and resolved at load time; when set, the '<parent>/email' "
            "directory must exist or settings validation will fail."
        )
    )
    
    # ---- Derived convenience properties ----
    
    @property
    def templates_enabled(self) -> bool:
        return self.MAIL_TEMPLATES_PARENT_DIR is not None
    
    @property
    def templates_dir(self) -> Path | None:
        
        if not self.templates_enabled:
            return None
        
        return (self.MAIL_TEMPLATES_PARENT_DIR / "email").expanduser().resolve()
    
    # ---- Normalizers & validation ----
    
    @field_validator("MAIL_SEND_TIMEOUT", mode="before")
    @classmethod
    def _noneify_timeout(cls, v:str) -> str | None:
        # Allow '', 'none', 'null' (case-insensitive) to disable the timeout via env
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"", "none", "null"}:
                return None
        return v
    
    @field_validator("MAIL_TEMPLATES_PARENT_DIR", mode="before")
    @classmethod
    def _noneify_mail_template_parent_dir(cls, v:str) -> str | None:
        # Treat "", "none", "null" as disabling templates
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() in {"none", "null"}:
                return None
        return v
    
    @field_validator("MAIL_TEMPLATES_PARENT_DIR", mode="after")
    @classmethod
    def _normalize_parent(cls, v: Path | None) -> Path | None:
        return v.expanduser().resolve() if isinstance(v, Path) else v
    
    @model_validator(mode="after")
    def _validate(self) -> "MailSettings":
        # TLS mode sanity
        if self.MAIL_STARTTLS and self.MAIL_SSL_TLS:
            raise ValueError("Set only one of MAIL_STARTTLS or MAIL_SSL_TLS, not both.")
        
        # If templates are enabled, require <parent>/email to exist
        if self.templates_enabled:
            td = self.templates_dir
            if not (td and td.is_dir()):
                raise ValueError(f"Templates directory does not exist: {td}")
        return self
    
    # ---- Configuration manager ----
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent / "instance" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

def build_mongo(env_files, enabled: bool = True) -> Optional[MongoConfig]:
    """
    Build MongoConfig from env files.
    If enabled=False, returns None (Mongo disabled).
    """
    if not enabled:
        return None
    return MongoConfig(_env_file=env_files)


def make_mongo_factory(env_files, enabled: bool = True) -> Callable[[], Optional[MongoConfig]]:
    """
    Pydantic's default_factory must be a zero-argument callable.
    So we "close over" env_files/enabled and return a 0-arg factory.
    """
    def _factory() -> Optional[MongoConfig]:
        return build_mongo(env_files, enabled)
    return _factory

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
    
    
    mail: MailSettings = Field(default_factory=MailSettings)
    
    MONGO: Optional[MongoConfig] = Field(default_factory=make_mongo_factory(
        env_files=str(Path(__file__).resolve().parent / "instance" / ".env"),
        enabled=True
    ))
    
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
