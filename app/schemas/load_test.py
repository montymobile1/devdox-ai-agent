from pydantic import BaseModel, Field,  field_validator
from typing import List, Dict, Optional, Any
import re
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass
from uuid import uuid4
from urllib.parse import urlparse


class LoadTestConfig:
    DEFAULT_TEST_TYPE = "locust"
    DEFAULT_BRANCH_NAME = "feature/load-tests"
    DEFAULT_COMMIT_MESSAGE_TEMPLATE = "feat: Add load tests for {repo_name}"
    PROCESSING_TIMEOUT_SECONDS = 300

    @staticmethod
    def get_branch_name(repo_name: str) -> str:
        return f"feature/load-tests-{repo_name}"

    @staticmethod
    def get_commit_message(repo_name: str) -> str:
        return f"feat: Add load tests for {repo_name}"

class LoadTestRequest(BaseModel):
    """Request wrapper for Load tests"""

    url: str = Field(
        ...,
        description="Swagger url of documentation",
        example="https://api.shop.example.com/v1/openapi.json",
        min_length=1
    )
    repo_alias_name:str = Field(..., description="The repository alias name", example="my-project")
    auth: bool = Field(
        ...,
        description="Whether the API requires authentication",
        example=True
    )

    output_path: Optional[str] = Field(
            default=None,
            description="Custom output path for generated tests. If not provided, uses repo_alias_name + '_test'",
            example="my-project-tests",
            max_length=200
        )

    spawn_rate: int = Field(
            default=10,
            description="Number of users to spawn per second during load test",
            example=10,
            ge=1,  # Greater than or equal to 1
            le=1000  # Less than or equal to 1000
        )

    run_time: str = Field(
            default="5m",
            description="Duration to run the load test (e.g., '5m', '30s', '2h')",
            example="5m",
            pattern=r'^\d+[smh]$'  # Pattern: number followed by s/m/h
        )

    host: str = Field(
            default="localhost",
            description="Target host for load testing (without protocol)",
            example="api.shop.example.com",
            min_length=1,
            max_length=255
        )

    custom_requirement: str = Field(
            default="",
            description="Custom authentication or requirements (empty if none)",
            example="Bearer <token>",
            max_length=1000
        )



    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
            """Validate that URL is properly formatted and accessible"""
            if not v or not v.strip():
                raise ValueError("URL cannot be empty")

            v = v.strip()

            if not v.startswith(('http://', 'https://')):
                v = f'https://{v}'

            # Parse URL to validate format
            try:
                parsed = urlparse(v)
                if not parsed.netloc:
                    raise ValueError("Invalid URL format")
                if parsed.scheme not in ['http', 'https']:
                    raise ValueError("URL must use http or https protocol")
            except Exception as e:
                raise ValueError(f"Invalid URL: {e}")

            # Check for common OpenAPI/Swagger patterns
            openapi_patterns = [
                'openapi.json', 'swagger.json', 'api-docs',
                'docs/json', 'v1/openapi', 'v2/openapi', 'v3/openapi'
            ]

            if not any(pattern in v.lower() for pattern in openapi_patterns):
                # Warning but don't fail - could be a valid custom endpoint
                pass
            return v

    @field_validator('repo_alias_name')
    @classmethod
    def validate_repo_alias_name(cls, v: str) -> str:
            """Validate repository alias name"""
            if not v or not v.strip():
                raise ValueError("Repository alias name cannot be empty")

            v = v.strip()
            # Check for valid identifier pattern
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', v):
                raise ValueError(
                    "Repository alias must start with a letter and contain only "
                    "letters, numbers, hyphens, and underscores"
                )

            # Reserved names check
            reserved_names = {'test', 'tmp', 'temp', 'admin', 'root', 'system'}
            if v.lower() in reserved_names:
                raise ValueError(f"'{v}' is a reserved name, please choose another")
            return v

    @field_validator('output_path')
    @classmethod
    def validate_output_path(cls, v: Optional[str]) -> Optional[str]:
            """Validate and clean output path"""
            if v is None or not v.strip():
                return None

            v = v.strip()

            # Basic path validation
            if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$', v):
                raise ValueError(
                    "Output path must contain only letters, numbers, hyphens, and underscores"
                )

            return v

    @field_validator('run_time')
    @classmethod
    def validate_run_time(cls, v: str) -> str:
            """Validate run time format and reasonable values"""
            if not v or not v.strip():
                raise ValueError("Run time cannot be empty")

            v = v.strip().lower()

            # Extract number and unit
            match = re.match(r'^(\d+)([smh])$', v)
            if not match:
                raise ValueError("Run time must be in format: number + unit (s/m/h). Example: '30s', '5m', '2h'")

            duration, unit = match.groups()
            duration = int(duration)

            # Validate reasonable ranges
            if unit == 's' and (duration < 10 or duration > 3600):  # 10 seconds to 1 hour
                raise ValueError("Seconds must be between 10 and 3600")
            elif unit == 'm' and (duration < 1 or duration > 60):  # 1 to 60 minutes
                raise ValueError("Minutes must be between 1 and 60")
            elif unit == 'h' and (duration < 1 or duration > 24):  # 1 to 24 hours
                raise ValueError("Hours must be between 1 and 24")
            return v

    @field_validator('host')
    @classmethod
    def validate_host(cls, v: str) -> str:
            """Validate host format"""
            if not v or not v.strip():
                raise ValueError("Host cannot be empty")

            v = v.strip().lower()
            # Remove protocol if accidentally included
            v = re.sub(r'^https?://', '', v)
            # Remove trailing slash
            v = v.rstrip('/')
            # Basic hostname validation
            if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$', v):
                raise ValueError("Invalid host format")

            # Check for localhost variations
            if v in ['localhost', '127.0.0.1', '0.0.0.0']:
                return v

            # Validate domain format
            if '.' in v:
                parts = v.split('.')
                if len(parts) < 2:
                    raise ValueError("Host must be a valid domain or IP address")

                # Check TLD
                if len(parts[-1]) < 2:
                    raise ValueError("Invalid top-level domain")

            return v

    @field_validator('custom_requirement')
    @classmethod
    def validate_custom_requirement(cls, v: str) -> str:
            """Clean and validate custom requirement"""
            if v is None:

                return ""

            v = str(v).strip()

            # Security check - don't allow potentially dangerous content
            dangerous_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
            if any(pattern in v.lower() for pattern in dangerous_patterns):
                raise ValueError("Custom requirement contains potentially dangerous content")
            return v



    def get_effective_output_path(self) -> str:
            """Get the effective output path with fallback logic"""
            if self.output_path and self.output_path.strip():
                return self.output_path.strip()
            return f"{self.repo_alias_name}_test"


    def get_run_time_seconds(self) -> int:
            """Convert run_time to seconds for internal use"""
            match = re.match(r'^(\d+)([smh])$', self.run_time.lower())
            if not match:
                return 300  # Default 5 minutes

            duration, unit = match.groups()
            duration = int(duration)

            if unit == 's':
                return duration
            elif unit == 'm':
                return duration * 60
            elif unit == 'h':
                return duration * 3600
            return 300  # Fallback

    def is_https_required(self) -> bool:
            """Check if HTTPS is required based on URL"""
            return self.url.startswith('https://')

    def get_base_url(self) -> str:
            """Extract base URL for testing"""
            protocol = 'https' if self.is_https_required() else 'http'
            return f"{protocol}://{self.host}"


class LoadTestStatus(Enum):
    """Status enumeration for load test operations"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LoadTestError(Exception):
    """Custom exception for load test operations"""
    def __init__(self, message: str, error_code: str = "LOAD_TEST_ERROR", details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class RepositoryValidationError(LoadTestError):
    """Repository, user, or token validation failed"""

    def __init__(
            self,
            message: str,
            error_code: str = "VALIDATION_ERROR",
            details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class GitOperationError(LoadTestError):
    """Git operations (create repo, branch, commit) failed"""

    def __init__(
            self,
            message: str,
            error_code: str = "GIT_OPERATION_ERROR",
            details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class TokenDecryptionError(LoadTestError):
    """Token decryption failed"""

    def __init__(
            self,
            message: str,
            error_code: str = "TOKEN_DECRYPTION_ERROR",
            details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class TestGenerationError(LoadTestError):
    """Test file generation failed"""

    def __init__(
            self,
            message: str,
            error_code: str = "TEST_GENERATION_ERROR",
            details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class LoadLocustPayload(BaseModel):
    repo_id: str
    token_id: str
    config: dict = Field(default_factory=dict)
    data: dict
    user_id: str
    priority: int = 1
    git_token: str
    git_provider: str
    auth_token: Optional[str] = None
    context_id: str = Field(default_factory=lambda: uuid4().hex)

class LoadTestResult(BaseModel):
    """Structured result for load test operations"""
    success: bool
    status: LoadTestStatus
    message: str
    created_files: List[Dict[str, Any]] = None
    error_details: Dict[str, Any] = None
    execution_time: float = 0.0
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            # Set timestamp to current UTC time
            object.__setattr__(self, 'timestamp', datetime.now(timezone.utc))

        if self.error_details is None:
            object.__setattr__(self, 'error_details', {})

        if self.created_files is None:
            self.created_files = []


