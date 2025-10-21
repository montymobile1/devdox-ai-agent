from typing import Annotated, List, Dict, Optional, Any, Tuple
from pathlib import Path
import asyncio
import logging
import time
from fastapi import Depends, HTTPException
from contextlib import asynccontextmanager

from pydantic import ValidationError
from models_src.repositories.repo import TortoiseRepoStore as RepoStore
from models_src.repositories.git_label import TortoiseGitLabelStore as GitLabelRepository
from app.schemas.load_test import LoadTestRequest, LoadTestResult, LoadTestStatus, LoadTestError, LoadLocustPayload
from together import Together
from app.config import settings, supabase_queue
from app.utils.auth import UserClaims
from app.utils.encryption import get_encryption_helper
from devdox_ai_locust import HybridLocustGenerator
from devdox_ai_locust.schemas.processing_result import SwaggerProcessingRequest
from devdox_ai_locust.utils.swagger_utils import get_api_schema
from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser, Endpoint

logger = logging.getLogger(__name__)

class LoadTestService:

    def __init__(self, repo_store: RepoStore, git_label_repository: GitLabelRepository):

        self.repo_store = repo_store
        self.git_label_repository = git_label_repository



        # Service configuration
        self.max_concurrent_operations = getattr(settings, 'MAX_CONCURRENT_LOAD_TESTS', 5)
        self.operation_timeout = getattr(settings, 'LOAD_TEST_TIMEOUT', 1800)  # 30 minutes
        self.max_file_size = getattr(settings, 'MAX_TEST_FILE_SIZE', 10 * 1024 * 1024)  # 10MB

        # Semaphore for controlling concurrent operations
        self._operation_semaphore = asyncio.Semaphore(self.max_concurrent_operations)

        logger.info(
            f"LoadTestService initialized with max_concurrent={self.max_concurrent_operations}, "
            f"timeout={self.operation_timeout}s"
        )




    @classmethod
    def with_dependency(
            cls,
            repo_store: Annotated[RepoStore, Depends()],
            git_label_repository: Annotated[GitLabelRepository, Depends()]
    ) -> "LoadTestService":
        """Dependency injection factory with validation"""
        if repo_store is None:
            logger.error("RepoStore dependency is None")
            raise HTTPException(status_code=500, detail="Repository store not available")

        logger.debug(f"Creating LoadTestService with repo_store: {type(repo_store).__name__}")
        if git_label_repository is None:
            logger.error("GitLabelRepository dependency is None")
            raise HTTPException(status_code=500, detail="GitLabelRepository not available")
        logger.debug(f"Creating LoadTestService with git_label_repository: {type(git_label_repository).__name__}")

        return cls(repo_store, git_label_repository)

    @asynccontextmanager
    async def _operation_context(self, operation_name: str, user_id: str, repo_name: str):
        """Context manager for tracking and timing operations"""
        start_time = time.time()
        operation_id = f"{operation_name}_{user_id}_{repo_name}_{int(start_time)}"

        logger.info(
            f"Starting operation: {operation_name}",
            extra={
                "operation_id": operation_id,
                "user_id": user_id,
                "repo_name": repo_name,
                "start_time": start_time
            }
        )

        try:
            async with self._operation_semaphore:
                yield operation_id
        except Exception as e:
            logger.error(
                f"Operation failed: {operation_name}",
                extra={
                    "operation_id": operation_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "execution_time": time.time() - start_time
                }
            )
            raise
        finally:
            execution_time = time.time() - start_time
            logger.info(
                f"Completed operation: {operation_name}",
                extra={
                    "operation_id": operation_id,
                    "execution_time": execution_time
                }
            )

    async def load_tests(
        self,
        user_claims: UserClaims,
        data: LoadTestRequest,
    )-> LoadTestResult:

        start_time = time.time()
        operation_name = "load_tests"

        # Validate inputs
        if not user_claims or not user_claims.sub:
            logger.error("Invalid user claims provided")
            return LoadTestResult(
                success=False,
                status=LoadTestStatus.FAILED,
                message="Invalid user authentication",
                error_details={"error_code": "INVALID_USER_CLAIMS"}
            )

        if not isinstance(data, LoadTestRequest):
            logger.error(f"Invalid data type: {type(data)}")
            return LoadTestResult(
                success=False,
                status=LoadTestStatus.FAILED,
                message="Invalid request data format",
                error_details={"error_code": "INVALID_REQUEST_FORMAT"}
            )

        async with self._operation_context(operation_name, user_claims.sub, data.repo_alias_name) as operation_id:

            logger.info(
                "Processing load test request",
                extra={
                    "operation_id": operation_id,
                    "user_id": user_claims.sub,
                    "repo_alias": data.repo_alias_name,
                    "swagger_url": data.url,
                    "auth_required": data.auth
                }
            )
        try:
            repo_info = await self.repo_store.find_by_user_and_alias_name(
                user_id=user_claims.sub, repo_alias_name=data.repo_alias_name
            )

            if not repo_info:

                logger.warning(
                    f"Repository not found for user {user_claims.sub} and alias name {data.repo_alias_name}",
                    extra={
                        "user_id": user_claims.sub,
                        "repo_alias": data.repo_alias_name
                    }
                )
                return LoadTestResult(
                    success=False,
                    status=LoadTestStatus.FAILED,
                    message="Repository not found or access denied",
                    error_details={
                        "error_code": "REPOSITORY_NOT_FOUND",
                        "repo_alias": data.repo_alias_name
                    }
                )
            token_info = await self.git_label_repository.find_by_token_id_and_user(repo_info.token_id, user_claims.sub)

            if not token_info:
                logger.warning(
                    "Git label not found for token",
                    extra={
                        "token_id": repo_info.token_id,
                        "user_id": user_claims.sub
                    }
                )
                return LoadTestResult(
                    success=False,
                    status=LoadTestStatus.FAILED,
                    message="Git label not found or access denied",
                    error_details={
                        "error_code": "GIT_LABEL_NOT_FOUND",
                        "token_id": repo_info.token_id
                    }
                )
            
            fernet_encryption_helper = get_encryption_helper()
            

            job_payload = LoadLocustPayload(
                repo_id = str(repo_info.repo_id),
                token_id = str(token_info.id),
                data = data.dict(),
                user_id = str(user_claims.sub),
                git_token = str(token_info.id),
                git_provider = token_info.git_hosting,
                auth_token=(
                    fernet_encryption_helper.encrypt(user_claims.git_token)
                    if (
                            token_info.git_hosting
                            and user_claims.git_provider
                            and token_info.git_hosting.lower() == user_claims.git_provider.lower()
                    )
                    else None
                )
            )
            payload = {
                "job_type": "load_locust",
                "payload": job_payload.dict(),
            }

            job_id = await supabase_queue.enqueue(
                "testing",
                payload=payload,
                priority=1,
                job_type="load_locust",
                user_id=user_claims.sub,
            )


            created_files = []
            execution_time= time.time() - start_time
            return LoadTestResult(
                success=True,
                status=LoadTestStatus.COMPLETED,
                message=f"Load test job enqueued (job_id={job_id})",
                created_files=created_files,
                execution_time=execution_time,
            )
        except LoadTestError:
            # Re-raise custom errors
            raise
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return LoadTestResult(
                success=False,
                status=LoadTestStatus.FAILED,
                message="Request validation failed",
                error_details={
                    "error_code": "VALIDATION_ERROR",
                    "validation_errors": e.errors()
                }
            )
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {self.operation_timeout}s")
            return LoadTestResult(
                success=False,
                status=LoadTestStatus.FAILED,
                message="Operation timed out",
                error_details={
                    "error_code": "OPERATION_TIMEOUT",
                    "timeout": self.operation_timeout
                }
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in load test generation: {e}",
                exc_info=True,
                extra={"operation_id": operation_id}
            )
            return LoadTestResult(
                success=False,
                status=LoadTestStatus.FAILED,
                message="Internal server error occurred",
                error_details={
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                execution_time=time.time() - start_time
            )

    async def _parse_api_schema(self, api_schema: Dict[str, Any]) -> Tuple[List[Endpoint], Dict[str, Any]]:
        """Parse API schema with comprehensive error handling"""
        parser = OpenAPIParser()

        try:
            # Parse schema in thread pool to avoid blocking
            _ = await asyncio.to_thread(parser.parse_schema, api_schema)
            endpoints = await asyncio.to_thread(parser.parse_endpoints)
            api_info = await asyncio.to_thread(parser.get_schema_info)

            logger.info(f"Successfully parsed API schema: {len(endpoints)} endpoints found")

            return endpoints, api_info

        except Exception as e:
            logger.error(f"Failed to parse API schema: {e}")
            raise LoadTestError(
                f"API schema parsing failed: {e}",
                "SCHEMA_PARSE_ERROR",
                {"error": str(e), "error_type": type(e).__name__}
            )

    async def _fetch_api_schema(self, swagger_url: str) -> Optional[Dict[str, Any]]:
        """Fetch API schema with timeout and validation"""
        try:
            source_request = SwaggerProcessingRequest(swagger_url=swagger_url)

            api_schema = await asyncio.wait_for(
                get_api_schema(source_request),
                timeout=60  # 1 minute timeout for schema fetch
            )

            if not api_schema:
                logger.warning(f"Empty API schema returned for URL: {swagger_url}")
                return None

            logger.info(f"Successfully fetched API schema from {swagger_url}")
            return api_schema

        except asyncio.TimeoutError:
            logger.error(f"Schema fetch timed out for URL: {swagger_url}")
            raise LoadTestError(
                "API schema fetch timed out",
                "SCHEMA_FETCH_TIMEOUT",
                {"swagger_url": swagger_url, "timeout": 60}
            )
        except Exception as e:
            logger.error(f"Failed to fetch API schema from {swagger_url}: {e}")
            raise LoadTestError(
                f"Failed to fetch API schema: {e}",
                "SCHEMA_FETCH_ERROR",
                {"swagger_url": swagger_url, "error": str(e)}
            )


    async def _generate_and_create_tests(self,
        endpoints: List[Endpoint],
        api_info: Dict[str, Any],
        output_dir: Path,
        custom_requirement: Optional[str] = "",
        host: Optional[str] = "0.0.0.0",
        auth: bool = False,
        operation_id:str=""
    ) -> List[Dict[Any, Any]]:
        """Generate tests using AI and create test files"""
        together_client = Together(api_key=settings.TOGETHER_API_KEY)


        generator = HybridLocustGenerator(ai_client=together_client)
        logger.info(
            f"Generating tests for {len(endpoints)} endpoints",
            extra={"operation_id": operation_id, "output_dir": str(output_dir)}
        )

        test_files, test_directories = await generator.generate_from_endpoints(
                endpoints=endpoints,
                api_info=api_info,
                custom_requirement=custom_requirement,
                target_host=host,
                include_auth=auth,
            )

        # Create test files

        created_files = []

        # Create workflow files
        if test_directories:
                workflows_dir = output_dir / "workflows"
                workflows_dir.mkdir(exist_ok=True)
                for file_workflow in test_directories:
                    workflow_files = await generator._create_test_files_safely(
                        file_workflow, workflows_dir
                    )
                    created_files.extend(workflow_files)

        # Create main test files
        if test_files:
                main_files = await generator._create_test_files_safely(
                    test_files, output_dir
                )
                created_files.extend(main_files)
        logger.info(
            f"Successfully created {len(created_files)} test files",
            extra={"operation_id": operation_id, "files": [f.get("filename") for f in created_files]}
        )

        return created_files

