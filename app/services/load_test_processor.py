# app/processors/load_test_processor.py
"""
Load test processor for handling load testing jobs
"""

import asyncio
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone
import logging
from app.infrastructure.base_processor import BaseProcessor
from app.schemas.load_test import LoadTestRequest, LoadTestConfig, RepositoryValidationError
from app.services.api_test_generator import APITestGenerator
from app.utils.encryption import FernetEncryptionHelper
from devdox_ai_git.repo_fetcher import RepoFetcher
from models_src.repositories.git_label import TortoiseGitLabelStore as GitLabelRepository
from models_src.repositories.repo import TortoiseRepoStore as RepoRepository
from models_src.repositories.user import TortoiseUserStore as UserRepository

logger = logging.getLogger(__name__)

class RepositoryValidationService:
    """Validates repository access and permissions"""

    def __init__(
            self,
            repo_repository: RepoRepository,
            user_repository:UserRepository,
            git_label_repository:GitLabelRepository,
    ):
        """
        Initialize validation service.

        Args:
            repo_repository: Repository data access object
            user_repository: User data access object
            git_label_repository: Git label data access object
        """
        self._repo_repo = repo_repository
        self._user_repo = user_repository
        self._label_repo = git_label_repository

    async def validate_repository_access(
            self,
            repo_id: str,
            user_id: str,
            token_id: str
    ) -> Tuple[Any, Any, Any]:
        """
        Validate that user has access to repository with given token.

        Performs three validations:
        1. Repository exists
        2. User exists
        3. User has valid git token

        Args:
            repo_id: Repository UUID
            user_id: User UUID
            token_id: Git token UUID

        Returns:
            Tuple of (repo_info, user_info, git_label)

        Raises:
            RepositoryValidationError: If any validation fails
        """
        logger.info(
            "Validating repository access",
            extra={
                "repo_id": repo_id,
                "user_id": user_id,
                "token_id": token_id
            }
        )

        # Validate repository exists
        repo_info = await self._repo_repo.find_by_repo_id(repo_id)
        if not repo_info:
            logger.warning(f"Repository not found: {repo_id}")
            raise RepositoryValidationError(
                f"Repository not found: {repo_id}",
                error_code="REPO_NOT_FOUND",
                details={"repo_id": repo_id}
            )

        # Validate user exists
        user_info = await self._user_repo.find_by_user_id(user_id)
        if not user_info:
            logger.warning(f"User not found: {user_id}")
            raise RepositoryValidationError(
                f"User not found: {user_id}",
                error_code="USER_NOT_FOUND",
                details={"user_id": user_id}
            )

        # Validate git label/token exists for user
        git_label = await self._label_repo.find_by_token_id_and_user(
            token_id=token_id,
            user_id=user_id
        )
        if not git_label:
            logger.warning(f"Git token not found for user: {token_id}")
            raise RepositoryValidationError(
                f"Git token not found for user: {token_id}",
                error_code="TOKEN_NOT_FOUND",
                details={"token_id": token_id, "user_id": user_id}
            )

        logger.info("Repository access validated successfully")
        return repo_info, user_info, git_label

def extract_files_for_commit(result: List[dict], repo_root: Path) -> Dict[str, str]:
    """Extract file_path and content from processing result"""
    files_for_commit = {}

    for file_info in result:
        final_path = file_info.get('final_path')
        if not final_path:
            logger.warning(f"Missing final_path for file: {file_info}")
            continue

        # Get relative path from repo root for Git
        relative_path = final_path.relative_to(repo_root)
        git_path = str(relative_path).replace('\\', '/')  # Ensure forward slashes
        try:
            # Read file content
            content = final_path.read_text(encoding='utf-8')
            files_for_commit[git_path] = content
        except (OSError, UnicodeDecodeError) as e:
            logger.error(f"Failed to read file {final_path}: {e}")
            raise
        


    return files_for_commit

def  get_token(token_db:str,encryption_salt:str, auth_token:str|None) -> str:
    if auth_token:
        return  FernetEncryptionHelper().decrypt(auth_token)
    else:
        return FernetEncryptionHelper().decrypt_for_user(
                    token_db,
                    salt_b64=FernetEncryptionHelper().encryption.decrypt(encryption_salt)
                )
def get_repo_info(git_provider, created_repo):
    if git_provider == "github":
        repo_full_name = created_repo.full_name
        default_branch = created_repo.default_branch
    elif git_provider == "gitlab":
        repo_full_name = created_repo.id
        default_branch = created_repo.default_branch
    return repo_full_name, default_branch



class LoadTestProcessor(BaseProcessor):
    """Processor for load testing jobs"""


    async def process(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process load test job

        Expected job_data structure:
        {
            "job_type": "load_locust",
            "payload": {
                "test": "sss",
                "context_id": "some-uuid-hex"
            }
        }
        """

        context_id = job_data.get("payload", {}).get("context_id")
        job_payload = job_data.get("payload", {})
        repo_id  = job_payload.get("repo_id")

        if not repo_id:
             raise ValueError("repo_id is required in job payload")
        token_id = job_payload.get("token_id")
        if not token_id:
             raise ValueError("token_id is required in job payload")
        user_id = job_payload.get("user_id")
        if not user_id:
             raise ValueError("user_id is required in job payload")
        git_provider = job_payload.get("git_provider")
        if not git_provider:
             raise ValueError("git_provider is required in job payload")
        auth_token = job_payload.get("auth_token")



        self.logger.info(f"Processing load test for context: {context_id}")

        start_time = datetime.now(timezone.utc)

        try:
            # Simulate load test execution (replace with actual logic)
            result, directory_test = await self._execute_load_test(context_id,job_payload)
            if directory_test:
                # Upload the directory to Supabase storage
                await self.setup_test_repository_workflow(result, directory_test, repo_id, token_id, user_id,git_provider,auth_token)

            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()

            return {
                "status": "completed",
                "context_id": context_id,
                "result": result,
                "processing_time": processing_time,
                "completed_at": end_time.isoformat()
            }

        except Exception as e:
            self.logger.error(f"Load test failed for context {context_id}: {e}")
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()

            return {
                "status": "failed",
                "context_id": context_id,
                "error": str(e),
                "processing_time": processing_time,
                "failed_at": end_time.isoformat()
            }

    async def _execute_load_test(self, context_id: str, job_payload: Dict[str, Any])  -> Tuple[Any, Path|None]:
        """Execute the actual load test"""
        # Replace this with your actual load testing logic
        # This could be:
        # - Executing Locust tests
        # - Running custom load testing scripts
        # - Calling external load testing services



        test_type = job_payload.get("test_type", LoadTestConfig.DEFAULT_TEST_TYPE)


        # Example: Run a Locust test file
        if test_type == LoadTestConfig.DEFAULT_TEST_TYPE:

            job_request = LoadTestRequest(**job_payload.get("data",{}))
            return await self._run_locust_test(job_request, context_id)

        else:
            # Simulate other test types
            await asyncio.sleep(5)  # Simulate processing time
            return {
                "test_type": test_type,
                "duration": 5,
                "requests_per_second": 100,
                "total_requests": 500,
                "success_rate": 0.98
            },None

    async def prepare_repository(self, repo_name: str) ->Path:

        repo_path = self.base_dir / repo_name
        try:

                    if repo_path.exists():
                            await asyncio.to_thread(shutil.rmtree, repo_path)
                    repo_path.mkdir(parents=True, exist_ok=True)

                    return repo_path
        except OSError as e:
            self.logger.exception(f"Failed to prepare repository {repo_name}: {e}")
        
            raise


    async def _run_locust_test(self, data:LoadTestRequest,context_id:str) -> Tuple[List[Dict[str, Any]], Path]:
        """Run Locust load test"""
        if not data:

            raise ValueError("LoadTestRequest data is required")

        if not context_id:

            raise ValueError("context_id is required")

        load_test_service = APITestGenerator()


        repo_name = data.get_effective_output_path()

        output_dir = await self.prepare_repository(repo_name)

        result =  await load_test_service.generate_tests_from_swagger_url(swagger_url=data.url,
                                                                          output_dir=output_dir,
                                                                          custom_requirement=data.custom_requirement or "",
                                                                          host=data.host or "localhost",
                                                                          auth=data.auth or False,
                                                                          operation_id=context_id
                                                                          )
        if result:
            return result, output_dir
        else:
            raise ValueError("Load test generation failed")



    async def setup_test_repository_workflow(self, result:Any, directory_test: Path, repo_id: str, token_id: str, user_id: str,git_provider:str,auth_token:str|None) -> None:
        """
        Create a new repository and commit generated test files.

        This method performs the following operations:
        1. Validates repository, user, and token access
        2. Decrypts authentication token
        3. Creates new repository with specified visibility
        4. Creates feature branch for tests
        5. Commits test files to the branch

        Args:
            result: List of generated test file information
            directory_test: Path to directory containing test files
            repo_id: UUID of the repository to replicate settings from
            token_id: UUID of the git authentication token
            user_id: UUID of the user creating the repository
            git_provider: Git provider name ('github', 'gitlab', etc.)

        Raises:
            RepositoryValidationError: If repo, user, or token validation fails
            TokenDecryptionError: If token decryption fails
            GitOperationError: If git operations (create/branch/commit) fail

        Returns:
            None. Logs success/failure status.
        """
        created_branch = False
        created_repo = False
        fetcher, _ = RepoFetcher().get_components(git_provider)


        validation_service = RepositoryValidationService(repo_repository=RepoRepository(),user_repository=UserRepository(),git_label_repository=GitLabelRepository())
        repo_info, user_info, git_label = await  validation_service.validate_repository_access(repo_id, user_id,
                                                                                              token_id)

        repo_name = directory_test.name
        if not fetcher:
            raise ValueError("Invalid git provider")

        try:
            decrypted_label_token = get_token(git_label.token_value, user_info.encryption_salt, auth_token)


            files = extract_files_for_commit(result, directory_test)

            created_repo = fetcher.create_repository(token=decrypted_label_token, name=repo_name, description="", visibility=repo_info.visibility)
            if created_repo:
                repo_full_name, default_branch = get_repo_info(git_provider, created_repo)


                created_branch = fetcher.create_branch(decrypted_label_token,repo_full_name,
                                                       LoadTestConfig.get_branch_name(repo_name),
                                                       default_branch)

                if created_branch:
                    self.logger.info(f"Branch feature_locust created successfully")


                    _ = fetcher.commit_files(decrypted_label_token, repo_full_name, LoadTestConfig.get_branch_name(repo_name), files,
                                                        LoadTestConfig.get_commit_message(repo_name),author_name=repo_info.repo_author_name, author_email=repo_info.repo_author_email)
                else:
                    self.logger.error(f"Branch feature_locust creation failed for {repo_id}")

                self.logger.info(f"Repository {repo_id} created successfully")
            else:
                self.logger.error(f"Repository {repo_id} creation failed")


        except Exception as e:

            # Rollback: delete branch if created

            if created_branch:
                try:

                    fetcher.delete_branch(decrypted_label_token, repo_full_name, LoadTestConfig.get_branch_name(repo_name))

                except Exception as rollback_error:

                    self.logger.error(f"Rollback failed delete branch: {rollback_error}")

            # Rollback: delete repo if created

            if created_repo:

                try:

                    fetcher.delete_repository(decrypted_label_token, repo_full_name)

                except Exception as rollback_error:

                    self.logger.error(f"Rollback failed delete repo: {rollback_error}")

            self.logger.error(f"Repository creation failed: {e}")
