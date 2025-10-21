# app/processors/load_test_processor.py
"""
Load test processor for handling load testing jobs
"""

import asyncio
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime, UTC
import logging
from app.infrastructure.base_processor import BaseProcessor
from app.infrastructure.queue_job_tracker.job_trace_metadata import JobTraceMetaData
from app.infrastructure.queue_job_tracker.job_tracker import JobLevels, JobTracker
from app.schemas.load_test import LoadTestRequest, LoadTestConfig, RepositoryValidationError
from app.services.api_test_generator import APITestGenerator
from app.utils.encryption import get_encryption_helper
from devdox_ai_git.repo_fetcher import RepoFetcher
from models_src.repositories.git_label import TortoiseGitLabelStore as GitLabelRepository
from models_src.repositories.repo import TortoiseRepoStore as RepoRepository
from models_src.repositories.user import TortoiseUserStore as UserRepository

logger = logging.getLogger(__name__)

delete_folder_message = "delete_folder_path failed"

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

def  get_token(token_db:str,encryption_salt:str, git_provider_label:str,git_provider:str, auth_token:str|None) -> str:
    fernet_encryption_helper = get_encryption_helper()
    
    if auth_token and git_provider == git_provider_label:
        return  fernet_encryption_helper.decrypt(auth_token)
    else:
        return fernet_encryption_helper.decrypt_for_user(
                    token_db,
                    salt_b64=fernet_encryption_helper.decrypt(encryption_salt)
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

    async def process(
        self,
        job_data: Dict[str, Any],
        trace: JobTraceMetaData | None = None,
        tracker: JobTracker | None = None,
    ) -> Dict[str, Any]:
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
        start_time = datetime.now(UTC)

        # hint tracker we're entering logical analysis phase
        if tracker:
            try:
                await tracker.update_step(JobLevels.LOAD_TESTS)
            except Exception:
                logger.warning("tracker.update_step(LOAD_TESTS) failed", exc_info=True)

        try:
            # Simulate load test execution (replace with actual logic)
            result, directory_test = await self._execute_load_test(
                context_id,
                job_payload,
                trace=trace,
                tracker=tracker,
            )

            if directory_test:
                # Upload the directory to Supabase storage
                await self.setup_test_repository_workflow(
                    result,
                    directory_test,
                    repo_id,
                    token_id,
                    user_id,
                    git_provider,
                    auth_token,
                    trace=trace,
                    tracker=tracker,
                )
                # Delete the directory after upload
                await self.delete_folder_path(directory_test, trace=trace)



            end_time = datetime.now(UTC)
            processing_time = (end_time - start_time).total_seconds()

            return {
                "status": "completed",
                "context_id": context_id,
                "result": result,
                "processing_time": processing_time,
                "completed_at": end_time.isoformat()
            }

        except Exception as e:
            self.logger.error(f"Load test failed for context {context_id}: {e}", exc_info=True)
            if trace:
                trace.record_error(e, summary="Load test processor failure")
            end_time = datetime.now(UTC)
            processing_time = (end_time - start_time).total_seconds()

            return {
                "status": "failed",
                "context_id": context_id,
                "error": str(e),
                "processing_time": processing_time,
                "failed_at": end_time.isoformat()
            }

    async def _execute_load_test(
        self,
        context_id: str,
        job_payload: Dict[str, Any],
        trace: JobTraceMetaData | None = None,
        tracker: JobTracker | None = None,
    ) -> Tuple[Any, Path | None]:
        """Execute the actual load test"""
        # Replace this with your actual load testing logic
        # This could be:
        # - Executing Locust tests
        # - Running custom load testing scripts
        # - Calling external load testing services



        test_type = job_payload.get("test_type", LoadTestConfig.DEFAULT_TEST_TYPE)


        # Example: Run a Locust test file
        if test_type == LoadTestConfig.DEFAULT_TEST_TYPE:
            # Track: we’re about to prepare workspace / run generation
            if tracker:
                try:
                    await tracker.update_step(JobLevels.WORKDIR)
                except Exception:
                    logger.warning("tracker.update_step(WORKDIR) failed", exc_info=True)

            job_request = LoadTestRequest(**job_payload.get("data",{}))
            return await self._run_locust_test(job_request, context_id, trace=trace, tracker=tracker)

        # Simulate other test types
        await asyncio.sleep(5)  # Simulate processing time
        return {
            "test_type": test_type,
            "duration": 5,
            "requests_per_second": 100,
            "total_requests": 500,
            "success_rate": 0.98
        }, None

    async def delete_folder_path(
            self,
            folder_path: str | Path,
            trace: JobTraceMetaData | None = None
    ) -> bool:
        """
        Delete a folder and all its contents asynchronously while preserving base_dir.

        Args:
            folder_path: Relative path from base_dir to delete (e.g., "test1" or "test1/test2")
            trace: Optional job trace metadata for error tracking
            tracker: Optional job tracker

        Returns:
            bool: True if deletion was successful, False if folder didn't exist

        Raises:
            OSError: If deletion fails
            ValueError: If attempting to delete base_dir itself

        Examples:
            # Delete app/repos/test1
            await delete_folder_path("test1")

            # Delete app/repos/test1/test2 (deletes test1 and test2)
            await delete_folder_path("test1/test2")

            # Using Path object
            await delete_folder_path(Path("test1/test2"))
        """
        # Convert to Path if string
        rel_path = Path(folder_path) if isinstance(folder_path, str) else folder_path

        # Get the topmost folder to delete (first part of the path)
        parts = rel_path.parts
        if not parts:
            error_msg = "Empty folder path provided"
            self.logger.error(error_msg)
            if trace:
                trace.record_error(ValueError(error_msg), summary=delete_folder_message)
            raise ValueError(error_msg)

        # Construct the full path to the topmost folder
        topmost_folder = parts[0]
        full_path = self.base_dir / topmost_folder
        print("full_path", full_path)

        # Safety check: ensure we're not trying to delete base_dir itself
        if full_path.resolve() == self.base_dir.resolve():
            error_msg = f"Cannot delete base directory: {self.base_dir}"
            self.logger.error(error_msg)
            if trace:
                trace.record_error(ValueError(error_msg), summary=delete_folder_message)
            raise ValueError(error_msg)

        # Safety check: ensure the path is within base_dir
        try:
            full_path.resolve().relative_to(self.base_dir.resolve())
        except ValueError:
            error_msg = f"Path {full_path} is outside base directory {self.base_dir}"
            self.logger.error(error_msg)
            if trace:
                trace.record_error(ValueError(error_msg), summary=delete_folder_message)
            raise ValueError(error_msg)

        try:
            if full_path.exists():
                self.logger.info(f"Deleting folder: {full_path}")
                await asyncio.to_thread(shutil.rmtree, full_path)
                self.logger.info(f"Successfully deleted: {full_path}")
                return True
            else:
                self.logger.debug(f"Folder does not exist, skipping deletion: {full_path}")
                return False

        except OSError as e:
            self.logger.exception(f"Failed to delete folder {full_path}: {e}")
            if trace:
                trace.record_error(e, summary=delete_folder_message)
            raise

    async def prepare_repository(
        self,
        repo_name: str,
        trace: JobTraceMetaData | None = None,
        tracker: JobTracker | None = None,
    ) -> Path:
        repo_path = self.base_dir / repo_name
        try:

            if repo_path.exists():
                await asyncio.to_thread(shutil.rmtree, repo_path)
            repo_path.mkdir(parents=True, exist_ok=True)

            return repo_path
        except OSError as e:
            self.logger.exception(f"Failed to prepare repository {repo_name}: {e}")
            if trace:
                trace.record_error(e, summary="prepare_repository failed")
            raise

    async def _run_locust_test(
        self,
        data: LoadTestRequest,
        context_id: str,
        trace: JobTraceMetaData | None = None,
        tracker: JobTracker | None = None,
    ) -> Tuple[List[Dict[str, Any]], Path]:
        """Run Locust load test"""
        if not data:

            raise ValueError("LoadTestRequest data is required")

        if not context_id:

            raise ValueError("context_id is required")

        load_test_service = APITestGenerator()


        repo_name = data.get_effective_output_path()

        # Track working directory creation
        if tracker:
            try:
                await tracker.update_step(JobLevels.WORKDIR)
            except Exception:
                logger.warning("tracker.update_step(WORKDIR) failed", exc_info=True)

        output_dir = await self.prepare_repository(repo_name, trace=trace, tracker=tracker)

        # Track: switch to “analysis/generation”
        if tracker:
            try:
                await tracker.update_step(JobLevels.SWAGGER_TEST_GENERATION)
            except Exception:
                logger.warning("tracker.update_step(SWAGGER_TEST_GENERATION) failed", exc_info=True)

        result = await load_test_service.generate_tests_from_swagger_url(
            swagger_url=data.url,
            output_dir=output_dir,
            custom_requirement=data.custom_requirement or "",
            host=data.host or "localhost",
            auth=data.auth or False,
            operation_id=context_id,
            db_type=data.db_type
        )
        if result:
            return result, output_dir
        else:
            raise ValueError("Load test generation failed")

    async def setup_test_repository_workflow(
        self,
        result: Any,
        directory_test: Path,
        repo_id: str,
        token_id: str,
        user_id: str,
        git_provider: str,
        auth_token: str | None,
        trace: JobTraceMetaData | None = None,
        tracker: JobTracker | None = None,
    ) -> None:
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

        fetcher, _ = RepoFetcher().get_components(git_provider)
        if not fetcher:
            raise ValueError("Invalid git provider")


        validation_service = RepositoryValidationService(repo_repository=RepoRepository(),user_repository=UserRepository(),git_label_repository=GitLabelRepository())
        
        
        # Track: prechecks (repo/user/token validation)
        if tracker:
            try:
                await tracker.update_step(JobLevels.PRECHECKS)
            except Exception:
                logger.warning("tracker.update_step(PRECHECKS) failed", exc_info=True)
        
        repo_info, user_info, git_label = await  validation_service.validate_repository_access(repo_id, user_id,
                                                                                              token_id)

        repo_name = directory_test.name
        
        # reflect extra metadata into trace if you want
        if trace:
            trace.add_metadata(repository_branch=None, repository_html_url=None, user_email= getattr(user_info, "email", None))
            #trace.add_metadata(user_email= getattr(user_info, "email", None))

        try:
            # Track: auth
            if tracker:
                try:
                    await tracker.update_step(JobLevels.AUTH)
                except Exception:
                    logger.warning("tracker.update_step(AUTH) failed", exc_info=True)
            
            decrypted_token = get_token(git_label.token_value,
                                        user_info.encryption_salt,
                                        git_label.git_hosting,
                                        git_provider,
                                        auth_token,
                                        )


            files = extract_files_for_commit(result, directory_test)

            # Track: context finalize (we’re about to create repo/branch/commit)
            if tracker:
                try:
                    await tracker.update_step(JobLevels.CONTEXT_FINALIZE)
                except Exception:
                    logger.warning("tracker.update_step(CONTEXT_FINALIZE) failed", exc_info=True)

            await self._execute_git_workflow(
                    fetcher=fetcher,
                    token=decrypted_token,
                    repo_name=repo_name,
                    repo_info=repo_info,
                    git_provider=git_provider,
                    files=files,
                    repo_id=repo_id,
                    trace=trace,
                    tracker=tracker,

                )
        except Exception as e:
            self.logger.error(f"Repository creation failed: {e}", exc_info=True)
            if trace:
                trace.record_error(e, summary="git workflow failed")
            raise

    def _rollback_git_operations(
            self,
            fetcher,
            token: str,
            repo_full_name: str,
            branch_name: str,
            created_branch: bool,
            created_repo
    ) -> None:
        """
        Rollback git operations in reverse order.

        Attempts to delete created branch and repository if they exist.
        Logs all rollback attempts and failures without raising exceptions.

        Args:
            fetcher: Git provider fetcher instance
            token: Decrypted authentication token
            repo_full_name: Full repository name/ID
            branch_name: Name of the branch to delete
            created_branch: Whether branch was successfully created
            created_repo: Created repository object (None if not created)

        Returns:
            None. All exceptions are caught and logged.
        """
        # Delete branch first (less destructive)
        if created_branch and repo_full_name and branch_name:
            try:
                fetcher.delete_branch(token, repo_full_name, branch_name)
                self.logger.info(f"Rolled back branch: {branch_name}")
            except Exception as rollback_error:
                self.logger.error(
                    f"Rollback failed (delete branch '{branch_name}'): {rollback_error}"
                )

        # Delete repository (more destructive)
        if created_repo and repo_full_name:
            try:
                fetcher.delete_repository(token, repo_full_name)
                self.logger.info(f"Rolled back repository: {repo_full_name}")
            except Exception as rollback_error:
                self.logger.error(
                    f"Rollback failed (delete repository '{repo_full_name}'): {rollback_error}"
                )

    async def _execute_git_workflow(
            self,
            fetcher,
            token: str,
            repo_name: str,
            repo_info,
            git_provider: str,
            files: list,
            repo_id: str,
            trace: JobTraceMetaData | None = None,
            tracker: JobTracker | None = None,
    ) -> None:
        """Execute git operations with automatic rollback on failure."""
        created_repo = None
        created_branch = False
        repo_full_name = ""
        branch_name = LoadTestConfig.get_branch_name(repo_name)

        try:
            # Create repository
            created_repo = fetcher.create_repository(
                token=token,
                name=repo_name,
                description="",
                visibility=repo_info.visibility
            )

            if not created_repo:
                self.logger.error(f"Repository {repo_id} creation failed")
                return

            self.logger.info(f"Repository {repo_id} created successfully")
            repo_full_name, default_branch = get_repo_info(git_provider, created_repo)

            if tracker:
                try:
                    await tracker.update_step(JobLevels.SOURCE_FETCH)
                except Exception:
                    logger.warning("tracker.update_step(SOURCE_FETCH) failed", exc_info=True)

            created_branch = fetcher.create_branch(
                token, repo_full_name, branch_name, default_branch
            )

            if not created_branch:
                self.logger.error(f"Branch {branch_name} creation failed for {repo_id}")
                raise ValueError(f"Failed to create branch {branch_name}")

            self.logger.info(f"Branch {branch_name} created successfully")

            # Track: VECTOR_STORE as a proxy for “commit assets”
            if tracker:
                try:
                    await tracker.update_step(JobLevels.VECTOR_STORE)
                except Exception:
                    logger.warning("tracker.update_step(VECTOR_STORE) failed", exc_info=True)

            fetcher.commit_files(
                token,
                repo_full_name,
                branch_name,
                files,
                LoadTestConfig.get_commit_message(repo_name),
                author_name=repo_info.repo_author_name,
                author_email=repo_info.repo_author_email
            )

            # Optionally update trace with repo page / branch
            if trace:
                repository_html_url = (
                        getattr(created_repo, "html_url", None) or  # GitHub
                        getattr(created_repo, "web_url", None)  # GitLab
                )
                trace.add_metadata(
                    repository_branch=branch_name,
                    repository_html_url=repository_html_url
                )

        except Exception:
            self._rollback_git_operations(
                fetcher=fetcher,
                token=token,
                repo_full_name=repo_full_name,
                branch_name=branch_name,
                created_branch=created_branch,
                created_repo=created_repo
            )
            raise