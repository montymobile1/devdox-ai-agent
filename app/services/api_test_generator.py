"""
API Test Generator - Unified class for API schema processing and test generation
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from together import AsyncTogether

from app.config import settings
from app.schemas.load_test import LoadTestError,DatabaseType
from devdox_ai_locust.schemas.processing_result import SwaggerProcessingRequest
from devdox_ai_locust.utils.swagger_utils import get_api_schema
from devdox_ai_locust.utils.open_ai_parser import OpenAPIParser, Endpoint
from devdox_ai_locust.hybrid_loctus_generator import HybridLocustGenerator


logger = logging.getLogger(__name__)


class APITestGenerator:
    """
    Unified class for API schema processing and test generation.

    Handles the complete workflow:
    1. Fetch API schema from Swagger URL
    2. Parse schema to extract endpoints and API info
    3. Generate test files using AI
    4. Create organized test file structure
    """

    def __init__(self, together_api_key: str = None):
        """
        Initialize the API test generator.

        Args:
            together_api_key: API key for Together AI service. Uses settings.TOGETHER_API_KEY if not provided.
        """
        self.together_api_key = together_api_key or settings.TOGETHER_API_KEY
        self.together_client = AsyncTogether(api_key=self.together_api_key)
        self.parser = OpenAPIParser()
        self.generator = HybridLocustGenerator(ai_client=self.together_client)

    async def generate_tests_from_swagger_url(
            self,
            swagger_url: str,
            output_dir: Path,
            custom_requirement: Optional[str] = "",
            host: Optional[str] = "0.0.0.0",
            auth: bool = False,
            operation_id: str = "",
            db_type: str = DatabaseType.EMPTY
    ) -> List[Dict[Any, Any]]:
        """
        Complete workflow: fetch schema, parse, and generate tests.

        Args:
            swagger_url: URL to fetch Swagger/OpenAPI schema from
            output_dir: Directory to create test files in
            custom_requirement: Custom requirements for test generation
            host: Target host for tests
            auth: Whether to include authentication in tests
            operation_id: Operation ID for tracking/logging

        Returns:
            List of created test file information

        Raises:
            LoadTestError: If any step in the process fails
        """
        logger.info(f"Starting test generation from Swagger URL: {swagger_url}")

        try:
            # Step 1: Fetch API schema
            api_schema = await self._fetch_api_schema(swagger_url)

            if not api_schema:
                raise LoadTestError(
                    "No API schema retrieved",
                    "EMPTY_SCHEMA",
                    {"swagger_url": swagger_url}
                )
            # Step 2: Parse schema
            endpoints, api_info = await self._parse_api_schema(api_schema)


            # Step 3: Generate and create tests
            created_files = await self._generate_and_create_tests(
                endpoints=endpoints,
                api_info=api_info,
                output_dir=output_dir,
                custom_requirement=custom_requirement,
                host=host,
                auth=auth,
                operation_id=operation_id,
                db_type=db_type
            )


            logger.info(
                f"Test generation completed successfully. Created {len(created_files)} files.",
                extra={
                    "operation_id": operation_id,
                    "swagger_url": swagger_url,
                    "total_files": len(created_files),
                    "db_type": db_type
                }
            )

            return created_files

        except LoadTestError:
            # Re-raise LoadTestErrors as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error during test generation: {e}", exc_info=True)
            raise LoadTestError(
                f"Test generation failed unexpectedly: {e}",
                "GENERATION_UNEXPECTED_ERROR",
                {
                    "swagger_url": swagger_url,
                    "operation_id": operation_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )

    async def _fetch_api_schema(self, swagger_url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch API schema with timeout and validation.

        Args:
            swagger_url: URL to fetch schema from

        Returns:
            Parsed API schema dictionary or None if empty

        Raises:
            LoadTestError: If fetch fails or times out
        """
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

    async def _parse_api_schema(self, api_schema: Dict[str, Any]) -> Tuple[List[Endpoint], Dict[str, Any]]:
        """
        Parse API schema with comprehensive error handling.

        Args:
            api_schema: Raw API schema dictionary

        Returns:
            Tuple of (endpoints list, api_info dictionary)

        Raises:
            LoadTestError: If parsing fails
        """
        try:
            # Parse schema in thread pool to avoid blocking
            _ = await asyncio.to_thread(self.parser.parse_schema, api_schema)
            endpoints = await asyncio.to_thread(self.parser.parse_endpoints)
            api_info = await asyncio.to_thread(self.parser.get_schema_info)

            logger.info(f"Successfully parsed API schema: {len(endpoints)} endpoints found")

            return endpoints, api_info

        except Exception as e:
            logger.error(f"Failed to parse API schema: {e}")
            raise LoadTestError(
                f"API schema parsing failed: {e}",
                "SCHEMA_PARSE_ERROR",
                {"error": str(e), "error_type": type(e).__name__}
            )

    async def _generate_and_create_tests(
            self,
            endpoints: List[Endpoint],
            api_info: Dict[str, Any],
            output_dir: Path,
            custom_requirement: Optional[str] = "",
            host: Optional[str] = "0.0.0.0",
            auth: bool = False,
            operation_id: str = "",
            db_type: str = DatabaseType.EMPTY
    ) -> List[Dict[Any, Any]]:
        """
        Generate tests using AI and create test files.

        Args:
            endpoints: List of parsed API endpoints
            api_info: API schema information
            output_dir: Directory to create files in
            custom_requirement: Custom requirements for generation
            host: Target host for tests
            auth: Include authentication
            operation_id: Operation tracking ID

        Returns:
            List of created file information dictionaries

        Raises:
            LoadTestError: If test generation or file creation fails
        """
        try:
            logger.info(
                f"Generating tests for {len(endpoints)} endpoints for {output_dir}",
                extra={"operation_id": operation_id, "output_dir": str(output_dir)}
            )

            # Generate test content
            test_files, test_directories = await self.generator.generate_from_endpoints(
                endpoints=endpoints,
                api_info=api_info,
                custom_requirement=custom_requirement,
                target_host=host,
                include_auth=auth,
                db_type=db_type
            )


            # Create test files
            created_files = []

            # Create workflow files
            if test_directories:
                workflows_dir = output_dir / "workflows"
                workflows_dir.mkdir(exist_ok=True)
                for file_workflow in test_directories:
                    workflow_files = await self.generator._create_test_files_safely(
                        file_workflow, workflows_dir
                    )
                    created_files.extend(workflow_files)

            # Create main test files
            if test_files:
                main_files = await self.generator._create_test_files_safely(
                    test_files, output_dir
                )
                created_files.extend(main_files)

            logger.info(
                f"Successfully created {len(created_files)} test files",
                extra={
                    "operation_id": operation_id,
                    "files": [f.get("filename") for f in created_files]
                }
            )

            return created_files

        except Exception as e:
            logger.error(f"Test generation and file creation failed: {e}", exc_info=True)
            raise LoadTestError(
                f"Test generation failed: {e}",
                "TEST_GENERATION_ERROR",
                {
                    "operation_id": operation_id,
                    "endpoint_count": len(endpoints),
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )

    async def parse_existing_schema(
            self,
            api_schema: Dict[str, Any]
    ) -> Tuple[List[Endpoint], Dict[str, Any]]:
        """
        Parse an existing API schema without fetching.

        Args:
            api_schema: Pre-loaded API schema dictionary

        Returns:
            Tuple of (endpoints list, api_info dictionary)
        """
        return await self._parse_api_schema(api_schema)

    async def generate_tests_from_schema(
            self,
            api_schema: Dict[str, Any],
            output_dir: Path,
            custom_requirement: Optional[str] = "",
            host: Optional[str] = "0.0.0.0",
            auth: bool = False,
            operation_id: str = "",
            db_type: str = DatabaseType.EMPTY
    ) -> List[Dict[Any, Any]]:
        """
        Generate tests from an existing API schema (skip fetch step).

        Args:
            api_schema: Pre-loaded API schema dictionary
            output_dir: Directory to create test files in
            custom_requirement: Custom requirements for test generation
            host: Target host for tests
            auth: Whether to include authentication in tests
            operation_id: Operation ID for tracking/logging

        Returns:
            List of created test file information
        """
        logger.info("Starting test generation from provided schema")

        try:
            # Parse schema
            endpoints, api_info = await self._parse_api_schema(api_schema)

            # Generate and create tests
            created_files = await self._generate_and_create_tests(
                endpoints=endpoints,
                api_info=api_info,
                output_dir=output_dir,
                custom_requirement=custom_requirement,
                host=host,
                auth=auth,
                operation_id=operation_id,
                db_type=db_type
            )

            logger.info(f"Test generation from schema completed. Created {len(created_files)} files.")
            return created_files

        except LoadTestError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during schema-based test generation: {e}")
            raise LoadTestError(
                f"Schema-based test generation failed: {e}",
                "SCHEMA_GENERATION_ERROR",
                {
                    "operation_id": operation_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )