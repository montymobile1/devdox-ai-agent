# app/processors/load_test_processor.py
"""
Load test processor for handling load testing jobs
"""

import asyncio
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple
from datetime import datetime, timezone
from app.infrastructure.base_processor import BaseProcessor
from app.schemas.load_test import LoadTestRequest
from app.services.api_test_generator import APITestGenerator


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

        self.logger.info(f"Processing load test for context: {context_id}")

        start_time = datetime.now(timezone.utc)

        try:
            # Simulate load test execution (replace with actual logic)
            result = await self._execute_load_test(context_id,job_payload)

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

    async def _execute_load_test(self, context_id: str, job_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the actual load test"""
        # Replace this with your actual load testing logic
        # This could be:
        # - Executing Locust tests
        # - Running custom load testing scripts
        # - Calling external load testing services



        test_type = job_payload.get("test_type", "locust")


        # Example: Run a Locust test file
        if test_type == "locust":

            job_request= LoadTestRequest(**job_payload.get("data",{}))
            return await self._run_locust_test(job_request,context_id)

        else:
            # Simulate other test types
            await asyncio.sleep(5)  # Simulate processing time
            return {
                "test_type": test_type,
                "duration": 5,
                "requests_per_second": 100,
                "total_requests": 500,
                "success_rate": 0.98
            }

    async def prepare_repository(self, repo_name: str) -> Tuple[Path, str]:
        repo_path = self.base_dir / repo_name
        if repo_path.exists():
            await asyncio.to_thread(shutil.rmtree, repo_path)
        repo_path.mkdir(parents=True, exist_ok=True)
        return repo_path


    async def _run_locust_test(self, data:LoadTestRequest,context_id:str) -> Dict[str, Any]:
        """Run Locust load test"""
        load_test_service = APITestGenerator()


        repo_name = data.get_effective_output_path()

        output_dir = await self.prepare_repository(repo_name)

        return await load_test_service.generate_tests_from_swagger_url(swagger_url=data.url,
                                                                          output_dir=output_dir,
                                                                          custom_requirement=data.custom_requirement or "",
                                                                          host=data.host or "localhost",
                                                                          auth=data.auth or False,
                                                                          operation_id=context_id
                                                                          )
