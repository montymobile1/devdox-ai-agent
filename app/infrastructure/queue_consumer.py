from app.infrastructure.supabase_queue import SupabaseQueue
from app.services.load_test_processor import LoadTestProcessor
import asyncio
import logging
from app.config import settings
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


class QueueConsumer:
    """Main queue consumer class"""

    def __init__(
            self,
            queue: SupabaseQueue,
            workers: int = 4,
            poll_interval: float = 2.0,
            max_processing_time: int = 1800,  # 30 minutes
    ):
        self.queue = queue
        self.workers = workers
        self.poll_interval = poll_interval
        self.max_processing_time = max_processing_time
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.processed_count = 0
        self.failed_count = 0
        self.logger = logging.getLogger(__name__)

        # Initialize task processors
        self.processors = {
            "load_tests": LoadTestProcessor(),
            "load_locust": LoadTestProcessor()

        }


    async def start(self):
        """Start the consumer with multiple workers"""
        self.running = True
        self.logger.info(f"Starting queue consumer with {self.workers} workers")

        # Create worker tasks
        for i in range(self.workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)

        # Start stats reporter
        stats_task = asyncio.create_task(self._stats_reporter())
        self.worker_tasks.append(stats_task)

        try:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Consumer error: {e}")
        finally:
            self.logger.info("Consumer stopped")

    async def stop(self):
        """Gracefully stop the consumer"""
        self.logger.info("Stopping queue consumer...")
        self.running = False

        # Cancel all worker tasks
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        await self.queue.close()
        self.logger.info("Consumer stopped gracefully")

    async def _worker(self, worker_id: str):
        """Individual worker that processes jobs from the queue"""
        self.logger.info(f"{worker_id} started")

        while self.running:
            try:
                # Try to get a job from the queue
                job = await self._dequeue_job("testing", ["load_locust", "load_tests"])

                if job:
                    await self._process_job(worker_id, job)
                else:
                    # No jobs available, wait before polling again
                    await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                self.logger.info(f"{worker_id} cancelled")
                raise
            except Exception as e:
                self.logger.error(f"{worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

        self.logger.info(f"{worker_id} stopped")

    async def _dequeue_job(self, queue_name: str, job_types: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Dequeue a job from the specified queue"""
        try:

            result = await self.queue.dequeue(queue_name, job_types=job_types)
            if result:
                await self._process_job(queue_name, result)
            else:
                # No jobs available, wait before checking again
                await asyncio.sleep(settings.QUEUE_POLLING_INTERVAL_SECONDS or 5)
                return None

            return result

        except Exception as e:
            self.logger.error(f"Failed to dequeue job: {e}")
            return None

    async def _process_job(self, worker_id: str, job_data: Dict[str, Any]):
        """Process a single job"""
        job_id = job_data.get("context_id") or job_data.get("pgmq_msg_id", "unknown")

        job_type = job_data.get("job_type", "unknown")


        start_time = datetime.now(timezone.utc)
        self.logger.info(f"{worker_id} processing job {job_id} of type {job_type}")

        try:
            # Get the appropriate processor
            processor = self.processors.get(job_type)

            if not processor:
                raise ValueError(f"No processor found for job type: {job_type}")

            # Process with timeout
            result = await asyncio.wait_for(
                processor.process(job_data),
                timeout=self.max_processing_time
            )


            # Mark job as completed
            success = await self.queue.complete_job(job_data, result)

            if success:
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                self.processed_count += 1
                self.logger.info(
                    f"{worker_id} completed job {job_id} in {processing_time:.2f}s",
                    extra={
                        "job_id": job_id,
                        "job_type": job_type,
                        "processing_time": processing_time,
                        "worker_id": worker_id
                    }
                )
            else:
                self.logger.error(f"{worker_id} failed to mark job {job_id} as completed")

        except asyncio.TimeoutError:
            self.logger.error(f"{worker_id} job {job_id} timed out after {self.max_processing_time}s")
            self.failed_count += 1
        except Exception as e:
            self.logger.error(f"{worker_id} failed to process job {job_id}: {e}", exc_info=True)
            self.failed_count += 1

            # For failed jobs, we let them timeout in the queue for retry
            # PGMQueue will make them available again after visibility timeout

    async def _stats_reporter(self):
        """Periodically report queue statistics"""
        while self.running:
            try:
                stats = await self.queue.get_queue_stats("testing")

                self.logger.info(
                    f"Queue stats - Queued: {stats.get('queued', 0)}, "
                    f"Processed: {self.processed_count}, Failed: {self.failed_count}"
                )
                await asyncio.sleep(30)  # Report every 30 seconds
            except Exception as e:
                self.logger.error(f"Stats reporter error: {e}")
                await asyncio.sleep(30)
