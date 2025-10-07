import traceback

from app.config import settings
from app.infrastructure.mailing_service import Template
from app.infrastructure.mailing_service.container import get_email_dispatcher
from app.infrastructure.mailing_service.models.context_shapes import ProjectAnalysisFailure, ProjectAnalysisSuccess
from app.infrastructure.queue_job_tracker.job_trace_metadata import JobTraceMetaData
from app.infrastructure.queue_job_tracker.job_tracker import JobLevels, JobTracker, JobTrackerManager
from app.infrastructure.supabase_queue import SupabaseQueue
from app.services.load_test_processor import LoadTestProcessor
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, UTC

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
        self.permanent_failures = 0
        self.retries_scheduled = 0
        self.logger = logging.getLogger(__name__)

        # Initialize task processors
        self.processors = {
            "load_tests": LoadTestProcessor(),
            "load_locust": LoadTestProcessor()
        }
        
        self.email_dispatch = get_email_dispatcher()

        self._tracker_manager = JobTrackerManager()

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

        queue_name = "testing"

        while self.running:
            try:
                # Try to get a job from the queue
                job = await self._dequeue_job(queue_name, ["load_locust", "load_tests"])

                if not job:
                    await asyncio.sleep(self.poll_interval)
                    continue

                # build a trace object for this job
                trace = JobTraceMetaData()
                # try to pick the enqueue time if present, otherwise keep default
                if job.get("scheduled_at"):
                    # string is OK; your validator accepts ISO and ensures tz-aware
                    trace.job_queued_at = job["scheduled_at"]

                payload = job.get("payload") or {}
                # enrich trace with known identifiers (optional)
                trace.add_metadata(
                    repo_id=payload.get("repo_id"),
                    user_id=payload.get("user_id"),
                    job_context_id=payload.get("context_id"),
                    job_type=job.get("job_type"),
                )

                # claim with JobTrackerManager
                tracker: Optional[JobTracker] = None
                claim = await self._tracker_manager.try_claim(
                    worker_id=worker_id,
                    message_id=str(job.get("pgmq_msg_id", job.get("id"))),
                    queue_name=job.get("queue_name", queue_name),
                )

                if claim.qualifies_for_tracking and claim.tracker:
                    tracker = claim.tracker
                    await tracker.start()
                    await tracker.update_step(JobLevels.DISPATCH)

                # hand off to processor
                await self._process_job(
                    worker_id,
                    job,
                    trace=trace,
                    tracker=tracker,
                )

            except asyncio.CancelledError:
                self.logger.info(f"{worker_id} cancelled")
                raise
            except Exception as e:
                self.logger.error(f"{worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

        self.logger.info(f"{worker_id} stopped")

    async def _dequeue_job(
        self,
        queue_name: str,
        job_types: Optional[List[str]] = None,
        trace: JobTraceMetaData | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Dequeue a job from the specified queue"""
        try:

            result = await self.queue.dequeue(queue_name, job_types=job_types)
            return result

        except Exception as e:
            self.logger.error(f"Failed to dequeue job: {e}")
            # record error into trace if provided
            if trace:
                trace.record_error(e, summary="Dequeue failure")
            return None

    async def _process_job(
        self,
        worker_id: str,
        job_data: Dict[str, Any],
        trace: JobTraceMetaData | None = None,
        tracker: JobTracker | None = None,
    ):
        """Process a single job with retry-before-fail handling."""
        job_id = job_data.get("context_id") or job_data.get("pgmq_msg_id", "unknown")

        job_type = job_data.get("job_type", "unknown")


        start_time = datetime.now(timezone.utc)

        # begin timing in the trace
        if trace:
            trace.mark_job_started(start_time)

        # mark PROCESSING step
        if tracker:
            try:
                await tracker.update_step(JobLevels.PROCESSING)
            except Exception:
                # don't fail job if bookkeeping fails
                self.logger.warning("tracker.update_step(PROCESSING) failed", exc_info=True)

        self.logger.info(f"{worker_id} processing job {job_id} of type {job_type}")

        try:
            # Get the appropriate processor
            processor = self.processors.get(job_type)

            if not processor:
                raise ValueError(f"No processor found for job type: {job_type}")

            # Process with timeout
            result = await asyncio.wait_for(
                processor.process(job_data, trace=trace, tracker=tracker),
                timeout=self.max_processing_time
            )

            # If processors ever return an explicit failure payload, treat as failure (retryable)
            if isinstance(result, dict) and result.get("status") == "failed":
                # synthesize an exception to capture in trace + retry handler
                
                exc = RuntimeError(result.get("error") or "Processor reported failed status")
                await self._retry_or_archive(job_data, exc, trace, tracker, "Processor reported failed status")
                return

            # Normal success path
            finished_at = datetime.now(UTC)
            if trace:
                trace.mark_job_finished(finished_at)

            # Mark job as completed
            success = await self.queue.complete_job(job_data, result)

            if success:
                # Update tracker for queue ack + completion
                if tracker:
                    try:
                        await tracker.update_step(JobLevels.QUEUE_ACK)
                        await tracker.completed()
                    except Exception:
                        self.logger.warning("tracker completion bookkeeping failed", exc_info=True)

                if trace:
                    trace.mark_job_settled(datetime.now(UTC))

                processing_time = (finished_at - start_time).total_seconds()
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
                
                await self._notify_success(job_tracer=trace)
                
            else:
                # If we failed to delete/ack the message, schedule a retry so it doesn't get stuck
                exc = RuntimeError("Queue complete_job returned False")
                await self._retry_or_archive(job_data, exc, trace, tracker, "Queue complete_job returned False")   # etc.


        except asyncio.TimeoutError as e:
            self.logger.error(f"{worker_id} job {job_id} timed out after {self.max_processing_time}s")
            await self._retry_or_archive(job_data, e, trace, tracker, "Processor timeout")
        except Exception as e:
            self.logger.error(f"{worker_id} failed to process job {job_id}: {e}", exc_info=True)
            await self._retry_or_archive(job_data, e, trace, tracker, "Processing failure")
    
    async def _retry_or_archive(self, job_data, exc, trace, tracker, summary: str):
        if trace:
            trace.record_error(exc, summary=summary)
        error_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        perma, handled_ok = await self.queue.fail_job(
            job_data=job_data,
            error=exc,
            job_tracker_instance=tracker,
            job_tracer=trace,
            error_trace=error_trace,
            retry=True,
        )
        self._record_failure_outcome(perma)
        
        if perma:
            await self._notify_permanent_failure(job_tracer=trace)
        
        return perma, handled_ok
    
    def _record_failure_outcome(self, perma: bool):
        if perma:
            self.permanent_failures += 1
        else:
            self.retries_scheduled += 1
    
    async def _stats_reporter(self):
        """Periodically report queue statistics"""
        while self.running:
            try:
                stats = await self.queue.get_queue_stats("testing")
                self.logger.info(
                    "Queue stats - Queued: %s, Processed: %s, PermanentFails: %s, RetriesScheduled: %s",
                    stats.get("queued", 0),
                    self.processed_count,
                    self.permanent_failures,
                    self.retries_scheduled,
                )
                await asyncio.sleep(30)  # Report every 30 seconds
            except Exception as e:
                self.logger.error(f"Stats reporter error: {e}")
                await asyncio.sleep(30)
    
    async def _notify_permanent_failure(self, job_tracer):
        try:
            
            serialized_model = job_tracer.model_dump()
            
            if not settings.mail.MAIL_AUDIT_RECIPIENTS:
                raise RuntimeError("MAIL_AUDIT_RECIPIENTS is not configured")
            
            context = ProjectAnalysisFailure(
                repository_html_url=serialized_model["repository_html_url"],
                user_email=serialized_model["user_email"],
                repository_branch=serialized_model["repository_branch"],
                job_context_id=serialized_model["job_context_id"],
                job_type=serialized_model["job_type"],
                job_queued_at=serialized_model["job_queued_at"],
                job_started_at=serialized_model["job_started_at"],
                job_finished_at=serialized_model["job_finished_at"],
                job_settled_at=serialized_model["job_settled_at"],
                error_type=serialized_model["error_type"],
                error_summary=serialized_model["error_summary"],
                error_chain=serialized_model["error_chain"],
                run_ms=serialized_model["run_ms"],
                total_ms=serialized_model["total_ms"],
                user_id=serialized_model["user_id"],
                repo_id=serialized_model["repo_id"],
            )
            
            email_dispatcher = get_email_dispatcher()
            await email_dispatcher.send_templated_html(
                to=settings.mail.MAIL_AUDIT_RECIPIENTS,
                template=Template.PROJECT_ANALYSIS_FAILURE,
                context=context,
            )
        except Exception:
            self.logger.warning("Notifier dev alert failed", exc_info=True)

    async def _notify_success(self, job_tracer):
        try:
            
            serialized_model = job_tracer.model_dump()
            
            context = ProjectAnalysisSuccess(
                repository_html_url=serialized_model.get("repository_html_url"),
                repository_branch=serialized_model.get("repository_branch"),
                job_type=serialized_model.get("job_type"),
                job_queued_at=serialized_model.get("job_queued_at")
            )
            
            email_dispatcher = get_email_dispatcher()
            await email_dispatcher.send_templated_html(
                to=[job_tracer.user_email],
                template=Template.PROJECT_ANALYSIS_SUCCESS,
                context=context,
            )
        except Exception:
            self.logger.warning("Notifier user success failed", exc_info=True)
    