from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, computed_field, ConfigDict, Field, field_serializer, field_validator, model_validator

from app.infrastructure.queue_job_tracker.trace_formatter import build_error_chain_for_template, make_plain_stacktrace


class JobTraceMetaData(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # ---- identifying fields ----
    repository_html_url: Optional[str] = Field(init=False, default=None)
    user_email: Optional[str] = Field(init=False, default=None)
    repository_branch: Optional[str] = Field(init=False, default=None)
    job_context_id: Optional[str] = Field(init=False, default=None)
    job_type: Optional[str] = Field(init=False, default=None)

    # ---- timing ----
    job_queued_at: Optional[datetime] = Field(init=False, default_factory=lambda: datetime.now(UTC))
    job_started_at: Optional[datetime] = Field(init=False, default=None)
    job_finished_at: Optional[datetime] = Field(init=False, default=None)
    job_settled_at: Optional[datetime] = Field(
        init=False, default=None,
        description="When the queue fully settled/acknowledged the job (post-processing done)",
    )
    
    # ---- error reporting ----
    error_type: Optional[str] = Field(
        init=False, default=None,
        description="Fully-qualified exception class, e.g. 'module.ValueError'"
    )
    error_stacktrace: Optional[str] = Field(
        init=False, default=None,
        description="Full chained traceback text (includes 'from e' causes)"
    )
    error_stacktrace_truncated: bool = Field(init=False, default=False)
    error_summary: Optional[str] = Field(
        init=False, default=None,
        description="Short human-readable message or log-friendly summary"
    )
    
    error_chain: Optional[List[Dict[str, Any]]] = Field(init=False, default=None)
    
    repo_id: Optional[str] = Field(init=False, default=None)
    user_id: Optional[str] = Field(init=False, default=None)
    
    # ---- computed properties --------------------------------------------------
    @computed_field
    @property
    def run_ms(self) -> Optional[int]:
        """Start → Finish"""
        if self.job_started_at is None or self.job_finished_at is None:
            return None
        td = self.job_finished_at - self.job_started_at
        return None if td is None else self._to_ms(td)
    
    @computed_field
    @property
    def total_ms(self) -> Optional[int]:
        """Queue → (Settled if available)."""
        if self.job_queued_at is None or self.job_settled_at is None:
            return None
        td = self.job_settled_at - self.job_queued_at
        return None if td is None else self._to_ms(td)
    
    @computed_field
    @property
    def has_error(self) -> bool:
        return any((self.error_type, self.error_stacktrace, self.error_summary))
    
    # ---- helpers -------------------------------------------------------------
    def add_metadata(
            self,
            repo_id: Optional[str] = None,
            user_id: Optional[str] = None,
            job_context_id: Optional[str] = None,
            job_type: Optional[str] = None,
            repository_branch: Optional[str] = None,
            repository_html_url: Optional[str] = None,
            user_email: Optional[str] = None
    ):
        
        if repo_id:
            self.repo_id = repo_id
        
        if user_id:
            self.user_id = user_id
        
        if job_context_id:
            self.job_context_id = job_context_id
        
        if job_type:
            self.job_type = job_type
        
        if repository_branch:
            self.repository_branch = repository_branch
        
        if repository_html_url:
            self.repository_html_url = repository_html_url
        
        if user_email:
            self.user_email = user_email
        
        return self
    
    def mark_job_started(self, when: Optional[datetime] = None, *, force: bool = False) -> "JobTraceMetaData":
        if self.job_started_at is not None and not force:
            return self
        self.job_started_at = when or datetime.now(UTC)
        return self
    
    def mark_job_finished(self, when: Optional[datetime] = None, *, force: bool = False) -> "JobTraceMetaData":
        if self.job_finished_at is not None and not force:
            return self
        self.job_finished_at = when or datetime.now(UTC)
        return self
    
    def mark_job_settled(self, when: Optional[datetime] = None, *, force: bool = False) -> "JobTraceMetaData":
        if self.job_settled_at is not None and not force:
            return self
        self.job_settled_at = when or datetime.now(UTC)
        return self
    
    def record_error(
        self,
        exc: Optional[BaseException]=None,
        summary: Optional[str] = None,
        include_locals: bool = False,
        max_chars: Optional[int] = 14000,
    ) -> "JobTraceMetaData":
        """
        	:param include_locals: sets whether to avoid leaking secrets, flip to True when debugging only.
        	:param max_chars: lets you cap stored stack size (useful if you’re persisting to DB/logs with limits).
        """
        
        derived_error_summary = None
        
        if exc:
            
            self.error_chain = build_error_chain_for_template(exc, include_location=True, msg_limit=200, include_locals=include_locals)
            
            self.error_type = " → ".join(n["func"] for n in self.error_chain) if self.error_chain else f"{exc.__class__.__module__}.{exc.__class__.__name__}"
            
            outer = self.error_chain[0] if self.error_chain else None
            derived_error_summary = f"{outer['type']}: {outer['msg']}" if outer else ""
            
            self.error_stacktrace, self.error_stacktrace_truncated = make_plain_stacktrace(exc, max_chars=max_chars)
        
        if summary:
            self.error_summary = summary
        elif derived_error_summary:
            self.error_summary = derived_error_summary
        elif self.error_type:
            self.error_summary = self.error_type
        else:
            self.error_summary = ""
        
        return self

    def clear_error(self) -> "JobTraceMetaData":
        self.error_type = self.error_stacktrace = self.error_summary = None
        self.error_stacktrace_truncated = False
        return self
    
    # --- private helper: timedelta -> milliseconds (rounded) ---
    def _to_ms(self, delta: timedelta) -> int:
        # microsecond precision in Python; rounding avoids floor bias
        return int(round(delta.total_seconds() * 1000))
    
    # ---- validation ----------------------------------------------------------
    @field_validator('job_queued_at', 'job_started_at', 'job_finished_at', 'job_settled_at', mode='before')
    @classmethod
    def _ensure_datetime_and_aware(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            v = v.replace('Z', '+00:00')
            v = datetime.fromisoformat(v)
        if not isinstance(v, datetime):
            raise TypeError('must be a datetime or ISO8601 string')
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError('datetime must be timezone-aware')
        return v
    
    @model_validator(mode='after')
    def _check_order(self):
        if self.job_started_at and self.job_started_at < self.job_queued_at:
            raise ValueError('job_started_at cannot be before job_queued_at')
        if self.job_finished_at:
            if self.job_started_at and self.job_finished_at < self.job_started_at:
                raise ValueError('job_finished_at cannot be before job_started_at')
            if self.job_finished_at < self.job_queued_at:
                raise ValueError('job_finished_at cannot be before job_queued_at')
        if self.job_settled_at:
            if self.job_finished_at and self.job_settled_at < self.job_finished_at:
                raise ValueError('job_settled_at cannot be before job_finished_at')
            if self.job_settled_at < self.job_queued_at:
                raise ValueError('job_settled_at cannot be before job_queued_at')
        return self

    # ---- serialization -------------------------------------------------------
    @field_serializer('job_queued_at', 'job_started_at', 'job_finished_at', 'job_settled_at')
    def _serialize_dt(self, dt: Optional[datetime]) -> Optional[str]:
        if dt is None:
            return None
        
        # Prefer 'Z' for UTC
        if dt.utcoffset() == timedelta(0):
            return dt.isoformat().replace('+00:00', 'Z')
        return dt.isoformat()
