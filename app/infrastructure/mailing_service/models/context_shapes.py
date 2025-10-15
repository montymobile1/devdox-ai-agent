from datetime import datetime
from math import isclose
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, computed_field, ConfigDict, Field, field_validator

from app.config import settings
from app.infrastructure.mailing_service.constants import EVIDENCE_AND_CONFIDENCE_DISPLAY_DATA_CAP


class BaseContextShape(BaseModel):
    """Marker base for all email context models."""

# --------------------------------------------
# Project Analysis Context
# --------------------------------------------

class ProjectAnalysisFailureTemplateLayout(BaseModel):
    """
    Fields that control width and spacing internally inside the template; no need to pass
    it manually when instance creating—just change it here.

    - `__template_layout_rail_w` -> `RAIL_W` (gutter width)
        Think of this as a fixed left "lane" reserved for the tree graphics (the dots,
        dashed line, and elbows).
            - Bigger RAIL_W -> more breathing room for the tree art; the text starts
              further to the right.
            - Smaller RAIL_W -> the text starts closer to the left edge.
            - Use when: long function names or you want the connectors to feel roomy.
            - Typical range: 120-220 px (default in examples: 120).

    - `__template_layout_step` -> `STEP` (spacing between levels)
        How far each deeper level appears to be indented inside the rail.
            - Bigger STEP -> clearer visual separation between levels, but the dots/elbows
              spread out horizontally.
            - Smaller STEP -> tighter, more compact levels.
            - Use when: levels look cramped (increase) or too stretched (decrease).
            - Typical range: 14-24 px (default: 14).

    - `__template_layout_max_levels` -> `MAX_LEVELS` (visual clamp)
        The maximum visible depth we draw inside the rail before we stop indenting further.
        If the real chain is deeper, we show a little "+N" badge indicating how many extra
        levels are hidden visually, but the content still lines up neatly because we don't
        keep pushing it to the right.
            - Higher MAX_LEVELS -> you see more real indentation before clamping.
            - Lower MAX_LEVELS -> keeps things tight for very deep stacks.
            - Use when: emails from some jobs get very deep and start to look messy -- lower
              this to keep layout tidy.
            - Typical range: 5-10 (default: 5).
    """
    
    # Changeable
    template_layout_rail_w:int = 120
    template_layout_step:int = 14
    template_layout_max_levels:int = 5

    # Do not change
    template_layout_rail_w__default:int = 120
    template_layout_step__default:int = 14
    template_layout_max_levels__default:int = 5

class ProjectAnalysisFailure(BaseContextShape, ProjectAnalysisFailureTemplateLayout):
    repo_id: Optional[str] = None
    user_id: Optional[str] = None
    repository_html_url: Optional[str] = None
    user_email: Optional[str] = None
    repository_branch: Optional[str] = None
    job_context_id: Optional[str] = None
    job_type: Optional[str] = None
    job_queued_at: Optional[str] = None
    job_started_at: Optional[str] = None
    job_finished_at: Optional[str] = None
    job_settled_at: Optional[str] = None
    error_type: Optional[str] = None
    error_stacktrace: Optional[str] = None
    error_stacktrace_truncated: Optional[bool] = None
    error_summary: Optional[str] = None
    error_chain: Optional[List[Dict[str, Any]]] = None
    run_ms: Optional[int] = None
    total_ms: Optional[int] = None

class ProjectAnalysisSuccess(BaseContextShape):
    repository_html_url: Optional[str] = None
    repository_branch: Optional[str] = None
    job_type: Optional[str] = None
    job_queued_at: Optional[str] = None

# --------------------------------------------
# Question and Answer Summary Context
# --------------------------------------------

class Project(BaseModel):
    """Required: name, repo_url (used directly in UI)."""
    name: str = Field(min_length=1)
    repo_url: str


class Meta(BaseModel):
    """
    generated_at_iso is required and parsed into datetime.
    model and prompt_version are optional per the Jinja guards.
    """
    model_config = ConfigDict(populate_by_name=True)

    generated_at: datetime = Field(alias="generated_at_iso")
    model: Optional[str] = None
    prompt_version: Optional[int] = None

    @field_validator("generated_at")
    @classmethod
    def _must_be_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("generated_at_iso must be timezone-aware (include offset).")
        return v

def compute_show_conf_and_evidence(confidence) -> bool:
    cap = EVIDENCE_AND_CONFIDENCE_DISPLAY_DATA_CAP
    eps = 1e-9
    
    if (confidence < cap) or isclose(confidence, cap, abs_tol=eps) or settings.API_DEBUG:
        return True
    
    return False

class QAPair(BaseModel):
    """
    All core fields are required.
    evidence_snippets is optional in the UI; default to [].
    """
    question: str = Field(min_length=1)
    answer: str
    insufficient_evidence: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_snippets: List[str] = Field(default_factory=list)
    
    #show_conf_and_evidence: Optional[bool] = Field(default=True)
    @computed_field
    @property
    def show_conf_and_evidence(self) -> bool:
        """Whether to show the confidence and evidence snippets in the html"""
        return compute_show_conf_and_evidence(self.confidence)
    
    
    # Convenience properties to mirror the template logic:
    @computed_field
    @property
    def inferred(self) -> bool:
        """True if the answer string begins with 'Inferred:' (UI shows a badge)."""
        return self.answer.startswith("Inferred:")

    @computed_field
    @property
    def normalized_answer(self) -> str:
        """Answer with leading 'Inferred:' removed (how the UI prints it)."""
        return self.answer[9:].lstrip() if self.inferred else self.answer

    @computed_field
    @property
    def confidence_pct(self) -> int:
        """0–100 integer for progress bar width / label."""
        return round(self.confidence * 100)


class QAReport(BaseContextShape):
    """
    Root document:
    - project: required
    - meta: required (with generated_at_iso required, others optional)
    - pairs: required list (can be empty)
    """
    project: Project
    meta: Meta
    pairs: List[QAPair]
    
    