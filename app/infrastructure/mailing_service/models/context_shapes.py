from typing import Any, Dict, List, Optional

from pydantic import BaseModel

class BaseContextShape(BaseModel):
    """Marker base for all email context models."""

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