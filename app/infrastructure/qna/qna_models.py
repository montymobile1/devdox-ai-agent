from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone

from app.infrastructure.qna.qna_utils import snippet_calculator

class QAPair(BaseModel):
	id: str = Field(..., description="Stable identifier for the question (e.g., 'goal')")
	question: str = Field(..., description="Human-readable question text")
	answer: str = Field(..., description="LLM-produced answer")
	
	confidence: float = Field(
		0.0,
		ge=0.0, le=1.0,
		description="Confidence score in [0.0, 1.0]. 1.0 = very confident. Its a score of how strongly this answer is grounded in the provided analysis text (not a global probability of truth). Higher = more direct, well-supported evidence."
	)
	
	
	insufficient_evidence: bool = Field(
		False,
		description="Set True if the analysis text does not provide enough support to answer confidently"
	)
	
	evidence_snippets: List[str] = Field(
		default_factory=list,
		min_length=0,
		max_length=2,
		description="Up to two short snippets (quotes/paraphrases) from the analysis that support the answer."
	)
	
	@field_validator("evidence_snippets")
	@classmethod
	def cap_snippets_len(cls, v: List[str]) -> List[str]:
		# defense-in-depth against oversized items
		return snippet_calculator(v)

class ProjectQnAPackage(BaseModel):
	project_name: str = Field(
		...,
		description="Human-friendly name of the project (as you want it to appear to users)."
	)
	
	repo_url: str = Field(  # switch to AnyUrl if you want strict URL validation
		...,
		description="Repository URL (e.g., https://github.com/org/repo or https://gitlab.com/group/proj)."
	)
	
	repo_id: str = Field(
		...,
		description="Your internal or provider identifier for the repo."
	)
	
	pairs: List[QAPair] = Field(
		default_factory=list,
		description="List of Q&A pairs produced for this repository."
	)
	
	generated_at: datetime = Field(
		default_factory=lambda: datetime.now(timezone.utc),
		description="UTC timestamp of when the Q&A package was generated."
	)
	
	# Optional provenance/debug fields (handy in logs/UI; safe to ignore if you want)
	model: Optional[str] = Field(
		default=None,
		description="LLM used to generate the answers (e.g., 'Llama-3.3-70B-Instruct-Turbo')."
	)
	
	prompt_version: Optional[str] = Field(
		default=None,
		description="Version tag of your prompt template (helps compare runs over time)."
	)
	
	prompt_tokens_hint: Optional[int] = Field(
		default=None,
		description="Optional count/estimate of prompt tokens used (for cost/usage tracking)."
	)
	
	raw_prompt: Optional[str] = Field(
		default=None,
		description="The exact prompt text sent to the LLM (useful for audits and debugging)."
	)
	
	raw_response: Optional[str] = Field(
		default=None,
		description="The raw LLM response before parsing (store sparingly if large)."
	)

