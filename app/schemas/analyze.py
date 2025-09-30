from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class AnalyseRequest(BaseModel):
    """Request wrapper for analysis"""
    
    # example: "How does the authentication system work?"
    questions: Optional[List[str]] = Field(
        default=[],
        max_length=settings.MAX_QUESTIONS,
        description="A list of questions to ask about the code repository",
    )
    
    # example: "my-project"
    relative_path: str = Field(
        ...,
        description="The relative path to the repository",
    )
    
    # Clean, dedupe, bound lengths, and cap count
    @field_validator("questions")
    @classmethod
    def _sanitize_questions(cls, v: List[str]) -> list[str]:
        cleaned: List[str] = []
        seen_ci = set()
        
        if not v:
            return []
        
        for q in v:
            if q is None:
                continue
            # normalize whitespace
            q = " ".join(str(q).split()).strip()
            if not q:
                continue

            key = q.casefold()
            if key in seen_ci:
                continue
            seen_ci.add(key)
            cleaned.append(q)

        if not cleaned:
            raise ValueError("At least one non-empty question is required.")
        return cleaned