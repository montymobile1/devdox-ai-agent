from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class AnalyseRequest(BaseModel):
    """Request wrapper for analysis"""
    
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class AnalyseRequest(BaseModel):
    """Request wrapper for analysis"""
    
 
    # questions: Optional[List[str]] = Field(
    #     default=[],
    #     max_length=settings.MAX_QUESTIONS,
    #     description="A list of questions to ask about the code repository",
    # )
    question: Optional[str] = Field(
        default="",
        max_length=300,
        description="A question to ask about the code repository",
    )
    
    repo_alias_name: str = Field(
        ...,
        description="The repository alias name given by the user",
    )
    
    # # Clean, dedupe, bound lengths, and cap count
    # @field_validator("questions")
    # @classmethod
    # def _sanitize_questions(cls, v: List[str]) -> list[str]:
    #     cleaned: List[str] = []
    #     seen_ci = set()
    #
    #     if v is None:
    #         return []
    #
    #     for q in v:
    #         if q is None:
    #             continue
    #         # normalize whitespace
    #         q = " ".join(str(q).split()).strip()
    #         if not q:
    #             continue
    #
    #         key = q.casefold()
    #         if key in seen_ci:
    #             continue
    #         seen_ci.add(key)
    #         cleaned.append(q)
    #
    #     if not cleaned:
    #         raise ValueError("At least one non-empty question is required.")
    #     return cleaned