from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import settings


class AnalyseRequest(BaseModel):
    """Request wrapper for analysis"""
    
    question: Optional[str] = Field(
        default="",
        max_length=300,
        description="A question to ask about the code repository",
    )
    
    repo_alias_name: str = Field(
        ...,
        description="The repository alias name given by the user",
    )