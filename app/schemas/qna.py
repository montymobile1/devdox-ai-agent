from pydantic import BaseModel, Field


class QnARequest(BaseModel):
    """Request wrapper for analysis"""
    repo_alias_name: str = Field(
        ...,
        description="The repository alias name",
    )