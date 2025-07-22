from fastapi import Body, Depends, Query, Path

from pydantic import BaseModel, Field

class AnalyseBase(BaseModel):
    """Base schema for GitLabel"""

    question: str = Field(
        ...,
        description="The question to ask about the code repository",
        example="How does the authentication system work?"
    )

    relative_path: str = Field(
        ...,
        description="The relative path to the repository",
        example="my-project"
    )


class AnalyseRequest(BaseModel):
    """Request wrapper for analysis"""

    question: str = Field(
        ...,
        description="The question to ask about the code repository",
        example="How does the authentication system work?"
    )

    relative_path: str = Field(
        ...,
        description="The relative path to the repository",
        example="my-project"
    )