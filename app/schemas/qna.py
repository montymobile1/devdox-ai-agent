from pydantic import BaseModel, Field


class QnARequest(BaseModel):
    """Request wrapper for analysis"""
    # example: "my-project"
    relative_path: str = Field(
        ...,
        description="The relative path to the repository",
    )