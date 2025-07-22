from fastapi import Body, Depends, Query, Path
from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated, Optional
from datetime import datetime
import uuid

from app.schemas.field_constants import (
    GIT_HOSTING_FIELD_DESCRIPTION,
    LABEL_FIELD_DESCRIPTION,
    TOKEN_VALUE_FIELD_DESCRIPTION,
)


class GitLabelBase(BaseModel):
    """Base schema for GitLabel"""

    label: str = Field(..., description=LABEL_FIELD_DESCRIPTION, max_length=100)
    # username: str = Field(
    #     ..., description="Username for the git hosting service", max_length=100
    # )
    git_hosting: str = Field(..., description=GIT_HOSTING_FIELD_DESCRIPTION)
    token_value: str = Field(..., description=TOKEN_VALUE_FIELD_DESCRIPTION)


class GitLabelUpdate(BaseModel):
    label: Optional[str] = Field(None, description=LABEL_FIELD_DESCRIPTION)
    git_hosting: Optional[str] = Field(None, description=GIT_HOSTING_FIELD_DESCRIPTION)
    username: Optional[str] = Field(
        None, description="Username for the git hosting service"
    )
    token_value: Optional[str] = Field(None, description=TOKEN_VALUE_FIELD_DESCRIPTION)


class GitLabelResponse(GitLabelBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(..., description="Unique identifier")
    user_id: str = Field(..., description="User identifier")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record last update timestamp")
    masked_token: str = Field(..., description="The masked repo token")
    username: str = Field(..., description="The repo username")


class GitLabelListResponse(BaseModel):
    items: list[GitLabelResponse]
    total: int
    page: int
    size: int


class ErrorResponse(BaseModel):
    """Schema for error responses"""

    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")




class AddGitTokenRequest:
    def __init__(
        self,
        payload: GitLabelBase = Body(...),
    ):
        self.payload = payload


class DeleteGitTokenRequest:
    def __init__(
        self, git_label_id: uuid.UUID = Path(..., description="The git label id")
    ):
        self.git_label_id = git_label_id


class GitLabelDBCreateDTO(BaseModel):
    label: str = Field(..., description=LABEL_FIELD_DESCRIPTION)
    user_id: str = Field(..., max_length=255, description="Authenticated user id")
    git_hosting: str = Field(
        ..., max_length=50, description=GIT_HOSTING_FIELD_DESCRIPTION
    )
    token_value: str = Field(..., description=TOKEN_VALUE_FIELD_DESCRIPTION)
    masked_token: str = Field(
        ..., description="Masked Access token for the git hosting service"
    )
    username: str = Field(..., description="Username for the git hosting service")

class GetGitLabelByLabelRequest:
    def __init__(
        self,
        label: str = Path(
            description="The label identifying the git labels to retrieve."
        ),
    ):
        self.label = label

class GetGitLabelsRequest:
    def __init__(
        self,
        git_hosting: Optional[str] = Query(
            None, description="Filter by git hosting service"
        ),
    ):
        self.git_hosting = git_hosting
