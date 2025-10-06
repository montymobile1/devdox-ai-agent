import logging
from typing import Annotated
from fastapi import APIRouter, Depends, status
from app.schemas.load_test import (LoadTestRequest, LoadTestResult)
from app.services.load_test import (
    LoadTestService
)
from app.utils.auth import get_mcp_aware_user_context, UserClaims

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/load_tests",
    response_model=LoadTestResult,
    status_code=status.HTTP_200_OK,
    summary="Load tests using locust repo",
    description="Load tests using locust repo based on output path",
    operation_id="load_tests"
)
async def load_tests(

        user_claims: Annotated[UserClaims, Depends(get_mcp_aware_user_context)],
        request: LoadTestRequest,
        service: Annotated[LoadTestService, Depends(LoadTestService.with_dependency)]
) -> LoadTestResult:
    """
    Load tests using locust repo
    """
    result =await service.load_tests(
        user_claims,
        request
    )
    return result