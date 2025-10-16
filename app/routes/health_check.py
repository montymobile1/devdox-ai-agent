import logging
from typing import  Any, Dict
from fastapi import APIRouter, status

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health-check",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Health check",
    operation_id="Health"
)
async def get_health_check(
) -> Dict[str, Any]:
    return {"status": "running"}