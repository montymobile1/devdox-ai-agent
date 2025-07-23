"""
This module defines the `/analyze` endpoint for the DevDox Agent.

The endpoint is responsible for analyzing a code repository based on a given relative path and user question.
It streams responses from the analysis service and aggregates them into a complete response.

Key functionalities:
- Authenticates and extracts user context via MCP-aware middleware.
- Accepts analysis requests with a question and relative path.
- Invokes the AnalyseService to process the code and generate a textual answer.

"""

import logging
from typing import Annotated, Any, Dict
from fastapi import APIRouter, Depends, status
from app.schemas.analyze import (AnalyseRequest)
from app.services.analyze import (
    AnalyseService
)
from app.utils.auth import get_mcp_aware_user_context, UserClaims

logger = logging.getLogger(__name__)

router = APIRouter()



@router.post(
    "/analyze",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Analyze repo",
    description="Analyze repo based on relative path",
    operation_id="analyze_code"
)
async def answer(
        user_claims: Annotated[UserClaims, Depends(get_mcp_aware_user_context)],
        request: AnalyseRequest,
        service: Annotated[AnalyseService, Depends(AnalyseService.with_dependency)],
) -> Dict[str, Any]:
    """
    Analyze code and return results
    """
    # Collect all chunks into a single response
    full_response = ""
    async for chunk in service.answer_question(
            user_claims,
            request.question,
            request.relative_path
    ):
        full_response += chunk

    return {
        "content": [
            {
                "type": "text",
                "text": full_response
            }
        ]
    }



