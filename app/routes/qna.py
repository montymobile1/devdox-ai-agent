import logging
from typing import Annotated, Any, Dict
from fastapi import APIRouter, Depends, status

from app.schemas.qna import QnARequest
from app.services.qna import GetAnswersResponse, QnAService
from app.utils.auth import get_mcp_aware_user_context, UserClaims

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/qna_summary",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Answers a set of default questions",
    description="This API uses a set of default defined questions to return a MCP compliant set of answers, while also sending an email to the user",
    operation_id="qna_summary"
)
async def answer(
        #user_claims: Annotated[UserClaims, Depends(get_mcp_aware_user_context)],
        request: QnARequest,
        service: Annotated[QnAService, Depends(QnAService.with_dependency)],
) -> Dict[str, Any]:
    
    user_claims = UserClaims(email="mohamadali.jaafar@montymobile.com", sub="user_2xioBPMzrTczyKDABvynLeToHst")
    
    get_answers_response:GetAnswersResponse = await service.get_answers(
        user_claims=user_claims,
        repo_alias_name=request.repo_alias_name
    )
    
    full_response = get_answers_response.format_qna_text
    
    return {
        "content": [
            {
                "type": "text",
                "text": full_response
            }
        ]
    }



