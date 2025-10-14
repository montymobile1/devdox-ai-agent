from typing import Annotated

from fastapi import Depends
from models_src.repositories.repo import TortoiseRepoStore
from pydantic import BaseModel
from together import AsyncTogether

from app.config import settings
from app.exceptions.custom_exceptions import QnAGenerationFailed, RepoAnalysisNotCompleted, ResourceNotFound
from app.exceptions.exception_constants import REPO_ANALYSIS_FAILED, REPO_ANALYSIS_NOT_REQUESTED, REPOSITORY_NOT_FOUND
from app.infrastructure.qna.formatters.qna_formatter_text import format_qna_text
from app.infrastructure.qna.qna_generator import generate_project_qna
from app.infrastructure.qna.qna_models import ProjectQnAPackage
from app.schemas.repo import RepoStatus
from app.utils.auth import UserClaims

class GetAnswersResponse(BaseModel):
	qna_pkg: ProjectQnAPackage | None = None
	format_qna_text: str | None = None

class QnAService:
	
	def __init__(self, repo_store: TortoiseRepoStore):
		
		self.repo_store = repo_store
	
	@classmethod
	def with_dependency(
			cls,
			repo_store: Annotated[TortoiseRepoStore, Depends()],
	) -> "QnAService":
		return cls(repo_store)
	
	async def get_answers(
			self,
			user_claims: UserClaims,
			repo_alias_name: str
	) -> GetAnswersResponse:
		
		# Figure out which repo we’re supposed to analyze for this user + path.
		# If we can’t find that repo, exit early
		repo_info = await self.repo_store.find_by_user_and_alias_name(
			user_id=user_claims.sub, repo_alias_name=repo_alias_name
		)
		
		if not repo_info:
			raise ResourceNotFound(reason=REPOSITORY_NOT_FOUND)
		
		if repo_info.status != RepoStatus.COMPLETED:
			if repo_info.status == RepoStatus.FAILED:
				raise RepoAnalysisNotCompleted(reason=REPO_ANALYSIS_FAILED)
			elif repo_info.status.strip() == "":
				raise RepoAnalysisNotCompleted(reason=REPO_ANALYSIS_NOT_REQUESTED)
			else:
				raise RepoAnalysisNotCompleted()
		
		async_together_client = AsyncTogether(api_key=settings.TOGETHER_API_KEY)
		
		qna_pkg:ProjectQnAPackage = await generate_project_qna(
			id_for_repo=str(repo_info.id),
			project_name=repo_info.repo_name,
			repo_url=repo_info.html_url,
			repo_system_reference=repo_info.repo_system_reference,
			together_client=async_together_client
		)
		
		if not qna_pkg:
			raise QnAGenerationFailed()
		
		formatted_qna = format_qna_text(qna_pkg, show_debug=False, ascii_bars=True)
		
		return GetAnswersResponse(
			qna_pkg=qna_pkg,
			format_qna_text=formatted_qna
		)