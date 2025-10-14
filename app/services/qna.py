import logging
from typing import Annotated, List, Optional

from fastapi import Depends
from models_src.repositories.repo import TortoiseRepoStore
from pydantic import BaseModel, EmailStr
from together import AsyncTogether

from app.config import settings
from app.infrastructure.mailing_service import Template
from app.infrastructure.mailing_service.container import get_email_dispatcher
from app.infrastructure.mailing_service.models.context_shapes import FailureNotice, FailureProject
from app.infrastructure.mailing_service.models.context_shapes import Meta, Project, QAPair, QAReport
from app.exceptions.custom_exceptions import QnAGenerationFailed, RepoAnalysisNotCompleted, ResourceNotFound
from app.exceptions.exception_constants import REPO_ANALYSIS_FAILED, REPO_ANALYSIS_NOT_REQUESTED, REPOSITORY_NOT_FOUND
from app.infrastructure.qna.formatters.qna_formatter_text import format_qna_text
from app.infrastructure.qna.qna_generator import generate_project_qna
from app.infrastructure.qna.qna_models import ProjectQnAPackage
from app.schemas.repo import RepoStatus
from app.utils.auth import UserClaims

logger = logging.getLogger(__name__)

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
	
	# ---------- public API ----------
	
	async def get_answers(
			self,
			user_claims: UserClaims,
			repo_alias_name: str
	) -> GetAnswersResponse:
		repo = await self._find_repo_or_raise(user_claims.sub, repo_alias_name)
		await self._ensure_completed_or_notify(repo, user_claims.email, repo_alias_name)
		
		qna_pkg = await self._generate_qna(repo)
		formatted_qna = format_qna_text(qna_pkg, show_debug=False, ascii_bars=True)
		
		await self._send_summary_if_possible(user_claims.email, qna_pkg)
		
		return GetAnswersResponse(qna_pkg=qna_pkg, format_qna_text=formatted_qna)
	
	# ---------- private helpers ----------
	
	async def _find_repo_or_raise(self, user_id: str, alias: str):
		repo = await self.repo_store.find_by_user_and_alias_name(
			user_id=user_id, repo_alias_name=alias
		)
		if not repo:
			raise ResourceNotFound(reason=REPOSITORY_NOT_FOUND)
		return repo
	
	async def _ensure_completed_or_notify(self, repo, user_email: Optional[str], project_name: str) -> None:
		"""If repo analysis isn’t completed, notify (email if possible) and raise."""
		if repo.status == RepoStatus.COMPLETED:
			return
		
		if not repo.status or repo.status != RepoStatus.COMPLETED:
			
			if not repo.status or repo.status.strip() == "":
				raise RepoAnalysisNotCompleted(reason=REPO_ANALYSIS_NOT_REQUESTED)
			
			if repo.status == RepoStatus.FAILED:
				raise RepoAnalysisNotCompleted(reason=REPO_ANALYSIS_FAILED)
			
		raise RepoAnalysisNotCompleted()

	
	async def _generate_qna(self, repo) -> ProjectQnAPackage:
		client = AsyncTogether(api_key=settings.TOGETHER_API_KEY)
		qna_pkg = await generate_project_qna(
			id_for_repo=str(repo.id),
			project_name=repo.repo_name,
			repo_url=repo.html_url,
			repo_system_reference=repo.repo_system_reference,
			together_client=client,
		)
		if not qna_pkg:
			raise QnAGenerationFailed()
		return qna_pkg
	
	async def _send_summary_if_possible(self, user_email: Optional[EmailStr], qna_pkg: ProjectQnAPackage) -> None:
		if not user_email:
			logger.info("Skipping QnA summary email: no user email on claims.")
			return
		try:
			await self.send_qna_summary_email(to_email=user_email, qna_pkg=qna_pkg)
		except Exception:
			logger.error("QnA summary email send failed")
			raise
	
	# ---------- existing mail senders (unchanged) ----------
	
	async def send_qna_summary_email(self, to_email: EmailStr, qna_pkg: ProjectQnAPackage):
		project_context_part = Project(name=qna_pkg.project_name, repo_url=qna_pkg.repo_url)
		meta_context_part = Meta(
			generated_at_iso=qna_pkg.generated_at,
			model=qna_pkg.model or "",
			prompt_version=qna_pkg.prompt_version or None,
		)
		pairs_context_part: List[QAPair] = [
			QAPair(
				question=p.question,
				answer=p.answer,
				confidence=p.confidence,
				insufficient_evidence=p.insufficient_evidence,
				evidence_snippets=p.evidence_snippets,
			)
			for p in qna_pkg.pairs
		]
		qa_report_context = QAReport(project=project_context_part, meta=meta_context_part, pairs=pairs_context_part)
		
		email_dispatcher = get_email_dispatcher()
		await email_dispatcher.send_templated_html(
			to=[to_email],
			template=Template.PROJECT_QNA_SUMMARY,
			context=qa_report_context,
			base_context_shape_config={"by_alias": True},
		)
	
	async def send_qna_summary_failure_email(
			self, to_email: EmailStr, project_name: str, error_message: str, project_repo_url: Optional[str] = None
	):
		project_context_part = FailureProject(name=project_name, repo_url=project_repo_url)
		project_failure_notice = FailureNotice(project=project_context_part, error_message=error_message)
		
		email_dispatcher = get_email_dispatcher()
		await email_dispatcher.send_templated_html(
			to=[to_email],
			template=Template.PROJECT_QNA_SUMMARY_FAILURE,
			context=project_failure_notice,
		)