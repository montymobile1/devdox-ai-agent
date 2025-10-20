import logging
from typing import Annotated, Any, List, Optional

from encryption_src.fernet.service import FernetEncryptionHelper
from fastapi import Depends
from models_src.repositories.code_chunks import TortoiseCodeChunksStore
from models_src.repositories.repo import TortoiseRepoStore
from models_src.repositories.user import TortoiseUserStore
from pydantic import BaseModel, EmailStr
from together import AsyncTogether

from app.config import settings
from app.infrastructure.mailing_service import EmailDispatcher, Template
from app.infrastructure.mailing_service.container import get_email_dispatcher
from app.infrastructure.mailing_service.models.context_shapes import Meta, Project, QAPair, QAReport
from app.exceptions.custom_exceptions import QnAGenerationFailed, RepoAnalysisNotCompleted, ResourceNotFound
from app.exceptions.exception_constants import REPO_ANALYSIS_FAILED, REPO_ANALYSIS_NOT_REQUESTED, REPOSITORY_NOT_FOUND
from app.infrastructure.qna.formatters.qna_formatter_text import format_qna_text
from app.infrastructure.qna.qna_generator import generate_project_qna, get_default_questions
from app.infrastructure.qna.qna_models import ProjectQnAPackage
from app.schemas.repo import RepoStatus
from app.utils.auth import UserClaims
from app.utils.encryption import get_encryption_helper

from cohere import AsyncClientV2 as CohereAsyncClientV2

logger = logging.getLogger(__name__)

BERT_EMBEDDING_MODEL_1 = "togethercomputer/m2-bert-80M-32k-retrieval"
TOTAL_K = 10  # context budget

class GetAnswersResponse(BaseModel):
	qna_pkg: ProjectQnAPackage | None = None
	format_qna_text: str | None = None
	user_email: EmailStr | None = None

class QnAService:
	
	def __init__(
			self,
			repo_store: TortoiseRepoStore,
			email_dispatcher: EmailDispatcher,
			code_chunks_store: TortoiseCodeChunksStore,
			user_store: TortoiseUserStore,
			encryption_service: FernetEncryptionHelper,
	):
		
		self.repo_store = repo_store
		self.email_dispatcher = email_dispatcher
		self.code_chunks_store = code_chunks_store
		self.user_store = user_store
		self.encryption_service = encryption_service
	
	@classmethod
	def with_dependency(
			cls,
			repo_store: Annotated[TortoiseRepoStore, Depends()],
			email_dispatcher: Annotated[EmailDispatcher, Depends(get_email_dispatcher)],
			code_chunks_store: Annotated[TortoiseCodeChunksStore, Depends()],
			user_store: Annotated[TortoiseUserStore, Depends()],
			encryption_service: Annotated[FernetEncryptionHelper, Depends(get_encryption_helper)],
			
	) -> "QnAService":
		return cls(
			repo_store=repo_store,
			email_dispatcher=email_dispatcher,
			code_chunks_store=code_chunks_store,
			user_store=user_store,
			encryption_service=encryption_service
		)
	
	# ---------- public API ----------
	
	async def get_answers(
			self,
			user_claims: UserClaims,
			repo_alias_name: str
	) -> GetAnswersResponse:
		repo = await self._find_repo_or_raise(user_claims.sub, repo_alias_name)
		self._ensure_completed_or_die(repo)
		
		qna_pkg = await self._generate_qna(repo)
		formatted_qna = format_qna_text(qna_pkg, show_debug=False, ascii_bars=True)
		
		await self._send_summary_if_possible(user_claims.email, qna_pkg)
		
		return GetAnswersResponse(user_email=user_claims.email, qna_pkg=qna_pkg, format_qna_text=formatted_qna)
	
	# ---------- private helpers ----------
	
	async def _generate_qna(self, repo) -> ProjectQnAPackage:
		
		client = AsyncTogether(api_key=settings.TOGETHER_API_KEY)
		
		questions = [q for _, q in get_default_questions()]
		
		# Build a grounded context from vectors (+ README).
		context_blob = await self._build_context_from_chunks(
			together=client,
			user_id=repo.user_id,
			repo_id=str(repo.id),
			questions=questions,
			top_k=TOTAL_K,
			truncate_chars=12000,
		)
		
		# Fallback: if nothing retrieved, use repo_system_reference
		effective_context = (context_blob or (repo.repo_system_reference or "")).strip()
		
		qna_pkg = await generate_project_qna(
			id_for_repo=str(repo.id),
			project_name=repo.repo_name,
			repo_url=repo.html_url,
			repo_system_reference=effective_context,  # <- grounded context now
			together_client=client,
			# (OPTIONAL) pass explicit questions here if you want to override defaults
			# questions=[("goal", "..."), ...],
		)
		if not qna_pkg:
			raise QnAGenerationFailed()
		return qna_pkg
	
	async def _find_repo_or_raise(self, user_id: str, alias: str):
		repo = await self.repo_store.find_by_user_and_alias_name(
			user_id=user_id, repo_alias_name=alias
		)
		if not repo:
			raise ResourceNotFound(reason=REPOSITORY_NOT_FOUND)
		return repo
	
	def _ensure_completed_or_die(self, repo) -> None:
		"""If repo analysis isn’t completed, raise."""
		if repo.status == RepoStatus.COMPLETED:
			return
			
		if not repo.status or repo.status.strip() == "":
			raise RepoAnalysisNotCompleted(reason=REPO_ANALYSIS_NOT_REQUESTED)
		
		if repo.status == RepoStatus.FAILED:
			raise RepoAnalysisNotCompleted(reason=REPO_ANALYSIS_FAILED)
			
		raise RepoAnalysisNotCompleted()

	async def _send_summary_if_possible(self, user_email: Optional[EmailStr], qna_pkg: ProjectQnAPackage) -> None:
		if not user_email:
			logger.info("Skipping QnA summary email: no user email on claims.")
			return
		try:
			await self.send_qna_summary_email(to_email=user_email, qna_pkg=qna_pkg)
		except Exception:
			logger.error("QnA summary email send failed")
			raise
	
	async def send_qna_summary_email(self, to_email: EmailStr, qna_pkg: ProjectQnAPackage):
		project_context_part = Project(name=qna_pkg.project_name, repo_url=qna_pkg.repo_url)
		meta_context_part = Meta(
			generated_at_iso=qna_pkg.generated_at
		)
		pairs_context_part: List[QAPair] = [
			QAPair(
				question=p.question,
				answer=p.answer,
				confidence=p.confidence,
				insufficient_evidence=p.insufficient_evidence,
				evidence_snippets=p.evidence_snippets
			)
			for p in qna_pkg.pairs
		]
		qa_report_context = QAReport(project=project_context_part, meta=meta_context_part, pairs=pairs_context_part)
		
		await self.email_dispatcher.send_templated_html(
			to=[to_email],
			template=Template.PROJECT_QNA_SUMMARY,
			context=qa_report_context,
			base_context_shape_config={"by_alias": True},
		)
	
	async def _embed_questions(self, client: AsyncTogether, questions: list[str]) -> list[list[float]]:
		resp = await client.embeddings.create(input=questions, model=BERT_EMBEDDING_MODEL_1)
		vecs = [d.embedding for d in resp.data]
		if not vecs:
			raise RuntimeError("empty embeddings")
		dim = len(vecs[0])
		if any(len(v) != dim for v in vecs) or dim != settings.VECTOR_SIZE:
			raise RuntimeError(f"Embedding dim mismatch: got {dim}, expected {settings.VECTOR_SIZE}")
		return vecs
	
	def _decrypt(self, encrypted: str, user_salt_b64: str) -> str:
		if not encrypted:
			return ""
		return self.encryption_service.decrypt_for_user(encrypted, user_salt_b64)
	
	async def _readme_blob(self, user_id: str, repo_id: str, salt_b64: str) -> str:
		rows = await self.code_chunks_store.get_repo_file_chunks(user_id, repo_id, file_name="readme")
		out = []
		for r in rows or []:
			ct = r.get("content")
			if ct:
				out.append(self._decrypt(ct, salt_b64))
		return "\n".join(out).strip()
	
	async def _cohere_rerank_optional(self, query: str, docs: list[str], top_n: int) -> list[int]:
		# Optional precise rerank if key present; otherwise keep original order
		if not settings.COHERE_API_KEY:
			return list(range(min(top_n, len(docs))))
		co = CohereAsyncClientV2(api_key=settings.COHERE_API_KEY)
		rr = await co.rerank(query=query, documents=docs, top_n=min(top_n, len(docs)), model="rerank-v3.5")
		# return indexes into docs
		return [getattr(x, "index", i) for i, x in enumerate(getattr(rr, "results", [])) if hasattr(x, "index")]
	
	async def _build_context_from_chunks(
			self,
			*,
			together: AsyncTogether,
			user_id: str,
			repo_id: str,
			questions: list[str],
			top_k: int = TOTAL_K,
			truncate_chars: int = 12000,
	) -> str:
		# 1) user + salt for decryption
		user = await self.user_store.find_by_user_id(user_id)
		if not user or not user.encryption_salt:
			return ""  # fail safe
		
		salt_b64 = self.encryption_service.decrypt(user.encryption_salt)
		
		# 2) embed questions
		q_vecs = await self._embed_questions(together, questions)
		
		# 3) vector search with fusion (SUM of per-question sims)
		rows: list[dict[str, Any]] = await self.code_chunks_store.get_user_repo_chunks_multi(
			user_id=user_id, repo_id=repo_id, query_embeddings=q_vecs,
			emb_dim=settings.VECTOR_SIZE, limit=top_k
		)
		if not rows:
			return await self._readme_blob(user_id, str(repo_id), salt_b64)
		
		# 4) decrypt and collect texts
		texts, file_names = [], []
		for r in rows:
			enc = r.get("content") or ""
			txt = self._decrypt(enc, salt_b64)
			if txt:
				texts.append(txt)
				file_names.append(r.get("file_name") or "unknown")
		
		if not texts:
			return await self._readme_blob(user_id, str(repo_id), salt_b64)
		
		# 5) optional fine rerank against a merged multi-question query
		merged_query = " | ".join(questions)
		try:
			order = await self._cohere_rerank_optional(merged_query, texts, top_n=min(len(texts), top_k))
			texts = [texts[i] for i in order]
			file_names = [file_names[i] for i in order]
		except Exception:
			logging.exception("Rerank failed; using vector order")
		
		# 6) prepend README as primer (if exists)
		readme = await self._readme_blob(user_id, str(repo_id), salt_b64)
		parts = []
		if readme:
			parts.append("### FILE: README ###\n" + readme)
		for fn, t in zip(file_names, texts):
			parts.append(f"### FILE: {fn} ###\n{t}")
		
		blob = "\n\n".join(parts).strip()
		if truncate_chars and len(blob) > truncate_chars:
			blob = blob[:truncate_chars]
		return blob