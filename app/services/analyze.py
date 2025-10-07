import logging
from typing import Annotated, Any, AsyncGenerator, Dict, List, Tuple
from fastapi import Depends
from models_src.dto.repo import RepoResponseDTO
from together import AsyncTogether
import asyncio
import json
from models_src.repositories.code_chunks import TortoiseCodeChunksStore as CodeChunksStore
from models_src.repositories.repo import TortoiseRepoStore as RepoStore
from cohere import AsyncClientV2 as CohereAsyncClientV2

from app.utils.auth import UserClaims
from app.config import settings


class AnalyseService:
    
    EMBEDDING_MODEL_1 = "togethercomputer/m2-bert-80M-32k-retrieval"
    EMBEDDING_MODEL_2 = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    
    TOTAL_K = 10  # luggage limit for LLM context
    
    def __init__(self, repo_store: RepoStore, code_chunks_store: CodeChunksStore):

        self.repo_store = repo_store
        self.code_chunks_store = code_chunks_store

    @classmethod
    def with_dependency(
        cls,
        repo_store: Annotated[RepoStore, Depends()],
        code_chunks_store: Annotated[CodeChunksStore, Depends()],
    ) -> "AnalyseService":
        return cls(repo_store, code_chunks_store)
    
    async def answer_question(
        self,
        user_claims: UserClaims,
        questions: List[str],
        relative_path: str
    ) -> AsyncGenerator[str, None]:
        async_together_client = AsyncTogether(api_key=settings.TOGETHER_API_KEY)
        
        # 1) Find repo
        # Figure out which repo we’re supposed to analyze for this user + path.
        # If we can’t find that repo, exit early
        repo_info = await self.repo_store.find_by_user_and_path(
            user_id=user_claims.sub, relative_path=relative_path
        )
        if not repo_info:
            yield "No relevant repo found."
            return

        # 2) Clean questions (keep separate)
        questions = [q.strip() for q in (questions or []) if q and q.strip()]
        if not questions:
            yield "No questions provided."
            return

        # 3) Batched, async embeddings
        try:
            q_vecs = await self._generate_embeddings_batch(
                together=async_together_client, questions=questions, model_api_string=self.EMBEDDING_MODEL_1
            )
        except Exception:
            logging.exception("Failed to generate embeddings batch")
            yield "Failed to generate embeddings for questions."
            return

        # Dimension guard
        # Basic safety check: ensure we got a VECTOR_SIZE-dim vector (what our DB expects).
        emb_dim = len(q_vecs[0])
        if any(len(v) != emb_dim for v in q_vecs):
            yield "Embedding dimensions inconsistent, aborting."
            return

        if emb_dim != settings.VECTOR_SIZE:
            yield f"Embedding dimension mismatch: DB={settings.VECTOR_SIZE}, model={emb_dim}."
            return

        # 4) ONE fused SQL query (no N+1)
        results = await self.code_chunks_store.get_user_repo_chunks_multi(
            user_id=user_claims.sub,
            repo_id=repo_info.id,
            query_embeddings=q_vecs,
            emb_dim=emb_dim,
            limit=self.TOTAL_K,
        )
        if not results:
            yield "No code chunks found."
            return

        # 5) README preface: If there’s README content for this repo, stream it first as helpful context.
        readme_content = await self._get_readme_content(user_claims.sub, str(repo_info.id))
        if readme_content:
            yield f"README/Setup information:\n{readme_content}"

        # 6) Rerank across questions (sum per-question relevance)
        # Improve ordering using a cross-encoder reranker (reads the text, not just vector math).
        # This looks at the documents versus EACH question and adds up relevance,
        # so items that answer multiple questions move higher.
        reranked = await self.rerank_results_multi(results, questions, top_k=self.TOTAL_K)
        if not reranked:
            yield json.dumps({"data": "No relevant reranked results found."})
            return

        # 7) Generate final answer (answers each question separately)
        # Ask the LLM to write the answer using those top chunks as context.
        # We pass the questions list so the model answers “Q1, Q2, …” in separate sections.
        async for chunk in self._process_reranked_results(
            together=async_together_client, reranked_results=reranked, questions=questions, repo_info=repo_info
        ):
            yield chunk
    
    async def _generate_embeddings_batch(
            self,
            together: AsyncTogether,
            questions: list[str],
            model_api_string: str | None = None,
    ) -> list[list[float]]:
        """Batched, async embeddings. Returns one vector per question, in order."""
        model = model_api_string or self.EMBEDDING_MODEL_1
        
        resp = await together.embeddings.create(input=questions, model=model)

        # Extract vectors and preserve keep order
        vecs = [item.embedding for item in resp.data]
        
        if not vecs:
            raise RuntimeError("Embeddings API returned no vectors.")
        
        # Check if all the vectors have the same dimension cause its required
        emb_dim = len(vecs[0])
        if any(len(v) != emb_dim for v in vecs):
            raise RuntimeError("Embeddings returned inconsistent dimensions.")
        
        return vecs
    
    # RERANKING PROCESS

    async def _process_reranked_results(
            self,
            together: AsyncTogether,
            reranked_results: List[Dict[str, Any]],
            questions: List[str],
            repo_info: RepoResponseDTO,
    ):
        combined_text = ""
        file_list: List[str] = []
        
        for result in reranked_results:
            file_name = result.get("file_name", "")
            text = result.get("content", "")
            combined_text += f"\n### FILE: {file_name} ###\n{text}\n"
            file_list.append(file_name)
        
        context_prompt = (
            "Create a single comprehensive documentation that covers all functionality "
            f"across these files: {', '.join(file_list)}.\n{combined_text}"
        )
        
        async for chunk in self._generate_documentation(
                together=together,
                context=context_prompt,
                questions=questions,
                previous_messages=[],
                repo_info=repo_info,
        ):
            yield chunk
    
    
    async def _generate_documentation(
            self,
            together: AsyncTogether,
            context: str,
            questions: List[str],
            previous_messages: List[dict] | None = None,
            repo_info: RepoResponseDTO | None = None,
    ):
        previous_messages = previous_messages or []
        
        if repo_info:
            project_name = repo_info.repo_name
            brief_description = repo_info.description
            languages = repo_info.language
        else:
            project_name = "Unknown Project"
            brief_description = "No description available"
            languages = "Unknown Languages"
        
        system_prompt = """
    You are an expert software technical writer.
    When multiple questions are provided, answer EACH in its own subsection (Q1, Q2, ...),
    grounded ONLY in the provided context. If context is insufficient for a question,
    say so explicitly under that subsection.
    Use markdown headings and lists.
    """.strip()
        
        rendered_qs = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        
        user_prompt = f"""
    ## Project
    - Repo Name: {project_name}
    - Description: {brief_description}
    - Languages: {languages}

    ## Context (retrieved code chunks)
    {context}

    ## Questions (answer each separately)
    {rendered_qs}
    """.strip()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            
            stream = await together.chat.completions.create(
                model=self.EMBEDDING_MODEL_2,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.7,
                top_k=40,
                repetition_penalty=1,
                stream=True,
            )
            
            async for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    if delta in ("json", "```"):
                        delta = ""
                    yield delta
                await asyncio.sleep(0)
        
        except Exception as e:
            logging.error(f"Async documentation generation failed: {e}")
    
    async def _get_readme_content(self, user_id: str, repo_id: str) -> str:
        """Extract README content collection into separate method"""
        read_me_results = await self.code_chunks_store.get_repo_file_chunks(
            user_id, repo_id, file_name="readme"
        )

        if not read_me_results:
            return ""

        readme_content = ""
        for chunk in read_me_results:
            content = chunk.get("content")
            if content:
                readme_content += content + "\n"

        return readme_content.strip()
    
    # Raranking related classes and helpers
    
    async def rerank_results_multi(
            self,
            results: List[Dict[str, Any]],
            questions: List[str],
            top_k: int = 10,
    ):
        if not results:
            return []
        
        indexed_docs, doc_texts = self._prepare_indexed_docs(results)
        if not doc_texts:
            logging.warning("No valid documents found for reranking")
            return results
        
        limit = min(len(doc_texts), max(top_k, 1))
        
        try:
            filtered_qs = self._filter_questions(questions)
            scores_in_doc_idx = await self._cohere_multi_rerank(filtered_qs, doc_texts, limit)
            scores_by_orig_idx = {indexed_docs[i][0]: s for i, s in scores_in_doc_idx.items()}
            return self._apply_scores_and_sort(results, scores_by_orig_idx, top_k)
        except Exception as e:
            logging.error(f"Multi-question reranking failed: {e}")
            return results
    
    
    @staticmethod
    def _prepare_indexed_docs(results: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, str]], List[str]]:
        """Return [(orig_idx, text)], and the parallel list of texts (doc_texts)."""
        indexed = [(i, (r.get("content") or "").strip()) for i, r in enumerate(results)]
        indexed = [(i, t) for i, t in indexed if t]
        return indexed, [t for _, t in indexed]
    
    @staticmethod
    def _filter_questions(questions: List[str]) -> List[str]:
        return [q.strip() for q in (questions or []) if q and q.strip()]
    
    async def _cohere_multi_rerank(
            self,
            questions: List[str],
            documents: List[str],
            limit: int,
    ) -> Dict[int, float]:
        """Return {doc_index_in_documents: aggregated_score}."""
        if not questions:
            return {}
        co = CohereAsyncClientV2(api_key=settings.COHERE_API_KEY)
        
        scores = [0.0] * len(documents)
        for q in questions:
            resp = await co.rerank(
                query=q,
                documents=documents,
                top_n=limit,
                model="rerank-v3.5",
            )
            for r in getattr(resp, "results", []):
                idx = getattr(r, "index", -1)
                if 0 <= idx < len(scores):
                    scores[idx] += float(getattr(r, "relevance_score", 0.0))
        
        # return only docs that received any score
        return {i: s for i, s in enumerate(scores) if s}
    
    @staticmethod
    def _apply_scores_and_sort(
            results: List[Dict[str, Any]],
            scores_by_orig_idx: Dict[int, float],
            top_k: int,
    ) -> List[Dict[str, Any]]:
        if not scores_by_orig_idx:
            logging.warning("No results received rerank scores (multi)")
            return results
        
        enriched = []
        for i, item in enumerate(results):
            if i in scores_by_orig_idx:
                x = dict(item)
                x["rerank_score"] = scores_by_orig_idx[i]
                enriched.append(x)
        
        enriched.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return enriched[:top_k]