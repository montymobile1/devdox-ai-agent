import logging
from typing import Annotated, Any, AsyncGenerator, Dict, List
from fastapi import Depends
from models_src.dto.repo import RepoResponseDTO
from together import Together
import cohere
import asyncio
import json
from models_src.repositories.code_chunks import TortoiseCodeChunksStore as CodeChunksStore
from models_src.repositories.repo import TortoiseRepoStore as RepoStore
from app.utils.auth import UserClaims
from app.config import settings


class AnalyseService:

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
        questions:List[str],
        relative_path:str
    )-> AsyncGenerator[str, None]:
        
        # Make a Together client so we can call the LLM/embeddings API.
        together_client = Together(api_key=settings.TOGETHER_API_KEY)
        readme_content = ""

        # Figure out which repo we’re supposed to analyze for this user + path.
        # If we can’t find that repo, exit early
        repo_info = await self.repo_store.find_by_user_and_path(
            user_id=user_claims.sub, relative_path=relative_path
        )

        if not repo_info:
            yield "No relevant repo found."
            return
        
        # Turn EACH question into an embedding vector (i.e., numbers the database can compare).
        # If the embedding API failed or if somehow no vectors were produced, stop, log it and exit early.
        q_vecs: List[List[float]] = []
        try:
            for q in questions:
                
                # generate_embedding returns a list with one vector for a single string,
                # so we grab the first item ([0]).
                vecs = self.generate_embedding(q)
                
                # Basic safety check: ensure we got a 768-dim vector (what our DB expects).
                if vecs and len(vecs[0]) == 768:
                    q_vecs.append(vecs[0])
        
        except Exception:
            logging.exception("Failed to generate question embeddings")
            yield "Failed to generate embeddings for questions."
            return
        
        if not q_vecs:
            yield "Failed to embed questions."
            return
        
        # For each question vector, ask the DB for the most similar code chunks.
        # We keep a global budget (TOTAL_K) to avoid flooding the LLM context.
        TOTAL_K = 10
        
        # Split that budget across however many questions we have (at least 1 each).
        k_per_q = max(1, TOTAL_K // len(q_vecs))
        
        # We'll merge (“fuse”) results from all the per-question searches here.
        # If the same chunk is relevant to multiple questions, its score will add up.
        fused: Dict[str, Dict[str, Any]] = {}
        for vec in q_vecs:
            
            # Vector search in the user's repo: returns chunks with a similarity 'score'.
            hits = await self.code_chunks_store.get_user_repo_chunks(
                user_id=user_claims.sub,
                repo_id=repo_info.id,
                query_embedding=vec,
                limit=k_per_q,
            )
            
            for h in (hits or []):
                
                # Create a stable key to identify a chunk across queries.
                # Prefer DB primary key if present, otherwise use a content-based fallback.
                
                key = str(h.get("id")) if h.get("id") else f"{h.get('file_name','')}::{hash((h.get('content','') or '')[:256])}"
                
                # Sum similarity scores so chunks that match multiple questions bubble up.
                fused_score = (fused.get(key, {}).get("fusion_score") or 0.0) + float(h.get("score") or 0.0)
                fused[key] = {**h, "fusion_score": fused_score}
        
        # Take the top TOTAL_K fused results (best overall across all questions), If nothing relevant was found, stop.
        results = sorted(fused.values(), key=lambda x: x["fusion_score"], reverse=True)[:TOTAL_K]
        
        if not results:
            yield "No code chunks found."
            return

        
        # If there’s README content for this repo, stream it first as helpful context.
        readme_content = await self._get_readme_content(user_claims.sub, str(repo_info.id))
        if readme_content:
            yield f"README/Setup information:\n{readme_content}"
        
        # Improve ordering using a cross-encoder reranker (reads the text, not just vector math).
        # This looks at the documents versus EACH question and adds up relevance,
        # so items that answer multiple questions move higher.
        reranked_results = self.rerank_results_multi(results, questions, top_k=TOTAL_K)
        
        # Finally, ask the LLM to write the answer using those top chunks as context.
        # We pass the questions list so the model answers “Q1, Q2, …” in separate sections.
        if reranked_results:
            async for chunk in self._process_reranked_results(
                    together_client, reranked_results, questions, repo_info
            ):
                # Stream small pieces (“chunks”) of the generated text back to the caller.
                yield chunk
        else:
            # If reranking somehow produced nothing, say so (very rare edge case).
            yield json.dumps({"data": "No relevant reranked results found."})
    
    def rerank_results(self, results: list, last_message_content: str, top_k: int = 10):
        """Rerank results based on multiple factors"""

        # Return early if no results to rerank
        if not results:
            return []

        # Return early if query is empty
        if not last_message_content or not last_message_content.strip():
            logging.warning("Empty query provided for reranking, returning original results")
            return results

        try:
            co_cohere = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
            documents = []

            for doc in results:
                content = doc.get("content", "")
                if not content.strip():  # Skip empty documents
                    continue
                documents.append(content)

            # If no valid documents found, return original results
            if not documents:
                logging.warning("No valid documents found for reranking")
                return results

            response = co_cohere.rerank(
                query=last_message_content,
                documents=documents,
                top_n=min(len(documents), top_k),
                model="rerank-v3.5"
            )

            # Check if reranking response is valid
            if not response or not hasattr(response, 'results') or not response.results:
                logging.warning("Invalid reranking response received")
                return results

            # Create a copy of results to avoid mutating the original
            reranked_results = results.copy()

            # Update documents with rerank scores
            for result in response.results:
                if result.index < len(reranked_results):
                    reranked_results[result.index]['rerank_score'] = result.relevance_score

            # Filter out results without rerank scores (in case some failed)
            valid_results = [r for r in reranked_results if 'rerank_score' in r]

            if not valid_results:
                logging.warning("No results received rerank scores")
                return results

            # Sort by rerank score and limit to top_k
            valid_results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

            return valid_results[:top_k]

        except Exception as e:
            logging.error(f"Cross-encoder reranking failed: {e}")
            # Return original results in case of failure
            return results
    
    async def _process_reranked_results(self, together_client, reranked_results, questions: List[str], repo_info):
        combined_text = ""
        file_list = []

        for result in reranked_results:
            file_name = result.get('file_name', '')
            text = result.get('content', '')
            combined_text += f"\n### FILE: {file_name} ###\n{text}\n"
            file_list.append(file_name)

        context_prompt = (
            f" Create a single comprehensive documentation that covers all functionality "
            f"across these files: {', '.join(file_list)}.\n{combined_text}"
        )

        async for chunk in self._generate_documentation(
            client=together_client,
            context=context_prompt,
            questions=questions,
            previous_messages=[],
            instructions="",
            repo_info=repo_info
        ):
            yield chunk
    
    def rerank_results_multi(self, results: List[Dict[str, Any]], questions: List[str], top_k: int = 10):
        """Rerank documents against MULTIPLE questions without concatenation."""
        if not results:
            return []

        documents: List[str] = []
        for doc in results:
            content = (doc.get("content") or "").strip()
            if content:
                documents.append(content)
        if not documents:
            logging.warning("No valid documents found for reranking")
            return results

        try:
            co_cohere = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
            agg_scores = [0.0] * len(documents)

            for q in questions:
                if not q.strip():
                    continue
                resp = co_cohere.rerank(
                    query=q,
                    documents=documents,
                    top_n=min(len(documents), max(top_k, 1)),
                    model="rerank-v3.5",
                )
                if getattr(resp, "results", None):
                    for r in resp.results:
                        idx = getattr(r, "index", None)
                        score = getattr(r, "relevance_score", 0.0)
                        if isinstance(idx, int) and 0 <= idx < len(agg_scores):
                            agg_scores[idx] += float(score)

            tmp = results.copy()
            for i, s in enumerate(agg_scores):
                if i < len(tmp):
                    tmp[i]["rerank_score"] = s

            valid = [d for d in tmp if "rerank_score" in d]
            if not valid:
                logging.warning("No results received rerank scores (multi)")
                return results

            valid.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return valid[:top_k]

        except Exception as e:
            logging.error(f"Multi-question reranking failed: {e}")
            return results

    def generate_embedding(self, question:str, model_api_string="togethercomputer/m2-bert-80M-32k-retrieval"):
        together_client = Together(api_key=settings.TOGETHER_API_KEY)
        outputs = together_client.embeddings.create(
            input=question,
            model=model_api_string,
        )
        return [x.embedding for x in outputs.data]

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
    
    async def _generate_documentation(
            self,
            client,
            context: str,
            questions: List[str],
            previous_messages: List[dict] = None,
            instructions: str = "",
            repo_info: RepoResponseDTO = None,
    ):
        if previous_messages is None:
            previous_messages = []
        
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
        """
        
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
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.7,
                top_k=40,
                repetition_penalty=1,
                stream=True,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    chunk_text = chunk.choices[0].delta.content
                    if chunk_text in ["json", "```"]:
                        chunk_text = ""
                    yield chunk_text
            await asyncio.sleep(0)
        except Exception as e:
            logging.error(f"API Error: {e}")
