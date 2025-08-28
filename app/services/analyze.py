import logging
from typing import Annotated, AsyncGenerator, List
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
        question:str,
        relative_path:str
    )-> AsyncGenerator[str, None]:
        together_client = Together(api_key=settings.TOGETHER_API_KEY)
        readme_content = ""

        # Get Repo
        repo_info = await self.repo_store.find_by_user_and_path(
            user_id=user_claims.sub, relative_path=relative_path
        )

        if not repo_info:
            yield "No relevant repo found."
            return

        query_embedding = self.generate_embedding(question)
        
        results = await self.code_chunks_store.get_user_repo_chunks(
            user_id=user_claims.sub,
            repo_id=repo_info.id,
            query_embedding=query_embedding[0],
            limit=5
        )
        
        if not results:
            yield "No code chunks found."
            return

        readme_content = await self._get_readme_content(user_claims.sub, str(repo_info.id))
        if readme_content:
            yield f"README/Setup information:\n{readme_content}"


        # Step 1: Rerank Results
        reranked_results = self.rerank_results(results, question, top_k=10)

        # Step 2: Stream from LLM
        if reranked_results:

                async for chunk in self._process_reranked_results(
                            together_client, reranked_results, question, repo_info
                ):
                    yield chunk

        else:
                yield json.dumps({"data": "No relevant reranked results found."})

    async def _process_reranked_results(self, together_client, reranked_results, question, repo_info):
        """Extract reranked results processing into separate method"""
        combined_text = ""
        file_list = []

        for result in reranked_results:
            file_name = result.get('file_name', '')
            text = result.get('content', '')
            combined_text += f"\n### FILE: {file_name} ###\n{text}\n"
            file_list.append(file_name)

        context_prompt = f" Create a single comprehensive documentation that covers all functionality across these files: {', '.join(file_list)}. \n{combined_text}"
        instructions_context = ""
        async for chunk in self._generate_documentation(
                client=together_client,
                context=context_prompt,
                question=question,
                previous_messages=[],
                instructions=instructions_context,
                repo_info=repo_info
        ):

            yield chunk


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

    async def _generate_documentation(self,
            client,
            context: str,
            question: str,
            previous_messages: List[dict] = None,
            instructions: str = "",
            repo_info:RepoResponseDTO=None,
    ):
        """Generate documentation using Together AI with streaming."""
        if previous_messages is None:
            previous_messages = []

        """Generate a summary of the chat conversation."""
        if repo_info:
            project_name = repo_info.repo_name
            brief_description = repo_info.description

            languages = repo_info.language

        else:
            project_name = "Unknown Project"
            brief_description = "No description available"
            languages = "Unknown Languages"


        system_prompt = """
        You are an expert software technical writer and AI assistant tasked with generating comprehensive, professional, and clear documentation for a software project.

        You will be given:
        - The name of the repository.
        - A brief project description.
        - A list of programming languages used in the codebase.
        - The contents of the README.md file.

        Your job is to analyze this input and generate a well-structured set of documentation that includes the following sections:

        ## 📘 Documentation Structure

        1. **Project Overview**
           - Summary of what the project does and its purpose.
           - Key technologies or languages used.

        2. **Features**
           - Key features and functionality the software provides.

        3. **Installation**
           - Step-by-step instructions on how to install or set up the project.

        4. **Usage**
           - Usage instructions and examples.
           - Include command-line examples or API calls if applicable.

        5. **Architecture**
           - High-level codebase overview.
           - Mention core components, structure, or modules.
           - Provide diagrams if the structure is inferable.

        6. **Configuration**
           - Environment variables, config files, or runtime settings.

        7. **Development**
           - Guide for contributing developers.
           - Include setup, testing, linting, and other dev practices.

        8. **API Documentation (if applicable)**
           - Overview of key endpoints, classes, or functions.
           - Mention Swagger/OpenAPI specs if available.

        9. **License**
           - Include the project license or inferred license from README.

        10. **References**
            - External links to related documentation, tools, or APIs.

        ## 🧠 Style Guidelines

        - Be concise, professional, and accurate.
        - Do not copy the README verbatim unless content is clearly suitable.
        - Use intelligent assumptions where content is missing and flag them with a `> NOTE:` annotation.
        - Use markdown formatting: headings (##, ###), lists, and code blocks.
        - Output should be usable directly in a `docs.md` or a documentation site.

        ## Input Format (to be injected before generation):

        - Repo Name: {repo_name}
        - Description: {project_description}
        - Languages: {languages_list}
        - README Content: {readme_content}
        - User instructions : {instructions}
        """.format(
            repo_name=project_name,
            project_description=brief_description,
            languages_list=languages,
            readme_content=context,
            instructions=instructions,
        )
        user_prompt = f"""


        Below is the chat after that:
        ---
        <new_chats>
         **Context:**
            {context}

            **Question:**
            {question}
        </new_chats>
        ---

        Please provide a summary of the chat till now including the historical summary of the chat.
        """


        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


        if previous_messages:
            messages = previous_messages + messages

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
