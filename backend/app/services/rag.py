import logging
from typing import List, Optional

from backend.app.models.schemas import (
    AskQuestionResponse,
    SourceTimestamp,
    RetrievedChunk,
)
from backend.app.services.vector_store import VectorStoreService
from backend.app.services.llm import LLMService

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a strictly grounded YouTube Video Intelligence & Q&A Assistant.
Your job is to answer user questions about the video using ONLY the provided transcript excerpts.

CRITICAL RULES TO PREVENT HALLUCINATIONS:
1. Base your answer EXCLUSIVELY on the provided transcript segments. Do NOT rely on prior world knowledge or make assumptions.
2. If the answer to the user's question is NOT covered in the provided transcript segments, respond with:
   "The requested topic was not found in this video's transcript."
3. When answering, cite the relevant timestamps (e.g., "At [05:12 - 07:40], the speaker explains...") so the user can jump to that exact section of the video.
4. Keep your answer concise, accurate, and faithful to what was spoken."""


class RAGService:
    """
    RAG Pipeline orchestrating vector retrieval, context construction,
    grounded prompt enforcement, and open-source LLM synthesis.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        llm_service: Optional[LLMService] = None,
    ):
        self.vector_store = vector_store or VectorStoreService.get_instance()
        self.llm_service = llm_service or LLMService()

    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks with clear timestamp boundaries for LLM ingestion."""
        context_parts = []
        for c in chunks:
            part = (
                f"[Segment {c.chunk_id} | Timestamps: {c.start_timestamp} - {c.end_timestamp}]\n"
                f"Content: {c.text}"
            )
            context_parts.append(part)
        return "\n\n".join(context_parts)

    def answer_question(
        self,
        video_id: str,
        question: str,
        top_k: int = 5,
        model: Optional[str] = None,
    ) -> AskQuestionResponse:
        """
        Execute grounded RAG workflow:
        1. Retrieve top-K relevant transcript chunks from ChromaDB.
        2. Construct timestamped context.
        3. Invoke local open-source LLM with anti-hallucination prompt.
        4. Return answer and structured timestamp source objects.
        """
        # Step 1: Vector Retrieval
        retrieved_chunks = self.vector_store.search_video_chunks(
            video_id=video_id,
            query=question,
            top_k=top_k,
        )

        if not retrieved_chunks:
            return AskQuestionResponse(
                video_id=video_id,
                question=question,
                answer="No transcript content found for this video.",
                sources=[],
                is_grounded=False,
            )

        # Step 2: Context Construction
        context_str = self.format_context(retrieved_chunks)
        user_prompt = (
            f"Here are the relevant transcript segments from the video:\n\n"
            f"{context_str}\n\n"
            f"User Question: {question}\n\n"
            f"Provide a grounded answer with timestamp references based only on the above transcript segments:"
        )

        # Step 3: LLM Synthesis
        answer = self.llm_service.generate(
            prompt=user_prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
            model=model,
            temperature=0.1,
        )

        # Step 4: Map source timestamps
        sources = [
            SourceTimestamp(
                chunk_id=c.chunk_id,
                start_time=c.start_time,
                end_time=c.end_time,
                start_timestamp=c.start_timestamp,
                end_timestamp=c.end_timestamp,
                text_snippet=c.text[:120] + "..." if len(c.text) > 120 else c.text,
            )
            for c in retrieved_chunks
        ]

        is_not_found = "not found in this video" in answer.lower()

        return AskQuestionResponse(
            video_id=video_id,
            question=question,
            answer=answer,
            sources=[] if is_not_found else sources,
            is_grounded=not is_not_found,
        )
