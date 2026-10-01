import logging
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from backend.app.models.schemas import (
    ProcessVideoRequest,
    ProcessVideoResponse,
    TranscriptSnippet,
    VideoMetadata,
    SearchRequest,
    SearchResponse,
    AskQuestionRequest,
    AskQuestionResponse,
    CheckTopicRequest,
    CheckTopicResponse,
    TopicsResponse,
)
from backend.app.services.youtube import (
    extract_video_id,
    fetch_video_metadata,
    fetch_transcript,
)
from backend.app.services.chunker import create_timestamped_chunks
from backend.app.services.vector_store import VectorStoreService
from backend.app.services.rag import RAGService
from backend.app.services.topic_service import TopicService
from backend.app.services.llm import LLMService
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health_check():
    """Health check endpoint to verify backend status."""
    return {"status": "ok", "service": "YouTube Video Intelligence API"}


@router.get("/models")
def get_available_models():
    """Return list of available local LLMs and default configurations."""
    llm = LLMService()
    models = llm.list_models()
    return {
        "available_models": models,
        "default_llm": settings.DEFAULT_LLM_MODEL,
        "default_embedding": settings.DEFAULT_EMBEDDING_MODEL,
        "ollama_connected": llm.is_available(),
    }


@router.post("/videos/process", response_model=ProcessVideoResponse)
def process_video(request: ProcessVideoRequest):
    """
    Process and vectorize a YouTube video:
    1. Parse and validate the video URL.
    2. Check if already processed to avoid duplicate computation.
    3. Fetch video metadata and timestamped transcript.
    4. Intelligently chunk the transcript preserving timestamps.
    5. Embed and store chunks in ChromaDB.
    """
    try:
        video_id = extract_video_id(request.youtube_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    vector_store = VectorStoreService.get_instance()
    already_indexed = vector_store.is_video_indexed(video_id)

    try:
        # Fetch metadata
        metadata = fetch_video_metadata(video_id)
        
        # Fetch transcript
        snippets = fetch_transcript(video_id)
        
        # Chunk transcript
        chunks = create_timestamped_chunks(
            snippets=snippets,
            chunk_size=request.chunk_size or 250,
            chunk_overlap=request.chunk_overlap or 40,
        )

        if not already_indexed:
            vector_store.index_video_chunks(video_id=video_id, chunks=chunks)
            logger.info(f"Successfully embedded and indexed video {video_id} ({len(chunks)} chunks).")
        else:
            logger.info(f"Video {video_id} was already indexed. Reusing existing vector index.")

        return ProcessVideoResponse(
            video=metadata,
            total_chunks=len(chunks),
            total_snippets=len(snippets),
            chunks=chunks,
            is_indexed=True,
            already_existed=already_indexed,
        )
    except ValueError as e:
        logger.warning(f"Error processing video {video_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error processing video {video_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")


@router.post("/videos/{video_id}/ask", response_model=AskQuestionResponse)
def ask_question_about_video(
    video_id: str,
    request: AskQuestionRequest,
    model: Optional[str] = Query(None, description="Optional LLM model override"),
):
    """
    Ask a grounded question about the video transcript.
    Strictly uses retrieved transcript segments and returns timestamp citations.
    """
    try:
        clean_id = extract_video_id(video_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    vector_store = VectorStoreService.get_instance()
    if not vector_store.is_video_indexed(clean_id):
        raise HTTPException(
            status_code=404,
            detail=f"Video '{clean_id}' is not indexed. Please submit it to /api/videos/process first.",
        )

    rag = RAGService(vector_store=vector_store)
    try:
        response = rag.answer_question(
            video_id=clean_id,
            question=request.question,
            top_k=request.top_k or 5,
            model=model,
        )
        return response
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error answering question for video {clean_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


@router.post("/videos/{video_id}/check-topic", response_model=CheckTopicResponse)
def check_topic_presence(
    video_id: str,
    request: CheckTopicRequest,
    model: Optional[str] = Query(None, description="Optional LLM model override"),
):
    """
    Check if a topic or concept is covered in the video.
    Returns YES / NO / PARTIALLY with explanation and timestamp references.
    """
    try:
        clean_id = extract_video_id(video_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    vector_store = VectorStoreService.get_instance()
    if not vector_store.is_video_indexed(clean_id):
        raise HTTPException(
            status_code=404,
            detail=f"Video '{clean_id}' is not indexed. Please submit it to /api/videos/process first.",
        )

    topic_service = TopicService(vector_store=vector_store)
    try:
        result = topic_service.verify_topic(
            video_id=clean_id,
            topic=request.topic,
            model=model,
        )
        return result
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error checking topic '{request.topic}' for video {clean_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Topic check failed: {str(e)}")


@router.get("/videos/{video_id}/topics", response_model=TopicsResponse)
def get_video_topics(
    video_id: str,
    model: Optional[str] = Query(None, description="Optional LLM model override"),
):
    """
    Extract major structured topics covered across the video timeline.
    """
    try:
        clean_id = extract_video_id(video_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    vector_store = VectorStoreService.get_instance()
    if not vector_store.is_video_indexed(clean_id):
        raise HTTPException(
            status_code=404,
            detail=f"Video '{clean_id}' is not indexed. Please submit it to /api/videos/process first.",
        )

    topic_service = TopicService(vector_store=vector_store)
    try:
        return topic_service.extract_topics(video_id=clean_id, model=model)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error extracting topics for video {clean_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Topic extraction failed: {str(e)}")


@router.post("/videos/{video_id}/search", response_model=SearchResponse)
def search_video_transcript(video_id: str, request: SearchRequest):
    """
    Perform semantic vector search across chunks for a specific YouTube video.
    Returns the top-K most relevant chunks with exact start and end timestamps.
    """
    try:
        clean_id = extract_video_id(video_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    vector_store = VectorStoreService.get_instance()
    if not vector_store.is_video_indexed(clean_id):
        raise HTTPException(
            status_code=404,
            detail=f"Video '{clean_id}' has not been processed or indexed yet. Call /api/videos/process first.",
        )

    results = vector_store.search_video_chunks(
        video_id=clean_id,
        query=request.query,
        top_k=request.top_k or 5,
    )

    return SearchResponse(
        video_id=clean_id,
        query=request.query,
        results=results,
    )


@router.get("/videos/{video_id}/status")
def get_video_status(video_id: str):
    """Check whether a video is already processed and indexed in the vector store."""
    try:
        clean_id = extract_video_id(video_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    vector_store = VectorStoreService.get_instance()
    indexed = vector_store.is_video_indexed(clean_id)
    return {
        "video_id": clean_id,
        "is_indexed": indexed,
    }


@router.get("/videos/{video_id}/transcript", response_model=List[TranscriptSnippet])
def get_raw_transcript(video_id: str):
    """Fetch raw timestamped transcript snippets for a video."""
    try:
        clean_id = extract_video_id(video_id)
        snippets = fetch_transcript(clean_id)
        return snippets
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos/{video_id}/metadata", response_model=VideoMetadata)
def get_metadata(video_id: str):
    """Fetch video metadata."""
    try:
        clean_id = extract_video_id(video_id)
        metadata = fetch_video_metadata(clean_id)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
