from pydantic import BaseModel, Field
from typing import List, Optional


class TranscriptSnippet(BaseModel):
    """Raw snippet returned from YouTube Transcript API."""
    text: str
    start: float
    duration: float
    end: float


class TranscriptChunk(BaseModel):
    """Processed and chunked transcript segment with timestamp boundaries."""
    chunk_id: int
    text: str
    start_time: float
    end_time: float
    start_timestamp: str
    end_timestamp: str
    word_count: int


class VideoMetadata(BaseModel):
    video_id: str
    url: str
    title: Optional[str] = None
    channel: Optional[str] = None
    duration: Optional[float] = None
    duration_str: Optional[str] = None
    thumbnail_url: Optional[str] = None


class ProcessVideoRequest(BaseModel):
    youtube_url: str = Field(..., description="Full YouTube video URL or ID")
    chunk_size: Optional[int] = Field(default=250, description="Target word count per chunk")
    chunk_overlap: Optional[int] = Field(default=40, description="Word overlap between chunks")


class ProcessVideoResponse(BaseModel):
    video: VideoMetadata
    total_chunks: int
    total_snippets: int
    chunks: List[TranscriptChunk]
    is_indexed: bool = True
    already_existed: bool = False


class RetrievedChunk(BaseModel):
    """Retrieved chunk from vector search with similarity score."""
    chunk_id: int
    text: str
    start_time: float
    end_time: float
    start_timestamp: str
    end_timestamp: str
    score: float  # cosine similarity score


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query or question")
    top_k: Optional[int] = Field(default=5, description="Number of chunks to retrieve")


class SearchResponse(BaseModel):
    video_id: str
    query: str
    results: List[RetrievedChunk]


class SourceTimestamp(BaseModel):
    chunk_id: int
    start_time: float
    end_time: float
    start_timestamp: str
    end_timestamp: str
    text_snippet: str


class AskQuestionRequest(BaseModel):
    question: str = Field(..., description="User question about the video")
    top_k: Optional[int] = Field(default=5, description="Number of context chunks to retrieve")


class AskQuestionResponse(BaseModel):
    video_id: str
    question: str
    answer: str
    sources: List[SourceTimestamp]
    is_grounded: bool = True


class TopicItem(BaseModel):
    topic: str
    start_time: float
    end_time: float
    start_timestamp: str
    end_timestamp: str
    subtopics: List[str] = []
    summary: Optional[str] = None


class TopicsResponse(BaseModel):
    video_id: str
    topics: List[TopicItem]


class CheckTopicRequest(BaseModel):
    topic: str = Field(..., description="Topic or concept to verify (e.g., 'LangGraph', 'RAG')")


class CheckTopicResponse(BaseModel):
    video_id: str
    topic: str
    is_covered: bool
    confidence: str  # YES, NO, PARTIALLY
    explanation: str
    relevant_timestamps: List[SourceTimestamp] = []
