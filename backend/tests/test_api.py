from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.app.main import app
from backend.app.models.schemas import TranscriptSnippet, VideoMetadata, RetrievedChunk

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "YouTube Video Intelligence" in response.text


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "YouTube Video Intelligence API"}


@patch("backend.app.api.routes.VectorStoreService")
@patch("backend.app.api.routes.fetch_video_metadata")
@patch("backend.app.api.routes.fetch_transcript")
def test_process_video_api(mock_fetch_transcript, mock_fetch_metadata, mock_vector_class):
    mock_vector_instance = MagicMock()
    mock_vector_instance.is_video_indexed.return_value = False
    mock_vector_instance.index_video_chunks.return_value = 1
    mock_vector_class.get_instance.return_value = mock_vector_instance

    mock_fetch_metadata.return_value = VideoMetadata(
        video_id="dQw4w9WgXcQ",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Test Video",
        channel="Test Channel",
        duration=120.0,
        duration_str="02:00",
        thumbnail_url="https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
    )

    mock_fetch_transcript.return_value = [
        TranscriptSnippet(text="Welcome to this video tutorial.", start=0.0, duration=3.0, end=3.0),
        TranscriptSnippet(text="We are discussing RAG and embeddings.", start=3.0, duration=4.0, end=7.0),
        TranscriptSnippet(text="Vector databases store embeddings efficiently.", start=7.0, duration=5.0, end=12.0),
    ]

    response = client.post(
        "/api/videos/process",
        json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "chunk_size": 10, "chunk_overlap": 2},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["video"]["video_id"] == "dQw4w9WgXcQ"
    assert data["video"]["title"] == "Test Video"
    assert data["total_snippets"] == 3
    assert data["total_chunks"] >= 1
    assert data["is_indexed"] is True
    assert data["already_existed"] is False


@patch("backend.app.api.routes.VectorStoreService")
def test_search_video_api(mock_vector_class):
    mock_vector_instance = MagicMock()
    mock_vector_instance.is_video_indexed.return_value = True
    mock_vector_instance.search_video_chunks.return_value = [
        RetrievedChunk(
            chunk_id=1,
            text="Vector search retrieves relevant context.",
            start_time=12.0,
            end_time=35.0,
            start_timestamp="00:12",
            end_timestamp="00:35",
            score=0.88,
        )
    ]
    mock_vector_class.get_instance.return_value = mock_vector_instance

    response = client.post(
        "/api/videos/dQw4w9WgXcQ/search",
        json={"query": "How does vector search work?", "top_k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "dQw4w9WgXcQ"
    assert len(data["results"]) == 1
    assert data["results"][0]["start_timestamp"] == "00:12"
    assert data["results"][0]["score"] == 0.88


def test_process_video_invalid_url():
    response = client.post(
        "/api/videos/process",
        json={"youtube_url": "https://not-youtube.com/page"},
    )
    assert response.status_code == 400
