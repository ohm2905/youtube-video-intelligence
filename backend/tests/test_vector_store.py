import pytest
import shutil
from backend.app.services.vector_store import VectorStoreService
from backend.app.models.schemas import TranscriptChunk


@pytest.fixture
def temp_vector_store(tmp_path):
    store_dir = str(tmp_path / "test_chroma")
    service = VectorStoreService(persist_directory=store_dir)
    yield service
    shutil.rmtree(store_dir, ignore_errors=True)


def test_vector_indexing_and_search(temp_vector_store):
    video_id = "test_vid_123"

    chunks = [
        TranscriptChunk(
            chunk_id=1,
            text="Retrieval augmented generation combines an LLM with external document retrieval from vector databases.",
            start_time=10.0,
            end_time=35.0,
            start_timestamp="00:10",
            end_timestamp="00:35",
            word_count=13,
        ),
        TranscriptChunk(
            chunk_id=2,
            text="Convolutional neural networks are commonly used for computer vision and image classification.",
            start_time=40.0,
            end_time=65.0,
            start_timestamp="00:40",
            end_timestamp="01:05",
            word_count=12,
        ),
        TranscriptChunk(
            chunk_id=3,
            text="Gradient descent optimizes the neural network weights by minimizing the loss function with backpropagation.",
            start_time=70.0,
            end_time=95.0,
            start_timestamp="01:10",
            end_timestamp="01:35",
            word_count=15,
        ),
    ]

    # Index chunks
    indexed_count = temp_vector_store.index_video_chunks(video_id, chunks)
    assert indexed_count == 3
    assert temp_vector_store.is_video_indexed(video_id) is True

    # Search for RAG
    results = temp_vector_store.search_video_chunks(video_id, query="What is retrieval augmented generation?", top_k=2)
    assert len(results) > 0
    top_result = results[0]
    assert top_result.chunk_id == 1
    assert "Retrieval augmented generation" in top_result.text
    assert top_result.start_timestamp == "00:10"
    assert top_result.end_timestamp == "00:35"
    assert top_result.score > 0.5

    # Search for CNN / Image recognition
    results_cnn = temp_vector_store.search_video_chunks(video_id, query="computer vision image recognition", top_k=1)
    assert len(results_cnn) == 1
    assert results_cnn[0].chunk_id == 2


def test_is_video_indexed_false(temp_vector_store):
    assert temp_vector_store.is_video_indexed("non_existent_vid") is False
