import pytest
from backend.app.services.chunker import format_timestamp, create_timestamped_chunks
from backend.app.models.schemas import TranscriptSnippet


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(45) == "00:45"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3600) == "01:00:00"
    assert format_timestamp(3665) == "01:01:05"
    assert format_timestamp(5132) == "01:25:32"


def test_create_timestamped_chunks():
    # Construct 10 mock snippets with 10 words each
    snippets = []
    for i in range(10):
        snippets.append(
            TranscriptSnippet(
                text=f"word1 word2 word3 word4 word5 word6 word7 word8 word9 snippet{i}",
                start=float(i * 10),
                duration=10.0,
                end=float((i + 1) * 10),
            )
        )

    # Total 100 words. Target chunk size 30 words, overlap 10 words.
    chunks = create_timestamped_chunks(snippets, chunk_size=30, chunk_overlap=10)

    assert len(chunks) > 0
    
    # Check first chunk properties
    first_chunk = chunks[0]
    assert first_chunk.chunk_id == 1
    assert first_chunk.start_time == 0.0
    assert first_chunk.start_timestamp == "00:00"
    assert first_chunk.end_time >= 30.0
    assert "snippet0" in first_chunk.text
    assert first_chunk.word_count >= 30

    # Ensure all chunks have valid timestamp sequences
    for chunk in chunks:
        assert chunk.start_time < chunk.end_time
        assert chunk.word_count > 0
        assert len(chunk.start_timestamp) >= 5
        assert len(chunk.end_timestamp) >= 5


def test_create_timestamped_chunks_empty():
    chunks = create_timestamped_chunks([])
    assert chunks == []
