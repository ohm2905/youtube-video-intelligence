from typing import List
from backend.app.models.schemas import TranscriptSnippet, TranscriptChunk


def format_timestamp(seconds: float) -> str:
    """
    Format seconds (e.g. 3724.5) to 'HH:MM:SS' or 'MM:SS'.
    Examples:
        75.2  -> '01:15'
        3724.0 -> '01:02:04'
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def create_timestamped_chunks(
    snippets: List[TranscriptSnippet],
    chunk_size: int = 250,
    chunk_overlap: int = 40,
) -> List[TranscriptChunk]:
    """
    Intelligently groups consecutive transcript snippets into semantic chunks while
    preserving exact start and end timestamps.
    
    Args:
        snippets: List of raw transcript snippets with start and end times.
        chunk_size: Target word count per chunk.
        chunk_overlap: Number of overlapping words between consecutive chunks.
        
    Returns:
        List of TranscriptChunk objects with chunk_id, text, start_time, end_time,
        and human-readable timestamps.
    """
    if not snippets:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    chunks: List[TranscriptChunk] = []
    chunk_id = 1
    
    snippet_idx = 0
    total_snippets = len(snippets)

    while snippet_idx < total_snippets:
        current_words: List[str] = []
        chunk_start_time = snippets[snippet_idx].start
        chunk_end_time = snippets[snippet_idx].end
        
        last_snippet_idx = snippet_idx

        # Accumulate snippets until target chunk_size is reached
        for i in range(snippet_idx, total_snippets):
            snip = snippets[i]
            words = snip.text.split()
            current_words.extend(words)
            chunk_end_time = snip.end
            last_snippet_idx = i

            if len(current_words) >= chunk_size:
                break

        chunk_text = " ".join(current_words)
        
        chunks.append(
            TranscriptChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_time=chunk_start_time,
                end_time=chunk_end_time,
                start_timestamp=format_timestamp(chunk_start_time),
                end_timestamp=format_timestamp(chunk_end_time),
                word_count=len(current_words),
            )
        )
        chunk_id += 1

        # If we reached the end of snippets, we're done
        if last_snippet_idx >= total_snippets - 1:
            break

        # Calculate next snippet_idx with overlap
        # Advance snippet_idx until we drop approximately (len(current_words) - chunk_overlap) words
        words_to_advance = max(1, len(current_words) - chunk_overlap)
        words_advanced = 0
        next_snippet_idx = snippet_idx

        while next_snippet_idx < last_snippet_idx and words_advanced < words_to_advance:
            words_in_snip = len(snippets[next_snippet_idx].text.split())
            words_advanced += words_in_snip
            next_snippet_idx += 1

        # Ensure forward progress
        if next_snippet_idx <= snippet_idx:
            next_snippet_idx = snippet_idx + 1

        snippet_idx = next_snippet_idx

    return chunks
