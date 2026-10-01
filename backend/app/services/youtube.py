import re
import logging
from typing import List, Optional, Any
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
import yt_dlp

from backend.app.models.schemas import TranscriptSnippet, VideoMetadata

logger = logging.getLogger(__name__)


def extract_video_id(url_or_id: str) -> str:
    """
    Extract the 11-character YouTube video ID from various URL formats or raw ID.
    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://www.youtube.com/v/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - Raw VIDEO_ID (e.g., dQw4w9WgXcQ)
    """
    url_or_id = url_or_id.strip()

    # Direct 11-character ID check
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id

    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/)([a-zA-Z0-9_-]{11})",
        r"[?&]v=([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract a valid YouTube video ID from: {url_or_id}")


def format_duration(seconds: Optional[float]) -> Optional[str]:
    """Format seconds into HH:MM:SS or MM:SS."""
    if seconds is None:
        return None
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fetch_video_metadata(video_id: str) -> VideoMetadata:
    """
    Fetch video metadata (title, channel, duration, thumbnail) without downloading video stream.
    Falls back gracefully if yt-dlp metadata extraction encounters issues.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title")
            channel = info.get("uploader") or info.get("channel")
            duration = info.get("duration")
            thumbnail = info.get("thumbnail")

            return VideoMetadata(
                video_id=video_id,
                url=url,
                title=title,
                channel=channel,
                duration=duration,
                duration_str=format_duration(duration),
                thumbnail_url=thumbnail,
            )
    except Exception as e:
        logger.warning(f"Failed to fetch metadata using yt-dlp for video {video_id}: {e}")
        return VideoMetadata(
            video_id=video_id,
            url=url,
            title="YouTube Video",
            channel="Unknown",
            duration=None,
            duration_str=None,
            thumbnail_url=f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        )


def _get_raw_transcript_data(video_id: str) -> List[dict]:
    """
    Internal helper to fetch raw transcript list from YouTubeTranscriptApi,
    supporting both modern instance-based and legacy classmethod-based API versions.
    """
    api = YouTubeTranscriptApi() if isinstance(YouTubeTranscriptApi, type) else YouTubeTranscriptApi

    transcript = None
    transcript_list = None

    if hasattr(api, "list"):
        transcript_list = api.list(video_id)
    elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    if transcript_list:
        try:
            transcript = transcript_list.find_manually_created_transcript(["en", "en-US", "en-GB"])
        except NoTranscriptFound:
            pass

        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(["en", "en-US", "en-GB"])
            except NoTranscriptFound:
                pass

        if not transcript:
            available = list(transcript_list)
            if available:
                transcript = available[0]

    if transcript:
        fetched = transcript.fetch()
        if hasattr(fetched, "to_raw_data"):
            return fetched.to_raw_data()
        return list(fetched)

    # Fallback to direct fetch
    if hasattr(api, "fetch"):
        fetched = api.fetch(video_id)
        if hasattr(fetched, "to_raw_data"):
            return fetched.to_raw_data()
        return list(fetched)
    elif hasattr(YouTubeTranscriptApi, "get_transcript"):
        return YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "en-GB"])

    raise NoTranscriptFound(video_id, ["en"], None)


def fetch_transcript(video_id: str) -> List[TranscriptSnippet]:
    """
    Fetch the transcript for a YouTube video, preserving text, start time, and duration.
    """
    try:
        raw_items = _get_raw_transcript_data(video_id)

        snippets: List[TranscriptSnippet] = []
        for item in raw_items:
            # Handle both dictionary and object formats
            if isinstance(item, dict):
                raw_text = item.get("text", "")
                start = float(item.get("start", 0.0))
                duration = float(item.get("duration", 0.0))
            else:
                raw_text = getattr(item, "text", "")
                start = float(getattr(item, "start", 0.0))
                duration = float(getattr(item, "duration", 0.0))

            text = " ".join(raw_text.split())
            if not text:
                continue

            end = round(start + duration, 2)
            snippets.append(
                TranscriptSnippet(
                    text=text,
                    start=round(start, 2),
                    duration=round(duration, 2),
                    end=end,
                )
            )

        if not snippets:
            raise ValueError(f"No usable text found in transcript for video {video_id}")

        return snippets

    except TranscriptsDisabled:
        raise ValueError(f"Subtitles/transcripts are disabled for video {video_id}.")
    except NoTranscriptFound:
        raise ValueError(f"No transcript found for video {video_id}.")
    except VideoUnavailable:
        raise ValueError(f"Video {video_id} is unavailable or private.")
    except Exception as e:
        logger.error(f"Error fetching transcript for {video_id}: {e}")
        raise ValueError(f"Failed to fetch transcript: {str(e)}")
