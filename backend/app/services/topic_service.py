import re
import json
import logging
from typing import List, Optional

from backend.app.models.schemas import (
    CheckTopicResponse,
    TopicsResponse,
    TopicItem,
    SourceTimestamp,
)
from backend.app.services.vector_store import VectorStoreService
from backend.app.services.llm import LLMService

logger = logging.getLogger(__name__)

TOPIC_VERIFY_SYSTEM_PROMPT = """You are a rigorous video content auditor.
Your job is to determine whether a specific topic or concept is actually covered in the provided transcript segments from a YouTube video.

Respond ONLY with a valid JSON object in this exact format:
{
  "is_covered": true or false,
  "confidence": "YES", "NO", or "PARTIALLY",
  "explanation": "Clear 1-3 sentence explanation citing the timestamps if covered, or explaining what was actually discussed instead."
}"""

TOPIC_EXTRACTION_SYSTEM_PROMPT = """You are an expert video curriculum analyzer.
Your job is to analyze transcript segments and identify the major structured topics covered across the video timeline.

Respond ONLY with a valid JSON object in this format:
{
  "topics": [
    {
      "topic": "Title of Topic",
      "start_timestamp": "HH:MM:SS or MM:SS",
      "end_timestamp": "HH:MM:SS or MM:SS",
      "subtopics": ["Subtopic 1", "Subtopic 2"],
      "summary": "Brief 1-sentence summary"
    }
  ]
}"""


class TopicService:
    """
    Service for automated topic extraction, topic timeline mapping,
    and semantic topic presence verification ('Is X covered?').
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        llm_service: Optional[LLMService] = None,
    ):
        self.vector_store = vector_store or VectorStoreService.get_instance()
        self.llm_service = llm_service or LLMService()

    def verify_topic(
        self,
        video_id: str,
        topic: str,
        top_k: int = 4,
        model: Optional[str] = None,
    ) -> CheckTopicResponse:
        """
        Verify whether a topic is covered in the video:
        1. Perform semantic search for topic chunks in ChromaDB.
        2. Prompt LLM to verify whether the concept is authentically covered.
        3. Return structured verification (YES / NO / PARTIALLY) with timestamps.
        """
        retrieved_chunks = self.vector_store.search_video_chunks(
            video_id=video_id,
            query=topic,
            top_k=top_k,
        )

        if not retrieved_chunks:
            return CheckTopicResponse(
                video_id=video_id,
                topic=topic,
                is_covered=False,
                confidence="NO",
                explanation=f"No matching content for topic '{topic}' was found in the video.",
                relevant_timestamps=[],
            )

        context_lines = []
        for c in retrieved_chunks:
            context_lines.append(
                f"[Segment {c.chunk_id} | Timestamps: {c.start_timestamp} - {c.end_timestamp} (Score: {c.score})]\n{c.text}"
            )
        context_str = "\n\n".join(context_lines)

        user_prompt = (
            f"Topic to verify: '{topic}'\n\n"
            f"Retrieved Transcript Segments:\n"
            f"{context_str}\n\n"
            f"Verify if the topic '{topic}' is covered in these excerpts. Output JSON only."
        )

        raw_response = self.llm_service.generate(
            prompt=user_prompt,
            system_prompt=TOPIC_VERIFY_SYSTEM_PROMPT,
            model=model,
            temperature=0.1,
        )

        # Parse JSON output safely
        is_covered = False
        confidence = "NO"
        explanation = raw_response

        try:
            # Handle markdown code fences if output by LLM
            clean_json = raw_response.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()

            parsed = json.loads(clean_json)
            is_covered = bool(parsed.get("is_covered", False))
            confidence = str(parsed.get("confidence", "NO")).upper()
            explanation = parsed.get("explanation", raw_response)
        except Exception as e:
            logger.warning(f"Could not parse LLM topic verification JSON directly: {e}. Attempting regex extraction.")
            # Extract fields via regex if json had trailing unclosed braces
            conf_match = re.search(r'"confidence"\s*:\s*"([^"]+)"', raw_response, re.IGNORECASE)
            cov_match = re.search(r'"is_covered"\s*:\s*(true|false)', raw_response, re.IGNORECASE)
            exp_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', raw_response, re.IGNORECASE)

            if conf_match:
                confidence = conf_match.group(1).upper()
            if cov_match:
                is_covered = cov_match.group(1).lower() == "true"
            if exp_match:
                explanation = exp_match.group(1)
            else:
                if "yes" in raw_response.lower() and "no" not in raw_response.lower():
                    is_covered = True
                    confidence = "YES"
                elif "no" in raw_response.lower():
                    is_covered = False
                    confidence = "NO"

        # Attach timestamps if covered or partially covered
        relevant_timestamps = []
        if is_covered or confidence in ["YES", "PARTIALLY"]:
            relevant_timestamps = [
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

        return CheckTopicResponse(
            video_id=video_id,
            topic=topic,
            is_covered=is_covered,
            confidence=confidence,
            explanation=explanation,
            relevant_timestamps=relevant_timestamps,
        )

    def extract_topics(
        self,
        video_id: str,
        model: Optional[str] = None,
    ) -> TopicsResponse:
        """
        Extract structured major topics and subtopics from the video transcript.
        """
        # Retrieve sample of chronological chunks from the video
        records = self.vector_store.collection.get(
            where={"video_id": video_id},
            include=["documents", "metadatas"],
        )

        if not records or not records["documents"]:
            return TopicsResponse(video_id=video_id, topics=[])

        # Sort documents by chunk_id
        sorted_items = sorted(
            zip(records["documents"], records["metadatas"]),
            key=lambda x: x[1]["chunk_id"],
        )

        # Form chronological summary text
        summary_snippets = []
        for doc, meta in sorted_items[:30]:  # up to first 30 chunks for topic overview
            summary_snippets.append(
                f"[{meta['start_timestamp']} - {meta['end_timestamp']}]: {doc[:200]}"
            )
        timeline_overview = "\n".join(summary_snippets)

        prompt = (
            f"Analyze this video transcript timeline and extract the main topics and subtopics:\n\n"
            f"{timeline_overview}\n\n"
            f"Return JSON matching the schema."
        )

        raw_response = self.llm_service.generate(
            prompt=prompt,
            system_prompt=TOPIC_EXTRACTION_SYSTEM_PROMPT,
            model=model,
            temperature=0.2,
        )

        topic_items: List[TopicItem] = []
        try:
            clean_json = raw_response.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()

            parsed = json.loads(clean_json)
            for item in parsed.get("topics", []):
                topic_items.append(
                    TopicItem(
                        topic=item.get("topic", "Topic"),
                        start_time=0.0,
                        end_time=0.0,
                        start_timestamp=item.get("start_timestamp", "00:00"),
                        end_timestamp=item.get("end_timestamp", "00:00"),
                        subtopics=item.get("subtopics", []),
                        summary=item.get("summary", ""),
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to parse topics JSON: {e}")

        return TopicsResponse(video_id=video_id, topics=topic_items)
