import os
import logging
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.app.core.config import settings
from backend.app.models.schemas import TranscriptChunk, RetrievedChunk
from backend.app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Manages vector storage and similarity retrieval using ChromaDB.
    Maintains chunk text and timestamp metadata for precise grounded retrieval.
    """

    _instance: Optional["VectorStoreService"] = None

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = "youtube_transcripts"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB initialized at '{self.persist_directory}', collection '{self.collection_name}'.")

    @classmethod
    def get_instance(cls, persist_directory: Optional[str] = None) -> "VectorStoreService":
        if cls._instance is None:
            cls._instance = cls(persist_directory)
        return cls._instance

    def is_video_indexed(self, video_id: str) -> bool:
        """Check if transcript chunks for the given video_id are already present."""
        existing = self.collection.get(
            where={"video_id": video_id},
            limit=1,
        )
        return bool(existing and existing["ids"])

    def index_video_chunks(
        self,
        video_id: str,
        chunks: List[TranscriptChunk],
        embedding_service: Optional[EmbeddingService] = None,
    ) -> int:
        """
        Embed and index chunks for a video.
        Skips indexing if the video has already been indexed.
        """
        if not chunks:
            return 0

        if self.is_video_indexed(video_id):
            logger.info(f"Video {video_id} already indexed in ChromaDB. Skipping re-indexing.")
            existing_count = len(self.collection.get(where={"video_id": video_id})["ids"])
            return existing_count

        embedder = embedding_service or EmbeddingService.get_instance()
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_documents(texts)

        ids = [f"{video_id}_chunk_{chunk.chunk_id}" for chunk in chunks]
        metadatas = [
            {
                "video_id": video_id,
                "chunk_id": chunk.chunk_id,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "start_timestamp": chunk.start_timestamp,
                "end_timestamp": chunk.end_timestamp,
                "word_count": chunk.word_count,
            }
            for chunk in chunks
        ]

        # Add in batches to avoid any payload limits
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            end_idx = i + batch_size
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )

        logger.info(f"Indexed {len(chunks)} chunks for video {video_id}.")
        return len(chunks)

    def search_video_chunks(
        self,
        video_id: str,
        query: str,
        top_k: int = 5,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> List[RetrievedChunk]:
        """
        Perform similarity search restricted to chunks of the given video_id.
        Returns top_k chunks sorted by relevance, with timestamps and cosine similarity scores.
        """
        embedder = embedding_service or EmbeddingService.get_instance()
        query_vector = embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"video_id": video_id},
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []

        if not results or not results["ids"] or not results["ids"][0]:
            return retrieved

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance = 1 - cosine_similarity
            similarity = round(max(0.0, 1.0 - dist), 4)
            retrieved.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    text=doc,
                    start_time=meta["start_time"],
                    end_time=meta["end_time"],
                    start_timestamp=meta["start_timestamp"],
                    end_timestamp=meta["end_timestamp"],
                    score=similarity,
                )
            )

        return retrieved

    def delete_video_chunks(self, video_id: str) -> int:
        """Delete all chunks for a video from the vector store."""
        records = self.collection.get(where={"video_id": video_id})
        ids = records.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)
