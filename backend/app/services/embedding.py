import logging
import torch
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service providing document chunk and query embeddings.
    Defaults to BAAI/bge-small-en-v1.5 with hardware acceleration (Metal/MPS on Mac, CUDA on Linux, or CPU).
    """

    _instance: Optional["EmbeddingService"] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.DEFAULT_EMBEDDING_MODEL
        self.device = self._detect_device()
        logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "EmbeddingService":
        if cls._instance is None or (model_name and cls._instance.model_name != model_name):
            cls._instance = cls(model_name)
        return cls._instance

    def _detect_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed list of document chunks. Documents are embedded as-is.
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed search question/query.
        Applies BGE query instruction prefix if using a BGE model.
        """
        query_text = query.strip()
        if "bge" in self.model_name.lower():
            # Standard instruction recommended by BAAI for BGE query retrieval
            query_text = f"Represent this sentence for searching relevant passages: {query_text}"

        embedding = self.model.encode(
            query_text,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()
