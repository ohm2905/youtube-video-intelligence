import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "YouTube Video Intelligence & Q&A Assistant"
    API_V1_STR: str = "/api"
    CORS_ORIGINS: List[str] = ["*"]

    # Chunking defaults
    DEFAULT_CHUNK_SIZE: int = 250  # target words per chunk
    DEFAULT_CHUNK_OVERLAP: int = 40  # overlapping words between chunks

    # Embedding & Vector Database
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    DEFAULT_EMBEDDING_MODEL: str = os.getenv("DEFAULT_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    TOP_K_RETRIEVAL: int = 5

    # Ollama LLM Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "llama3.2:3b")

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
