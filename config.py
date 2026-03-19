"""
Configuration management for Eye on AI Chatbot.
Loads settings from environment variables / .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent
load_dotenv(_project_root / ".env")


class Config:
    """Central configuration loaded from environment variables."""

    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Google Drive
    GOOGLE_CREDENTIALS_PATH: str = os.getenv(
        "GOOGLE_CREDENTIALS_PATH",
        str(_project_root / "credentials.json"),
    )
    DRIVE_FOLDER_ID: str = os.getenv(
        "DRIVE_FOLDER_ID", "178HN2Eldzshqt4pQwalArOfz9xL1XWbB"
    )

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(_project_root / "chroma_data")
    )

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "8"))

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS: list[str] = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")
    ]

    # Embedding model (fixed — used for both ingestion and retrieval)
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ChromaDB collection name
    COLLECTION_NAME: str = "eye_on_ai_episodes"

    # Ingestion state file (tracks which docs have been processed)
    INGESTION_STATE_PATH: str = os.getenv(
        "INGESTION_STATE_PATH",
        str(_project_root / "ingestion_state.json"),
    )


config = Config()
