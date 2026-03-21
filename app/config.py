import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    DRIVE_FOLDER_ID: str = os.getenv("DRIVE_FOLDER_ID", "178HN2Eldzshqt4pQwalArOfz9xL1XWbB")
    GOOGLE_SERVICE_ACCOUNT_FILE: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials/service-account.json")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    SYNC_INTERVAL: int = int(os.getenv("SYNC_INTERVAL", "3600"))
    TOP_K: int = int(os.getenv("TOP_K", "10"))


settings = Settings()
