"""
FastAPI server for the Eye on AI Podcast Chatbot.

Endpoints:
    GET  /health        — Health check
    POST /chat          — Chat with the podcast knowledge base
    GET  /stats         — Collection statistics
    GET  /              — Serves the frontend
"""

import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from config import config
from embeddings import get_query_embedding
from vector_store import get_chroma_client, get_collection, query_chunks
from llm import generate_response

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Eye on AI Chatbot",
    description="RAG chatbot for the Eye on AI podcast transcripts",
    version="1.0.0",
)

# CORS — allow embedding from any origin (configurable)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory conversation store (simple — not meant for high-scale production)
# ---------------------------------------------------------------------------
_conversations: dict[str, list[dict[str, str]]] = {}

# ---------------------------------------------------------------------------
# ChromaDB — initialize once at startup
# ---------------------------------------------------------------------------
_chroma_client = None
_collection = None


@app.on_event("startup")
def startup_event() -> None:
    global _chroma_client, _collection
    _chroma_client = get_chroma_client()
    _collection = get_collection(_chroma_client)
    count = _collection.count()
    logger.info("ChromaDB collection loaded with %d chunks", count)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None


class SourceItem(BaseModel):
    episode: str
    snippet: str


class ChatResponse(BaseModel):
    response: str
    sources: list[SourceItem]
    conversation_id: str


class HealthResponse(BaseModel):
    status: str
    collection_count: int
    llm_provider: str
    llm_model: str


class StatsResponse(BaseModel):
    collection_count: int
    llm_provider: str
    llm_model: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    count = _collection.count() if _collection else 0
    return HealthResponse(
        status="ok",
        collection_count=count,
        llm_provider=config.LLM_PROVIDER,
        llm_model=config.LLM_MODEL,
    )


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    """Collection statistics."""
    count = _collection.count() if _collection else 0
    return StatsResponse(
        collection_count=count,
        llm_provider=config.LLM_PROVIDER,
        llm_model=config.LLM_MODEL,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Chat endpoint — accepts a message, retrieves relevant transcript chunks,
    and generates a response using the configured LLM.
    """
    if _collection is None or _collection.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents have been ingested yet. Run the ingestion pipeline first.",
        )

    # Manage conversation
    conv_id = req.conversation_id or str(uuid.uuid4())
    history = _conversations.get(conv_id, [])

    try:
        # 1. Embed the query
        logger.info("Processing query: %s", req.message[:100])
        query_embedding = get_query_embedding(req.message)

        # 2. Retrieve relevant chunks
        results = query_chunks(_collection, query_embedding)

        documents: list[str] = results["documents"][0] if results["documents"] else []
        metadatas: list[dict[str, Any]] = results["metadatas"][0] if results["metadatas"] else []
        distances: list[float] = results["distances"][0] if results["distances"] else []

        if not documents:
            return ChatResponse(
                response="I couldn't find any relevant information in the podcast transcripts for your question. "
                         "Could you try rephrasing or asking about a different topic?",
                sources=[],
                conversation_id=conv_id,
            )

        logger.info(
            "Retrieved %d chunks (distances: %.3f - %.3f)",
            len(documents),
            min(distances) if distances else 0,
            max(distances) if distances else 0,
        )

        # 3. Generate LLM response
        response_text = generate_response(
            question=req.message,
            context_chunks=documents,
            chunk_metadatas=metadatas,
            conversation_history=history if history else None,
        )

        # 4. Build sources list (deduplicate by episode)
        seen_episodes: set[str] = set()
        sources: list[SourceItem] = []
        for doc, meta in zip(documents, metadatas):
            episode_name = meta.get("episode_title", meta.get("doc_name", "Unknown"))
            if episode_name not in seen_episodes:
                seen_episodes.add(episode_name)
                # Take first ~200 chars as snippet
                snippet = doc[:200].strip()
                if len(doc) > 200:
                    snippet += "..."
                sources.append(SourceItem(episode=episode_name, snippet=snippet))

        # 5. Update conversation history
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": response_text})
        # Keep history manageable
        if len(history) > 20:
            history = history[-20:]
        _conversations[conv_id] = history

        return ChatResponse(
            response=response_text,
            sources=sources,
            conversation_id=conv_id,
        )

    except Exception as e:
        logger.error("Error processing chat request: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------
_static_dir = Path(__file__).resolve().parent / "static"

if _static_dir.exists():
    @app.get("/")
    def serve_index() -> FileResponse:
        return FileResponse(_static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Run with uvicorn
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info",
    )
