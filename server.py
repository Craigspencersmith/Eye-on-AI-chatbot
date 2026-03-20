"""
FastAPI server for the Eye on AI Podcast Chatbot.

Endpoints:
    GET  /health        — Health check
    POST /chat          — Chat with the podcast knowledge base
    GET  /stats         — Collection statistics
    GET  /search        — Search episodes by guest, number, or keyword
    GET  /episodes      — List all episodes
    GET  /              — Serves the frontend
"""

import logging
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from config import config
from embeddings import get_query_embedding
from vector_store import get_chroma_client, get_collection, query_chunks, hybrid_search
from llm import generate_response
from episode_index import (
    load_episode_index,
    search_episodes,
    format_episode_index_for_context,
)

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
    version="2.0.0",
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
# ChromaDB + Episode Index — initialize once at startup
# ---------------------------------------------------------------------------
_chroma_client = None
_collection = None
_episode_index: list[dict[str, Any]] = []


@app.on_event("startup")
def startup_event() -> None:
    global _chroma_client, _collection, _episode_index
    _chroma_client = get_chroma_client()
    _collection = get_collection(_chroma_client)
    count = _collection.count()
    logger.info("ChromaDB collection loaded with %d chunks", count)

    # Load episode index
    _episode_index = load_episode_index()
    logger.info("Episode index loaded with %d episodes", len(_episode_index))


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
    episode_count: int
    llm_provider: str
    llm_model: str


class StatsResponse(BaseModel):
    collection_count: int
    episode_count: int
    llm_provider: str
    llm_model: str


class EpisodeItem(BaseModel):
    doc_id: str
    title: str
    guest_name: str
    episode_number: str
    episode_date: str
    episode_topic: str


class SearchResponse(BaseModel):
    results: list[EpisodeItem]
    total: int


# ---------------------------------------------------------------------------
# Factual query detection
# ---------------------------------------------------------------------------

_FACTUAL_PATTERNS = [
    r"\bhow many\b.*\b(episode|time|guest|appear|been on)\b",
    r"\blist\b.*\b(episode|guest|all)\b",
    r"\bwhich episode\b",
    r"\bwhat episode\b",
    r"\bhas .+ been (a |on |the )?guest\b",
    r"\bhow often\b",
    r"\bcount\b.*\bepisode\b",
    r"\ball episodes?\b",
    r"\bevery episode\b",
]


def _is_factual_query(message: str) -> bool:
    """Detect if a query is asking factual/listing questions about episodes."""
    msg_lower = message.lower()
    return any(re.search(p, msg_lower) for p in _FACTUAL_PATTERNS)


def _extract_guest_from_query(message: str) -> str | None:
    """Try to extract a person's name from the query for index lookup."""
    # Common patterns: "how many times has Geoff Hinton been..."
    # "list all episodes with Yann LeCun"
    patterns = [
        r"(?:has|with|featuring|by|about|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s+(?:been|episode|appear|guest))",
    ]
    for p in patterns:
        match = re.search(p, message)
        if match:
            name = match.group(1).strip()
            # Filter out common false positives
            if name.lower() not in {"the", "a", "an", "all", "every", "which", "what", "how"}:
                return name
    return None


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
        episode_count=len(_episode_index),
        llm_provider=config.LLM_PROVIDER,
        llm_model=config.LLM_MODEL,
    )


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    """Collection statistics."""
    count = _collection.count() if _collection else 0
    return StatsResponse(
        collection_count=count,
        episode_count=len(_episode_index),
        llm_provider=config.LLM_PROVIDER,
        llm_model=config.LLM_MODEL,
    )


@app.get("/episodes")
def list_episodes() -> list[EpisodeItem]:
    """Return the full episode index."""
    return [
        EpisodeItem(
            doc_id=ep.get("doc_id", ""),
            title=ep.get("title", ""),
            guest_name=ep.get("guest_name", ""),
            episode_number=ep.get("episode_number", ""),
            episode_date=ep.get("episode_date", ""),
            episode_topic=ep.get("episode_topic", ""),
        )
        for ep in _episode_index
    ]


@app.get("/search", response_model=SearchResponse)
def search(
    guest: str | None = Query(None, description="Guest name (partial, case-insensitive)"),
    episode: int | None = Query(None, description="Episode number"),
    q: str | None = Query(None, description="Keyword search across title, guest, topic"),
) -> SearchResponse:
    """Search episodes by guest name, episode number, or keyword."""
    if not guest and episode is None and not q:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one search parameter: guest, episode, or q",
        )

    results = search_episodes(
        _episode_index, guest=guest, episode_number=episode, keyword=q
    )

    items = [
        EpisodeItem(
            doc_id=ep.get("doc_id", ""),
            title=ep.get("title", ""),
            guest_name=ep.get("guest_name", ""),
            episode_number=ep.get("episode_number", ""),
            episode_date=ep.get("episode_date", ""),
            episode_topic=ep.get("episode_topic", ""),
        )
        for ep in results
    ]

    return SearchResponse(results=items, total=len(items))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Chat endpoint — accepts a message, retrieves relevant transcript chunks,
    and generates a response using the configured LLM.

    For factual queries (counting episodes, listing guests), the episode
    index is included as additional context.
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

        # 2. Retrieve relevant chunks (hybrid: semantic + keyword fallback)
        results = hybrid_search(
            _collection, query_embedding, req.message, top_k=config.TOP_K
        )

        documents: list[str] = results["documents"][0] if results["documents"] else []
        metadatas: list[dict[str, Any]] = results["metadatas"][0] if results["metadatas"] else []
        distances: list[float] = results["distances"][0] if results["distances"] else []

        if not documents:
            return ChatResponse(
                response="I searched the podcast transcript database but couldn't find any "
                         "relevant information for your question. Could you try rephrasing, "
                         "or ask about a different topic covered on the show?",
                sources=[],
                conversation_id=conv_id,
            )

        logger.info(
            "Retrieved %d chunks (distances: %.3f - %.3f)",
            len(documents),
            min(distances) if distances else 0,
            max(distances) if distances else 0,
        )

        # 3. Check if this is a factual query — if so, include episode index
        episode_context: str | None = None
        if _is_factual_query(req.message):
            logger.info("Detected factual query — including episode index")
            # Try to find relevant episodes by guest name
            guest_name = _extract_guest_from_query(req.message)
            if guest_name:
                matching = search_episodes(_episode_index, guest=guest_name)
                if matching:
                    episode_context = format_episode_index_for_context(matching)
                else:
                    # Include full index if no specific match
                    episode_context = format_episode_index_for_context(_episode_index)
            else:
                episode_context = format_episode_index_for_context(_episode_index)

        # 4. Generate LLM response
        response_text = generate_response(
            question=req.message,
            context_chunks=documents,
            chunk_metadatas=metadatas,
            conversation_history=history if history else None,
            episode_index_context=episode_context,
        )

        # 5. Build sources list (deduplicate by episode)
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

        # 6. Update conversation history
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
