"""
FastAPI application — REST API + chat widget serving.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.chat import chat
from app.indexer import index_new_docs
from app.vectorstore import get_indexed_count
from app.drive_sync import get_total_doc_count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Background sync task
sync_task = None


async def periodic_sync():
    """Background task to periodically sync new docs from Drive."""
    while True:
        try:
            logger.info("Running periodic Drive sync...")
            result = await asyncio.get_event_loop().run_in_executor(None, index_new_docs)
            if result["processed"] > 0:
                logger.info(
                    f"Sync complete: {result['processed']} docs, {result['chunks']} chunks"
                )
            else:
                logger.info("Sync complete: no new documents")
            if result["errors"]:
                for err in result["errors"]:
                    logger.error(f"Sync error: {err}")
        except Exception as e:
            logger.error(f"Periodic sync failed: {e}")

        await asyncio.sleep(settings.SYNC_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sync_task
    # Start background sync
    sync_task = asyncio.create_task(periodic_sync())
    logger.info("Background sync started")
    yield
    # Cleanup
    if sync_task:
        sync_task.cancel()
        try:
            await sync_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Eye on AI Chatbot",
    description="RAG chatbot over Eye on AI podcast transcripts",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow embedding from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models ---


class ChatRequest(BaseModel):
    question: str
    conversation_history: list[dict] | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class SyncResponse(BaseModel):
    processed: int
    chunks: int
    errors: list[str]


class StatusResponse(BaseModel):
    indexed_chunks: int
    drive_docs: int | None = None


# --- Routes ---


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    """Chat with the podcast transcripts."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: chat(req.question, req.conversation_history)
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sync", response_model=SyncResponse)
async def api_sync():
    """Manually trigger a Drive sync."""
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, index_new_docs)
        return SyncResponse(**result)
    except Exception as e:
        logger.error(f"Sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status", response_model=StatusResponse)
async def api_status():
    """Get indexing status."""
    try:
        chunks = get_indexed_count()
        try:
            docs = await asyncio.get_event_loop().run_in_executor(
                None, get_total_doc_count
            )
        except Exception:
            docs = None
        return StatusResponse(indexed_chunks=chunks, drive_docs=docs)
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def serve_widget():
    """Serve the chat widget."""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())
