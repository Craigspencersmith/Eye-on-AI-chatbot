"""
Vector store — ChromaDB wrapper for storing and querying transcript embeddings.
"""

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_client: Optional[chromadb.ClientAPI] = None
_collection = None
_openai: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name="transcripts",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using OpenAI."""
    client = get_openai_client()
    # OpenAI allows up to 2048 texts per request; batch if needed
    all_embeddings = []
    batch_size = 500
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def add_chunks(chunks: list[dict], doc_id: str):
    """
    Add chunks to ChromaDB.
    chunks: list of {"text": str, "metadata": {"source": str, "chunk_index": int}}
    doc_id: Google Drive file ID (used to build unique chunk IDs)
    """
    if not chunks:
        return

    collection = get_collection()

    ids = [f"{doc_id}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    logger.info(f"Embedding {len(texts)} chunks for {chunks[0]['metadata']['source']}...")
    embeddings = embed_texts(texts)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info(f"Stored {len(texts)} chunks in ChromaDB")


def delete_doc_chunks(doc_id: str):
    """Remove all chunks for a given doc from ChromaDB."""
    collection = get_collection()
    # Get all IDs matching this doc
    results = collection.get(where={"source": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])


def query(question: str, top_k: Optional[int] = None) -> list[dict]:
    """
    Query the vector store. Returns list of:
    {"text": str, "source": str, "score": float}
    """
    if top_k is None:
        top_k = settings.TOP_K

    collection = get_collection()

    # Check if collection has any documents
    if collection.count() == 0:
        return []

    question_embedding = embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "score": 1 - dist,  # cosine distance → similarity
            }
        )
    return output


def get_indexed_count() -> int:
    """Get total number of chunks in the collection."""
    try:
        return get_collection().count()
    except Exception:
        return 0
