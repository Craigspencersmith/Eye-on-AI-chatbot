"""
ChromaDB vector store management.

Provides a thin wrapper around ChromaDB for storing and querying
episode transcript chunks with their embeddings and metadata.
"""

import logging
from typing import Any

import chromadb

from config import config

logger = logging.getLogger(__name__)


def get_chroma_client() -> chromadb.ClientAPI:
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def get_collection(client: chromadb.ClientAPI | None = None) -> chromadb.Collection:
    """Get or create the episode chunks collection."""
    if client is None:
        client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_chunks(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    """
    Add document chunks to the collection.
    Handles ChromaDB's batch size limit by splitting into groups of 5000.
    """
    batch_size = 5000

    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )
        logger.info("Added chunks %d-%d to ChromaDB", i + 1, end)


def delete_doc_chunks(collection: chromadb.Collection, doc_id: str) -> None:
    """Delete all chunks belonging to a specific document (by doc_id metadata)."""
    # ChromaDB where filter to find all chunks for this doc
    results = collection.get(where={"doc_id": doc_id}, include=[])
    if results["ids"]:
        collection.delete(ids=results["ids"])
        logger.info("Deleted %d existing chunks for doc %s", len(results["ids"]), doc_id)


def query_chunks(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Query the collection for the most relevant chunks.

    Returns ChromaDB query results dict with keys:
    ids, documents, metadatas, distances
    """
    top_k = top_k or config.TOP_K

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results
