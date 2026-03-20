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


def keyword_search_chunks(
    collection: chromadb.Collection,
    keyword: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Keyword-based search using ChromaDB's where_document $contains filter.

    Useful as a fallback when semantic search misses exact terms,
    proper nouns, or project names that don't embed well.

    Returns ChromaDB get results re-shaped to match query_chunks format.
    """
    top_k = top_k or config.TOP_K

    results = collection.get(
        where_document={"$contains": keyword},
        include=["documents", "metadatas"],
        limit=top_k,
    )

    # Reshape to match query() output format so callers can use either
    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    # Wrap in outer list to match ChromaDB query format (list of lists)
    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [[0.0] * len(ids)],  # No distance score for keyword search
    }


def hybrid_search(
    collection: chromadb.Collection,
    query_embedding: list[float],
    query_text: str,
    top_k: int | None = None,
    distance_threshold: float = 1.2,
) -> dict[str, Any]:
    """
    Hybrid search: semantic first, keyword fallback.

    Strategy:
    1. Run semantic (vector) search
    2. If best semantic distance is poor (above threshold), also run
       keyword search and merge results (semantic first, keyword appended,
       deduplicated by chunk id)
    3. Also run keyword search if the query contains quoted terms or
       looks like a proper noun / specific term

    Returns results in the same format as query_chunks.
    """
    import re

    top_k = top_k or config.TOP_K

    # 1. Semantic search
    semantic = query_chunks(collection, query_embedding, top_k=top_k)

    sem_ids = semantic["ids"][0] if semantic["ids"] else []
    sem_docs = semantic["documents"][0] if semantic["documents"] else []
    sem_metas = semantic["metadatas"][0] if semantic["metadatas"] else []
    sem_dists = semantic["distances"][0] if semantic["distances"] else []

    # 2. Decide if we need keyword fallback
    best_distance = min(sem_dists) if sem_dists else 999.0
    needs_keyword = best_distance > distance_threshold

    # Also keyword-search if query has quoted terms or short specific terms
    quoted = re.findall(r'["\'](.+?)["\']', query_text)
    # Extract potential proper nouns / specific terms (capitalized words, single
    # distinctive words that aren't common English)
    _COMMON = {
        "the", "a", "an", "is", "are", "was", "were", "what", "who", "how",
        "when", "where", "why", "tell", "me", "about", "all", "any", "some",
        "this", "that", "have", "has", "had", "do", "does", "did", "can",
        "could", "will", "would", "should", "be", "been", "being", "i",
        "you", "we", "they", "it", "my", "your", "our", "their", "from",
        "with", "for", "and", "or", "but", "not", "no", "in", "on", "at",
        "to", "of", "by", "search", "find", "episodes", "episode", "podcast",
        "transcript", "transcripts", "guest", "guests",
    }
    query_words = query_text.strip().split()
    specific_terms = [
        w for w in query_words
        if w.lower() not in _COMMON and len(w) > 2
    ]

    keywords_to_try: list[str] = []
    if quoted:
        keywords_to_try.extend(quoted)
    if needs_keyword and specific_terms:
        # Try the full specific phrase first, then individual terms
        keywords_to_try.append(" ".join(specific_terms))
        if len(specific_terms) > 1:
            keywords_to_try.extend(specific_terms)

    if not keywords_to_try:
        return semantic

    # 3. Run keyword searches and merge
    seen_ids = set(sem_ids)
    merged_ids = list(sem_ids)
    merged_docs = list(sem_docs)
    merged_metas = list(sem_metas)
    merged_dists = list(sem_dists)

    for kw in keywords_to_try:
        if len(merged_ids) >= top_k * 2:
            break
        try:
            kw_results = keyword_search_chunks(collection, kw, top_k=top_k)
            kw_ids = kw_results["ids"][0] if kw_results["ids"] else []
            kw_docs = kw_results["documents"][0] if kw_results["documents"] else []
            kw_metas = kw_results["metadatas"][0] if kw_results["metadatas"] else []

            for i, cid in enumerate(kw_ids):
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    merged_ids.append(cid)
                    merged_docs.append(kw_docs[i])
                    merged_metas.append(kw_metas[i])
                    merged_dists.append(0.5)  # Synthetic mid-range score
        except Exception as e:
            logger.warning("Keyword search for '%s' failed: %s", kw, e)

    logger.info(
        "Hybrid search: %d semantic + %d keyword-added = %d total chunks",
        len(sem_ids),
        len(merged_ids) - len(sem_ids),
        len(merged_ids),
    )

    return {
        "ids": [merged_ids],
        "documents": [merged_docs],
        "metadatas": [merged_metas],
        "distances": [merged_dists],
    }
