"""
Indexing pipeline — orchestrates Drive sync + chunking + embedding.
"""

import logging

from app.drive_sync import get_new_or_updated_docs, mark_synced
from app.chunker import chunk_text
from app.vectorstore import add_chunks

logger = logging.getLogger(__name__)


def index_new_docs() -> dict:
    """
    Pull new/updated docs from Drive, chunk, embed, and store.
    Returns {"processed": int, "chunks": int, "errors": list[str]}
    """
    docs = get_new_or_updated_docs()

    if not docs:
        logger.info("No new or updated documents to index.")
        return {"processed": 0, "chunks": 0, "errors": []}

    logger.info(f"Found {len(docs)} new/updated documents to index.")

    total_chunks = 0
    errors = []
    synced_docs = []

    for doc in docs:
        try:
            chunks = chunk_text(doc["content"], doc["name"])
            if chunks:
                add_chunks(chunks, doc["id"])
                total_chunks += len(chunks)
            synced_docs.append(doc)
            logger.info(f"Indexed {doc['name']}: {len(chunks)} chunks")
        except Exception as e:
            error_msg = f"Failed to index {doc['name']}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    # Mark successfully processed docs as synced
    if synced_docs:
        mark_synced(synced_docs)

    return {
        "processed": len(synced_docs),
        "chunks": total_chunks,
        "errors": errors,
    }
