"""
Full ingestion pipeline: Google Drive → chunk → embed → ChromaDB.

Supports both full and incremental (sync) ingestion.
Tracks ingestion state in a JSON file to enable efficient updates.
"""

import json
import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Any

from config import config
from drive_client import get_drive_service, list_google_docs, export_doc_as_text
from chunker import chunk_text, count_tokens
from embeddings import get_embeddings
from vector_store import get_chroma_client, get_collection, add_chunks, delete_doc_chunks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ingestion state persistence
# ---------------------------------------------------------------------------

def load_state() -> dict[str, Any]:
    """Load the ingestion state file (tracks doc_id → modifiedTime)."""
    state_path = Path(config.INGESTION_STATE_PATH)
    if state_path.exists():
        with open(state_path, "r") as f:
            return json.load(f)
    return {"docs": {}}


def save_state(state: dict[str, Any]) -> None:
    """Persist ingestion state to disk."""
    with open(config.INGESTION_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Core ingestion logic
# ---------------------------------------------------------------------------

def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{doc_id}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_episode_metadata(doc_name: str) -> dict[str, str]:
    """
    Extract episode title, date, guest name(s), episode number, and topic
    from the document name.

    Common patterns seen in podcast transcripts:
        "Episode 123 - Guest Name on Topic"
        "Ep 123: Guest Name discusses Topic"
        "Guest Name - Topic (2024-01-15)"
        "Guest Name on Topic"
        "#123 Guest Name"
    """
    import re

    metadata: dict[str, str] = {"episode_title": doc_name}

    # Clean up the name for processing
    clean = doc_name.strip()

    # Try to extract a date in various formats
    date_match = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})", clean)
    if date_match:
        metadata["episode_date"] = date_match.group(1)
        # Remove date from clean string for further parsing
        clean = clean[:date_match.start()] + clean[date_match.end():]
        clean = re.sub(r"[()]", "", clean).strip()

    # Try to extract episode number
    ep_match = re.search(r"(?:ep(?:isode)?|#)\s*(\d+)", clean, re.IGNORECASE)
    if ep_match:
        metadata["episode_number"] = ep_match.group(1)
        # Remove episode prefix from clean string
        clean = clean[:ep_match.start()] + clean[ep_match.end():]
        clean = re.sub(r"^[\s\-:]+", "", clean).strip()

    # Try to extract guest name and topic
    # Pattern: "Guest Name on Topic" or "Guest Name - Topic"
    # or "Guest Name discusses Topic" or "Guest Name: Topic"
    separators = [
        (r"\s+on\s+", "on"),
        (r"\s+discusses?\s+", "discusses"),
        (r"\s+talks?\s+about\s+", "talks about"),
        (r"\s*[-–—]\s*", "-"),
        (r"\s*:\s*", ":"),
    ]

    guest_name = ""
    topic = ""

    for sep_pattern, _ in separators:
        parts = re.split(sep_pattern, clean, maxsplit=1)
        if len(parts) == 2:
            candidate_guest = parts[0].strip()
            candidate_topic = parts[1].strip()
            # A guest name should be 2-60 chars and look like a name
            # (not all numbers, not too short)
            if (2 < len(candidate_guest) < 60
                    and not candidate_guest.isdigit()
                    and len(candidate_guest.split()) <= 6):
                guest_name = candidate_guest
                topic = candidate_topic
                break

    # If no separator worked, the whole clean string might be a guest name
    if not guest_name and clean and not clean.isdigit():
        # Check if it looks like a person's name (2-5 words, reasonable length)
        words = clean.split()
        if 2 <= len(words) <= 5 and len(clean) < 60:
            guest_name = clean

    if guest_name:
        # Clean up common prefixes/suffixes
        guest_name = re.sub(r"^(with|featuring|feat\.?|ft\.?)\s+", "", guest_name, flags=re.IGNORECASE).strip()
        metadata["guest_name"] = guest_name

    if topic:
        # Remove trailing punctuation
        topic = topic.strip(" .,;:")
        metadata["episode_topic"] = topic

    return metadata


def ingest_document(
    doc: dict[str, Any],
    service: Any,
    collection: Any,
) -> int:
    """
    Ingest a single Google Doc: export → chunk → embed → store.

    Returns the number of chunks created.
    """
    doc_id = doc["id"]
    doc_name = doc["name"]

    logger.info("Ingesting: %s (id=%s)", doc_name, doc_id)

    # 1. Export as plain text
    text = export_doc_as_text(doc_id, service)
    if not text.strip():
        logger.warning("Document '%s' is empty, skipping", doc_name)
        return 0

    total_tokens = count_tokens(text)
    logger.info("  Document has %d tokens", total_tokens)

    # 2. Chunk the text
    chunks = chunk_text(text)
    if not chunks:
        logger.warning("No chunks produced for '%s', skipping", doc_name)
        return 0

    logger.info("  Produced %d chunks", len(chunks))

    # 3. Delete any existing chunks for this doc (for re-ingestion)
    delete_doc_chunks(collection, doc_id)

    # 4. Prepare metadata
    ep_meta = _extract_episode_metadata(doc_name)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        chunk_id = _make_chunk_id(doc_id, i)
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append(
            {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **ep_meta,
            }
        )

    # 5. Generate embeddings
    logger.info("  Generating embeddings...")
    embeddings = get_embeddings(documents)

    # 6. Store in ChromaDB
    add_chunks(collection, ids, documents, embeddings, metadatas)
    logger.info("  Stored %d chunks in ChromaDB", len(chunks))

    return len(chunks)


def run_ingestion(full: bool = False) -> dict[str, Any]:
    """
    Run the ingestion pipeline.

    Args:
        full: If True, re-ingest everything. If False, only new/modified docs.

    Returns:
        Summary dict with counts.
    """
    state = load_state()
    service = get_drive_service()
    client = get_chroma_client()
    collection = get_collection(client)

    # List all docs in the folder
    docs = list_google_docs(service)
    logger.info("Found %d documents in Google Drive", len(docs))

    stats = {
        "total_docs_found": len(docs),
        "docs_ingested": 0,
        "docs_skipped": 0,
        "total_chunks": 0,
        "errors": 0,
    }

    for doc in docs:
        doc_id = doc["id"]
        modified_time = doc["modifiedTime"]

        # Check if we need to process this doc
        if not full and doc_id in state["docs"]:
            if state["docs"][doc_id]["modifiedTime"] == modified_time:
                logger.debug("Skipping unchanged doc: %s", doc["name"])
                stats["docs_skipped"] += 1
                continue

        try:
            num_chunks = ingest_document(doc, service, collection)
            stats["docs_ingested"] += 1
            stats["total_chunks"] += num_chunks

            # Update state (include extracted metadata for episode index)
            ep_meta = _extract_episode_metadata(doc["name"])
            state["docs"][doc_id] = {
                "name": doc["name"],
                "modifiedTime": modified_time,
                "chunks": num_chunks,
                "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "guest_name": ep_meta.get("guest_name", ""),
                "episode_number": ep_meta.get("episode_number", ""),
                "episode_date": ep_meta.get("episode_date", ""),
                "episode_topic": ep_meta.get("episode_topic", ""),
            }
            save_state(state)

        except Exception as e:
            logger.error("Error ingesting '%s': %s", doc["name"], e, exc_info=True)
            stats["errors"] += 1

    logger.info(
        "Ingestion complete: %d ingested, %d skipped, %d errors, %d total chunks",
        stats["docs_ingested"],
        stats["docs_skipped"],
        stats["errors"],
        stats["total_chunks"],
    )

    # Regenerate the episode index after ingestion
    try:
        from episode_index import build_episode_index
        build_episode_index()
        logger.info("Episode index regenerated successfully")
    except Exception as e:
        logger.error("Failed to regenerate episode index: %s", e, exc_info=True)

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    full = "--full" in sys.argv

    if full:
        logger.info("Starting FULL ingestion (re-processing all documents)...")
    else:
        logger.info("Starting incremental ingestion (new/modified docs only)...")

    stats = run_ingestion(full=full)

    print("\n" + "=" * 60)
    print("Ingestion Summary")
    print("=" * 60)
    print(f"  Documents found:    {stats['total_docs_found']}")
    print(f"  Documents ingested: {stats['docs_ingested']}")
    print(f"  Documents skipped:  {stats['docs_skipped']}")
    print(f"  Total chunks:       {stats['total_chunks']}")
    print(f"  Errors:             {stats['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
