"""
Backfill episode metadata for all transcripts in Google Drive.

Connects to Google Drive, exports each document, and uses OpenAI
gpt-4o-mini to extract guest names, episode numbers, dates, and topics
from the transcript content.  Updates (or creates) ingestion_state.json
and regenerates episodes.json.

Usage:
    python backfill_metadata.py                    # uses default state path
    python backfill_metadata.py --state-path ./ingestion_state.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from config import config
from drive_client import get_drive_service, list_google_docs, export_doc_as_text
from metadata_extractor import (
    extract_metadata_from_transcript,
    extract_metadata_from_filename,
)
from episode_index import build_episode_index

logger = logging.getLogger(__name__)

# Delay between OpenAI API calls to stay well within rate limits.
_API_DELAY_SECONDS = 0.5


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _load_state(state_path: str) -> dict[str, Any]:
    """Load ingestion state, returning an empty skeleton if not found."""
    path = Path(state_path)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    logger.info("No existing state file at %s — starting fresh", state_path)
    return {"docs": {}}


def _save_state(state: dict[str, Any], state_path: str) -> None:
    """Persist ingestion state to disk."""
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Core backfill logic
# ---------------------------------------------------------------------------

def backfill(state_path: str) -> dict[str, int]:
    """
    Iterate over every Google Doc in the Drive folder, extract metadata
    via the LLM, and merge it into the ingestion state file.

    Returns a summary dict with counts.
    """
    stats = {
        "total": 0,
        "metadata_found": 0,
        "errors": 0,
        "skipped_empty": 0,
    }

    # --- Load existing state ---
    state = _load_state(state_path)

    # --- Connect to Google Drive ---
    service = get_drive_service()
    docs = list_google_docs(service)
    stats["total"] = len(docs)
    logger.info("Found %d documents in Google Drive", len(docs))

    for i, doc in enumerate(docs, 1):
        doc_id = doc["id"]
        doc_name = doc["name"]

        logger.info("[%d/%d] Processing: %s", i, stats["total"], doc_name)

        try:
            # Export document text
            text = export_doc_as_text(doc_id, service)
            if not text.strip():
                logger.warning("  → empty document, skipping")
                stats["skipped_empty"] += 1
                continue

            # Extract metadata via LLM
            meta = extract_metadata_from_transcript(text, doc_name)

            # Fall back to filename if LLM returned nothing useful
            if not meta.get("guest_name") and not meta.get("episode_topic"):
                logger.info("  → LLM extraction sparse, augmenting from filename")
                fallback = extract_metadata_from_filename(doc_name)
                for key in ("guest_name", "episode_number", "episode_date", "episode_topic"):
                    if not meta.get(key) and fallback.get(key):
                        meta[key] = fallback[key]

            has_useful_meta = bool(
                meta.get("guest_name") or meta.get("episode_topic")
            )
            if has_useful_meta:
                stats["metadata_found"] += 1

            logger.info(
                "  → guest=%s | ep=%s | date=%s | topic=%s",
                meta.get("guest_name", "—") or "—",
                meta.get("episode_number", "—") or "—",
                meta.get("episode_date", "—") or "—",
                (meta.get("episode_topic", "—") or "—")[:60],
            )

            # Merge into state — preserve existing fields (chunks, ingested_at, etc.)
            existing = state["docs"].get(doc_id, {})
            existing.update({
                "name": doc_name,
                "modifiedTime": doc.get("modifiedTime", existing.get("modifiedTime", "")),
                "guest_name": meta.get("guest_name", ""),
                "episode_number": meta.get("episode_number", ""),
                "episode_date": meta.get("episode_date", ""),
                "episode_topic": meta.get("episode_topic", ""),
            })
            state["docs"][doc_id] = existing

            # Save after each doc so progress isn't lost on crash
            _save_state(state, state_path)

            # Rate limit
            time.sleep(_API_DELAY_SECONDS)

        except Exception as exc:
            logger.error("  → ERROR: %s", exc, exc_info=True)
            stats["errors"] += 1

    # --- Rebuild episode index ---
    logger.info("Regenerating episode index...")
    try:
        output_path = str(Path(state_path).parent / "episodes.json")
        build_episode_index(state_path=state_path, output_path=output_path)
        logger.info("Episode index written to %s", output_path)
    except Exception as exc:
        logger.error("Failed to rebuild episode index: %s", exc, exc_info=True)

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Backfill episode metadata from transcript content via LLM.",
    )
    parser.add_argument(
        "--state-path",
        default=config.INGESTION_STATE_PATH,
        help="Path to ingestion_state.json (default: from config)",
    )
    args = parser.parse_args()

    logger.info("Starting metadata backfill (state: %s)", args.state_path)
    stats = backfill(args.state_path)

    print("\n" + "=" * 60)
    print("Metadata Backfill Summary")
    print("=" * 60)
    print(f"  Total documents:       {stats['total']}")
    print(f"  Metadata extracted:    {stats['metadata_found']}")
    print(f"  Empty (skipped):       {stats['skipped_empty']}")
    print(f"  Errors:                {stats['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
