"""
Sync script — checks Google Drive for new or updated documents
and ingests only those. Designed to be run via cron or as a periodic task.

Usage:
    python sync.py              # Incremental sync
    python sync.py --full       # Full re-ingestion
    python sync.py --dry-run    # Show what would be synced without doing it
"""

import logging
import sys

from config import config
from drive_client import get_drive_service, list_google_docs
from ingest import load_state, run_ingestion

logger = logging.getLogger(__name__)


def dry_run() -> None:
    """Show which documents would be ingested without actually doing it."""
    state = load_state()
    service = get_drive_service()
    docs = list_google_docs(service)

    new_docs = []
    modified_docs = []
    unchanged_docs = []

    for doc in docs:
        doc_id = doc["id"]
        modified_time = doc["modifiedTime"]

        if doc_id not in state.get("docs", {}):
            new_docs.append(doc)
        elif state["docs"][doc_id]["modifiedTime"] != modified_time:
            modified_docs.append(doc)
        else:
            unchanged_docs.append(doc)

    print("\n" + "=" * 60)
    print("Sync Dry Run")
    print("=" * 60)
    print(f"  Total documents:    {len(docs)}")
    print(f"  New (not ingested): {len(new_docs)}")
    print(f"  Modified:           {len(modified_docs)}")
    print(f"  Unchanged:          {len(unchanged_docs)}")

    if new_docs:
        print("\nNew documents:")
        for d in new_docs:
            print(f"  + {d['name']}")

    if modified_docs:
        print("\nModified documents:")
        for d in modified_docs:
            print(f"  ~ {d['name']}")

    print("=" * 60)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if "--dry-run" in sys.argv:
        dry_run()
        return

    full = "--full" in sys.argv

    if full:
        logger.info("Running full re-sync...")
    else:
        logger.info("Running incremental sync...")

    stats = run_ingestion(full=full)

    print("\n" + "=" * 60)
    print("Sync Summary")
    print("=" * 60)
    print(f"  Documents found:    {stats['total_docs_found']}")
    print(f"  Documents synced:   {stats['docs_ingested']}")
    print(f"  Documents skipped:  {stats['docs_skipped']}")
    print(f"  Total chunks:       {stats['total_chunks']}")
    print(f"  Errors:             {stats['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
