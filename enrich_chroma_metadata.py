"""
Enrich ChromaDB chunk metadata with rich extracted episode metadata.

Reads the per-episode JSON files from metadata/ and updates each chunk
in the 'transcripts' ChromaDB collection with structured fields:
  - episode_number, episode_date, title, guest_name, guest_organization
  - topic_summary, ai_subfields, industries, policy_areas, geographic_focus
  - interview_style

This enables metadata-filtered queries and richer context in RAG responses.

Usage:
    python enrich_chroma_metadata.py              # enrich all episodes
    python enrich_chroma_metadata.py --episode 235  # single episode
    python enrich_chroma_metadata.py --dry-run    # show what would happen
"""

import argparse
import json
import logging
import re
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)

METADATA_DIR = Path(__file__).parent / "metadata"
CHROMA_DIR = Path(__file__).parent / "chroma_data"
COLLECTION_NAME = "transcripts"


def flatten_rich_metadata(rich: dict) -> dict:
    """
    Flatten the rich metadata JSON into a flat dict suitable for ChromaDB metadata.
    Joins lists as comma-separated strings.
    """
    profile = rich.get("episode_profile", {})
    tags = rich.get("thematic_tags", {})
    dynamics = rich.get("discourse_dynamics", {})
    meta_info = rich.get("_meta", {})

    flat = {}

    # Episode profile
    flat["title"] = profile.get("title", "")
    flat["guest_name"] = profile.get("guest_name", "")
    flat["guest_title"] = profile.get("guest_title", "")
    flat["guest_organization"] = profile.get("guest_organization", "")
    flat["topic_summary"] = profile.get("topic_summary", "")

    # Provenance
    flat["episode_number"] = str(meta_info.get("episode_number", ""))
    flat["episode_date"] = meta_info.get("episode_date", "")
    flat["rss_title"] = meta_info.get("rss_title", "")

    # Thematic tags (joined as comma-separated)
    flat["ai_subfields"] = ", ".join(tags.get("ai_subfields", []))
    flat["industries"] = ", ".join(tags.get("industries", []))
    flat["policy_areas"] = ", ".join(tags.get("policy_areas", []))
    flat["geographic_focus"] = ", ".join(tags.get("geographic_focus", []))

    # Discourse
    flat["interview_style"] = dynamics.get("interview_style", "")
    flat["tone"] = dynamics.get("tone", "")

    # Named entities (top organizations and technologies)
    entities = rich.get("named_entities", {})
    flat["organizations"] = ", ".join(entities.get("organizations", [])[:10])
    flat["technologies"] = ", ".join(entities.get("technologies", [])[:10])

    # Clean: remove empty strings to keep metadata lean
    flat = {k: v for k, v in flat.items() if v}

    return flat


def enrich_collection(
    episodes: list[int] | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Update ChromaDB chunk metadata with rich episode metadata.
    """
    stats = {"episodes_processed": 0, "chunks_updated": 0, "errors": 0, "skipped": 0}

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(COLLECTION_NAME)
    logger.info("Connected to ChromaDB collection '%s' (%d chunks)", COLLECTION_NAME, col.count())

    # Determine which episodes to process
    if episodes is None:
        metadata_files = sorted(METADATA_DIR.glob("episode_*.json"))
    else:
        metadata_files = []
        for ep in episodes:
            path = METADATA_DIR / f"episode_{ep:03d}.json"
            if path.exists():
                metadata_files.append(path)
            else:
                logger.warning("No metadata file for episode %d", ep)
                stats["errors"] += 1

    logger.info("Will process %d episode metadata files", len(metadata_files))

    for i, meta_path in enumerate(metadata_files, 1):
        m = re.match(r"episode_(\d+)\.json", meta_path.name)
        if not m:
            continue
        ep_num = int(m.group(1))
        source_key = f"episode_{ep_num:03d}"

        # Load rich metadata
        try:
            with open(meta_path) as f:
                rich = json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            logger.error("Ep %d: failed to load metadata: %s", ep_num, exc)
            stats["errors"] += 1
            continue

        if "_error" in rich:
            logger.warning("Ep %d: metadata has errors, skipping", ep_num)
            stats["skipped"] += 1
            continue

        flat_meta = flatten_rich_metadata(rich)

        if dry_run:
            if i <= 3:
                logger.info("Ep %d: would update with: %s", ep_num, list(flat_meta.keys()))
            stats["episodes_processed"] += 1
            continue

        # Get all chunk IDs for this episode
        chunk_ids = []
        existing_metas = []
        for source_variant in [source_key, f"episode_{ep_num}"]:
            try:
                results = col.get(
                    where={"source": source_variant},
                    include=["metadatas"],
                )
                if results.get("ids"):
                    chunk_ids = results["ids"]
                    existing_metas = results["metadatas"]
                    break
            except Exception:
                continue

        if not chunk_ids:
            logger.debug("Ep %d: no chunks found in collection", ep_num)
            stats["skipped"] += 1
            continue

        # Build updated metadata for each chunk (preserve existing + add new)
        updated_metas = []
        for existing in existing_metas:
            merged = {**existing, **flat_meta}
            updated_metas.append(merged)

        # Batch update
        BATCH_SIZE = 500
        for batch_start in range(0, len(chunk_ids), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(chunk_ids))
            col.update(
                ids=chunk_ids[batch_start:batch_end],
                metadatas=updated_metas[batch_start:batch_end],
            )

        stats["episodes_processed"] += 1
        stats["chunks_updated"] += len(chunk_ids)

        if i % 50 == 0 or i == len(metadata_files):
            logger.info(
                "Progress: %d/%d episodes (%d chunks updated)",
                i, len(metadata_files), stats["chunks_updated"],
            )

    return stats


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Enrich ChromaDB chunks with rich episode metadata."
    )
    parser.add_argument("--episode", type=int, help="Process a single episode")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    args = parser.parse_args()

    episodes = [args.episode] if args.episode else None

    print("Enriching ChromaDB metadata")
    print(f"  Metadata dir: {METADATA_DIR}")
    print(f"  ChromaDB: {CHROMA_DIR}")
    print(f"  Collection: {COLLECTION_NAME}")
    if args.dry_run:
        print("  MODE: DRY RUN")
    print()

    stats = enrich_collection(episodes=episodes, dry_run=args.dry_run)

    print()
    print("=" * 50)
    print("Results:")
    print(f"  Episodes processed: {stats['episodes_processed']}")
    print(f"  Chunks updated:     {stats['chunks_updated']}")
    print(f"  Skipped:            {stats['skipped']}")
    print(f"  Errors:             {stats['errors']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
