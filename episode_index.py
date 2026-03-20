"""
Episode index builder.

Reads the ingestion state file and builds a structured JSON index
of all episodes with metadata (guest names, topics, dates, episode numbers).
This index is used by the chat endpoint to answer factual queries like
"how many times has X been a guest" or "list all episodes about Y".
"""

import json
import logging
from pathlib import Path
from typing import Any

from config import config

logger = logging.getLogger(__name__)


def build_episode_index(
    state_path: str | None = None,
    output_path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build a structured episode index from the ingestion state file.

    Reads ingestion_state.json and produces episodes.json with
    deduplicated, sorted episode entries.

    Returns the list of episode dicts.
    """
    state_path = state_path or config.INGESTION_STATE_PATH
    output_path = output_path or str(
        Path(config.INGESTION_STATE_PATH).parent / "episodes.json"
    )

    # Load ingestion state
    state_file = Path(state_path)
    if not state_file.exists():
        logger.warning("Ingestion state file not found: %s", state_path)
        return []

    with open(state_file, "r") as f:
        state = json.load(f)

    docs = state.get("docs", {})
    episodes: list[dict[str, Any]] = []

    for doc_id, info in docs.items():
        episode: dict[str, Any] = {
            "doc_id": doc_id,
            "title": info.get("name", "Unknown"),
            "guest_name": info.get("guest_name", ""),
            "episode_number": info.get("episode_number", ""),
            "episode_date": info.get("episode_date", ""),
            "episode_topic": info.get("episode_topic", ""),
            "chunks": info.get("chunks", 0),
        }
        episodes.append(episode)

    # Sort by episode number (numeric) if available, else by title
    def sort_key(ep: dict[str, Any]) -> tuple[int, str]:
        num = ep.get("episode_number", "")
        try:
            return (0, str(int(num)).zfill(5))
        except (ValueError, TypeError):
            return (1, ep.get("title", ""))

    episodes.sort(key=sort_key)

    # Write the index
    with open(output_path, "w") as f:
        json.dump(episodes, f, indent=2)

    logger.info(
        "Built episode index with %d episodes → %s", len(episodes), output_path
    )

    return episodes


def load_episode_index(
    index_path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load the episode index from disk.

    Returns an empty list if the file doesn't exist.
    """
    index_path = index_path or str(
        Path(config.INGESTION_STATE_PATH).parent / "episodes.json"
    )

    path = Path(index_path)
    if not path.exists():
        logger.warning("Episode index not found: %s", index_path)
        return []

    with open(path, "r") as f:
        return json.load(f)


def search_episodes(
    episodes: list[dict[str, Any]],
    guest: str | None = None,
    episode_number: int | None = None,
    keyword: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search the episode index by guest name, episode number, or keyword.

    All matching is case-insensitive. Guest name uses partial matching.
    Keyword searches across title, guest_name, and episode_topic.

    Returns matching episodes.
    """
    results = episodes

    if guest:
        guest_lower = guest.lower()
        results = [
            ep for ep in results
            if guest_lower in ep.get("guest_name", "").lower()
            or guest_lower in ep.get("title", "").lower()
        ]

    if episode_number is not None:
        results = [
            ep for ep in results
            if ep.get("episode_number", "") == str(episode_number)
        ]

    if keyword:
        kw_lower = keyword.lower()
        results = [
            ep for ep in results
            if kw_lower in ep.get("title", "").lower()
            or kw_lower in ep.get("guest_name", "").lower()
            or kw_lower in ep.get("episode_topic", "").lower()
        ]

    return results


def format_episode_index_for_context(episodes: list[dict[str, Any]]) -> str:
    """
    Format the episode index as a text block suitable for including
    in the LLM context. Compact format to minimize token usage.
    """
    if not episodes:
        return "No episodes found in the index."

    lines: list[str] = [
        f"EPISODE INDEX ({len(episodes)} episodes):",
        "---",
    ]

    for ep in episodes:
        parts: list[str] = []
        if ep.get("episode_number"):
            parts.append(f"Ep {ep['episode_number']}")
        parts.append(ep.get("title", "Unknown"))
        if ep.get("guest_name"):
            parts.append(f"[Guest: {ep['guest_name']}]")
        if ep.get("episode_date"):
            parts.append(f"({ep['episode_date']})")
        if ep.get("episode_topic"):
            parts.append(f"Topic: {ep['episode_topic']}")

        lines.append(" | ".join(parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    episodes = build_episode_index()
    print(f"\nBuilt index with {len(episodes)} episodes")

    # Print summary
    guests = set(
        ep["guest_name"] for ep in episodes if ep.get("guest_name")
    )
    print(f"Unique guests identified: {len(guests)}")
    if guests:
        for g in sorted(guests)[:20]:
            count = sum(1 for ep in episodes if ep.get("guest_name") == g)
            print(f"  {g}: {count} episode(s)")
        if len(guests) > 20:
            print(f"  ... and {len(guests) - 20} more")
