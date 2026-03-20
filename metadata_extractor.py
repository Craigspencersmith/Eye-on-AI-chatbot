"""
Shared metadata extraction for Eye on AI podcast transcripts.

Provides two strategies for extracting episode metadata (guest name,
episode number, date, topic):

1. **Transcript-based** — sends the first ~3000 chars of a transcript
   to OpenAI gpt-4o-mini and parses structured JSON from the response.
2. **Filename-based** — regex heuristics on the document name (fallback).

Both `backfill_metadata.py` and `ingest.py` import from this module.
"""

import json
import logging
import re
from typing import Any

from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)

# How many characters from the start of the transcript to send to the LLM.
_TRANSCRIPT_PREFIX_LEN = 3000

# The extraction prompt — tuned for the Eye on AI podcast format.
_EXTRACTION_PROMPT = """\
You are a metadata extraction assistant for the "Eye on AI" podcast, \
hosted by Craig S. Smith. You will be given the opening portion of a \
podcast transcript. Extract the following fields as JSON:

{
  "guest_name": "<full name(s) of the guest(s) being interviewed, comma-separated if multiple; empty string if no guest / solo episode>",
  "episode_number": "<episode number as a string, e.g. '123'; empty string if not mentioned>",
  "episode_date": "<date the episode aired in YYYY-MM-DD format; empty string if not determinable>",
  "episode_topic": "<a concise 5-15 word summary of the main topic discussed>"
}

Rules:
- The host is Craig S. Smith (do NOT list him as a guest).
- If multiple guests appear, list all names separated by commas.
- For episode_number, look for patterns like "Episode 123", "Ep. 123", "#123", or similar.
- For episode_date, look for explicit date mentions (e.g. "January 15, 2024", "2024-01-15"). \
  If only a month and year are mentioned, use the first of the month (e.g. "2024-01-01"). \
  If no date is mentioned at all, leave it as an empty string.
- For episode_topic, summarize the main subject of the conversation based on the introduction.
- Return ONLY the JSON object, no markdown fences, no extra text.
"""


def extract_metadata_from_transcript(
    text: str,
    doc_name: str,
) -> dict[str, str]:
    """
    Use OpenAI gpt-4o-mini to extract episode metadata from the opening
    portion of a transcript.

    Args:
        text: Full transcript text (only the first ~3000 chars are sent).
        doc_name: The Google Doc name (included for context).

    Returns:
        Dict with keys: guest_name, episode_number, episode_date, episode_topic.
        Values default to empty strings on failure.
    """
    empty_result: dict[str, str] = {
        "guest_name": "",
        "episode_number": "",
        "episode_date": "",
        "episode_topic": "",
    }

    prefix = text[:_TRANSCRIPT_PREFIX_LEN]
    if not prefix.strip():
        logger.warning("Empty transcript for '%s', cannot extract metadata", doc_name)
        return empty_result

    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Document name: {doc_name}\n\n"
                        f"Transcript opening:\n{prefix}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=300,
        )

        raw = response.choices[0].message.content or ""
        # Strip markdown fences if the model adds them anyway
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed: dict[str, Any] = json.loads(raw)

        # Normalise to strings and only keep expected keys
        result: dict[str, str] = {}
        for key in ("guest_name", "episode_number", "episode_date", "episode_topic"):
            val = parsed.get(key, "")
            result[key] = str(val).strip() if val else ""

        logger.debug(
            "Extracted metadata for '%s': guest=%s, ep=%s, date=%s, topic=%s",
            doc_name,
            result["guest_name"],
            result["episode_number"],
            result["episode_date"],
            result["episode_topic"],
        )
        return result

    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse LLM JSON for '%s': %s (raw: %s)",
            doc_name, exc, raw[:200] if 'raw' in dir() else "N/A",
        )
        return empty_result

    except Exception as exc:
        logger.warning(
            "LLM metadata extraction failed for '%s': %s", doc_name, exc,
        )
        return empty_result


def extract_metadata_from_filename(doc_name: str) -> dict[str, str]:
    """
    Extract episode metadata from the document filename using regex heuristics.

    This is the fallback strategy when the LLM-based extraction is unavailable.

    Common filename patterns:
        "Episode 123 - Guest Name on Topic"
        "Ep 123: Guest Name discusses Topic"
        "#123 Guest Name"
        "Guest Name - Topic (2024-01-15)"
    """
    metadata: dict[str, str] = {
        "guest_name": "",
        "episode_number": "",
        "episode_date": "",
        "episode_topic": "",
    }

    clean = doc_name.strip()

    # --- Date ---
    date_match = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})", clean)
    if date_match:
        metadata["episode_date"] = date_match.group(1)
        clean = clean[:date_match.start()] + clean[date_match.end():]
        clean = re.sub(r"[()]", "", clean).strip()

    # --- Episode number ---
    ep_match = re.search(r"(?:ep(?:isode)?|#)\s*(\d+)", clean, re.IGNORECASE)
    if ep_match:
        metadata["episode_number"] = ep_match.group(1)
        clean = clean[:ep_match.start()] + clean[ep_match.end():]
        clean = re.sub(r"^[\s\-:]+", "", clean).strip()

    # --- Guest name & topic ---
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
            if (
                2 < len(candidate_guest) < 60
                and not candidate_guest.isdigit()
                and len(candidate_guest.split()) <= 6
            ):
                guest_name = candidate_guest
                topic = candidate_topic
                break

    if not guest_name and clean and not clean.isdigit():
        words = clean.split()
        if 2 <= len(words) <= 5 and len(clean) < 60:
            guest_name = clean

    if guest_name:
        guest_name = re.sub(
            r"^(with|featuring|feat\.?|ft\.?)\s+",
            "",
            guest_name,
            flags=re.IGNORECASE,
        ).strip()
        metadata["guest_name"] = guest_name

    if topic:
        metadata["episode_topic"] = topic.strip(" .,;:")

    return metadata
