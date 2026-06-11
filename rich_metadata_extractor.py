"""
Rich metadata extraction for Eye on AI podcast transcripts.

Sends full transcripts to gpt-4o-mini and extracts a comprehensive metadata
schema including: episode profile, thematic tags, key arguments, named entities,
content chapters, citations, bias/affiliation flags, and discourse dynamics.

Designed to be run as a batch process across all 336 episodes.
Saves results to metadata/ directory as individual JSON files per episode.

Usage:
    python rich_metadata_extractor.py                  # process all episodes
    python rich_metadata_extractor.py --episode 150    # process single episode
    python rich_metadata_extractor.py --start 1 --end 50  # process range
    python rich_metadata_extractor.py --dry-run        # show what would be processed
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from google.oauth2 import service_account
from googleapiclient.discovery import build

from config import config

logger = logging.getLogger(__name__)

# --- Configuration ---
METADATA_DIR = Path(__file__).parent / "metadata"
EPISODE_DATES_PATH = Path(__file__).parent / "episode_dates.json"
DRIVE_FOLDER_ID = "1jHl78MEIdfT3hHFPA-sh5QTVrghDK1ZY"
SERVICE_ACCOUNT_PATH = Path(__file__).parent / "credentials" / "service-account.json"

# Rate limiting
API_DELAY_SECONDS = 1.0  # Delay between OpenAI calls
MODEL = "gpt-4o-mini"
MAX_TRANSCRIPT_TOKENS = 100_000  # Safety limit for very long transcripts

# --- Extraction prompt ---
EXTRACTION_PROMPT = """\
You are a metadata extraction specialist for the "Eye on AI" podcast, hosted by \
Craig S. Smith (former NYT correspondent). You will be given the FULL transcript \
of a podcast episode along with its known metadata (episode number, date, RSS title).

Extract comprehensive structured metadata as JSON with the following schema:

{
  "episode_profile": {
    "title": "<concise episode title, 5-15 words>",
    "guest_name": "<full name(s), comma-separated if multiple; empty if solo>",
    "guest_title": "<guest's role/title at time of interview>",
    "guest_organization": "<guest's primary organization/company>",
    "topic_summary": "<2-3 sentence summary of the main discussion>"
  },
  "thematic_tags": {
    "ai_subfields": ["<list of AI subfields discussed: e.g. NLP, computer vision, reinforcement learning, generative AI, robotics>"],
    "industries": ["<industries/sectors: e.g. healthcare, defense, education, finance>"],
    "policy_areas": ["<policy topics: e.g. regulation, ethics, safety, governance, open source>"],
    "geographic_focus": ["<countries/regions prominently discussed>"]
  },
  "key_arguments": [
    {
      "claim": "<a key argument or prediction made by the guest, 1-2 sentences>",
      "speaker": "<who made this claim: guest name or 'Craig Smith'>",
      "context": "<brief context for why this claim was made>"
    }
  ],
  "named_entities": {
    "people": ["<other people mentioned by name (not host/guest)>"],
    "organizations": ["<companies, labs, universities, agencies mentioned>"],
    "technologies": ["<specific AI models, tools, frameworks, datasets mentioned>"],
    "publications": ["<papers, books, reports referenced>"]
  },
  "content_chapters": [
    {
      "title": "<chapter title, 3-8 words>",
      "summary": "<1-2 sentence summary of this section>",
      "approximate_position": "<early/middle/late in the episode>"
    }
  ],
  "citations": [
    {
      "reference": "<paper title, book, or report mentioned>",
      "authors": "<author(s) if mentioned>",
      "context": "<why it was referenced>"
    }
  ],
  "affiliations_and_bias": {
    "guest_affiliations": ["<organizations the guest is affiliated with>"],
    "funding_sources": ["<funding sources mentioned, if any>"],
    "potential_conflicts": "<note any obvious commercial interests or conflicts; 'none apparent' if none>"
  },
  "discourse_dynamics": {
    "interview_style": "<descriptive: e.g. 'technical deep-dive', 'policy discussion', 'career retrospective', 'product demo'>",
    "tone": "<e.g. 'collegial', 'adversarial', 'educational', 'celebratory'>",
    "notable_moments": ["<any particularly striking quotes, disagreements, or insights>"]
  }
}

Rules:
- The host is Craig S. Smith. Do NOT list him as a guest.
- For key_arguments, extract 3-7 of the most significant claims or predictions.
- For content_chapters, identify 3-6 logical sections of the conversation.
- For named_entities, only include entities actually discussed (not passing mentions).
- For citations, only include formally referenced works (papers, books, reports).
- If a field has no applicable content, use an empty list [] or empty string "".
- Return ONLY the JSON object. No markdown fences, no extra text.
"""


# --- Google Drive access ---
def get_drive_service():
    """Get authenticated Google Drive service."""
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    creds = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_PATH), scopes=scopes
    )
    return build('drive', 'v3', credentials=creds)


def get_transcript_files(service) -> dict[int, dict]:
    """
    List all transcript files in the Drive folder.
    Returns {episode_number: {id, name}} mapping.
    """
    files = {}
    page_token = None

    while True:
        results = service.files().list(
            q=f"'{DRIVE_FOLDER_ID}' in parents",
            fields="nextPageToken, files(id, name)",
            pageSize=100,
            pageToken=page_token
        ).execute()

        for f in results.get('files', []):
            # Extract episode number from filename like "123 Transcript.txt" or "235 Tyler Xuan Saltsman.txt"
            m = re.match(r'(\d+)\s', f['name'])
            if m:
                ep_num = int(m.group(1))
                files[ep_num] = {'id': f['id'], 'name': f['name']}

        page_token = results.get('nextPageToken')
        if not page_token:
            break

    return files


def download_transcript(file_id: str, service) -> str:
    """Download and decode a transcript file from Google Drive."""
    content = service.files().get_media(fileId=file_id).execute()
    # Files are UTF-16 encoded
    try:
        return content.decode('utf-16')
    except (UnicodeDecodeError, UnicodeError):
        return content.decode('utf-8', errors='replace')


# --- LLM extraction ---
def extract_rich_metadata(
    transcript: str,
    episode_number: int,
    episode_date: str,
    rss_title: str,
) -> dict[str, Any]:
    """
    Send full transcript to gpt-4o-mini and extract rich metadata.

    Args:
        transcript: Full episode transcript text.
        episode_number: Episode number.
        episode_date: Publication date (YYYY-MM-DD).
        rss_title: RSS feed title for the episode.

    Returns:
        Parsed metadata dict, or error dict on failure.
    """
    # Truncate very long transcripts to stay within context
    if len(transcript) > MAX_TRANSCRIPT_TOKENS * 4:
        transcript = transcript[:MAX_TRANSCRIPT_TOKENS * 4]
        logger.warning("Ep %d: transcript truncated to ~%d tokens", episode_number, MAX_TRANSCRIPT_TOKENS)

    context_header = (
        f"Episode Number: {episode_number}\n"
        f"Publication Date: {episode_date}\n"
        f"RSS Title: {rss_title}\n"
        f"---\n"
    )

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": context_header + transcript},
            ],
            temperature=0.1,
            max_tokens=4000,
        )

        raw = response.choices[0].message.content or ""
        raw = raw.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)

        # Add provenance fields
        parsed["_meta"] = {
            "episode_number": episode_number,
            "episode_date": episode_date,
            "rss_title": rss_title,
            "extraction_model": MODEL,
            "extraction_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_tokens": response.usage.prompt_tokens if response.usage else None,
            "output_tokens": response.usage.completion_tokens if response.usage else None,
        }

        return parsed

    except json.JSONDecodeError as exc:
        logger.error("Ep %d: JSON parse error: %s", episode_number, exc)
        return {"_error": f"JSON parse error: {exc}", "_raw": raw[:500]}

    except Exception as exc:
        logger.error("Ep %d: extraction failed: %s", episode_number, exc)
        return {"_error": str(exc)}


# --- Batch processing ---
def process_episodes(
    episodes: list[int],
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, int]:
    """
    Process a list of episode numbers: download transcript, extract metadata, save.

    Args:
        episodes: List of episode numbers to process.
        dry_run: If True, only show what would be processed.
        force: If True, reprocess even if metadata file already exists.

    Returns:
        Stats dict with counts.
    """
    stats = {"total": len(episodes), "processed": 0, "skipped": 0, "errors": 0}

    # Load episode dates
    with open(EPISODE_DATES_PATH) as f:
        episode_dates = json.load(f)

    # Ensure output directory exists
    METADATA_DIR.mkdir(exist_ok=True)

    if dry_run:
        print(f"DRY RUN: Would process {len(episodes)} episodes")
        for ep in episodes[:10]:
            output_path = METADATA_DIR / f"episode_{ep:03d}.json"
            exists = output_path.exists()
            print(f"  Ep {ep:3d}: {'EXISTS (skip)' if exists and not force else 'PROCESS'}")
        if len(episodes) > 10:
            print(f"  ... and {len(episodes) - 10} more")
        return stats

    # Connect to Drive
    logger.info("Connecting to Google Drive...")
    service = get_drive_service()
    transcript_files = get_transcript_files(service)
    logger.info("Found %d transcript files in Drive", len(transcript_files))

    for i, ep_num in enumerate(episodes, 1):
        output_path = METADATA_DIR / f"episode_{ep_num:03d}.json"

        # Skip if already processed (unless --force)
        if output_path.exists() and not force:
            logger.debug("Ep %d: already processed, skipping", ep_num)
            stats["skipped"] += 1
            continue

        # Check if transcript exists
        if ep_num not in transcript_files:
            logger.warning("Ep %d: no transcript file found in Drive", ep_num)
            stats["errors"] += 1
            continue

        # Get episode date and title from RSS
        ep_info = episode_dates.get(str(ep_num), {})
        ep_date = ep_info.get("date", "unknown")
        rss_title = ep_info.get("title", "unknown")

        logger.info("[%d/%d] Processing episode %d (%s)...", i, stats["total"], ep_num, ep_date)

        try:
            # Download transcript
            transcript = download_transcript(transcript_files[ep_num]['id'], service)
            if not transcript.strip():
                logger.warning("Ep %d: empty transcript", ep_num)
                stats["errors"] += 1
                continue

            # Extract rich metadata
            metadata = extract_rich_metadata(transcript, ep_num, ep_date, rss_title)

            # Save
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            if "_error" in metadata:
                logger.warning("Ep %d: extraction had errors: %s", ep_num, metadata["_error"])
                stats["errors"] += 1
            else:
                stats["processed"] += 1
                tokens_in = metadata.get("_meta", {}).get("input_tokens", "?")
                tokens_out = metadata.get("_meta", {}).get("output_tokens", "?")
                logger.info("  -> OK (tokens: %s in, %s out)", tokens_in, tokens_out)

            # Rate limit
            time.sleep(API_DELAY_SECONDS)

        except Exception as exc:
            logger.error("Ep %d: unexpected error: %s", ep_num, exc, exc_info=True)
            stats["errors"] += 1

    return stats


# --- CLI ---
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Extract rich metadata from Eye on AI transcripts."
    )
    parser.add_argument("--episode", type=int, help="Process a single episode number")
    parser.add_argument("--start", type=int, default=1, help="Start episode (inclusive)")
    parser.add_argument("--end", type=int, default=338, help="End episode (inclusive)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--force", action="store_true", help="Reprocess existing files")
    args = parser.parse_args()

    # Determine which episodes to process
    if args.episode:
        episodes = [args.episode]
    else:
        # All transcript episodes (1-338 minus 120, 121)
        episodes = [ep for ep in range(args.start, args.end + 1) if ep not in (120, 121)]

    print(f"Rich Metadata Extraction")
    print(f"  Episodes: {len(episodes)} ({episodes[0]}-{episodes[-1]})")
    print(f"  Model: {MODEL}")
    print(f"  Output: {METADATA_DIR}/")
    print()

    stats = process_episodes(episodes, dry_run=args.dry_run, force=args.force)

    print()
    print("=" * 50)
    print("Results:")
    print(f"  Total:     {stats['total']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped:   {stats['skipped']}")
    print(f"  Errors:    {stats['errors']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
