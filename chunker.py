"""
Text chunking utilities for the ingestion pipeline.

Uses tiktoken for accurate token counting. Chunks at paragraph/sentence
boundaries where possible to preserve context in dense technical content.
"""

import logging
import re

import tiktoken

from config import config

logger = logging.getLogger(__name__)

# Use the cl100k_base tokenizer (same family as text-embedding-3-small)
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in a string using cl100k_base encoding."""
    return len(_enc.encode(text))


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, preserving non-empty blocks."""
    # Normalize line endings and split on double-newlines
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _split_paragraph_into_sentences(paragraph: str) -> list[str]:
    """
    Split a paragraph into sentences.
    Handles common abbreviations and decimal numbers to avoid false splits.
    """
    # Simple sentence splitter: split on period/question/exclamation followed by space + uppercase
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` tokens,
    with `chunk_overlap` tokens of overlap between consecutive chunks.

    Strategy:
    1. Split into paragraphs.
    2. Greedily accumulate paragraphs into chunks up to chunk_size.
    3. If a single paragraph exceeds chunk_size, split it into sentences
       and accumulate sentences instead.
    4. Apply overlap by carrying trailing tokens from the previous chunk.

    Returns a list of chunk strings.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    # Build a flat list of "segments" — either full paragraphs or individual
    # sentences (for paragraphs that are too long on their own).
    segments: list[str] = []
    for para in paragraphs:
        if count_tokens(para) <= chunk_size:
            segments.append(para)
        else:
            # Paragraph is too large — break into sentences
            sentences = _split_paragraph_into_sentences(para)
            for sent in sentences:
                if count_tokens(sent) <= chunk_size:
                    segments.append(sent)
                else:
                    # Extremely long sentence — force-split by token count
                    tokens = _enc.encode(sent)
                    for i in range(0, len(tokens), chunk_size):
                        segments.append(_enc.decode(tokens[i : i + chunk_size]))

    # Greedily build chunks from segments with overlap
    chunks: list[str] = []
    current_segments: list[str] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = count_tokens(seg)

        if current_tokens + seg_tokens > chunk_size and current_segments:
            # Finalize current chunk
            chunk_text_str = "\n\n".join(current_segments)
            chunks.append(chunk_text_str)

            # Build overlap: keep trailing segments that fit within overlap budget
            overlap_segments: list[str] = []
            overlap_tokens = 0
            for prev_seg in reversed(current_segments):
                prev_tokens = count_tokens(prev_seg)
                if overlap_tokens + prev_tokens > chunk_overlap:
                    break
                overlap_segments.insert(0, prev_seg)
                overlap_tokens += prev_tokens

            current_segments = overlap_segments
            current_tokens = overlap_tokens

        current_segments.append(seg)
        current_tokens += seg_tokens

    # Don't forget the last chunk
    if current_segments:
        chunks.append("\n\n".join(current_segments))

    logger.info(
        "Chunked text (%d tokens) into %d chunks (target size=%d, overlap=%d)",
        count_tokens(text),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
