"""
Embedding utilities using OpenAI text-embedding-3-small.

Handles batching to stay within API limits.
"""

import logging
from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)

# OpenAI embedding API allows up to 2048 inputs per request,
# but we'll use smaller batches to be safe with rate limits.
_BATCH_SIZE = 100


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using OpenAI text-embedding-3-small.

    Automatically batches large lists.
    Returns a list of embedding vectors (same order as input).
    """
    if not texts:
        return []

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        logger.info(
            "Embedding batch %d-%d of %d texts",
            i + 1,
            min(i + _BATCH_SIZE, len(texts)),
            len(texts),
        )

        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=batch,
        )

        # Response embeddings are returned in order of input
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def get_query_embedding(query: str) -> list[float]:
    """Generate a single embedding for a search query."""
    embeddings = get_embeddings([query])
    return embeddings[0]
