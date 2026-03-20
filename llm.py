"""
LLM abstraction layer — supports both OpenAI and Anthropic.

Provides a unified interface for chat completions with
streaming support and conversation history management.
"""

import logging
from typing import Any

from config import config

logger = logging.getLogger(__name__)

# System prompt for the Eye on AI chatbot
SYSTEM_PROMPT = """You are an expert assistant for the "Eye on AI" podcast, \
hosted by Craig S. Smith. This podcast features in-depth conversations with \
leading AI researchers, engineers, and thought leaders about the latest \
developments in artificial intelligence, machine learning, and their real-world \
applications.

You have access to a searchable database of transcripts from 300+ podcast \
episodes. When a user asks a question, relevant transcript excerpts are \
automatically retrieved from this database and provided below as context.

When answering:

1. Be accurate and specific — cite the episode or guest when the context allows.
2. Synthesize information across multiple episodes when relevant.
3. If the retrieved transcripts don't contain enough information to answer the \
question, say "I couldn't find information about that in the podcast transcripts" \
rather than implying the user needs to provide anything. The database may not \
have a match for every topic.
4. Explain technical concepts clearly — the audience ranges from AI professionals \
to curious enthusiasts.
5. When quoting or closely paraphrasing, indicate which episode the information \
comes from.
6. Keep answers conversational but informative, matching the podcast's style.
7. When an EPISODE INDEX is provided, use it to answer factual questions about \
how many times a guest has appeared, which episodes cover a topic, or to list \
episodes. The episode index is authoritative for episode metadata (guest names, \
episode numbers, dates). Transcript excerpts are authoritative for content.
8. Never tell the user to "provide" or "share" transcripts — you already have \
access to the full database. If you can't find something, it means the search \
didn't return relevant results, not that the user forgot to give you something."""


def _build_messages(
    question: str,
    context_chunks: list[str],
    chunk_metadatas: list[dict[str, Any]],
    conversation_history: list[dict[str, str]] | None = None,
    episode_index_context: str | None = None,
) -> list[dict[str, str]]:
    """Build the message list for the LLM, including context and history."""
    # Format context chunks with source info
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(context_chunks, chunk_metadatas), 1):
        episode = meta.get("episode_title", meta.get("doc_name", "Unknown episode"))
        date = meta.get("episode_date", "")
        guest = meta.get("guest_name", "")
        header = f"[Source {i}: {episode}"
        if guest:
            header += f" — Guest: {guest}"
        if date:
            header += f" ({date})"
        header += "]"
        context_parts.append(f"{header}\n{chunk}")

    context_block = "\n\n---\n\n".join(context_parts)

    messages: list[dict[str, str]] = []

    # Add conversation history (if any)
    if conversation_history:
        for msg in conversation_history[-10:]:  # Keep last 10 turns
            messages.append(msg)

    # Build the user message with context
    parts: list[str] = []

    # Include episode index if available (for factual queries)
    if episode_index_context:
        parts.append(
            "Here is the EPISODE INDEX for the Eye on AI podcast "
            "(use this to answer factual questions about episodes, guests, "
            "and appearances):\n\n"
            f"{episode_index_context}"
        )
        parts.append("---")

    parts.append(
        "Here are relevant transcript excerpts retrieved from the Eye on AI "
        "podcast database:\n\n"
        f"{context_block}"
    )
    parts.append("---")
    parts.append(f"Based on the above context, please answer this question:\n{question}")

    user_message = "\n\n".join(parts)
    messages.append({"role": "user", "content": user_message})

    return messages


def chat_openai(
    question: str,
    context_chunks: list[str],
    chunk_metadatas: list[dict[str, Any]],
    conversation_history: list[dict[str, str]] | None = None,
    episode_index_context: str | None = None,
) -> str:
    """Generate a response using OpenAI."""
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    messages = _build_messages(
        question, context_chunks, chunk_metadatas,
        conversation_history, episode_index_context,
    )

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        temperature=0.3,
        max_tokens=2048,
    )

    return response.choices[0].message.content or ""


def chat_anthropic(
    question: str,
    context_chunks: list[str],
    chunk_metadatas: list[dict[str, Any]],
    conversation_history: list[dict[str, str]] | None = None,
    episode_index_context: str | None = None,
) -> str:
    """Generate a response using Anthropic."""
    from anthropic import Anthropic

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    messages = _build_messages(
        question, context_chunks, chunk_metadatas,
        conversation_history, episode_index_context,
    )

    response = client.messages.create(
        model=config.LLM_MODEL,
        system=SYSTEM_PROMPT,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )

    return response.content[0].text


def generate_response(
    question: str,
    context_chunks: list[str],
    chunk_metadatas: list[dict[str, Any]],
    conversation_history: list[dict[str, str]] | None = None,
    episode_index_context: str | None = None,
) -> str:
    """
    Generate a chat response using the configured LLM provider.

    Dispatches to OpenAI or Anthropic based on config.LLM_PROVIDER.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "openai":
        return chat_openai(
            question, context_chunks, chunk_metadatas,
            conversation_history, episode_index_context,
        )
    elif provider == "anthropic":
        return chat_anthropic(
            question, context_chunks, chunk_metadatas,
            conversation_history, episode_index_context,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai' or 'anthropic'.")
