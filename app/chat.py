"""
Chat — RAG pipeline: retrieve relevant chunks, build prompt, call LLM.
"""

import logging
from typing import Optional

from openai import OpenAI

from app.config import settings
from app.vectorstore import query as vector_query

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Eye on AI podcast assistant. You answer questions about the Eye on AI podcast, which covers AI research, applications, and industry developments across 300+ episodes.

You have access to transcript excerpts from the podcast. Use them to give accurate, specific answers. When referencing information, mention which episode it came from when possible.

Guidelines:
- Be accurate — only cite information from the provided transcript excerpts
- If the transcripts don't contain enough information to fully answer, say so
- When multiple episodes discuss a topic, synthesize across them
- Keep answers conversational but informative
- If asked about something clearly outside the podcast's scope, note that and answer if you can

If no relevant transcript excerpts are provided, let the user know that you couldn't find relevant content in the transcripts."""


def build_context(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks."""
    if not chunks:
        return "No relevant transcript excerpts found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["source"]
        text = chunk["text"]
        score = chunk.get("score", 0)
        context_parts.append(f"[Excerpt {i} — {source} (relevance: {score:.2f})]\n{text}")

    return "\n\n---\n\n".join(context_parts)


def chat(question: str, conversation_history: Optional[list[dict]] = None) -> dict:
    """
    RAG chat: retrieve context, build prompt, get LLM response.
    Returns {"answer": str, "sources": list[str]}
    """
    # Retrieve relevant chunks
    chunks = vector_query(question)
    context = build_context(chunks)

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history if provided (last 10 exchanges max)
    if conversation_history:
        messages.extend(conversation_history[-20:])  # last 10 Q&A pairs

    # Add the current question with context
    user_message = f"""Here are relevant transcript excerpts from the Eye on AI podcast:

{context}

---

User question: {question}"""

    messages.append({"role": "user", "content": user_message})

    # Call LLM
    if settings.LLM_PROVIDER == "openai":
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )
        answer = response.choices[0].message.content
    elif settings.LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        # Extract system message and convert format
        system_msg = messages[0]["content"]
        chat_messages = messages[1:]
        response = client.messages.create(
            model=settings.LLM_MODEL,
            system=system_msg,
            messages=chat_messages,
            temperature=0.3,
            max_tokens=2000,
        )
        answer = response.content[0].text
    else:
        raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")

    # Collect unique sources
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {"answer": answer, "sources": sources}
