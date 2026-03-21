"""
Text chunking — splits transcript text into overlapping chunks for embedding.
"""

from app.config import settings


def chunk_text(text: str, doc_name: str) -> list[dict]:
    """
    Split text into overlapping chunks.
    Returns list of {"text": str, "metadata": {"source": doc_name, "chunk_index": int}}
    """
    chunk_size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    chunks = []

    # Clean up whitespace
    text = text.strip()
    if not text:
        return []

    start = 0
    chunk_index = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break first
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break + 2
            else:
                # Look for sentence end
                for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                    sent_break = text.rfind(sep, start, end)
                    if sent_break > start + chunk_size // 2:
                        end = sent_break + len(sep)
                        break

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append(
                {
                    "text": chunk_text_str,
                    "metadata": {
                        "source": doc_name,
                        "chunk_index": chunk_index,
                    },
                }
            )
            chunk_index += 1

        start = end - overlap
        if start <= 0 and chunk_index > 0:
            break

    return chunks
