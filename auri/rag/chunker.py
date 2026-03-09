"""
Text chunker for RAG ingestion.

Splits text into paragraph-based chunks. Long paragraphs are split further
at sentence boundaries. Each chunk carries its source path and position index
for citation rendering.
"""

from __future__ import annotations


def chunk_text(text: str, source: str, max_chars: int = 800) -> list[dict]:
    """
    Split text into chunks suitable for embedding.

    Returns a list of dicts: {text, source, chunk_index}.
    chunk_index is a string like "3" or "3.1" (sub-chunk of paragraph 3).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict] = []

    for para_idx, para in enumerate(paragraphs):
        if len(para) <= max_chars:
            chunks.append({
                "text": para,
                "source": source,
                "chunk_index": str(para_idx),
            })
        else:
            # Split at sentence boundaries (". " heuristic)
            sentences = para.replace("\n", " ").split(". ")
            current = ""
            sub_idx = 0
            for sent in sentences:
                candidate = current + sent + ". "
                if len(candidate) > max_chars and current:
                    chunks.append({
                        "text": current.rstrip(),
                        "source": source,
                        "chunk_index": f"{para_idx}.{sub_idx}",
                    })
                    sub_idx += 1
                    current = sent + ". "
                else:
                    current = candidate
            if current.strip():
                chunks.append({
                    "text": current.rstrip(),
                    "source": source,
                    "chunk_index": f"{para_idx}.{sub_idx}",
                })

    return chunks
