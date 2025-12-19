from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    text: str
    char_start: int
    char_end: int


def normalize_text(raw: str) -> str:
    # Keep logic close to your repo's style: collapse blank lines, strip CR, trim
    return raw.replace("\r", "").replace("\n\n", "\n").strip()


def sliding_window_chunks(text: str, chunk_chars: int, overlap_chars: int) -> List[TextChunk]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must satisfy 0 <= overlap_chars < chunk_chars")

    chunks: List[TextChunk] = []
    start = 0
    i = 0
    step = chunk_chars - overlap_chars

    while start < len(text):
        end = min(start + chunk_chars, len(text))
        seg = text[start:end]
        if seg.strip():
            chunks.append(TextChunk(chunk_id=i, text=seg, char_start=start, char_end=end))
            i += 1
        if end == len(text):
            break
        start += step

    return chunks