
from typing import List
def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    if not text:
        return []
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks[:50]  # hard cap
