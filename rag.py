"""
RAG core: read documents, chunk, embed, retrieve, and answer with source citations.

Embeddings run locally (sentence-transformers) so your data never leaves the machine for
retrieval. Answer generation uses a swappable LLM (Groq or OpenAI). If no key is set, the
app still works in retrieval-only mode so you can see grounding without any API cost.
"""
from __future__ import annotations
import io
from dataclasses import dataclass

import numpy as np

_EMBEDDER = None


def _embedder():
    """Lazy-load the local embedding model (all-MiniLM-L6-v2, 384-dim)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


# ---------------------------------------------------------------- reading files
def read_document(name: str, data: bytes) -> str:
    """Extract text from a .pdf / .txt / .md file given its bytes."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    # txt / md / anything utf-8-ish
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = 170, overlap: int = 40) -> list[str]:
    """Split text into overlapping word-based chunks (keeps context across boundaries)."""
    words = text.split()
    if not words:
        return []
    chunks, step = [], max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


# ---------------------------------------------------------------- the index
@dataclass
class Passage:
    text: str
    source: str


class RAGIndex:
    """A tiny cosine-similarity vector index over document chunks."""

    def __init__(self):
        self.passages: list[Passage] = []
        self._matrix: np.ndarray | None = None

    def add_documents(self, docs: list[tuple[str, str]]) -> int:
        """docs: list of (text, source_name). Returns number of chunks added."""
        added = 0
        for text, source in docs:
            for ch in chunk_text(text):
                self.passages.append(Passage(ch, source))
                added += 1
        if self.passages:
            self._matrix = _embedder().encode(
                [p.text for p in self.passages], normalize_embeddings=True
            ).astype("float32")
        return added

    def retrieve(self, query: str, k: int = 4) -> list[tuple[Passage, float]]:
        if self._matrix is None or not self.passages:
            return []
        q = _embedder().encode([query], normalize_embeddings=True).astype("float32")[0]
        scores = self._matrix @ q  # cosine similarity (vectors are normalized)
        top = np.argsort(-scores)[:k]
        return [(self.passages[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------- answering
SYSTEM = (
    "You are a precise assistant. Answer the question using ONLY the provided context. "
    "Cite the sources you used inline like [1], [2]. If the answer is not in the context, "
    "say clearly that you don't have that information. Never invent facts."
)


def build_prompt(query: str, retrieved: list[tuple[Passage, float]]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] (source: {p.source})\n{p.text}" for i, (p, _) in enumerate(retrieved)
    )
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (cite sources):"


def answer(query: str, retrieved: list[tuple[Passage, float]], provider: str, api_key: str) -> str:
    if not retrieved:
        return "No documents indexed yet. Upload a file or load the sample docs in the sidebar."
    prompt = build_prompt(query, retrieved)

    if provider == "Groq" and api_key:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content

    if provider == "OpenAI" and api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content

    # retrieval-only fallback (no API key), still demonstrates grounding
    lines = ["_No LLM key set, showing the retrieved source passages. "
             "Add a Groq (free) or OpenAI key in the sidebar for generated answers._\n"]
    for i, (p, score) in enumerate(retrieved):
        snippet = p.text[:450] + ("..." if len(p.text) > 450 else "")
        lines.append(f"**[{i + 1}] {p.source}** (match {score:.2f})\n\n{snippet}")
    return "\n\n".join(lines)
