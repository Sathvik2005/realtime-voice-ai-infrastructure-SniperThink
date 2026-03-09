"""
RAG Retrieval Pipeline

Combines EmbeddingEngine + VectorStore into a single async function that
the AudioRouter calls at runtime:

    context = await retrieve_context(transcript)

The function is designed to be a no-op if the knowledge base is empty
or the store has not been initialised, so the rest of the pipeline
is never blocked by RAG failures.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from rag.embeddings import EmbeddingEngine
from rag.vector_store import VectorStore, Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_embedder: Optional[EmbeddingEngine] = None
_store: Optional[VectorStore] = None

SIMILARITY_THRESHOLD = 0.35   # Minimum score to include a chunk
MAX_CONTEXT_CHARS = 1_200     # Limit context injected into LLM prompt


async def _ensure_initialised() -> None:
    """Lazy-init embedder + vector store on first use."""
    global _embedder, _store

    if _embedder is None:
        _embedder = EmbeddingEngine()
        await _embedder.load()

    if _store is None:
        _store = VectorStore(embedding_dim=384)
        await _store.load_or_create()

        # Auto-ingest knowledge base files on first run
        kb_path = os.environ.get("KNOWLEDGE_BASE_PATH", "rag/knowledge_base")
        if Path(kb_path).exists() and _store.size == 0:
            await ingest_directory(kb_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def retrieve_context(query: str, top_k: int = 4) -> str:
    """
    Given a user transcript, retrieve the most relevant knowledge chunks
    and return them as a formatted string for LLM context injection.

    Returns an empty string if nothing relevant is found.
    """
    try:
        await _ensure_initialised()

        if _store.size == 0:
            return ""

        query_vec = await _embedder.embed_query(query)
        results = await _store.search(query_vec, top_k=top_k)

        chunks = [
            doc.text
            for doc, score in results
            if score >= SIMILARITY_THRESHOLD
        ]

        if not chunks:
            return ""

        context = "\n\n".join(chunks)
        # Truncate to avoid blowing up the LLM context window
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "…"

        logger.debug(f"RAG: injecting {len(chunks)} chunk(s), {len(context)} chars")
        return context

    except Exception as exc:
        logger.warning(f"RAG retrieval error (non-fatal): {exc}")
        return ""


async def ingest_texts(texts: List[str], source: str = "manual") -> int:
    """
    Embed and index a list of text strings.
    Returns the number of documents added.
    """
    await _ensure_initialised()

    documents = [
        Document(id=str(uuid.uuid4()), text=t, source=source)
        for t in texts
        if t.strip()
    ]
    if not documents:
        return 0

    embeddings = await _embedder.embed([d.text for d in documents])
    await _store.add(documents, embeddings)
    await _store.save()
    logger.info(f"Ingested {len(documents)} document(s) from '{source}'")
    return len(documents)


async def ingest_directory(directory: str) -> int:
    """
    Walk a directory and ingest all .txt and .md files.
    Each file is split into ~300-word chunks and indexed.
    """
    total = 0
    for path in sorted(Path(directory).rglob("*")):
        if path.suffix in {".txt", ".md"} and path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = _chunk_text(text, max_words=300)
            added = await ingest_texts(chunks, source=str(path))
            total += added
    logger.info(f"Ingested {total} chunk(s) from '{directory}'")
    return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_words: int = 300, overlap: int = 30) -> List[str]:
    """Split text into overlapping word-count windows."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_words - overlap  # Overlap for context continuity

    return chunks
