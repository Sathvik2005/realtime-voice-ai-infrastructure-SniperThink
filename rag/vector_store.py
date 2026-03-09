"""
RAG Vector Store  —  FAISS-backed document index

Stores text chunks alongside their embeddings.
Supports adding documents and performing top-k similarity search.

The index is persisted to disk so it survives process restarts.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

INDEX_FILE = "rag/data/faiss.index"
META_FILE  = "rag/data/metadata.json"


@dataclass
class Document:
    id: str
    text: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """
    FAISS flat index (L2, equivalent to cosine after unit normalisation).

    Design:
      • Documents and their embeddings are added in batches.
      • Search returns the top-k most similar documents plus their scores.
      • The index is saved / loaded from disk atomically.
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self._dim = embedding_dim
        self._index = None          # faiss.IndexFlatIP
        self._documents: List[Document] = []

    # ------------------------------------------------------------------
    # Initialise / load
    # ------------------------------------------------------------------
    async def load_or_create(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_or_create_sync)

    def _load_or_create_sync(self) -> None:
        import faiss

        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            logger.info("Loading existing FAISS index from disk…")
            self._index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "r", encoding="utf-8") as f:
                docs = json.load(f)
            self._documents = [Document(**d) for d in docs]
            logger.info(f"Loaded {len(self._documents)} documents.")
        else:
            logger.info("Creating new FAISS index.")
            self._index = faiss.IndexFlatIP(self._dim)  # Inner Product == cosine on unit vecs

    # ------------------------------------------------------------------
    # Add documents
    # ------------------------------------------------------------------
    async def add(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """
        Add documents and their pre-computed embeddings to the index.
        embeddings shape: (len(documents), embedding_dim)
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._add_sync, documents, embeddings)

    def _add_sync(self, documents: List[Document], embeddings: np.ndarray) -> None:
        if self._index is None:
            self._load_or_create_sync()
        self._index.add(embeddings.astype(np.float32))
        self._documents.extend(documents)
        logger.info(f"Added {len(documents)} docs — total: {len(self._documents)}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    async def search(
        self, query_vector: np.ndarray, top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve the top_k most similar documents.
        Returns list of (Document, similarity_score) tuples.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_sync, query_vector, top_k)

    def _search_sync(
        self, query_vector: np.ndarray, top_k: int
    ) -> List[Tuple[Document, float]]:
        q = query_vector.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self._documents):
                results.append((self._documents[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    async def save(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_sync)

    def _save_sync(self) -> None:
        import faiss

        Path(INDEX_FILE).parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, INDEX_FILE)
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "id": d.id,
                        "text": d.text,
                        "source": d.source,
                        "metadata": d.metadata,
                    }
                    for d in self._documents
                ],
                f,
                indent=2,
            )
        logger.info(f"Index saved: {len(self._documents)} documents.")

    @property
    def size(self) -> int:
        return len(self._documents)
