"""
RAG Embeddings

Converts text documents (and queries) into dense vector embeddings
using a sentence-transformer model.

The model is loaded lazily on first use and cached as a class variable
so it is only loaded once per process, even across multiple calls.
"""

import asyncio
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Lightweight model — ~90 MB, runs well on CPU, 768-dim embeddings
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingEngine:
    """
    Thin async wrapper around sentence-transformers.

    Usage:
        engine = EmbeddingEngine()
        vectors = await engine.embed(["text one", "text two"])
    """

    _model = None  # Shared model across instances

    @classmethod
    async def load(cls, model_name: str = DEFAULT_MODEL) -> None:
        if cls._model is not None:
            return
        loop = asyncio.get_event_loop()
        cls._model = await loop.run_in_executor(None, cls._load_model, model_name)
        logger.info(f"Embedding model loaded: {model_name}")

    @staticmethod
    def _load_model(model_name: str):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)

    async def embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into a 2-D float32 numpy array.
        Shape: (len(texts), embedding_dim)
        """
        if self._model is None:
            await self.load()

        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, convert_to_numpy=True)
        )
        # Normalise to unit vectors for cosine similarity via dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (vectors / norms).astype(np.float32)

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns 1-D float32 array."""
        result = await self.embed([text])
        return result[0]
