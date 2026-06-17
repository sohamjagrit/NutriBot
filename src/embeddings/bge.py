"""BGE embedder (better quality, recommended)."""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.logging_config import get_logger
from .base import Embedder

logger = get_logger(__name__)


class BGEEmbedder(Embedder):
    """BGE (Beijing General Embeddings) - high quality embeddings.

    - Model: BAAI/bge-base-en-v1.5
    - Dimension: 768
    - Speed: Fast
    - Quality: Strong retrieval performance
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cpu"):
        """Initialize BGE embedder.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use (cpu, cuda, mps)
        """
        logger.info(f"Loading BGE embedder from {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._device = device
        logger.info(f"BGE embedder loaded successfully on {device}")

    @property
    def dimension(self) -> int:
        """Embedding dimension (768 for BGE-base)."""
        return 768

    @property
    def model_name(self) -> str:
        """Model name."""
        return self._model_name

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector of shape (1024,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            Embedding matrix of shape (len(texts), 1024)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with instruction prefix (BGE-specific).

        Args:
            query: Query text to embed

        Returns:
            Query embedding vector
        """
        # BGE recommends using instruction prefix for queries
        instruction = "Represent this sentence for searching relevant passages: "
        embedding = self.model.encode(
            instruction + query,
            convert_to_numpy=True
        )
        return embedding
