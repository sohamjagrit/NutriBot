"""Base embedder class."""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class Embedder(ABC):
    """Abstract base class for embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model."""
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector of shape (dimension,)
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Embed multiple text strings efficiently.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            Embedding matrix of shape (len(texts), dimension)
        """
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query (can be different from embed for some models).

        Args:
            query: Query text to embed

        Returns:
            Query embedding vector
        """
        return self.embed(query)
