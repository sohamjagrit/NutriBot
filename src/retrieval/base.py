"""Base retriever class and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    content: str
    score: float
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Retriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of retrieval results
        """
        pass
