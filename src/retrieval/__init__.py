"""Retrieval module for NutriBot."""

from .base import Retriever, RetrievalResult
from .semantic import SemanticRetriever
from .hybrid_pinecone import HybridPineconeRetriever

__all__ = [
    "Retriever",
    "RetrievalResult",
    "SemanticRetriever",
    "HybridPineconeRetriever",
]
