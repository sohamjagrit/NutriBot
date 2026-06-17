"""Embeddings module for NutriBot."""

from .base import Embedder
from .bge import BGEEmbedder
from .factory import EmbedderFactory

__all__ = ["Embedder", "BGEEmbedder", "EmbedderFactory"]
