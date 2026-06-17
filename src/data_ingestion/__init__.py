"""Data ingestion module for knowledge base setup."""

from .s3_utils import S3Manager
from .chunk_loader import load_chunks

__all__ = ["S3Manager", "load_chunks"]
