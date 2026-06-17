"""Shared loader for parsed chunks (S3 with local fallback).

Used by both the embedding pipeline (scripts/embed_and_upload.py) and the
RAG pipeline's hybrid retriever, which needs all chunks in memory for BM25.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from config.settings import RAGConfig
from src.data_ingestion.s3_utils import S3Manager
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

LOCAL_CHUNKS_PATH = "data/processed/parsed_chunks.jsonl"


def _parse_jsonl(raw: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_chunks(config: RAGConfig) -> List[Dict[str, Any]]:
    """Load parsed chunks, preferring S3 and falling back to the local file.

    Returns an empty list if neither source is available.
    """
    # ── S3 ──────────────────────────────────────────────────────────────
    try:
        s3 = S3Manager(
            bucket_name=config.s3_config.bucket_name,
            region=config.s3_config.region,
            access_key=config.s3_config.access_key,
            secret_key=config.s3_config.secret_key,
        )
        s3_key = f"{config.s3_config.parsed_prefix}/chunks/parsed_chunks.jsonl"
        response = s3.s3_client.get_object(Bucket=s3.bucket_name, Key=s3_key)
        chunks = _parse_jsonl(response["Body"].read().decode("utf-8"))
        if chunks:
            logger.info(f"Loaded {len(chunks)} chunks from s3://{s3.bucket_name}/{s3_key}")
            return chunks
        logger.warning("S3 returned no chunks, falling back to local file")
    except Exception as e:
        logger.warning(f"S3 chunk load failed ({e}); falling back to local file")

    # ── Local fallback ──────────────────────────────────────────────────
    local = Path(LOCAL_CHUNKS_PATH)
    if local.exists():
        chunks = _parse_jsonl(local.read_text())
        logger.info(f"Loaded {len(chunks)} chunks from {local}")
        return chunks

    logger.error("No chunks available from S3 or local file")
    return []
