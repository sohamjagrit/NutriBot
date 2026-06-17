#!/usr/bin/env python3
"""
Embedding pipeline: S3 chunks → BGE-base embeddings → Pinecone.

Flow:
1. Load 1,345 parsed chunks from S3 (parsed/chunks/parsed_chunks.jsonl)
2. Embed with BAAI/bge-base-en-v1.5 in batches of 32
3. Upsert to Pinecone with text content stored in metadata
4. Verify upload
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_config
from src.embeddings import EmbedderFactory
from src.data_ingestion import S3Manager
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

PINECONE_METADATA_TEXT_KEY = "text"
PINECONE_BATCH_SIZE = 100  # vectors per upsert call


def load_chunks_from_s3(config) -> List[Dict[str, Any]]:
    """Load parsed chunks from S3."""
    s3 = S3Manager(
        bucket_name=config.s3_config.bucket_name,
        region=config.s3_config.region,
        access_key=config.s3_config.access_key,
        secret_key=config.s3_config.secret_key,
    )

    s3_key = f"{config.s3_config.parsed_prefix}/chunks/parsed_chunks.jsonl"
    logger.info(f"Loading chunks from s3://{config.s3_config.bucket_name}/{s3_key}")

    response = s3.s3_client.get_object(Bucket=s3.bucket_name, Key=s3_key)
    raw = response["Body"].read().decode("utf-8")

    chunks = [json.loads(line) for line in raw.splitlines() if line.strip()]
    logger.info(f"Loaded {len(chunks)} chunks from S3")
    return chunks


def embed_chunks(embedder, chunks: List[Dict[str, Any]]) -> list:
    """Embed all chunk texts, return list of float lists."""
    logger.info(f"Embedding {len(chunks)} chunks with {embedder.model_name}...")
    texts = [chunk["content"] for chunk in chunks]
    embeddings = embedder.embed_batch(texts, show_progress=True)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings.tolist()


def build_pinecone_vectors(
    chunks: List[Dict[str, Any]],
    embeddings: list,
) -> List[Dict]:
    """Build Pinecone upsert records with text content in metadata."""
    vectors = []
    for chunk, vector in zip(chunks, embeddings):
        metadata = {
            PINECONE_METADATA_TEXT_KEY: chunk["content"],
            "source_file": chunk.get("source_file", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "token_count": chunk.get("token_count", 0),
            "block_type": chunk.get("metadata", {}).get("block_type", "section"),
        }
        # Pinecone metadata values must be str/int/float/bool/list-of-str
        vectors.append({
            "id": chunk["chunk_id"],
            "values": vector,
            "metadata": metadata,
        })
    return vectors


def upsert_to_pinecone(pc_index, vectors: List[Dict], batch_size: int = PINECONE_BATCH_SIZE):
    """Upsert vectors to Pinecone in batches."""
    total = len(vectors)
    logger.info(f"Upserting {total} vectors to Pinecone (batch size {batch_size})...")

    for i in range(0, total, batch_size):
        batch = vectors[i: i + batch_size]
        pc_index.upsert(vectors=batch)
        logger.info(f"  Upserted {min(i + batch_size, total)}/{total}")

    logger.info("Upsert complete")


def verify(pc_index, embedder, sample_query: str = "What foods are high in protein?"):
    """Query Pinecone to confirm retrieval works."""
    logger.info(f"\nVerification query: '{sample_query}'")
    query_vec = embedder.embed_query(sample_query).tolist()
    results = pc_index.query(vector=query_vec, top_k=3, include_metadata=True)

    for i, match in enumerate(results["matches"], 1):
        text_preview = match["metadata"].get(PINECONE_METADATA_TEXT_KEY, "")[:120].replace("\n", " ")
        logger.info(f"  {i}. score={match['score']:.3f} | {text_preview}...")

    stats = pc_index.describe_index_stats()
    logger.info(f"\nFinal index stats: {stats.total_vector_count} vectors, dim={stats.dimension}")
    return stats.total_vector_count


def main():
    logger.info("=" * 60)
    logger.info("Embed & Upload Pipeline  (BGE-base → Pinecone)")
    logger.info("=" * 60)

    config = get_config()
    start = time.time()

    # ── 1. Load chunks ──────────────────────────────────────────
    logger.info("\n[1/4] Loading chunks from S3...")
    chunks = load_chunks_from_s3(config)
    if not chunks:
        logger.error("No chunks found. Run the Docling parsing job first.")
        return False

    # ── 2. Embed ────────────────────────────────────────────────
    logger.info("\n[2/4] Embedding chunks...")
    embedder = EmbedderFactory.create(config.embedding_config)
    logger.info(f"Model: {embedder.model_name}  |  Dim: {embedder.dimension}")
    embeddings = embed_chunks(embedder, chunks)

    # ── 3. Upsert to Pinecone ───────────────────────────────────
    logger.info("\n[3/4] Connecting to Pinecone and upserting...")
    try:
        from pinecone import Pinecone
    except ImportError:
        logger.error("pinecone package not installed. Run: pip install pinecone-client")
        return False

    if not config.pinecone_config.api_key:
        logger.error("PINECONE_API_KEY not set in .env")
        return False

    pc = Pinecone(api_key=config.pinecone_config.api_key)
    pc_index = pc.Index(config.pinecone_config.index_name)

    vectors = build_pinecone_vectors(chunks, embeddings)
    upsert_to_pinecone(pc_index, vectors)

    # ── 4. Verify ───────────────────────────────────────────────
    logger.info("\n[4/4] Verifying retrieval...")
    time.sleep(2)  # give Pinecone a moment to index
    count = verify(pc_index, embedder)

    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"Vectors in Pinecone: {count}")
    logger.info("Next: uvicorn app:app --reload")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
