"""Hybrid retrieval: BM25 (local) + Pinecone (semantic) fused with RRF."""

from typing import List, Dict, Any
import time
import numpy as np
from rank_bm25 import BM25Okapi

from src.embeddings import Embedder
from config.settings import PineconeConfig, RetrieverConfig
from src.utils.logging_config import get_logger
from .base import Retriever, RetrievalResult
from .semantic import SemanticRetriever

logger = get_logger(__name__)

RRF_K = 60  # standard constant — dampens how much top ranks dominate the fusion


class HybridPineconeRetriever(Retriever):
    """Hybrid retrieval: BM25 keyword search + Pinecone semantic search, merged via RRF.

    BM25 runs entirely in memory on chunk texts (no embeddings needed).
    Pinecone handles semantic ANN search.
    Reciprocal Rank Fusion merges the two ranked lists by rank position, so the
    two different score scales (BM25 term-frequency vs. cosine similarity) never
    have to be normalized against each other.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embedder: Embedder,
        pinecone_config: PineconeConfig,
        retriever_config: RetrieverConfig,
    ):
        """
        Args:
            chunks: All parsed chunks (loaded from S3 or local) — used for the BM25 index
            embedder: BGE embedder for query embedding (used by the semantic side)
            pinecone_config: Pinecone connection config
            retriever_config: top_k, weights, threshold
        """
        self.chunks = chunks
        self.retriever_config = retriever_config

        # chunk_id → chunk dict, for content lookup on BM25-only hits
        self._by_id: Dict[str, Dict] = {c["chunk_id"]: c for c in chunks}

        # BM25 index — built on lowercased whitespace tokens, fast in-memory
        logger.info(f"Building BM25 index on {len(chunks)} chunks...")
        tokenized = [c["content"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index ready")

        # Pinecone semantic retriever
        self.semantic = SemanticRetriever(embedder, pinecone_config, retriever_config)

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """Run BM25 + Pinecone and merge with RRF.

        Fetches top_k * 3 from each source so the union has enough candidates,
        then RRF re-ranks and the final top_k are returned. No score-threshold
        filtering: RRF ranks by position, not absolute relevance, so a strong
        match found by only one retriever is kept rather than discarded.
        """
        if top_k is None:
            top_k = self.retriever_config.top_k

        fetch_k = top_k * 3  # cast a wider net before RRF collapses it back to top_k
        start = time.time()

        # ── BM25: chunk_id → rank (0-based) ─────────────────────────────────
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:fetch_k]
        bm25_ranks: Dict[str, int] = {
            self.chunks[idx]["chunk_id"]: rank for rank, idx in enumerate(bm25_top)
        }

        # ── Pinecone semantic: chunk_id → rank, plus the result objects ─────
        semantic_results = self.semantic.retrieve(query, top_k=fetch_k)
        semantic_ranks: Dict[str, int] = {}
        semantic_data: Dict[str, RetrievalResult] = {}
        for rank, result in enumerate(semantic_results):
            chunk_id = result.metadata.get("chunk_id")
            if chunk_id is None:
                continue
            semantic_ranks[chunk_id] = rank
            semantic_data[chunk_id] = result

        # ── RRF fusion over the union of candidate ids ──────────────────────
        all_ids = set(bm25_ranks) | set(semantic_ranks)
        rrf_scores: Dict[str, float] = {}
        for chunk_id in all_ids:
            score = 0.0
            if chunk_id in bm25_ranks:
                score += 1.0 / (RRF_K + bm25_ranks[chunk_id])
            if chunk_id in semantic_ranks:
                score += 1.0 / (RRF_K + semantic_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # ── Build results (prefer Pinecone content; fall back to local) ─────
        results = []
        for chunk_id, rrf_score in ranked:
            if chunk_id in semantic_data:
                sem = semantic_data[chunk_id]
                content = sem.content
                metadata = {**sem.metadata, "rrf_score": rrf_score,
                            "in_bm25": chunk_id in bm25_ranks,
                            "in_semantic": True}
            elif chunk_id in self._by_id:
                chunk = self._by_id[chunk_id]
                content = chunk["content"]
                metadata = {
                    "chunk_id": chunk_id,
                    "source_file": chunk.get("source_file", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "rrf_score": rrf_score,
                    "in_bm25": True,
                    "in_semantic": False,
                }
            else:
                continue

            results.append(RetrievalResult(content=content, score=rrf_score, metadata=metadata))

        elapsed = time.time() - start
        logger.info(
            f"Hybrid retrieval: {len(results)} results in {elapsed:.2f}s "
            f"(BM25 candidates: {len(bm25_ranks)}, Pinecone candidates: {len(semantic_ranks)}, "
            f"union: {len(all_ids)})"
        )
        return results
