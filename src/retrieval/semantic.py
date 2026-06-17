"""Semantic retrieval using Pinecone vector database."""

from typing import List
import time
from src.embeddings import Embedder
from config.settings import PineconeConfig, RetrieverConfig
from src.utils.logging_config import get_logger
from .base import Retriever, RetrievalResult

logger = get_logger(__name__)

PINECONE_METADATA_TEXT_KEY = "text"


class SemanticRetriever(Retriever):
    """Semantic retrieval using Pinecone vector database."""

    def __init__(self, embedder: Embedder, config: PineconeConfig, retriever_config: RetrieverConfig):
        self.embedder = embedder
        self.config = config
        self.retriever_config = retriever_config
        self._index = None

        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=config.api_key)
            self._index = pc.Index(config.index_name)
            stats = self._index.describe_index_stats()
            logger.info(
                f"Pinecone index '{config.index_name}' ready: "
                f"{stats.total_vector_count} vectors, dim={stats.dimension}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        if top_k is None:
            top_k = self.retriever_config.top_k

        if self._index is None:
            logger.warning("Pinecone index not initialized, returning empty results")
            return []

        start = time.time()
        try:
            query_vec = self.embedder.embed_query(query).tolist()
            response = self._index.query(
                vector=query_vec,
                top_k=top_k,
                include_metadata=True,
            )

            results = []
            for match in response["matches"]:
                text = match["metadata"].get(PINECONE_METADATA_TEXT_KEY, "")
                metadata = {k: v for k, v in match["metadata"].items() if k != PINECONE_METADATA_TEXT_KEY}
                # Expose the Pinecone vector id (== chunk_id) so callers like
                # the hybrid retriever can fuse this list with BM25 by id.
                metadata["chunk_id"] = match["id"]
                results.append(RetrievalResult(
                    content=text,
                    score=match["score"],
                    metadata=metadata,
                ))

            logger.info(f"Retrieved {len(results)} docs in {time.time() - start:.2f}s")
            return results

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
