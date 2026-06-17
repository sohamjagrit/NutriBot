"""Cross-encoder reranker.

A bi-encoder (BGE) embeds query and documents *separately*, so its similarity
score is only an approximation. A cross-encoder reads the query and each
candidate document *together* in a single forward pass, producing a far more
accurate relevance score. It is too slow to run over the whole corpus, so it
only re-scores the handful of candidates the retriever already shortlisted.
"""

from typing import List
import time

from config.settings import RerankerConfig
from src.retrieval.base import RetrievalResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Re-score retrieval candidates with a cross-encoder, return the best top_k."""

    def __init__(self, config: RerankerConfig = None):
        self.config = config or RerankerConfig()

        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder reranker: {self.config.model_name}")
        self.model = CrossEncoder(self.config.model_name, device=self.config.device)
        logger.info(f"Reranker loaded on {self.config.device}")

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Re-score candidates against the query and return the top_k.

        The original retriever score is preserved in metadata as
        `retriever_score`; `score` is overwritten with the cross-encoder score.
        """
        if not candidates:
            return []

        start = time.time()

        pairs = [(query, c.content) for c in candidates]
        scores = self.model.predict(pairs)  # higher = more relevant

        reranked = []
        for candidate, score in zip(candidates, scores):
            metadata = {**candidate.metadata, "retriever_score": candidate.score}
            reranked.append(RetrievalResult(
                content=candidate.content,
                score=float(score),
                metadata=metadata,
            ))

        reranked.sort(key=lambda r: r.score, reverse=True)
        top = reranked[:top_k]

        logger.info(
            f"Reranked {len(candidates)} → {len(top)} candidates in "
            f"{time.time() - start:.2f}s"
        )
        return top
