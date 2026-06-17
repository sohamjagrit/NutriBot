"""Metrics collection and tracking for RAG pipeline."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    retrieval_latency: float = 0.0
    llm_latency: float = 0.0
    total_latency: float = 0.0
    num_retrieved_docs: int = 0
    similarity_scores: List[float] = field(default_factory=list)
    response_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Collect and analyze metrics for RAG pipeline."""

    def __init__(self):
        self.queries: List[QueryMetrics] = []

    def record_query(self, metrics: QueryMetrics) -> None:
        """Record metrics for a query."""
        self.queries.append(metrics)

    def average_retrieval_latency(self) -> float:
        """Get average retrieval latency in milliseconds."""
        if not self.queries:
            return 0.0
        return sum(q.retrieval_latency for q in self.queries) / len(self.queries) * 1000

    def average_llm_latency(self) -> float:
        """Get average LLM latency in milliseconds."""
        if not self.queries:
            return 0.0
        return sum(q.llm_latency for q in self.queries) / len(self.queries) * 1000

    def average_total_latency(self) -> float:
        """Get average total latency in milliseconds."""
        if not self.queries:
            return 0.0
        return sum(q.total_latency for q in self.queries) / len(self.queries) * 1000

    def average_similarity_score(self) -> float:
        """Get average similarity score of retrieved documents."""
        all_scores = []
        for q in self.queries:
            all_scores.extend(q.similarity_scores)
        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "total_queries": len(self.queries),
            "avg_retrieval_latency_ms": round(self.average_retrieval_latency(), 2),
            "avg_llm_latency_ms": round(self.average_llm_latency(), 2),
            "avg_total_latency_ms": round(self.average_total_latency(), 2),
            "avg_similarity_score": round(self.average_similarity_score(), 3),
            "avg_docs_retrieved": round(sum(q.num_retrieved_docs for q in self.queries) / len(self.queries) if self.queries else 0, 2),
        }
