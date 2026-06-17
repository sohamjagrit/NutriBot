"""Main RAG pipeline orchestrator."""

import time
from typing import Optional, List
from config.settings import get_config, RAGConfig
from src.embeddings import EmbedderFactory, Embedder
from src.retrieval import (
    SemanticRetriever,
    HybridPineconeRetriever,
    RetrievalResult,
)
from src.data_ingestion import load_chunks
from src.prompting import PromptManager
from src.utils.logging_config import get_logger
from src.utils.metrics import MetricsCollector, QueryMetrics

logger = get_logger(__name__)

# Returned verbatim when retrieval finds nothing relevant (off-topic / out-of-context).
REFUSAL_MESSAGE = (
    "I'm sorry, that's outside what I can help with. I can only answer nutrition "
    "questions using my knowledge base, and I couldn't find relevant information "
    "for this. Try asking about foods, nutrients, vitamins, or diet."
)


class NutritionRAG:
    """Complete RAG pipeline for nutrition chatbot."""

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize RAG pipeline.

        Args:
            config: RAG configuration
        """
        self.config = config or get_config()
        self.metrics = MetricsCollector()

        logger.info("Initializing NutritionRAG pipeline")

        # Initialize embedder
        self.embedder = EmbedderFactory.create(self.config.embedding_config)
        logger.info(f"Embedder initialized: {self.embedder.model_name} ({self.embedder.dimension}d)")

        # Initialize retriever (Pinecone-backed)
        self.retriever = self._init_retriever()

        # Initialize reranker (optional cross-encoder second stage)
        self.reranker = self._init_reranker()

        # Initialize LLM
        self._init_llm()
        logger.info("RAG pipeline initialized successfully")

    def _init_retriever(self):
        """Build the retriever based on config.

        Hybrid (default): BM25 over all chunks + Pinecone semantic, fused with RRF.
        Falls back to pure Pinecone semantic if hybrid is disabled or chunks
        can't be loaded for the BM25 index.
        """
        if self.config.retriever_config.use_hybrid_search:
            chunks = load_chunks(self.config)
            if chunks:
                logger.info(f"Using hybrid retriever (BM25 + Pinecone) over {len(chunks)} chunks")
                return HybridPineconeRetriever(
                    chunks,
                    self.embedder,
                    self.config.pinecone_config,
                    self.config.retriever_config,
                )
            logger.warning("Hybrid requested but no chunks available; using semantic-only")

        logger.info("Using semantic retriever (Pinecone)")
        return SemanticRetriever(
            self.embedder,
            self.config.pinecone_config,
            self.config.retriever_config,
        )

    def _init_reranker(self):
        """Build the cross-encoder reranker if enabled, else None."""
        if not self.config.retriever_config.use_reranking:
            logger.info("Reranking disabled")
            return None
        try:
            from src.reranking import CrossEncoderReranker
            reranker = CrossEncoderReranker(self.config.reranker_config)
            logger.info("Reranker initialized")
            return reranker
        except Exception as e:
            logger.warning(f"Failed to initialize reranker, continuing without it: {e}")
            return None

    def _init_llm(self) -> None:
        """Initialize LLM based on configuration."""
        if self.config.llm_config.provider == "groq":
            try:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(
                    model_name=self.config.groq_config.model_name,
                    api_key=self.config.groq_config.api_key,
                    temperature=self.config.groq_config.temperature,
                )
                logger.info(f"Groq LLM initialized: {self.config.groq_config.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq LLM: {e}")
                self.llm = None
        else:
            logger.warning(f"Unknown LLM provider: {self.config.llm_config.provider}")
            self.llm = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieve documents for a query, with optional cross-encoder reranking.

        When reranking is on: fetch `rerank_candidates` from the retriever, then
        the cross-encoder re-scores them and the best `top_k` are returned.

        Args:
            query: Query text
            top_k: Number of documents to return (defaults to config.top_k)

        Returns:
            List of retrieval results
        """
        if self.retriever is None:
            logger.error("Retriever not initialized")
            return []

        final_k = top_k if top_k is not None else self.config.retriever_config.top_k

        if self.reranker is not None:
            # Cast a wider net, then let the cross-encoder pick the best.
            fetch_k = max(self.config.retriever_config.rerank_candidates, final_k)
            candidates = self.retriever.retrieve(query, top_k=fetch_k)
            return self.reranker.rerank(query, candidates, top_k=final_k)

        return self.retriever.retrieve(query, top_k=final_k)

    def _is_out_of_context(self, retrieved: List[RetrievalResult]) -> bool:
        """Decide if retrieval surfaced nothing relevant enough to answer from.

        Returns True when there are no results, or (when reranking is on) the
        best candidate scores below the configured relevance threshold.
        """
        if not retrieved:
            return True
        if self.reranker is not None:
            return retrieved[0].score < self.config.reranker_config.min_relevance_score
        return False

    def generate_response(self, query: str, context: str, prompt_type: str = "standard") -> str:
        """Generate response using LLM.

        Args:
            query: User query
            context: Retrieved context
            prompt_type: Type of system prompt

        Returns:
            Generated response
        """
        if self.llm is None:
            logger.error("LLM not initialized")
            return "LLM not available"

        # Build prompt
        system_prompt = PromptManager.get_system_prompt(prompt_type)
        rag_prompt = PromptManager.build_rag_prompt(query, context, prompt_type)

        # Generate response
        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=rag_prompt),
            ]
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        prompt_type: str = "standard"
    ) -> dict:
        """Complete RAG query: retrieve documents and generate response.

        Args:
            question: User question
            top_k: Number of documents to retrieve
            prompt_type: Type of system prompt

        Returns:
            Dictionary with question, context, and response
        """
        query_start = time.time()

        # Retrieve
        retrieve_start = time.time()
        retrieved = self.retrieve(question, top_k)
        retrieve_time = time.time() - retrieve_start

        # Prepare context
        context_parts = []
        scores = []
        for i, result in enumerate(retrieved):
            context_parts.append(f"[Source {i+1}] {result.content}")
            scores.append(result.score)

        context = "\n\n".join(context_parts)

        # Relevance gate: if nothing relevant was retrieved, refuse without
        # calling the LLM. Only applied when reranking is on, since its scores
        # are calibrated logits (off-topic queries score strongly negative).
        if self._is_out_of_context(retrieved):
            top_score = retrieved[0].score if retrieved else None
            logger.info(f"Out-of-context query refused (top_score={top_score})")
            response = REFUSAL_MESSAGE
            llm_time = 0.0
        else:
            # Generate response
            llm_start = time.time()
            response = self.generate_response(question, context, prompt_type)
            llm_time = time.time() - llm_start

        total_time = time.time() - query_start

        # Record metrics
        metrics = QueryMetrics(
            query=question,
            retrieval_latency=retrieve_time,
            llm_latency=llm_time,
            total_latency=total_time,
            num_retrieved_docs=len(retrieved),
            similarity_scores=scores,
        )
        self.metrics.record_query(metrics)

        logger.info(f"Query completed in {total_time:.2f}s (retrieval: {retrieve_time:.2f}s, LLM: {llm_time:.2f}s)")

        return {
            "question": question,
            "retrieved_docs": [r.content for r in retrieved],
            "scores": scores,
            "context": context,
            "response": response,
            "metrics": {
                "retrieval_latency_s": retrieve_time,
                "llm_latency_s": llm_time,
                "total_latency_s": total_time,
            }
        }

    def get_metrics_summary(self) -> dict:
        """Get metrics summary for all queries.

        Returns:
            Dictionary of aggregated metrics
        """
        return self.metrics.get_summary()
