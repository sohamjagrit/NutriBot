"""Configuration management for NutriBot RAG pipeline."""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "BAAI/bge-base-en-v1.5"
    model_type: str = "bge"
    dimension: int = 768
    normalize_embeddings: bool = True
    batch_size: int = 32
    device: str = "cpu"  # cpu, cuda, mps


@dataclass
class RetrieverConfig:
    """Retrieval configuration."""
    top_k: int = 5
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    similarity_threshold: float = 0.3
    use_hybrid_search: bool = True
    use_reranking: bool = True
    # When reranking, fetch this many candidates from the retriever, then the
    # cross-encoder re-scores them and the best `top_k` are kept.
    rerank_candidates: int = 20


@dataclass
class RerankerConfig:
    """Cross-encoder reranker configuration."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"  # cpu, cuda, mps
    # Relevance gate: if the best reranked candidate scores below this, the
    # query is treated as out-of-context and refused without calling the LLM.
    # ms-marco scores are logits — relevant matches are positive/high, clearly
    # off-topic queries score strongly negative.
    min_relevance_score: float = -4.0


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "groq"  # groq, local, openai
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9


@dataclass
class PineconeConfig:
    """Pinecone vector database configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    env: str = field(default_factory=lambda: os.getenv("PINECONE_ENV", ""))
    index_name: str = "nutrition-index"
    metric: str = "cosine"  # cosine, euclidean, dotproduct


@dataclass
class GroqConfig:
    """Groq LLM configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class HuggingFaceConfig:
    """HuggingFace configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("HUGGINGFACE_API_KEY", ""))
    cache_dir: str = ".cache/huggingface"


@dataclass
class S3Config:
    """AWS S3 configuration."""
    bucket_name: str = field(default_factory=lambda: os.getenv("S3_BUCKET_NAME", "nutrition-data"))
    region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    raw_prefix: str = "raw/pdf's"
    parsed_prefix: str = "parsed"
    access_key: str = field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", ""))


@dataclass
class RAGConfig:
    """Main RAG pipeline configuration."""
    # Components
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever_config: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker_config: RerankerConfig = field(default_factory=RerankerConfig)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    pinecone_config: PineconeConfig = field(default_factory=PineconeConfig)
    groq_config: GroqConfig = field(default_factory=GroqConfig)
    huggingface_config: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    s3_config: S3Config = field(default_factory=S3Config)

    # Paths
    data_dir: str = "data"
    log_dir: str = "logs"
    cache_dir: str = ".cache"

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self):
        """Validate configuration after initialization."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)


def get_config() -> RAGConfig:
    """Get RAG configuration (singleton pattern)."""
    return RAGConfig()
