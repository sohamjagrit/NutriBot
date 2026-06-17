"""Factory for creating embedder instances."""

from typing import Optional
from config.settings import EmbeddingConfig
from src.utils.logging_config import get_logger
from .base import Embedder
from .bge import BGEEmbedder

logger = get_logger(__name__)


class EmbedderFactory:
    """Factory for creating embedder instances."""

    _EMBEDDERS = {
        "bge": BGEEmbedder,
    }

    @classmethod
    def create(cls, config: Optional[EmbeddingConfig] = None) -> Embedder:
        """Create embedder from config.

        Args:
            config: Embedding configuration

        Returns:
            Embedder instance

        Raises:
            ValueError: If embedder type is not supported
        """
        if config is None:
            config = EmbeddingConfig()

        embedder_type = config.model_type.lower()

        if embedder_type not in cls._EMBEDDERS:
            raise ValueError(
                f"Unknown embedder type: {embedder_type}. "
                f"Supported types: {list(cls._EMBEDDERS.keys())}"
            )

        embedder_class = cls._EMBEDDERS[embedder_type]
        logger.info(f"Creating {embedder_type} embedder")

        return embedder_class(
            model_name=config.model_name,
            device=config.device
        )

    @classmethod
    def register(cls, name: str, embedder_class: type) -> None:
        """Register a new embedder type.

        Args:
            name: Name of the embedder
            embedder_class: Embedder class to register
        """
        cls._EMBEDDERS[name.lower()] = embedder_class
        logger.info(f"Registered embedder type: {name}")
