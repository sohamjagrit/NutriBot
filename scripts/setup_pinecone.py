#!/usr/bin/env python3
"""
Setup Pinecone index for NutriBot.

This script:
1. Connects to Pinecone using your API key
2. Creates the nutrition-index if it doesn't exist
3. Verifies the setup
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    print("ERROR: pinecone package not installed")
    print("Install with: pip install pinecone-client")
    sys.exit(1)


def setup_pinecone():
    """Setup Pinecone index."""
    config = get_config()

    logger.info("=" * 60)
    logger.info("Pinecone Setup for NutriBot")
    logger.info("=" * 60)

    # Get credentials
    api_key = config.pinecone_config.api_key
    env = config.pinecone_config.env
    index_name = config.pinecone_config.index_name

    if not api_key:
        logger.error("PINECONE_API_KEY not set in .env")
        logger.info("1. Get your API key from https://app.pinecone.io")
        logger.info("2. Add to .env: PINECONE_API_KEY=your_key")
        return False

    if not env:
        logger.error("PINECONE_ENV not set in .env")
        logger.info("1. Check your environment at https://app.pinecone.io")
        logger.info("2. Add to .env: PINECONE_ENV=us-east-1-aws (or your region)")
        return False

    logger.info(f"Using API Key: {api_key[:8]}...")
    logger.info(f"Using Environment: {env}")
    logger.info(f"Using Index Name: {index_name}")

    try:
        # Initialize Pinecone client
        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        logger.info("✓ Connected to Pinecone")

        # List existing indexes
        logger.info("Checking for existing indexes...")
        existing_indexes = pc.list_indexes()
        existing_names = [idx.name for idx in existing_indexes]
        logger.info(f"Existing indexes: {existing_names if existing_names else 'None'}")

        # Check if index already exists
        if index_name in existing_names:
            logger.info(f"✓ Index '{index_name}' already exists")

            # Get index stats
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            logger.info(f"  Dimension: {stats.dimension}")
            logger.info(f"  Total vectors: {stats.total_vector_count}")

            return True

        # Create index if it doesn't exist
        logger.info(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=768,  # BGE-base embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=env.replace("-aws", "")  # Extract region from env string
            )
        )

        logger.info(f"✓ Index '{index_name}' created successfully")
        logger.info("  Waiting for index to be ready...")

        # Wait for index to be ready (with timeout)
        import time
        max_wait = 60
        start = time.time()

        while time.time() - start < max_wait:
            try:
                index = pc.Index(index_name)
                stats = index.describe_index_stats()
                logger.info(f"✓ Index is ready")
                logger.info(f"  Dimension: {stats.dimension}")
                logger.info(f"  Total vectors: {stats.total_vector_count}")
                break
            except Exception as e:
                logger.debug(f"Index not ready yet: {e}")
                time.sleep(2)

        return True

    except Exception as e:
        logger.error(f"Failed to setup Pinecone: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Verify API key at https://app.pinecone.io")
        logger.error("2. Verify environment value (e.g., us-east-1-aws)")
        logger.error("3. Check your Pinecone free tier limits")
        return False


def verify_pinecone():
    """Verify Pinecone connection."""
    config = get_config()

    logger.info("=" * 60)
    logger.info("Verifying Pinecone Setup")
    logger.info("=" * 60)

    api_key = config.pinecone_config.api_key
    index_name = config.pinecone_config.index_name

    if not api_key:
        logger.error("✗ PINECONE_API_KEY not configured")
        return False

    try:
        pc = Pinecone(api_key=api_key)

        # Check if index exists
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]

        if index_name not in index_names:
            logger.error(f"✗ Index '{index_name}' not found")
            logger.info(f"  Available indexes: {index_names if index_names else 'None'}")
            return False

        logger.info(f"✓ Index '{index_name}' exists")

        # Get index stats
        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        logger.info(f"✓ Index details:")
        logger.info(f"  Dimension: {stats.dimension}")
        logger.info(f"  Total vectors: {stats.total_vector_count}")
        logger.info(f"  Namespace count: {len(stats.namespaces) if stats.namespaces else 0}")

        return True

    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        return False


def main():
    """Run setup and verification."""
    logger.info("Starting Pinecone setup...")

    # Setup
    if not setup_pinecone():
        logger.error("Setup failed")
        return False

    # Verify
    if not verify_pinecone():
        logger.error("Verification failed")
        return False

    logger.info("=" * 60)
    logger.info("✓ Pinecone setup complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Embed chunks & upload: python scripts/embed_and_upload.py")
    logger.info("2. Run the API: uvicorn app:app --reload")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
