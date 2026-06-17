FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock* .

# Install dependencies using uv (creates /app/.venv)
RUN uv sync --frozen --no-dev

# Put the venv on PATH so `python` and `uvicorn` resolve to the installed deps
# (uv installs into /app/.venv, not the system interpreter).
ENV PATH="/app/.venv/bin:$PATH"

# Copy project files
COPY app.py .
COPY .env.example .env
COPY config/ config/
COPY src/ src/
COPY static/ static/
# Note: no data/ copied — chunks load from S3 and vectors from Pinecone at runtime.

# Pre-download models so the container works offline.
# Cached at /root/.cache/huggingface/hub
#   - BGE-base: embeddings (queries + documents)
#   - ms-marco-MiniLM cross-encoder: reranking
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')" \
 && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
