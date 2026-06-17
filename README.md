# NutriBot — Nutrition RAG Chatbot

A production-minded Retrieval-Augmented Generation (RAG) chatbot that answers
nutrition questions grounded in a parsed nutrition textbook. Built on
BGE embeddings + Pinecone + a cross-encoder reranker + Groq, served over FastAPI.

## How it works

```
                        OFFLINE  (run when the knowledge base changes)
  PDF ──Docling parse──▶ semantic chunks ──BGE-base embed──▶ Pinecone (768-dim)

                        ONLINE  (every user query)
  question
     │  BGE-base embed (query prefix)
     ▼
  Hybrid retrieval ── BM25 (local)  +  Pinecone semantic ──▶ fused with RRF  (top 20)
     │
     ▼
  Cross-encoder rerank (ms-marco-MiniLM-L-6-v2)  ──▶ top 5
     │
     ▼
  Augment prompt with the 5 chunks  ──▶  Groq (llama-3.3-70b-versatile)  ──▶  answer
```

- **Embeddings:** `BAAI/bge-base-en-v1.5` (768 dims), runs locally / baked into the image.
- **Vector DB:** Pinecone (`nutrition-index`, cosine, 768 dims).
- **Retrieval:** hybrid BM25 + semantic, merged with Reciprocal Rank Fusion.
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the top candidates.
- **LLM:** Groq `llama-3.3-70b-versatile`.
- **API:** FastAPI.

## Project layout

```
app.py                          FastAPI app (lifespan loads models once)
config/settings.py              All configuration (dataclasses)
scripts/
  setup_pinecone.py             Create the Pinecone index (768 dims)
  embed_and_upload.py           S3 chunks → BGE embed → upsert to Pinecone
src/
  rag_pipeline.py               Orchestrator: retrieve → rerank → generate
  embeddings/                   BGE embedder + factory
  retrieval/                    semantic (Pinecone) + hybrid (BM25+Pinecone, RRF)
  reranking/                    cross-encoder reranker
  prompting/                    system prompts + RAG prompt builder
  data_ingestion/               Docling parser, semantic chunker, S3, chunk loader
  utils/                        logging, metrics, text helpers
```

## Setup

1. **Install dependencies** (uv):
   ```bash
   uv sync
   ```

2. **Configure secrets** — copy `.env.example` to `.env` and fill in:
   ```
   GROQ_API_KEY=...
   PINECONE_API_KEY=...
   PINECONE_ENV=us-east-1-aws
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   AWS_REGION=us-east-1
   S3_BUCKET_NAME=nutrition-usda-foods
   ```

## Build the knowledge base (offline, one-time per data change)

```bash
# 1. Create the Pinecone index (768 dims, cosine)
uv run python scripts/setup_pinecone.py

# 2. Embed parsed chunks from S3 and upload to Pinecone
uv run python scripts/embed_and_upload.py
```

(Parsing raw PDFs into chunks is handled by `src/data_ingestion/` via Docling;
chunks are expected at `s3://<bucket>/parsed/chunks/parsed_chunks.jsonl`.)

## Run the API

```bash
uv run uvicorn app:app --reload
# open http://localhost:8000/docs
```

> `uvicorn: command not found` means the venv isn't active. Use `uv run uvicorn ...`
> or `source .venv/bin/activate` first.

### Endpoints

| Method | Path        | Purpose                                  |
|--------|-------------|------------------------------------------|
| GET    | `/health`   | Liveness (used by load-balancer checks)  |
| POST   | `/chat`     | Ask a question → grounded answer + sources |
| GET    | `/metrics`  | Aggregated latency / retrieval stats     |
| POST   | `/feedback` | Submit a 1–5 rating on an answer         |
| GET    | `/docs`     | Interactive Swagger UI                    |

Example:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How does vitamin C help with iron absorption?"}'
```

## Docker

The image bakes in both models (BGE-base + the cross-encoder) so the container
runs without downloading anything at startup.

```bash
docker build -t nutrition-chatbot:v1 .
docker run -p 8000:8000 --env-file .env nutrition-chatbot:v1
```

## Configuration knobs (`config/settings.py`)

- `RetrieverConfig.top_k` — chunks passed to the LLM (default 5)
- `RetrieverConfig.use_reranking` / `rerank_candidates` — reranking on/off and width (default on / 20)
- `RetrieverConfig.use_hybrid_search` — hybrid vs. pure semantic (default hybrid)
- `RerankerConfig.model_name` — cross-encoder model
- `EmbeddingConfig` / `GroqConfig` — embedder and LLM settings

See `ROADMAP.md` for the full deployment plan (Docker → AWS EC2 → ALB → auto-scaling).
