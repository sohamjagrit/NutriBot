# NutriBot — Nutrition RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers nutrition questions
grounded in a parsed nutrition textbook. It retrieves relevant passages, reranks
them, and has the LLM answer **only** from that context — refusing off-topic
questions and citing its sources.

```
                        OFFLINE  (run when the knowledge base changes)
  PDF ──Docling parse──▶ semantic chunks ──BGE-base embed──▶ Pinecone (768-dim)

                        ONLINE  (every user query)
  question
     │  BGE-base embed (query prefix)
     ▼
  Hybrid retrieval ── BM25 (local) + Pinecone semantic ──▶ fused with RRF  (top 20)
     │
     ▼
  Cross-encoder rerank (ms-marco-MiniLM-L-6-v2)  ──▶ top 5
     │
     ▼
  Relevance gate ──▶ if nothing relevant, refuse (skip the LLM)
     │
     ▼
  Augment prompt with the 5 chunks  ──▶  Groq (llama-3.3-70b)  ──▶  cited answer
```

---

## Design decisions

The interesting part of this project is *why* each piece was chosen. Here's the
reasoning behind the architecture.

### Embeddings — BGE-base (768-d), run locally
`BAAI/bge-base-en-v1.5` sits at the quality/cost sweet spot: it matches or beats
older 1536-d API embeddings on retrieval benchmarks while being small enough
(~440 MB) to run on CPU and bake into the container. We run it ourselves rather
than calling an embedding API because embedding is a batch job — there's no
reason to pay per-token or add a network dependency for something a CPU does in
milliseconds. The dimension (768) is a fixed property of the model, and the
Pinecone index must match it exactly.

> BGE distinguishes *queries* from *documents*: queries get an instruction
> prefix (`embed_query`), documents don't. Skipping the prefix measurably hurts
> retrieval, so it's handled in the embedder.

### Vector store — Pinecone
A managed vector DB removes the operational burden of running and scaling our
own index. We store the **chunk text inside the vector metadata**, so a single
query returns both the match and its content — no second lookup. The vector id
*is* the chunk id, which lets the hybrid retriever fuse Pinecone results with
BM25 results by id.

### Retrieval — hybrid (BM25 + semantic), fused with RRF
Semantic search alone misses exact terms (specific nutrients, codes, names);
keyword search alone misses meaning ("ascorbic acid" vs "vitamin C"). Running
**both** and merging covers each one's blind spot. We merge with **Reciprocal
Rank Fusion** (`score = Σ 1/(60 + rank)`) rather than a weighted score sum,
because RRF combines by *rank position* — so BM25's term-frequency scale and
cosine similarity never have to be normalized against each other. BM25 runs
in-memory over the chunks; Pinecone handles the semantic side.

### Reranking — cross-encoder second stage
A bi-encoder (BGE) embeds the query and each document *separately*, so its
similarity is only an approximation. A **cross-encoder** reads the query and a
candidate *together* in one pass, producing a far more accurate relevance score.
It's too slow to run over the whole corpus, so it only re-scores the ~20
candidates retrieval already shortlisted, then keeps the best 5. We use
`ms-marco-MiniLM-L-6-v2` (~80 MB, CPU-fast) over a heavier reranker because this
runs on *every* query and latency matters more than a marginal quality gain.

### Grounding — gate + strict prompt + citations
Two independent layers keep answers honest:
1. **Relevance gate** — if the best reranked score falls below a threshold, the
   query is treated as out-of-context and refused *without calling the LLM*
   (cheap, deterministic). Cross-encoder scores are calibrated logits, so
   off-topic queries score strongly negative — a clean signal.
2. **Strict prompt** — the system prompt forbids outside knowledge, requires
   inline `[Source N]` citations, and mandates a refusal sentence when the
   context doesn't answer the question.

Citations + the returned source chunks make every answer auditable: you can
trace a claim back to the passage it came from.

### LLM — Groq (hosted)
Generation is delegated to a hosted model (`llama-3.3-70b`) for fast inference
without managing GPUs. The LLM is the one component we *don't* self-host,
because token cost is tiny at this scale and hosted latency is excellent.

### Serving — FastAPI, models loaded once
Models (embedder, reranker) and the BM25 index are expensive to build, so they
load **once at startup** (FastAPI lifespan) and are reused across requests —
never per-request. A self-contained static page is served from the same app, so
there's no separate frontend build or dependency.

### Offline vs online separation
Parsing and embedding are a **batch job** run only when the knowledge base
changes; serving is a **long-running process**. They share code but run on
different schedules, which is why ingestion deps (Docling) are an optional
extra rather than part of the serving image.

---

## Project layout

```
app.py                   FastAPI app — loads models once, serves API + UI
config/settings.py       All configuration as dataclasses
scripts/
  setup_pinecone.py      Create the Pinecone index (768 dims)
  embed_and_upload.py    S3 chunks → BGE embed → upsert to Pinecone
src/
  rag_pipeline.py        Orchestrator: retrieve → rerank → gate → generate
  embeddings/            BGE embedder + factory
  retrieval/             semantic (Pinecone) + hybrid (BM25 + Pinecone, RRF)
  reranking/             cross-encoder reranker
  prompting/             system prompts + RAG prompt builder
  data_ingestion/        Docling parser, semantic chunker, S3, chunk loader
  utils/                 logging, metrics, text helpers
static/index.html        Self-contained chat UI
```

## Setup

```bash
uv sync                     # serving dependencies
uv sync --extra ingestion   # add only if parsing new PDFs (pulls Docling)
cp .env.example .env        # then fill in your keys
```

`.env` needs: `GROQ_API_KEY`, `PINECONE_API_KEY`, `PINECONE_ENV`,
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET_NAME`.

## Build the knowledge base (once per data change)

```bash
uv run python scripts/setup_pinecone.py     # create the 768-dim index
uv run python scripts/embed_and_upload.py   # embed S3 chunks → Pinecone
```

## Run

```bash
uv run uvicorn app:app --reload     # http://localhost:8000/
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Chat UI |
| GET | `/health` | Liveness |
| POST | `/chat` | `{question}` → grounded answer + sources |
| GET | `/metrics` | Latency / retrieval stats |
| POST | `/feedback` | `{query, rating 1-5, notes}` |
| GET | `/docs` | Swagger UI |

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How does vitamin C help with iron absorption?"}'
```

## Docker

The image bakes in both models (BGE-base + the cross-encoder) and uses CPU-only
torch, so the container starts without downloading anything.

```bash
docker build -t nutrition-chatbot .
docker run -p 8000:8000 --env-file .env nutrition-chatbot
```

## Configuration (`config/settings.py`)

- `RetrieverConfig.top_k` — chunks passed to the LLM (default 5)
- `RetrieverConfig.use_reranking` / `rerank_candidates` — reranking on/off and width
- `RetrieverConfig.use_hybrid_search` — hybrid vs pure semantic
- `RerankerConfig.min_relevance_score` — refusal gate threshold (lower = more permissive)
- `EmbeddingConfig` / `GroqConfig` — embedder and LLM settings
