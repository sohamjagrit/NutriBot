"""NutriBot FastAPI application."""

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import get_config, RAGConfig
from src.rag_pipeline import NutritionRAG
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Global state, populated once at startup and reused across all requests.
rag: Optional[NutritionRAG] = None
startup_time: Optional[datetime] = None
config: Optional[RAGConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once at startup, reuse for every request.

    Models (BGE embedder, cross-encoder reranker) and the BM25 index over all
    chunks are expensive to build, so they are loaded here a single time rather
    than per-request. Code before `yield` runs on startup, code after on shutdown.
    """
    global rag, startup_time, config

    logger.info("Starting NutriBot FastAPI server...")
    config = get_config()
    rag = NutritionRAG(config)
    startup_time = datetime.now()
    logger.info(f"RAG pipeline ready (retriever={type(rag.retriever).__name__}, "
                f"embedder={rag.embedder.model_name}, "
                f"reranker={'on' if rag.reranker else 'off'})")

    yield

    logger.info("Shutting down NutriBot server...")


# Initialize FastAPI app
app = FastAPI(
    title="NutriBot",
    description="Production RAG Chatbot for Nutrition Q&A",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    """Chat endpoint request model."""

    question: str = Field(..., description="User's nutrition question")
    prompt_type: str = Field(
        default="standard",
        description="Type of system prompt (standard, conversational, detailed)",
    )
    top_k: Optional[int] = Field(default=None, description="Number of documents to retrieve")


class ChatResponse(BaseModel):
    """Chat endpoint response model."""

    question: str
    answer: str
    sources: list[str]
    scores: list[float]
    metrics: dict


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    uptime_seconds: float
    retriever_type: str
    embedder: str


class MetricsResponse(BaseModel):
    """Metrics response model."""

    total_queries: int
    avg_retrieval_latency_ms: float
    avg_llm_latency_ms: float
    avg_total_latency_ms: float
    avg_similarity_score: float
    avg_docs_retrieved: float


class FeedbackRequest(BaseModel):
    """Feedback endpoint request model."""

    query: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    notes: Optional[str] = Field(default=None, description="Optional feedback notes")


# Static assets (the chat UI lives in static/index.html)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Routes
@app.get("/", tags=["UI"], include_in_schema=False)
async def root():
    """Serve the chat UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api", tags=["Info"])
async def api_info():
    """API metadata and endpoint listing."""
    return {
        "app": "NutriBot",
        "version": "1.0.0",
        "description": "Production RAG Chatbot for Nutrition Q&A",
        "endpoints": {
            "ui": "/",
            "health": "/health",
            "chat": "/chat (POST)",
            "metrics": "/metrics",
            "feedback": "/feedback (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health", tags=["Health"])
async def health() -> HealthResponse:
    """Health check endpoint."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    uptime = (datetime.now() - startup_time).total_seconds()

    retriever_type = type(rag.retriever).__name__ if rag.retriever else "None"

    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        retriever_type=retriever_type,
        embedder=rag.embedder.model_name,
    )


@app.post("/chat", tags=["Chat"], response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat endpoint: answer nutrition questions."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = rag.query(request.question, top_k=request.top_k, prompt_type=request.prompt_type)

        return ChatResponse(
            question=result["question"],
            answer=result["response"],
            sources=result["retrieved_docs"],
            scores=result["scores"],
            metrics=result["metrics"],
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")


@app.get("/metrics", tags=["Metrics"], response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Get aggregated metrics."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    summary = rag.get_metrics_summary()

    return MetricsResponse(
        total_queries=summary.get("total_queries", 0),
        avg_retrieval_latency_ms=summary.get("avg_retrieval_latency_ms", 0),
        avg_llm_latency_ms=summary.get("avg_llm_latency_ms", 0),
        avg_total_latency_ms=summary.get("avg_total_latency_ms", 0),
        avg_similarity_score=summary.get("avg_similarity_score", 0),
        avg_docs_retrieved=summary.get("avg_docs_retrieved", 0),
    )


@app.post("/feedback", tags=["Feedback"])
async def feedback(request: FeedbackRequest):
    """Submit feedback on a response."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        # Create feedback directory if it doesn't exist
        feedback_dir = Path("data/feedback")
        feedback_dir.mkdir(parents=True, exist_ok=True)

        # Save feedback to file
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "query": request.query,
            "rating": request.rating,
            "notes": request.notes,
        }

        feedback_file = feedback_dir / f"feedback_{int(time.time())}.json"

        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)

        logger.info(f"Feedback saved: rating={request.rating}, query={request.query[:50]}...")

        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail="Error saving feedback")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Return a consistent JSON error body with the correct HTTP status code."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "detail": exc.detail,
            "status_code": exc.status_code,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
