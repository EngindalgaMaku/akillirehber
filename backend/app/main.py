"""
RAG Educational Chatbot - FastAPI Backend
Main application entry point with CORS configuration
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import (
    admin,
    auth,
    chat,
    chunking,
    courses,
    course_settings,
    course_prompt_templates,
    document,
    documents,
    embeddings,
    giskard,
    llm_models,
    ragas,
    semantic_similarity,
    system,
    system_settings,
    test_generation,
)
from app.api.benchmark import router as benchmark_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("Starting RAG Educational Chatbot API...")
    yield
    # Shutdown
    print("Shutting down RAG Educational Chatbot API...")


app = FastAPI(
    title="RAG Educational Chatbot API",
    description="Backend API for RAG-based educational chatbot system",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS configuration for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
        "http://frontend:3000",  # Docker network
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(benchmark_router)
app.include_router(courses.router)
app.include_router(course_settings.router)
app.include_router(course_prompt_templates.router)
app.include_router(documents.router)
app.include_router(document.router)  # Legacy upload endpoint
app.include_router(chunking.router)
app.include_router(embeddings.router)
app.include_router(chat.router)
app.include_router(ragas.router)
app.include_router(semantic_similarity.router)
app.include_router(giskard.router)
app.include_router(system.router)
app.include_router(system_settings.router)
app.include_router(llm_models.router)
app.include_router(test_generation.router)


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "RAG Educational Chatbot API",
        "status": "running",
        "version": "0.2.0",
    }


@app.get("/diagnostics")
async def system_diagnostics():
    """System diagnostics endpoint"""
    from datetime import datetime, timezone
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "database": "connected",
            "weaviate": "unknown",
            "embedding": "unknown"
        },
        "metrics": {
            "total_documents": 0,
            "total_chunks": 0,
            "processing_queue_size": 0
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "environment": settings.environment}
