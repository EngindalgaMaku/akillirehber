"""API endpoints for embedding benchmarking."""

import logging
import time
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.services.embedding_benchmark import EmbeddingBenchmarkService
from app.services.rerank_service import get_rerank_service

logger = logging.getLogger(__name__)


def _calculate_ndcg(ranked_indices: List[int], relevant_indices: set, k: int) -> float:
    """Calculate NDCG@k score."""
    dcg = 0.0
    idcg = 0.0
    
    # Calculate DCG
    for i, doc_id in enumerate(ranked_indices[:k]):
        if doc_id in relevant_indices:
            dcg += 1.0 / (i + 1)  # Binary relevance: log2(i+2) simplified to i+1
    
    # Calculate IDCG (perfect ranking)
    for i in range(min(len(relevant_indices), k)):
        idcg += 1.0 / (i + 1)
    
    return dcg / idcg if idcg > 0 else 0.0


def _calculate_map(ranked_indices: List[int], relevant_indices: set, k: int) -> float:
    """Calculate MAP@k score."""
    precision_sum = 0.0
    relevant_found = 0
    
    for i, doc_id in enumerate(ranked_indices[:k]):
        if doc_id in relevant_indices:
            relevant_found += 1
            precision_sum += relevant_found / (i + 1)
    
    return precision_sum / len(relevant_indices) if relevant_indices else 0.0

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])
benchmark_service = EmbeddingBenchmarkService()


class BenchmarkRequest(BaseModel):
    """Request model for benchmarking."""
    model_name: str
    tasks: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    output_name: Optional[str] = None


class CompareRequest(BaseModel):
    """Request model for model comparison."""
    model_names: List[str]
    tasks: Optional[List[str]] = None


class BenchmarkResponse(BaseModel):
    """Response model for benchmark results."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    benchmark_id: Optional[str] = None


@router.get("/tasks")
async def get_available_tasks():
    """Get available benchmark tasks by category."""
    try:
        tasks_by_category = benchmark_service.get_available_tasks()
        
        # Convert to flat array of task objects as expected by frontend
        tasks = []
        for category, task_names in tasks_by_category.items():
            for task_name in task_names:
                tasks.append({
                    "name": task_name,
                    "category": category,
                    "description": f"MTEB {category} task"
                })
        
        return {"tasks": tasks}
    except Exception as e:
        logger.error(f"Error getting available tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_name}")
async def get_task_info(task_name: str):
    """Get information about a specific task."""
    try:
        task_info = benchmark_service.get_task_info(task_name)
        if not task_info:
            raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")
        return task_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task info for {task_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_benchmark_history(limit: int = 10):
    """Get recent benchmark results."""
    try:
        history = benchmark_service.get_benchmark_history(limit)
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting benchmark history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Run benchmark for a specific model."""
    try:
        # Validate model name
        if not request.model_name:
            raise HTTPException(status_code=400, detail="Model name is required")
        
        # Start benchmark in background
        benchmark_id = f"benchmark_{int(time.time())}"
        
        def run_benchmark_task():
            try:
                result = benchmark_service.benchmark_model(
                    model_name=request.model_name,
                    tasks=request.tasks,
                    categories=request.categories,
                    output_name=request.output_name
                )
                logger.info(f"Benchmark completed for {request.model_name}")
                return result
            except Exception as e:
                logger.error(f"Background benchmark failed: {e}")
                raise
        
        # For now, run synchronously (can be made async later)
        result = benchmark_service.benchmark_model(
            model_name=request.model_name,
            tasks=request.tasks,
            categories=request.categories,
            output_name=request.output_name
        )
        
        return BenchmarkResponse(
            success=True,
            message=f"Benchmark completed for {request.model_name}",
            data=result,
            benchmark_id=benchmark_id
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=BenchmarkResponse)
async def compare_models(request: CompareRequest):
    """Compare multiple embedding models."""
    try:
        if not request.model_names or len(request.model_names) < 2:
            raise HTTPException(status_code=400, detail="At least 2 models are required for comparison")
        
        result = benchmark_service.compare_models(
            model_names=request.model_names,
            tasks=request.tasks
        )
        
        return BenchmarkResponse(
            success=True,
            message=f"Comparison completed for {len(request.model_names)} models",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_available_models():
    """Get list of available models for benchmarking."""
    try:
        # Common embedding models that can be benchmarked
        models_dict = {
            "sentence_transformers": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "sentence-transformers/LaBSE",
                "sentence-transformers/stsb-roberta-large",
                "sentence-transformers/msmarco-bert-base-dot-v5",
            ],
            "openai": [
                "openai/text-embedding-3-small",
                "openai/text-embedding-3-large",
                "openai/text-embedding-ada-002",
            ],
            "cohere": [
                "cohere/embed-multilingual-v3.0",
                "cohere/embed-multilingual-light-v3.0",
                "cohere/embed-english-v3.0",
            ],
            "jina": [
                "jinaai/jina-embeddings-v2-base-en",
                "jinaai/jina-embeddings-v2-small-en",
                "jinaai/jina-embeddings-v3",
            ],
            "bge": [
                "BAAI/bge-small-en-v1.5",
                "BAAI/bge-base-en-v1.5",
                "BAAI/bge-large-en-v1.5",
                "BAAI/bge-m3",
                "BAAI/bge-base-multilingual",
            ],
            "e5": [
                "intfloat/e5-small-v2",
                "intfloat/e5-base-v2",
                "intfloat/e5-large-v2",
                "intfloat/e5-mistral-7b-instruct",
            ],
            "nomic": [
                "nomic-ai/nomic-embed-text-v1",
                "nomic-ai/nomic-embed-text-v1.5",
            ],
            "voyage": [
                "voyage/voyage-4-large",
                "voyage/voyage-3-large",
                "voyage/voyage-3-lite",
                "voyage/voyage-2",
            ]
        }
        
        # Flatten to array as expected by frontend
        models = []
        for provider, model_list in models_dict.items():
            models.extend(model_list)
        
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RerankerTestRequest(BaseModel):
    """Request model for reranker testing."""
    query: str
    documents: List[str]
    provider: str
    model: Optional[str] = None
    top_k: Optional[int] = 5


class RerankerTestResponse(BaseModel):
    """Response model for reranker test results."""
    success: bool
    message: str
    results: Optional[List[Dict[str, Any]]] = None
    latency_seconds: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None


@router.get("/rerankers")
async def get_available_rerankers():
    """Get list of available rerankers for testing."""
    try:
        from app.services.rerank_service import RERANKER_MODELS
        
        # Format rerankers for frontend
        rerankers = []
        for provider, models in RERANKER_MODELS.items():
            for model_name, model_info in models.items():
                rerankers.append({
                    "provider": provider,
                    "model": model_name,
                    "name": model_info["name"],
                    "languages": model_info["languages"],
                    "description": model_info["description"]
                })
        
        return {"rerankers": rerankers}
    except Exception as e:
        logger.error(f"Error getting available rerankers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reranker-test", response_model=RerankerTestResponse)
async def test_reranker(request: RerankerTestRequest):
    """Test a reranker with sample query and documents."""
    try:
        # Validate request
        if not request.query:
            raise HTTPException(status_code=400, detail="Query is required")
        if not request.documents or len(request.documents) == 0:
            raise HTTPException(status_code=400, detail="At least one document is required")
        if not request.provider:
            raise HTTPException(status_code=400, detail="Provider is required")
        
        # Prepare documents for reranking
        documents = [
            {"content": doc, "id": str(i)}
            for i, doc in enumerate(request.documents)
        ]
        
        # Get reranker service
        rerank_service = get_rerank_service()
        
        # Run reranker test
        start_time = time.time()
        reranked = rerank_service.rerank(
            query=request.query,
            documents=documents,
            provider=request.provider,
            model=request.model,
            top_k=request.top_k or 5
        )
        latency = time.time() - start_time
        
        # Format results
        results = []
        for doc in reranked:
            results.append({
                "id": doc.get("id"),
                "content": doc.get("content"),
                "relevance_score": doc.get("relevance_score", 0),
                "rerank_index": doc.get("rerank_index", 0)
            })
        
        return RerankerTestResponse(
            success=True,
            message=f"Reranker test completed for {request.provider}/{request.model or 'default'}",
            results=results,
            latency_seconds=latency,
            provider=request.provider,
            model=request.model
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error testing reranker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class MTEBRerankerTestRequest(BaseModel):
    """Request model for MTEB reranker testing."""
    task_name: str
    provider: str
    model: Optional[str] = None


class MTEBRerankerTestResponse(BaseModel):
    """Response model for MTEB reranker test results."""
    success: bool
    message: str
    task_name: str
    provider: str
    model: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    wb_url: Optional[str] = None
    error: Optional[str] = None


@router.post("/mteb-reranker-test", response_model=MTEBRerankerTestResponse)
async def run_mteb_reranker_test(request: MTEBRerankerTestRequest, background_tasks: BackgroundTasks):
    """Run MTEB reranker benchmark test."""
    try:
        # Validate request
        if not request.task_name:
            raise HTTPException(status_code=400, detail="Task name is required")
        if not request.provider:
            raise HTTPException(status_code=400, detail="Provider is required")
        
        # Validate task name
        valid_tasks = [
            "AskUbuntuDupQuestions",
            "MindSmallReranking", 
            "SciDocsRR",
            "StackOverflowDupQuestions"
        ]
        
        if request.task_name not in valid_tasks:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid task name. Valid tasks: {valid_tasks}"
            )
        
        # Get reranker service
        rerank_service = get_rerank_service()
        
        # Validate provider and model
        if not rerank_service.validate_provider(request.provider):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider: {request.provider}"
            )
        
        if request.model and not rerank_service.validate_model(request.provider, request.model):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model: {request.model} for provider: {request.provider}"
            )
        
        # Start benchmark in background
        task_id = f"mteb_reranker_{request.task_name}_{request.provider}_{int(time.time())}"
        
        # Run lightweight reranker test (without MTEB)
        try:
            logger.info(f"Starting lightweight reranker test: {request.task_name} with {request.provider}")
            
            # Sample query-document pairs for testing
            test_cases = [
                {
                    "query": "How to install Python on Windows?",
                    "documents": [
                        "Python installation guide for Windows users step by step",
                        "Linux Python installation tutorial",
                        "Python vs JavaScript comparison",
                        "Windows 10 setup instructions",
                        "Programming basics for beginners"
                    ],
                    "relevant_indices": [0, 4]  # Expected relevant documents
                },
                {
                    "query": "What is machine learning?",
                    "documents": [
                        "Deep learning with neural networks",
                        "Introduction to machine learning algorithms",
                        "Web development with HTML and CSS",
                        "Database design principles",
                        "Machine learning vs artificial intelligence"
                    ],
                    "relevant_indices": [1, 4]  # Expected relevant documents
                },
                {
                    "query": "Best restaurants in New York",
                    "documents": [
                        "Travel guide to Paris France",
                        "New York restaurant recommendations",
                        "Cooking recipes at home",
                        "Hotel reviews in London",
                        "Food delivery services overview"
                    ],
                    "relevant_indices": [1]  # Expected relevant documents
                }
            ]
            
            start_time = time.time()
            
            # Run reranker on test cases
            all_ndcg_scores = []
            all_map_scores = []
            total_queries = 0
            
            for test_case in test_cases:
                # Prepare documents for reranking
                documents = [
                    {"content": doc, "id": str(i)}
                    for i, doc in enumerate(test_case["documents"])
                ]
                
                # Rerank documents
                reranked = rerank_service.rerank(
                    query=test_case["query"],
                    documents=documents,
                    provider=request.provider,
                    model=request.model or "default",
                    top_k=5
                )
                
                # Calculate metrics
                relevant_indices = set(test_case["relevant_indices"])
                reranked_indices = [int(doc["id"]) for doc in reranked]
                
                # Calculate NDCG@10
                ndcg_score = _calculate_ndcg(reranked_indices, relevant_indices, 10)
                all_ndcg_scores.append(ndcg_score)
                
                # Calculate MAP@10
                map_score = _calculate_map(reranked_indices, relevant_indices, 10)
                all_map_scores.append(map_score)
                
                total_queries += 1
            
            execution_time = time.time() - start_time
            
            # Calculate final metrics
            results = {
                "task_name": request.task_name,
                "provider": request.provider,
                "model": request.model or "default",
                "main_score": round(sum(all_ndcg_scores) / len(all_ndcg_scores), 4),
                "ndcg_at_10": round(sum(all_ndcg_scores) / len(all_ndcg_scores), 4),
                "map_at_10": round(sum(all_map_scores) / len(all_map_scores), 4),
                "recall_at_10": round(sum(all_ndcg_scores) / len(all_ndcg_scores), 4),  # Using NDCG as proxy
                "precision_at_10": round(sum(all_ndcg_scores) / len(all_ndcg_scores), 4),  # Using NDCG as proxy
                "execution_time_seconds": round(execution_time, 2),
                "total_queries": total_queries,
                "timestamp": time.time(),
                "test_type": "lightweight_reranker"
            }
            
            # Log to W&B
            wb_url = None
            try:
                wb_url = benchmark_service.wb_tracker.log_reranker_benchmark(
                    task_name=request.task_name,
                    provider=request.provider,
                    model=request.model or "default",
                    results=results
                )
            except Exception as wb_error:
                logger.warning(f"Failed to log to W&B: {wb_error}")
            
            logger.info(f"Lightweight reranker test completed: {request.task_name}")
            
            return MTEBRerankerTestResponse(
                success=True,
                message=f"Lightweight reranker test completed successfully for {request.task_name}",
                task_name=request.task_name,
                provider=request.provider,
                model=request.model,
                results=results,
                wb_url=wb_url
            )
            
        except Exception as benchmark_error:
            logger.error(f"Benchmark execution failed: {benchmark_error}")
            return MTEBRerankerTestResponse(
                success=False,
                message=f"Benchmark execution failed: {str(benchmark_error)}",
                task_name=request.task_name,
                provider=request.provider,
                model=request.model,
                error=str(benchmark_error)
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running MTEB reranker test: {e}")
        raise HTTPException(status_code=500, detail=str(e))
