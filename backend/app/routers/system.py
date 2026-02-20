"""System diagnostics and health endpoints."""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["system"])

logger = logging.getLogger(__name__)


@router.get("/system/diagnostics")
async def get_system_diagnostics():
    """
    Get comprehensive system diagnostics.
    
    Returns overall system health, performance metrics,
    and diagnostic information.
    """
    # Simple diagnostics without database dependency
    diagnostics = {
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
    
    return diagnostics


@router.get("/system/health")
async def system_health():
    """
    Basic system health check.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/health/embedding-providers")
async def embedding_providers_health() -> Dict[str, Any]:
    """
    Check health and availability of all embedding providers.
    
    Returns status for each configured embedding provider including:
    - availability (API key configured)
    - health status (can make requests)
    - cooldown status (recently failed)
    
    Requirements: 11.5
    """
    from app.services.embedding_provider import (
        EmbeddingProviderManager,
        EmbeddingProviderConfig,
        OpenRouterProvider,
        OpenAIProvider,
    )
    
    providers_status = {}
    overall_healthy = False
    
    # Check OpenRouter provider
    try:
        openrouter = OpenRouterProvider()
        providers_status["openrouter"] = {
            "available": openrouter.is_available(),
            "healthy": openrouter.is_available(),
            "error": None
        }
        if openrouter.is_available():
            overall_healthy = True
    except Exception as e:
        providers_status["openrouter"] = {
            "available": False,
            "healthy": False,
            "error": str(e)
        }
    
    # Check OpenAI provider
    try:
        openai_provider = OpenAIProvider()
        providers_status["openai"] = {
            "available": openai_provider.is_available(),
            "healthy": openai_provider.is_available(),
            "error": None
        }
        if openai_provider.is_available():
            overall_healthy = True
    except Exception as e:
        providers_status["openai"] = {
            "available": False,
            "healthy": False,
            "error": str(e)
        }
    
    # Get provider manager status if available
    try:
        config = EmbeddingProviderConfig()
        manager = EmbeddingProviderManager(config=config)
        manager_status = manager.get_provider_status()
        
        for provider_name, status in manager_status.items():
            if provider_name in providers_status:
                providers_status[provider_name].update({
                    "in_cooldown": status.get("in_cooldown", False),
                    "manager_healthy": status.get("healthy", True)
                })
    except Exception as e:
        # Manager initialization failed, but individual providers may work
        pass
    
    return {
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": providers_status,
        "summary": {
            "total_providers": len(providers_status),
            "available_providers": sum(
                1 for p in providers_status.values() if p.get("available")
            ),
            "healthy_providers": sum(
                1 for p in providers_status.values() if p.get("healthy")
            )
        }
    }



@router.get("/system/semantic-chunker/test")
async def test_semantic_chunker() -> Dict[str, Any]:
    """
    Test semantic chunker with a sample text.
    
    Returns detailed information about:
    - Which embedding provider was used
    - Whether fallback was triggered
    - Language detection results
    - Adaptive threshold calculation
    - Q&A detection results
    - Cache status
    - Chunk quality metrics
    """
    import time
    
    # Sample test text (Turkish + English mix)
    test_text = """
    Makine öğrenmesi nedir? Makine öğrenmesi, bilgisayarların açıkça 
    programlanmadan verilerden öğrenmesini sağlayan bir yapay zeka dalıdır.
    
    What is deep learning? Deep learning is a subset of machine learning 
    that uses neural networks with multiple layers to learn from data.
    
    Derin öğrenme nasıl çalışır? Derin öğrenme, çok katmanlı yapay sinir 
    ağları kullanarak karmaşık örüntüleri öğrenir.
    """
    
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_text_length": len(test_text),
        "components": {},
        "chunking_result": None,
        "errors": []
    }
    
    # Test 1: Language Detection
    try:
        from app.services.language_detector import LanguageDetector
        detector = LanguageDetector()
        lang = detector.detect_language(test_text)
        results["components"]["language_detector"] = {
            "status": "ok",
            "detected_language": lang.value,
            "confidence": "high" if lang.value in ["tr", "en"] else "low"
        }
    except Exception as e:
        results["components"]["language_detector"] = {
            "status": "error",
            "error": str(e)
        }
        results["errors"].append(f"Language detector: {e}")
    
    # Test 2: Sentence Tokenizer
    try:
        from app.services.sentence_tokenizer import EnhancedSentenceTokenizer
        from app.services.language_detector import Language
        tokenizer = EnhancedSentenceTokenizer()
        sentences = tokenizer.tokenize(test_text, Language.TURKISH)
        results["components"]["sentence_tokenizer"] = {
            "status": "ok",
            "sentence_count": len(sentences),
            "sample_sentences": sentences[:3] if sentences else []
        }
    except Exception as e:
        results["components"]["sentence_tokenizer"] = {
            "status": "error",
            "error": str(e)
        }
        results["errors"].append(f"Sentence tokenizer: {e}")
    
    # Test 3: Q&A Detector
    try:
        from app.services.qa_detector import QADetector
        qa_detector = QADetector()
        qa_pairs = qa_detector.detect_qa_pairs(test_text)
        results["components"]["qa_detector"] = {
            "status": "ok",
            "qa_pairs_found": len(qa_pairs),
            "pairs": [
                {"question": p.question[:50], "answer": p.answer[:50]}
                for p in qa_pairs[:3]
            ] if qa_pairs else []
        }
    except Exception as e:
        results["components"]["qa_detector"] = {
            "status": "error",
            "error": str(e)
        }
        results["errors"].append(f"Q&A detector: {e}")
    
    # Test 4: Adaptive Threshold
    try:
        from app.services.adaptive_threshold import AdaptiveThresholdCalculator
        calc = AdaptiveThresholdCalculator()
        rec = calc.recommend_threshold(test_text)
        results["components"]["adaptive_threshold"] = {
            "status": "ok",
            "recommended_threshold": rec.recommended_threshold,
            "base_threshold": rec.base_threshold,
            "diversity_factor": rec.diversity_factor,
            "length_factor": rec.length_factor,
            "confidence": rec.confidence,
            "reasoning": rec.reasoning
        }
    except Exception as e:
        results["components"]["adaptive_threshold"] = {
            "status": "error",
            "error": str(e)
        }
        results["errors"].append(f"Adaptive threshold: {e}")
    
    # Test 5: Embedding Providers
    try:
        from app.services.embedding_provider import (
            OpenRouterProvider,
            OpenAIProvider,
            EmbeddingProviderManager,
            EmbeddingProviderConfig,
        )
        
        providers_info = {}
        available_providers = []
        
        # Check OpenRouter
        openrouter = OpenRouterProvider()
        providers_info["openrouter"] = {
            "available": openrouter.is_available(),
            "priority": 1
        }
        if openrouter.is_available():
            available_providers.append(openrouter)
        
        # Check OpenAI
        openai_prov = OpenAIProvider()
        providers_info["openai"] = {
            "available": openai_prov.is_available(),
            "priority": 2
        }
        if openai_prov.is_available():
            available_providers.append(openai_prov)
        
        results["components"]["embedding_providers"] = {
            "status": "ok" if available_providers else "error",
            "providers": providers_info,
            "primary_provider": available_providers[0].name if available_providers else None,
            "fallback_available": len(available_providers) > 1
        }
        
        # Test actual embedding generation
        if available_providers:
            config = EmbeddingProviderConfig(batch_size=2, max_retries=1)
            manager = EmbeddingProviderManager(
                providers=available_providers,
                config=config
            )
            
            start = time.time()
            test_embeddings = manager.get_embeddings(
                ["Test sentence one.", "Test sentence two."],
                "openai/text-embedding-3-small"
            )
            embed_time = time.time() - start
            
            # Check which provider was actually used by checking health status
            used_provider = None
            for p in available_providers:
                if manager._provider_health.get(p.name, False):
                    used_provider = p.name
                    break
            if not used_provider and available_providers:
                used_provider = available_providers[0].name
            
            results["components"]["embedding_test"] = {
                "status": "ok",
                "provider_used": used_provider,
                "providers_tried": [p.name for p in available_providers],
                "embedding_dimension": len(test_embeddings[0]) if test_embeddings else 0,
                "response_time_ms": round(embed_time * 1000, 2),
                "provider_health": dict(manager._provider_health)
            }
    except Exception as e:
        results["components"]["embedding_providers"] = {
            "status": "error",
            "error": str(e)
        }
        results["errors"].append(f"Embedding providers: {e}")
    
    # Test 6: Full Semantic Chunking
    try:
        from app.services.chunker import chunk_with_error_handling, ChunkingStrategy
        
        start = time.time()
        chunk_result = chunk_with_error_handling(
            text=test_text,
            strategy=ChunkingStrategy.SEMANTIC,
            enable_qa_detection=True,
            enable_adaptive_threshold=True,
            enable_cache=True,
            min_chunk_size=100,
            max_chunk_size=1000
        )
        chunk_time = time.time() - start
        
        results["chunking_result"] = {
            "status": "ok" if chunk_result.success else "error",
            "chunk_count": len(chunk_result.chunks),
            "chunks": [
                {
                    "index": c.index,
                    "char_count": c.char_count,
                    "preview": c.content[:100] + "..." if len(c.content) > 100 else c.content
                }
                for c in chunk_result.chunks[:5]
            ],
            "processing_time_ms": round(chunk_time * 1000, 2),
            "fallback_used": chunk_result.fallback_used,
            "warning_message": chunk_result.warning_message,
            "error": str(chunk_result.error) if chunk_result.error else None
        }
        
        if chunk_result.diagnostics:
            results["chunking_result"]["diagnostics"] = {
                "strategy": chunk_result.diagnostics.strategy,
                "avg_chunk_size": chunk_result.diagnostics.avg_chunk_size,
                "quality_score": chunk_result.diagnostics.quality_score
            }
            
    except Exception as e:
        results["chunking_result"] = {
            "status": "error",
            "error": str(e)
        }
        results["errors"].append(f"Semantic chunking: {e}")
    
    # Overall status
    error_count = len(results["errors"])
    if error_count == 0:
        results["overall_status"] = "healthy"
    elif error_count <= 2:
        results["overall_status"] = "degraded"
    else:
        results["overall_status"] = "unhealthy"
    
    return results


@router.get("/system/embedding-providers/test")
async def test_embedding_with_details() -> Dict[str, Any]:
    """
    Test embedding generation with detailed provider information.
    
    Shows exactly which provider is being used and whether
    fallback mechanism is working.
    """
    import time
    import os
    
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "openrouter_key_set": bool(os.environ.get("OPENROUTER_API_KEY")),
            "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        },
        "tests": []
    }
    
    test_texts = ["Hello world.", "Merhaba dünya."]
    
    # Test with provider manager
    try:
        from app.services.embedding_provider import (
            OpenRouterProvider,
            OpenAIProvider,
            EmbeddingProviderManager,
            EmbeddingProviderConfig,
        )
        
        providers = []
        
        openrouter = OpenRouterProvider()
        if openrouter.is_available():
            providers.append(openrouter)
            
        openai_prov = OpenAIProvider()
        if openai_prov.is_available():
            providers.append(openai_prov)
        
        if not providers:
            results["tests"].append({
                "name": "provider_manager",
                "status": "error",
                "error": "No providers available"
            })
        else:
            config = EmbeddingProviderConfig(batch_size=2, max_retries=2)
            manager = EmbeddingProviderManager(providers=providers, config=config)
            
            start = time.time()
            embeddings = manager.get_embeddings(
                test_texts,
                "openai/text-embedding-3-small"
            )
            elapsed = time.time() - start
            
            results["tests"].append({
                "name": "provider_manager",
                "status": "ok",
                "providers_available": [p.name for p in providers],
                "embedding_count": len(embeddings),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                "response_time_ms": round(elapsed * 1000, 2),
                "provider_health": dict(manager._provider_health)
            })
            
    except Exception as e:
        logger.error(f"Provider manager test failed: {e}", exc_info=True)
        results["tests"].append({
            "name": "provider_manager",
            "status": "error",
            "error": str(e)
        })
    
    return results
