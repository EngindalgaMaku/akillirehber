"""API router for text chunking operations."""

import asyncio
import json
import time
import logging
from typing import List, Union, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.chunking import (
    Chunk,
    ChunkRequest,
    ChunkResponse,
    ChunkResponseWithQuality,
    ChunkStats,
    ChunkQualityMetricsResponse,
    QualityReportResponse,
    ChunkingStrategy,
    ChunkingProgressEvent,
)
from app.services.chunker import (
    ChunkerService,
    chunk_with_error_handling,
    ChunkingResult,
)

router = APIRouter(prefix="/api", tags=["chunking"])

# Initialize the chunker service
chunker_service = ChunkerService()

logger = logging.getLogger(__name__)


def calculate_stats(chunks: List[Chunk]) -> ChunkStats:
    """Calculate statistics from a list of chunks."""
    if not chunks:
        return ChunkStats(
            total_chunks=0,
            total_characters=0,
            avg_chunk_size=0.0,
            min_chunk_size=0,
            max_chunk_size=0,
        )

    char_counts = [chunk.char_count for chunk in chunks]
    total_chars = sum(char_counts)

    return ChunkStats(
        total_chunks=len(chunks),
        total_characters=total_chars,
        avg_chunk_size=total_chars / len(chunks),
        min_chunk_size=min(char_counts),
        max_chunk_size=max(char_counts),
    )


@router.post(
    "/chunk",
    response_model=Union[ChunkResponseWithQuality, ChunkResponse],
    responses={
        200: {
            "description": "Successfully chunked text",
            "content": {
                "application/json": {
                    "examples": {
                        "basic": {
                            "summary": "Basic chunking response",
                            "value": {
                                "chunks": [
                                    {
                                        "index": 0,
                                        "content": "First chunk content...",
                                        "start_position": 0,
                                        "end_position": 100,
                                        "char_count": 100,
                                        "has_overlap": False
                                    }
                                ],
                                "stats": {
                                    "total_chunks": 1,
                                    "total_characters": 100,
                                    "avg_chunk_size": 100.0,
                                    "min_chunk_size": 100,
                                    "max_chunk_size": 100
                                },
                                "strategy_used": "semantic"
                            }
                        },
                        "with_quality": {
                            "summary": "Response with quality metrics",
                            "value": {
                                "chunks": [{"index": 0, "content": "..."}],
                                "stats": {"total_chunks": 1},
                                "strategy_used": "semantic",
                                "quality_metrics": [
                                    {
                                        "chunk_index": 0,
                                        "semantic_coherence": 0.85,
                                        "sentence_count": 3,
                                        "topic_consistency": 0.9,
                                        "has_questions": False,
                                        "has_qa_pairs": False
                                    }
                                ],
                                "quality_report": {
                                    "total_chunks": 1,
                                    "avg_coherence": 0.85,
                                    "overall_quality_score": 0.82,
                                    "recommendations": ["Chunk quality is good."]
                                },
                                "detected_language": "en",
                                "adaptive_threshold_used": 0.52
                            }
                        }
                    }
                }
            }
        },
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)
async def chunk_text(
    request: ChunkRequest
) -> Union[ChunkResponseWithQuality, ChunkResponse]:
    """Process text and return chunks with metadata.

    This endpoint accepts text and chunking parameters, processes the text
    using the specified strategy, and returns the resulting chunks along
    with statistics.

    ## Strategies

    - **fixed_size**: Character-based chunking with overlap
    - **recursive**: Hierarchical splitting using separators
    - **sentence**: NLP-based sentence boundary detection
    - **semantic**: Embedding-based semantic similarity chunking (recommended)
    - **late_chunking**: Long-context embedding model approach
    - **agentic**: LLM-driven semantic segmentation

    ## Enhanced Semantic Chunking Features

    When using the `semantic` strategy, the following features are available:

    - **Q&A Detection**: Automatically detects and keeps question-answer pairs
      together in the same chunk
    - **Adaptive Threshold**: Calculates optimal similarity threshold based on
      text characteristics (vocabulary diversity, sentence length)
    - **Embedding Caching**: Caches embeddings for improved performance
    - **Quality Metrics**: Optional quality analysis including coherence scores
      and recommendations

    ## Configuration Options

    - `enable_qa_detection`: Enable Q&A pair detection (default: true)
    - `enable_adaptive_threshold`: Enable adaptive threshold (default: true)
    - `enable_cache`: Enable embedding caching (default: true)
    - `include_quality_metrics`: Include quality analysis in response
    - `min_chunk_size`: Minimum characters per chunk (default: 150)
    - `max_chunk_size`: Maximum characters per chunk (default: 2000)
    - `buffer_size`: Context sentences for semantic chunking (default: 1)
    """
    start_time = time.time()

    try:
        # Use enhanced semantic chunking for semantic strategy
        if request.strategy == ChunkingStrategy.SEMANTIC:
            return await _handle_semantic_chunking(request, start_time)

        # Build kwargs based on strategy for other strategies
        kwargs = {
            "chunk_size": request.chunk_size,
            "overlap": request.overlap,
        }

        # Add strategy-specific parameters
        if request.separators is not None:
            kwargs["separators"] = request.separators

        if request.similarity_threshold is not None:
            kwargs["similarity_threshold"] = request.similarity_threshold

        if request.embedding_model is not None:
            kwargs["embedding_model"] = request.embedding_model

        if request.llm_model is not None:
            kwargs["llm_model"] = request.llm_model

        # Perform chunking
        chunks = chunker_service.chunk_text(
            text=request.text,
            strategy=request.strategy,
            **kwargs
        )

        # Calculate statistics
        stats = calculate_stats(chunks)

        processing_time = (time.time() - start_time) * 1000

        if request.include_quality_metrics:
            return ChunkResponseWithQuality(
                chunks=chunks,
                stats=stats,
                strategy_used=request.strategy,
                processing_time_ms=processing_time,
            )

        return ChunkResponse(
            chunks=chunks,
            stats=stats,
            strategy_used=request.strategy,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Required dependency not installed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Chunking error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


async def _handle_semantic_chunking(
    request: ChunkRequest,
    start_time: float
) -> Union[ChunkResponseWithQuality, ChunkResponse]:
    """Handle semantic chunking with enhanced features."""
    detected_language = None
    adaptive_threshold = None

    # Use error handling wrapper for robust chunking
    result: ChunkingResult = chunk_with_error_handling(
        text=request.text,
        strategy=ChunkingStrategy.SEMANTIC,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        similarity_threshold=request.similarity_threshold,
        embedding_model=request.embedding_model,
        enable_qa_detection=request.enable_qa_detection,
        enable_adaptive_threshold=request.enable_adaptive_threshold,
        enable_cache=request.enable_cache,
        min_chunk_size=request.min_chunk_size,
        max_chunk_size=request.max_chunk_size,
        buffer_size=request.buffer_size,
    )

    if not result.success and not result.chunks:
        raise HTTPException(
            status_code=500,
            detail=result.error.message if result.error else "Chunking failed"
        )

    # Detect language for response
    if request.enable_qa_detection or request.enable_adaptive_threshold:
        try:
            from app.services.language_detector import LanguageDetector
            detector = LanguageDetector()
            lang = detector.detect_language(request.text)
            detected_language = lang.value
        except Exception:
            pass

    # Get adaptive threshold if used
    if request.enable_adaptive_threshold and request.similarity_threshold is None:
        try:
            from app.services.adaptive_threshold import AdaptiveThresholdCalculator
            calc = AdaptiveThresholdCalculator()
            rec = calc.recommend_threshold(request.text)
            adaptive_threshold = rec.recommended_threshold
        except Exception:
            pass

    # Calculate statistics
    stats = calculate_stats(result.chunks)
    processing_time = (time.time() - start_time) * 1000

    # Calculate quality metrics if requested
    quality_metrics = None
    quality_report = None

    if request.include_quality_metrics and result.chunks:
        try:
            from app.services.chunk_quality import ChunkQualityAnalyzer

            analyzer = ChunkQualityAnalyzer()
            chunk_texts = [c.content for c in result.chunks]

            # Calculate metrics for each chunk
            metrics_list = []
            for i, chunk_text in enumerate(chunk_texts):
                metrics = analyzer.calculate_chunk_metrics(chunk_text, i)
                metrics_list.append(ChunkQualityMetricsResponse(
                    chunk_index=metrics.chunk_index,
                    semantic_coherence=metrics.semantic_coherence,
                    sentence_count=metrics.sentence_count,
                    topic_consistency=metrics.topic_consistency,
                    has_questions=metrics.has_questions,
                    has_qa_pairs=metrics.has_qa_pairs,
                ))
            quality_metrics = metrics_list

            # Generate quality report
            report = analyzer.generate_quality_report(chunk_texts)
            quality_report = QualityReportResponse(
                total_chunks=report.total_chunks,
                avg_coherence=report.avg_coherence,
                min_coherence=report.min_coherence,
                max_coherence=report.max_coherence,
                chunks_below_threshold=report.chunks_below_threshold,
                inter_chunk_similarities=report.inter_chunk_similarities,
                merge_recommendations=report.merge_recommendations,
                split_recommendations=report.split_recommendations,
                overall_quality_score=report.overall_quality_score,
                recommendations=report.recommendations,
            )
        except Exception as e:
            logger.warning(f"Failed to calculate quality metrics: {e}")

    return ChunkResponseWithQuality(
        chunks=result.chunks,
        stats=stats,
        strategy_used=ChunkingStrategy.SEMANTIC,
        quality_metrics=quality_metrics,
        quality_report=quality_report,
        detected_language=detected_language,
        adaptive_threshold_used=adaptive_threshold,
        processing_time_ms=processing_time,
        fallback_used=result.fallback_used,
        warning_message=result.warning_message,
    )


def _create_sse_event(event: ChunkingProgressEvent) -> str:
    """Create SSE formatted event string."""
    data = event.model_dump_json()
    return f"data: {data}\n\n"


async def _stream_semantic_chunking(
    request: ChunkRequest,
) -> AsyncGenerator[str, None]:
    """Stream semantic chunking progress via SSE."""
    start_time = time.time()
    text_length = len(request.text)
    
    try:
        # Stage 1: Initialization (0-5%)
        yield _create_sse_event(ChunkingProgressEvent(
            event_type="progress",
            stage="initialization",
            progress=0,
            message="İşlem başlatılıyor...",
            details={"text_length": text_length}
        ))
        await asyncio.sleep(0.1)
        
        # Stage 2: Language Detection (5-10%)
        yield _create_sse_event(ChunkingProgressEvent(
            event_type="progress",
            stage="language_detection",
            progress=5,
            message="Dil algılanıyor...",
        ))
        
        detected_language = None
        if request.enable_qa_detection or request.enable_adaptive_threshold:
            try:
                from app.services.language_detector import LanguageDetector
                detector = LanguageDetector()
                lang = detector.detect_language(request.text)
                detected_language = lang.value
            except Exception:
                pass
        
        yield _create_sse_event(ChunkingProgressEvent(
            event_type="progress",
            stage="language_detection",
            progress=10,
            message=f"Dil algılandı: {detected_language or 'bilinmiyor'}",
            details={"detected_language": detected_language}
        ))
        await asyncio.sleep(0.1)
        
        # Stage 3: Sentence Tokenization (10-20%)
        yield _create_sse_event(ChunkingProgressEvent(
            event_type="progress",
            stage="tokenization",
            progress=15,
            message="Metin cümlelere ayrılıyor...",
        ))
        await asyncio.sleep(0.1)
        
        # Stage 4: Adaptive Threshold (20-25%)
        adaptive_threshold = None
        if request.enable_adaptive_threshold and request.similarity_threshold is None:
            yield _create_sse_event(ChunkingProgressEvent(
                event_type="progress",
                stage="threshold_calculation",
                progress=20,
                message="Adaptif eşik hesaplanıyor...",
            ))
            try:
                from app.services.adaptive_threshold import AdaptiveThresholdCalculator
                calc = AdaptiveThresholdCalculator()
                rec = calc.recommend_threshold(request.text)
                adaptive_threshold = rec.recommended_threshold
            except Exception:
                pass
        
        yield _create_sse_event(ChunkingProgressEvent(
            event_type="progress",
            stage="threshold_calculation",
            progress=25,
            message=f"Eşik değeri: {adaptive_threshold or request.similarity_threshold or 0.5:.2f}",
            details={"threshold": adaptive_threshold or request.similarity_threshold}
        ))
        await asyncio.sleep(0.1)
        
        # Stage 5: Embedding Generation (25-60%)
        # This is the longest stage for large documents
        estimated_sentences = text_length // 100  # Rough estimate
        yield _create_sse_event(ChunkingProgressEvent(
            event_type="progress",
            stage="embedding_generation",
            progress=30,
            message=f"Embedding'ler oluşturuluyor (~{estimated_sentences} cümle)...",
            details={"estimated_sentences": estimated_sentences}
        ))
        
        # Simulate progress during embedding (actual work happens in chunk_with_error_handling)
        for pct in [35, 40, 45, 50, 55]:
            await asyncio.sleep(0.2)
            yield _create_sse_event(ChunkingProgressEvent(
                event_type="progress",
                stage="embedding_generation",
                progress=pct,
                message="Embedding'ler oluşturuluyor...",
            ))
        
        # Stage 6: Chunking (60-80%)
        yield _create_sse_event(ChunkingProgressEvent(
            event_type="progress",
            stage="chunking",
            progress=60,
            message="Metin parçalanıyor...",
        ))
        
        # Perform actual chunking
        result: ChunkingResult = chunk_with_error_handling(
            text=request.text,
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            similarity_threshold=request.similarity_threshold,
            embedding_model=request.embedding_model,
            enable_qa_detection=request.enable_qa_detection,
            enable_adaptive_threshold=request.enable_adaptive_threshold,
            enable_cache=request.enable_cache,
            min_chunk_size=request.min_chunk_size,
            max_chunk_size=request.max_chunk_size,
            buffer_size=request.buffer_size,
        )
        
        if not result.success and not result.chunks:
            yield _create_sse_event(ChunkingProgressEvent(
                event_type="error",
                stage="chunking",
                progress=60,
                message=result.error.message if result.error else "Chunking failed",
            ))
            return
        
        yield _create_sse_event(ChunkingProgressEvent(
            event_type="progress",
            stage="chunking",
            progress=80,
            message=f"{len(result.chunks)} parça oluşturuldu",
            details={"chunk_count": len(result.chunks)}
        ))
        await asyncio.sleep(0.1)
        
        # Stage 7: Quality Metrics (80-95%)
        quality_metrics = None
        quality_report = None
        
        if request.include_quality_metrics and result.chunks:
            yield _create_sse_event(ChunkingProgressEvent(
                event_type="progress",
                stage="quality_analysis",
                progress=85,
                message="Kalite metrikleri hesaplanıyor...",
            ))
            
            try:
                from app.services.chunk_quality import ChunkQualityAnalyzer
                analyzer = ChunkQualityAnalyzer()
                chunk_texts = [c.content for c in result.chunks]
                
                # Calculate metrics for each chunk with progress
                metrics_list = []
                total_chunks = len(chunk_texts)
                for i, chunk_text in enumerate(chunk_texts):
                    metrics = analyzer.calculate_chunk_metrics(chunk_text, i)
                    metrics_list.append(ChunkQualityMetricsResponse(
                        chunk_index=metrics.chunk_index,
                        semantic_coherence=metrics.semantic_coherence,
                        sentence_count=metrics.sentence_count,
                        topic_consistency=metrics.topic_consistency,
                        has_questions=metrics.has_questions,
                        has_qa_pairs=metrics.has_qa_pairs,
                    ))
                    
                    # Update progress every few chunks
                    if i % 5 == 0 or i == total_chunks - 1:
                        pct = 85 + (i / total_chunks) * 8
                        yield _create_sse_event(ChunkingProgressEvent(
                            event_type="progress",
                            stage="quality_analysis",
                            progress=pct,
                            message=f"Kalite analizi: {i+1}/{total_chunks} parça",
                            details={"analyzed": i+1, "total": total_chunks}
                        ))
                
                quality_metrics = metrics_list
                
                # Generate quality report
                report = analyzer.generate_quality_report(chunk_texts)
                quality_report = QualityReportResponse(
                    total_chunks=report.total_chunks,
                    avg_coherence=report.avg_coherence,
                    min_coherence=report.min_coherence,
                    max_coherence=report.max_coherence,
                    chunks_below_threshold=report.chunks_below_threshold,
                    inter_chunk_similarities=report.inter_chunk_similarities,
                    merge_recommendations=report.merge_recommendations,
                    split_recommendations=report.split_recommendations,
                    overall_quality_score=report.overall_quality_score,
                    recommendations=report.recommendations,
                )
            except Exception as e:
                logger.warning(f"Failed to calculate quality metrics: {e}")
        
        # Stage 8: Finalization (95-100%)
        yield _create_sse_event(ChunkingProgressEvent(
            event_type="progress",
            stage="finalization",
            progress=95,
            message="Sonuçlar hazırlanıyor...",
        ))
        
        # Calculate final stats
        stats = calculate_stats(result.chunks)
        processing_time = (time.time() - start_time) * 1000
        
        # Build final response
        final_response = ChunkResponseWithQuality(
            chunks=result.chunks,
            stats=stats,
            strategy_used=ChunkingStrategy.SEMANTIC,
            quality_metrics=quality_metrics,
            quality_report=quality_report,
            detected_language=detected_language,
            adaptive_threshold_used=adaptive_threshold,
            processing_time_ms=processing_time,
            fallback_used=result.fallback_used,
            warning_message=result.warning_message,
        )
        
        # Send complete event with result
        yield _create_sse_event(ChunkingProgressEvent(
            event_type="complete",
            stage="complete",
            progress=100,
            message="İşlem tamamlandı!",
            result=json.loads(final_response.model_dump_json())
        ))
        
    except Exception as e:
        logger.error(f"Streaming chunking error: {e}", exc_info=True)
        yield _create_sse_event(ChunkingProgressEvent(
            event_type="error",
            stage="error",
            progress=0,
            message=f"Hata: {str(e)}",
        ))


@router.post("/chunk/stream")
async def chunk_text_stream(request: ChunkRequest):
    """Stream chunking progress via Server-Sent Events.
    
    This endpoint provides real-time progress updates during chunking,
    especially useful for large documents where processing takes time.
    
    ## Event Types
    
    - **progress**: Intermediate progress update
    - **complete**: Final result with all data
    - **error**: Error occurred during processing
    
    ## Stages
    
    1. initialization (0-5%)
    2. language_detection (5-10%)
    3. tokenization (10-20%)
    4. threshold_calculation (20-25%)
    5. embedding_generation (25-60%)
    6. chunking (60-80%)
    7. quality_analysis (80-95%)
    8. finalization (95-100%)
    
    ## Usage
    
    ```javascript
    const eventSource = new EventSource('/api/chunk/stream', {
        method: 'POST',
        body: JSON.stringify(request)
    });
    
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.progress, data.message);
    };
    ```
    """
    if request.strategy != ChunkingStrategy.SEMANTIC:
        # For non-semantic strategies, use regular endpoint
        raise HTTPException(
            status_code=400,
            detail="Streaming only supported for semantic chunking"
        )
    
    return StreamingResponse(
        _stream_semantic_chunking(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
