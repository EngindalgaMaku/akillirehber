"""Chat API endpoints for RAG-based conversation."""

from typing import List, Optional

import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import User, Document, Chunk, ChatMessageDB
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkReference,
    ChatHistoryResponse,
    ChatHistoryClearResponse,
)
from app.services.auth_service import get_current_user
from app.services.course_service import (
    verify_course_access,
    get_or_create_settings,
)
from app.services.embedding_service import get_embedding_service
from app.services.llm_service import (
    get_llm_service,
    LLMProviderError,
    LLMConfigurationError,
    LLMAPIError,
)
from app.services.weaviate_service import get_weaviate_service, SearchResult
from app.services.chat_validation_service import ChatValidationService
from app.services.rerank_service import get_rerank_service

router = APIRouter(prefix="/api", tags=["chat"])


def clean_context_text(text: str) -> str:
    """Temizlenmiş context metni döndürür.
    
    Chunk'lardaki gereksiz whitespace karakterlerini temizler:
    - Birden fazla \n -> tek boşluk
    - \t -> tek boşluk  
    - Birden fazla boşluk -> tek boşluk
    - Başta/sonda boşluk -> kaldır
    
    Bu iyileştirme hem LLM'in daha iyi cevap vermesini hem de
    RAGAS metriklerinin daha yüksek olmasını sağlar.
    """
    import re
    
    if not text:
        return text
    
    # \n ve \t karakterlerini boşluğa çevir
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    # Birden fazla boşluğu tek boşluğa indir
    text = re.sub(r'\s+', ' ', text)
    
    # Başta ve sonda boşluk varsa kaldır
    text = text.strip()
    
    return text


def build_context(results: List[SearchResult], db: Session) -> str:
    """Build context string from search results with enhanced validation and cleaning."""
    if not results:
        return ""

    context_parts = []
    for i, result in enumerate(results, 1):
        # Get document name with validation
        doc = db.query(Document).filter(Document.id == result.document_id).first()
        if not doc:
            # Skip results that reference non-existent documents
            continue

        doc_name = doc.original_filename

        # Validate chunk exists and content matches
        chunk = (
            db.query(Chunk)
            .filter(Chunk.id == result.chunk_id)
            .filter(Chunk.document_id == result.document_id)
            .first()
        )

        # Use database content if available for consistency
        content = chunk.content if chunk else result.content
        
        # 🔥 YENİ: Context'i temizle - whitespace karakterlerini kaldır
        content = clean_context_text(content)

        context_parts.append(
            f"[Kaynak {i}: {doc_name}]\n{content}\n"
        )

    return "\n".join(context_parts)


def build_references(
    results: List[SearchResult], db: Session
) -> List[ChunkReference]:
    """Build chunk references from search results with enhanced validation."""
    references = []
    for result in results:
        # Validate document exists
        doc = db.query(Document).filter(Document.id == result.document_id).first()
        if not doc:
            # Skip results that reference non-existent documents
            continue

        doc_name = doc.original_filename

        # Validate chunk exists
        chunk = (
            db.query(Chunk)
            .filter(Chunk.id == result.chunk_id)
            .filter(Chunk.document_id == result.document_id)
            .first()
        )

        # Use database content if available for consistency
        content = chunk.content if chunk else result.content

        # Truncate content for preview
        preview = content[:200] + "..." if len(content) > 200 else content

        references.append(ChunkReference(
            document_id=result.document_id,
            document_name=doc_name,
            chunk_index=result.chunk_index,
            content_preview=preview,
            full_content=content,
            score=result.score
        ))

    return references


@router.post("/courses/{course_id}/chat", response_model=ChatResponse)
async def chat_with_course(
    course_id: int,
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Chat with course materials using RAG.

    Performs hybrid search on course documents and generates
    a response using the retrieved context.
    """
    verify_course_access(db, course_id, current_user)

    # Validate chunk availability before processing
    chat_validation = ChatValidationService(db)
    chunk_validation = chat_validation.validate_chunk_availability(course_id)
    
    # Debug log for troubleshooting
    print(f"Chunk validation for course {course_id}: {chunk_validation}")

    if not chunk_validation.get("valid", False):
        message = (
            "Bu derste henüz sohbet için hazır doküman bulunmuyor. "
            "Lütfen dokümanların işlendiğinden emin olun veya öğretmeninize "
            "danışın."
        )
        db.add(
            ChatMessageDB(
                course_id=course_id,
                user_id=current_user.id,
                role="user",
                content=request.message,
            )
        )
        db.add(
            ChatMessageDB(
                course_id=course_id,
                user_id=current_user.id,
                role="assistant",
                content=message,
                sources=[],
            )
        )
        db.commit()
        return ChatResponse(message=message, sources=[])

    # Get course settings for LLM configuration
    settings = get_or_create_settings(db, course_id)
    
    # Debug log for embedding model
    print(f"Course embedding model: {settings.default_embedding_model}")

    # Get query embedding using course's embedding model
    embedding_service = get_embedding_service()
    
    # DEBUG: Log the embedding model being used
    print(f"[CHAT DEBUG] Using embedding model: {settings.default_embedding_model}")
    print(f"[CHAT DEBUG] Course ID: {course_id}")
    
    query_vector = embedding_service.get_embedding(
        request.message,
        model=settings.default_embedding_model
    )
    
    # Debug log for query vector
    print(f"[CHAT DEBUG] Query vector length: {len(query_vector) if query_vector else 0}")

    # Search for relevant chunks using course settings
    weaviate_service = get_weaviate_service()
    
    # Use course settings for search parameters
    search_alpha = settings.search_alpha
    search_top_k = settings.search_top_k

    if request.search_type == "vector":
        results = weaviate_service.vector_search(
            course_id=course_id,
            query_vector=query_vector,
            limit=search_top_k
        )
    elif request.search_type == "keyword":
        results = weaviate_service.keyword_search(
            course_id=course_id,
            query=request.message,
            limit=search_top_k
        )
    else:  # hybrid
        results = weaviate_service.hybrid_search(
            course_id=course_id,
            query=request.message,
            query_vector=query_vector,
            alpha=search_alpha,
            limit=search_top_k
        )

    # Apply reranking if enabled
    if settings.enable_reranker and settings.reranker_provider:
        try:
            start_time = time.time()
            
            # Determine initial top_k for reranking
            reranker_top_k = settings.reranker_top_k or 10
            
            # If we have results, prepare them for reranking
            if results:
                original_count = len(results)
                original_scores = [r.score for r in results]
                
                # Convert SearchResult objects to dicts for reranker
                documents_for_reranking = [
                    {
                        "id": str(r.chunk_id),
                        "content": r.content,
                        "score": r.score,
                        "document_id": r.document_id,
                        "chunk_index": r.chunk_index
                    }
                    for r in results
                ]
                
                # Call reranker service
                rerank_service = get_rerank_service()
                reranked_docs = rerank_service.rerank(
                    query=request.message,
                    documents=documents_for_reranking,
                    provider=settings.reranker_provider,
                    model=settings.reranker_model,
                    top_k=min(reranker_top_k, len(documents_for_reranking))
                )
                
                # Convert reranked results back to SearchResult objects
                results = [
                    SearchResult(
                        chunk_id=int(doc["id"]),
                        document_id=doc["document_id"],
                        content=doc["content"],
                        chunk_index=doc["chunk_index"],
                        score=doc.get("relevance_score", doc.get("score", 0))
                    )
                    for doc in reranked_docs
                ]
                
                # Calculate metrics
                rerank_latency = time.time() - start_time
                new_scores = [r.score for r in results]
                avg_score_improvement = (
                    (sum(new_scores) / len(new_scores)) - (sum(original_scores) / len(original_scores))
                    if new_scores and original_scores else 0
                )
                
                print(f"[RERANKER] Success: {settings.reranker_provider}/{settings.reranker_model or 'default'}")
                print(f"[RERANKER] Latency: {rerank_latency:.3f}s")
                print(f"[RERANKER] Results: {original_count} -> {len(results)}")
                print(f"[RERANKER] Avg score improvement: {avg_score_improvement:+.4f}")
        except Exception as e:
            # Log error but continue with original results (fallback already handled in rerank_service)
            print(f"[RERANKER] Failed: {e}")
            print("[RERANKER] Falling back to original search results")

    # Filter results by minimum relevance score
    min_score = getattr(settings, 'min_relevance_score', 0.0) or 0.0
    
    # Debug: Print scores before filtering
    if results:
        scores = [r.score for r in results]
        print(f"[DEBUG] Search scores before filtering: {scores}")
        print(f"[DEBUG] Min relevance score setting: {min_score}")
    
    if min_score > 0 and results:
        original_count = len(results)
        results = [r for r in results if r.score >= min_score]
        print(f"[DEBUG] Filtered from {original_count} to {len(results)} results")

    # Validate source attribution for results (temporarily disabled)
    # TODO: Fix content mismatch between DB and vector store
    """
    if results:
        attribution_validation = chat_validation.validate_source_attribution(
            course_id, results
        )
        # Filter out invalid results
        if not attribution_validation.get("valid", False):
            # Keep only valid results
            valid_results = []
            for i, result in enumerate(results):
                detail = attribution_validation.get("details", [])[i] if i < len(attribution_validation.get("details", [])) else {}
                if detail.get("valid", False):
                    valid_results.append(result)
            results = valid_results
    """

    # Build context and references
    context = build_context(results, db)
    references = build_references(results, db)

    # Handle no results
    if not results:
        message = (
            "Bu konuyla ilgili ders materyallerinde bilgi bulunamadı. "
            "Lütfen farklı bir soru sorun veya öğretmeninize danışın."
        )
        db.add(
            ChatMessageDB(
                course_id=course_id,
                user_id=current_user.id,
                role="user",
                content=request.message,
            )
        )
        db.add(
            ChatMessageDB(
                course_id=course_id,
                user_id=current_user.id,
                role="assistant",
                content=message,
                sources=[],
            )
        )
        db.commit()
        return ChatResponse(message=message, sources=[])

    # Persist user message
    db.add(
        ChatMessageDB(
            course_id=course_id,
            user_id=current_user.id,
            role="user",
            content=request.message,
        )
    )

    # Build messages for LLM
    # Use custom system prompt from course settings if available, otherwise use default
    default_system_prompt = """You are an educational assistant. Answer student questions using the provided context information.

Rules:
1. Use only the information from the provided context
2. If the context doesn't contain the answer, say so clearly
3. Provide accurate and helpful responses
4. Keep your answers clear and understandable
5. Reference sources when necessary"""
    
    active_template = getattr(settings, "active_prompt_template", None)
    if active_template is not None and getattr(
        active_template,
        "content",
        None,
    ):
        system_prompt = active_template.content
    else:
        system_prompt = (
            settings.system_prompt if settings.system_prompt else default_system_prompt
        )

    messages = [{"role": "system", "content": system_prompt}]

# ... (rest of the code remains the same)
    # Add chat history
    if request.history:
        for msg in request.history[-10:]:  # Last 10 messages
            messages.append({"role": msg.role, "content": msg.content})

    # Add current message with context
    user_message = f"""Context:
{context}

Question: {request.message}"""

    messages.append({"role": "user", "content": user_message})

    # Call LLM
    try:
        # Get LLM service from course settings
        llm_service = get_llm_service(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )

        start_time = time.time()
        assistant_message = llm_service.generate_response(messages)
        response_time_ms = int((time.time() - start_time) * 1000)

        db.add(
            ChatMessageDB(
                course_id=course_id,
                user_id=current_user.id,
                role="assistant",
                content=assistant_message,
                sources=[r.model_dump() for r in references],
                response_time_ms=response_time_ms,
            )
        )
        db.commit()

        return ChatResponse(
            message=assistant_message,
            sources=references
        )

    except (LLMProviderError, LLMConfigurationError, LLMAPIError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM error: {str(e)}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        ) from e


@router.get(
    "/courses/{course_id}/chat/history",
    response_model=ChatHistoryResponse,
)
async def get_chat_history(
    course_id: int,
    limit: int = 20,
    before_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    verify_course_access(db, course_id, current_user)

    limit = max(1, min(limit, 100))

    query = (
        db.query(ChatMessageDB)
        .filter(ChatMessageDB.course_id == course_id)
        .filter(ChatMessageDB.user_id == current_user.id)
    )

    if before_id is not None:
        query = query.filter(ChatMessageDB.id < before_id)

    rows = query.order_by(ChatMessageDB.id.desc()).limit(limit + 1).all()

    has_more = len(rows) > limit
    rows = rows[:limit]
    rows.reverse()

    for row in rows:
        if row.sources is not None and not isinstance(row.sources, list):
            row.sources = None

    return ChatHistoryResponse(messages=rows, has_more=has_more)


@router.delete(
    "/courses/{course_id}/chat/history",
    response_model=ChatHistoryClearResponse,
)
async def clear_chat_history(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    verify_course_access(db, course_id, current_user)

    deleted_count = (
        db.query(ChatMessageDB)
        .filter(ChatMessageDB.course_id == course_id)
        .filter(ChatMessageDB.user_id == current_user.id)
        .delete(synchronize_session=False)
    )
    db.commit()

    return ChatHistoryClearResponse(success=True, deleted_count=deleted_count)


@router.get("/courses/{course_id}/chat/diagnostics")
async def get_chat_diagnostics(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get comprehensive chat integration diagnostics for a course.

    Returns detailed information about chunk availability, data consistency,
    search functionality, and source attribution.
    """
    verify_course_access(db, course_id, current_user)

    chat_validation = ChatValidationService(db)
    diagnostics = chat_validation.get_chat_integration_diagnostics(course_id)

    return diagnostics


@router.get("/courses/{course_id}/chat/validate-chunks")
async def validate_chunk_availability(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Validate chunk availability for chat functionality.

    Checks if documents have been processed and chunks are available
    for chat functionality.
    """
    verify_course_access(db, course_id, current_user)

    chat_validation = ChatValidationService(db)
    validation_result = chat_validation.validate_chunk_availability(course_id)

    return validation_result


@router.get("/courses/{course_id}/chat/validate-consistency")
async def validate_data_consistency(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Validate data consistency between database and vector store.

    Checks if chunk counts match between PostgreSQL and Weaviate
    for all documents in the course.
    """
    verify_course_access(db, course_id, current_user)

    chat_validation = ChatValidationService(db)
    consistency_result = chat_validation.validate_chunk_data_consistency(course_id)

    return consistency_result


@router.post("/courses/{course_id}/chat/test-search")
async def test_search_functionality(
    course_id: int,
    test_query: str = "test",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Test chat search functionality with a sample query.

    Tests vector, keyword, and hybrid search to ensure all
    search types are working correctly.
    """
    verify_course_access(db, course_id, current_user)

    chat_validation = ChatValidationService(db)
    search_test = chat_validation.test_chat_search_functionality(
        course_id, test_query
    )

    return search_test
