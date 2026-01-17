# Design Document - Reranker Integration

## Overview

This document describes the design for integrating reranker functionality into the RAG system. Rerankers improve search quality by re-scoring retrieved documents based on their semantic relevance to the query. The system will support multiple reranker providers (Cohere, Alibaba, Weaviate native) with optional usage configured per course.

**Key Design Principles:**
1. **Optional by Default**: Reranking is opt-in to maintain backward compatibility
2. **Provider Flexibility**: Support multiple reranker providers with consistent interface
3. **Graceful Degradation**: System continues working if reranker fails
4. **Performance First**: Minimize latency impact through caching and optimization
5. **Weaviate Integration**: Leverage Weaviate's native reranker capabilities

## Architecture

### High-Level Flow

```
User Query
    ↓
Hybrid Search (retrieve reranker_top_k results)
    ↓
[If reranker enabled]
    ↓
Reranker Service (re-score and re-order)
    ↓
Top search_top_k results
    ↓
LLM Generation
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG Service                            │
├─────────────────────────────────────────────────────────────┤
│  search_with_reranking()                                    │
│    ├─ hybrid_search(top_k=reranker_top_k)                   │
│    ├─ rerank_service.rerank(query, documents)               │
│    └─ return top search_top_k                               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   Reranker Service                          │
├─────────────────────────────────────────────────────────────┤
│  rerank(query, documents, provider, model, top_k)           │
│    ├─ route_to_provider()                                   │
│    ├─ cache_check()                                         │
│    ├─ provider.rerank()                                     │
│    ├─ cache_store()                                         │
│    └─ return ranked_documents                               │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Cohere     │  │   Alibaba    │  │   Weaviate   │
│   Reranker   │  │   Reranker   │  │   Reranker   │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Components and Interfaces

### 1. RerankService

**Purpose**: Unified interface for reranking across multiple providers

**Class Structure**:
```python
class RerankService:
    """Service for reranking search results using various providers."""
    
    def __init__(self):
        self._cohere_api_key = os.environ.get("COHERE_API_KEY")
        self._dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
        self._cohere_client = None
        self._alibaba_client = None
        self._cache = TTLCache(maxsize=1000, ttl=300)  # 5 min cache
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        provider: str,
        model: str = None,
        top_k: int = None
    ) -> List[Dict]:
        """Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of documents with 'content' field
            provider: Reranker provider (cohere/alibaba/weaviate)
            model: Provider-specific model name
            top_k: Number of top results to return
            
        Returns:
            Reranked documents with relevance scores
        """
        pass
    
    def _get_cohere_client(self):
        """Get or create Cohere client."""
        pass
    
    def _get_alibaba_client(self):
        """Get or create Alibaba client."""
        pass
    
    def _rerank_cohere(self, query, documents, model, top_k):
        """Rerank using Cohere API."""
        pass
    
    def _rerank_alibaba(self, query, documents, model, top_k):
        """Rerank using Alibaba API."""
        pass
    
    def _rerank_weaviate(self, query, documents, model, top_k):
        """Rerank using Weaviate native reranker."""
        pass
    
    def _generate_cache_key(self, query, documents, provider, model):
        """Generate cache key for reranking results."""
        pass
```

### 2. Cohere Reranker Implementation

**API**: Cohere Rerank API v2
**Models**: 
- `rerank-english-v3.0` (English only)
- `rerank-multilingual-v3.0` (100+ languages)

**Implementation**:
```python
def _rerank_cohere(
    self,
    query: str,
    documents: List[Dict],
    model: str = "rerank-multilingual-v3.0",
    top_k: int = None
) -> List[Dict]:
    """Rerank using Cohere API.
    
    Cohere API accepts:
    - query: str
    - documents: List[str] or List[Dict with 'text' field]
    - model: str
    - top_n: int (optional)
    - return_documents: bool (default True)
    
    Returns documents with relevance_score field.
    """
    client = self._get_cohere_client()
    
    # Extract text content from documents
    doc_texts = [doc.get('content', '') for doc in documents]
    
    # Call Cohere rerank API
    response = client.rerank(
        model=model,
        query=query,
        documents=doc_texts,
        top_n=top_k,
        return_documents=True
    )
    
    # Map results back to original documents
    reranked = []
    for result in response.results:
        doc = documents[result.index].copy()
        doc['relevance_score'] = result.relevance_score
        doc['rerank_index'] = result.index
        reranked.append(doc)
    
    return reranked
```

### 3. Alibaba Reranker Implementation

**API**: Alibaba DashScope Rerank API
**Models**: 
- `gte-rerank` (Chinese optimized)
- `gte-rerank-hybrid` (Multilingual)

**Implementation**:
```python
def _rerank_alibaba(
    self,
    query: str,
    documents: List[Dict],
    model: str = "gte-rerank",
    top_k: int = None
) -> List[Dict]:
    """Rerank using Alibaba DashScope API.
    
    Alibaba API structure similar to Cohere.
    """
    # Similar implementation to Cohere
    # Use Alibaba DashScope SDK
    pass
```

### 4. Weaviate Native Reranker

**Integration**: Weaviate reranker module
**Providers**: Cohere, Transformers, VoyageAI, JinaAI

**Configuration**:
```python
# Collection configuration with reranker
collection_config = {
    "class": "Document",
    "moduleConfig": {
        "reranker-cohere": {
            "model": "rerank-multilingual-v3.0"
        }
    }
}

# Query with reranking
results = client.query.get(
    "Document",
    ["content", "title"]
).with_hybrid(
    query=query,
    alpha=0.5
).with_additional([
    "distance",
    "rerank(query: $query) { score }"
]).with_limit(top_k).do()
```

**Implementation**:
```python
def _rerank_weaviate(
    self,
    query: str,
    documents: List[Dict],
    model: str = None,
    top_k: int = None
) -> List[Dict]:
    """Rerank using Weaviate native reranker.
    
    Note: This requires documents to already be in Weaviate.
    For external documents, use Cohere/Alibaba directly.
    """
    # Weaviate reranking happens during query
    # This method is for consistency but may not be used
    # if reranking is done in the search query itself
    pass
```

### 5. Course Settings Schema

**Database Migration**:
```python
# Add reranker fields to course_settings table
def upgrade():
    op.add_column('course_settings',
        sa.Column('enable_reranker', sa.Boolean(), default=False))
    op.add_column('course_settings',
        sa.Column('reranker_provider', sa.String(50), nullable=True))
    op.add_column('course_settings',
        sa.Column('reranker_model', sa.String(100), nullable=True))
    op.add_column('course_settings',
        sa.Column('reranker_top_k', sa.Integer(), default=100))
```

**Pydantic Schema**:
```python
class CourseSettingsUpdate(BaseModel):
    # ... existing fields ...
    enable_reranker: Optional[bool] = False
    reranker_provider: Optional[str] = None  # cohere/alibaba/weaviate
    reranker_model: Optional[str] = None
    reranker_top_k: Optional[int] = 100
    
    @validator('reranker_provider')
    def validate_provider(cls, v):
        if v and v not in ['cohere', 'alibaba', 'weaviate']:
            raise ValueError('Invalid reranker provider')
        return v
    
    @validator('reranker_top_k')
    def validate_top_k(cls, v):
        if v and (v < 10 or v > 1000):
            raise ValueError('reranker_top_k must be between 10 and 1000')
        return v
```

### 6. RAG Service Integration

**Updated Search Flow**:
```python
async def search_with_context(
    self,
    query: str,
    course_id: int,
    conversation_history: List[Dict] = None
) -> Dict:
    """Search with optional reranking."""
    
    # Get course settings
    settings = await get_course_settings(course_id)
    
    # Determine top_k for initial retrieval
    initial_top_k = (
        settings.reranker_top_k 
        if settings.enable_reranker 
        else settings.search_top_k
    )
    
    # Perform hybrid search
    search_results = await self.weaviate_service.hybrid_search(
        query=query,
        course_id=course_id,
        top_k=initial_top_k,
        alpha=settings.search_alpha
    )
    
    # Apply reranking if enabled
    if settings.enable_reranker and search_results:
        try:
            reranked = await self.rerank_service.rerank(
                query=query,
                documents=search_results,
                provider=settings.reranker_provider,
                model=settings.reranker_model,
                top_k=settings.search_top_k
            )
            search_results = reranked
            logger.info(f"Reranked {len(search_results)} results")
        except Exception as e:
            logger.error(f"Reranking failed: {e}, using original results")
            # Fall back to original results
            search_results = search_results[:settings.search_top_k]
    
    # Continue with LLM generation
    return await self.generate_response(query, search_results, settings)
```

## Data Models

### Reranked Document Structure

```python
{
    "id": "doc_123",
    "content": "Document text content...",
    "title": "Document Title",
    "metadata": {...},
    "original_score": 0.85,  # Original hybrid search score
    "relevance_score": 0.92,  # Reranker score
    "rerank_index": 0,  # Position after reranking
    "original_index": 5  # Original position before reranking
}
```

### Reranker Configuration

```python
RERANKER_MODELS = {
    "cohere": {
        "rerank-english-v3.0": {
            "name": "Cohere Rerank English v3.0",
            "languages": ["en"],
            "max_documents": 1000,
            "max_query_length": 2048
        },
        "rerank-multilingual-v3.0": {
            "name": "Cohere Rerank Multilingual v3.0",
            "languages": ["100+"],
            "max_documents": 1000,
            "max_query_length": 2048
        }
    },
    "alibaba": {
        "gte-rerank": {
            "name": "Alibaba GTE Rerank",
            "languages": ["zh", "en"],
            "max_documents": 500,
            "max_query_length": 1024
        },
        "gte-rerank-hybrid": {
            "name": "Alibaba GTE Rerank Hybrid",
            "languages": ["100+"],
            "max_documents": 500,
            "max_query_length": 1024
        }
    },
    "weaviate": {
        "rerank-cohere": {
            "name": "Weaviate Cohere Reranker",
            "languages": ["100+"],
            "requires_module": "reranker-cohere"
        }
    }
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Reranker Score Ordering

*For any* query and document set, after reranking, documents should be ordered by descending relevance_score.

**Validates: Requirements 7.4**

### Property 2: Document Preservation

*For any* reranking operation, all input documents should be present in the output (no documents lost or added).

**Validates: Requirements 8.3**

### Property 3: Fallback Consistency

*For any* reranking failure, the system should return the original search results without modification.

**Validates: Requirements 9.1**

### Property 4: Top-K Constraint

*For any* reranking operation with top_k parameter, the output should contain at most top_k documents.

**Validates: Requirements 7.4**

### Property 5: Cache Key Uniqueness

*For any* two different queries or document sets, the cache keys should be different.

**Validates: Requirements 10.2**

### Property 6: Configuration Validation

*For any* course settings update with enable_reranker=true, the system should reject if provider or model is not specified.

**Validates: Requirements 6.6**

### Property 7: Provider Routing

*For any* reranking request, the system should route to the correct provider based on the provider parameter.

**Validates: Requirements 1.2**

### Property 8: Score Range

*For any* reranked document, the relevance_score should be between 0 and 1.

**Validates: Requirements 8.2**

### Property 9: Metadata Preservation

*For any* reranking operation, original document metadata should be preserved in the output.

**Validates: Requirements 8.3**

### Property 10: Empty Input Handling

*For any* reranking request with empty document list, the system should return an empty list without error.

**Validates: Requirements 8.4**

## Error Handling

### Error Scenarios and Responses

1. **Missing API Key**
   - Error: `ValueError("COHERE_API_KEY not configured")`
   - Fallback: Return original search results
   - Log: Warning level

2. **Invalid Provider**
   - Error: `ValueError("Invalid reranker provider: {provider}")`
   - Fallback: Return original search results
   - Log: Error level

3. **API Rate Limit**
   - Error: `RateLimitError`
   - Fallback: Return original search results
   - Log: Warning level
   - Action: Implement exponential backoff

4. **API Timeout**
   - Timeout: 5 seconds
   - Fallback: Return original search results
   - Log: Warning level

5. **Invalid Model**
   - Error: `ValueError("Model {model} not available for {provider}")`
   - Fallback: Use default model for provider
   - Log: Warning level

6. **Network Error**
   - Error: `ConnectionError`
   - Fallback: Return original search results
   - Log: Error level
   - Action: Retry once

### Error Handling Pattern

```python
async def rerank_with_fallback(
    self,
    query: str,
    documents: List[Dict],
    settings: CourseSettings
) -> List[Dict]:
    """Rerank with graceful fallback."""
    
    if not settings.enable_reranker:
        return documents[:settings.search_top_k]
    
    try:
        # Attempt reranking
        reranked = await self.rerank_service.rerank(
            query=query,
            documents=documents,
            provider=settings.reranker_provider,
            model=settings.reranker_model,
            top_k=settings.search_top_k
        )
        
        # Log success metrics
        self._log_rerank_success(reranked, documents)
        
        return reranked
        
    except ValueError as e:
        # Configuration error
        logger.error(f"Reranker configuration error: {e}")
        return documents[:settings.search_top_k]
        
    except (RateLimitError, TimeoutError) as e:
        # Temporary error
        logger.warning(f"Reranker temporary error: {e}")
        return documents[:settings.search_top_k]
        
    except Exception as e:
        # Unexpected error
        logger.error(f"Reranker unexpected error: {e}", exc_info=True)
        return documents[:settings.search_top_k]
```

## Testing Strategy

### Unit Tests

1. **RerankService Tests**
   - Test provider routing
   - Test cache key generation
   - Test each provider implementation
   - Test error handling

2. **Cohere Reranker Tests**
   - Test API call formatting
   - Test response parsing
   - Test error scenarios
   - Test model selection

3. **Alibaba Reranker Tests**
   - Test API call formatting
   - Test response parsing
   - Test Chinese content handling

4. **Configuration Tests**
   - Test settings validation
   - Test default values
   - Test invalid configurations

### Property-Based Tests

Each correctness property should have a corresponding property-based test:

```python
@given(
    query=st.text(min_size=1, max_size=100),
    documents=st.lists(
        st.fixed_dictionaries({
            'content': st.text(min_size=1),
            'id': st.text(min_size=1)
        }),
        min_size=1,
        max_size=50
    )
)
def test_reranker_score_ordering(query, documents):
    """Property 1: Documents ordered by descending score."""
    reranked = rerank_service.rerank(
        query=query,
        documents=documents,
        provider="cohere",
        model="rerank-multilingual-v3.0"
    )
    
    scores = [doc['relevance_score'] for doc in reranked]
    assert scores == sorted(scores, reverse=True)
```

### Integration Tests

1. **End-to-End Search with Reranking**
   - Test complete search flow
   - Verify reranking improves relevance
   - Test fallback scenarios

2. **Provider Switching**
   - Test switching between providers
   - Verify consistent behavior

3. **Performance Tests**
   - Measure reranking latency
   - Test with various document counts
   - Verify cache effectiveness

### Test Configuration

- Minimum 100 iterations per property test
- Mock external API calls in unit tests
- Use real APIs in integration tests (with test keys)
- Tag tests: `@pytest.mark.reranker`

## Performance Considerations

### Latency Targets

- Reranking 100 documents: < 2 seconds
- Reranking 500 documents: < 5 seconds
- Cache hit: < 10ms

### Optimization Strategies

1. **Caching**
   - Cache reranking results for 5 minutes
   - Use query + document IDs as cache key
   - LRU eviction policy

2. **Batching**
   - Batch multiple reranking requests when possible
   - Respect provider batch limits

3. **Async Processing**
   - Use async/await for API calls
   - Parallel processing where possible

4. **Early Termination**
   - If reranking takes > 5s, return original results
   - Implement timeout at provider level

### Monitoring Metrics

```python
RERANKER_METRICS = {
    "rerank_calls_total": Counter,
    "rerank_latency_seconds": Histogram,
    "rerank_errors_total": Counter,
    "rerank_cache_hits": Counter,
    "rerank_cache_misses": Counter,
    "rerank_score_improvement": Histogram
}
```

## Security Considerations

1. **API Key Management**
   - Store API keys in environment variables
   - Never log API keys
   - Validate keys on startup

2. **Input Validation**
   - Sanitize query input
   - Validate document content
   - Limit document count per request

3. **Rate Limiting**
   - Implement per-user rate limits
   - Track API usage per course
   - Alert on unusual patterns

## Deployment Considerations

### Environment Variables

```bash
# Cohere
COHERE_API_KEY=your-cohere-key

# Alibaba
DASHSCOPE_API_KEY=your-dashscope-key

# Reranker Configuration
RERANKER_CACHE_TTL=300  # seconds
RERANKER_TIMEOUT=5  # seconds
RERANKER_MAX_DOCUMENTS=1000
```

### Database Migration

```bash
# Generate migration
alembic revision --autogenerate -m "Add reranker settings"

# Apply migration
alembic upgrade head
```

### Weaviate Configuration

```yaml
# docker-compose.yml
weaviate:
  environment:
    ENABLE_MODULES: 'reranker-cohere'
    COHERE_APIKEY: ${COHERE_API_KEY}
```

## Future Enhancements

1. **Additional Providers**
   - OpenRouter reranker (if available)
   - VoyageAI reranker
   - JinaAI reranker

2. **Advanced Features**
   - Multi-stage reranking
   - Ensemble reranking (combine multiple providers)
   - Custom reranker models

3. **Analytics**
   - A/B testing framework
   - Reranking effectiveness metrics
   - User satisfaction tracking

4. **Optimization**
   - Adaptive top_k selection
   - Smart caching strategies
   - Provider cost optimization

## References

- [Cohere Rerank API Documentation](https://docs.cohere.com/docs/reranking-quickstart)
- [Weaviate Reranker Module](https://weaviate.io/developers/weaviate/search/rerank)
- [Alibaba DashScope Documentation](https://help.aliyun.com/zh/dashscope/)
- [Reranking Best Practices](https://docs.cohere.com/v1/docs/reranking-with-cohere)
