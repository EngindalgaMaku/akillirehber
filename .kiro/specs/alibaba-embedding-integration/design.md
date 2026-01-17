# Design Document: Alibaba Embedding Integration

## Overview

This design document outlines the integration of Alibaba's text-embedding-v4 model into the existing RAG system. The system currently supports only OpenRouter's OpenAI embedding models. This enhancement adds Alibaba Cloud's DashScope as an alternative embedding provider, leveraging its OpenAI-compatible API for seamless integration.

The integration follows a provider-routing pattern where the embedding service detects the model prefix and routes requests to the appropriate API endpoint (OpenRouter for "openai/*" models, DashScope for "alibaba/*" models).

## Architecture

### Current Architecture

The system uses a singleton `EmbeddingService` class that:
- Connects to OpenRouter API using OpenAI SDK
- Generates embeddings for single texts and batches
- Returns embedding dimensions for known models
- Stores embedding model selection in `CourseSettings`

### Proposed Architecture

The enhanced architecture maintains backward compatibility while adding multi-provider support:

```
┌─────────────────────────────────────────────────────────┐
│                   Course Settings                        │
│  - default_embedding_model: "openai/*" or "alibaba/*"   │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              EmbeddingService (Enhanced)                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │  get_embedding(text, model)                       │  │
│  │    ├─ if model.startswith("alibaba/")            │  │
│  │    │    └─> _get_alibaba_client()                │  │
│  │    └─ else                                        │  │
│  │         └─> _get_openrouter_client()             │  │
│  └───────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │ OpenRouter Client   │  │  Alibaba DashScope       │  │
│  │ (OpenAI SDK)        │  │  Client (OpenAI SDK)     │  │
│  │                     │  │                          │  │
│  │ Base URL:           │  │  Base URL:               │  │
│  │ openrouter.ai/api/v1│  │  dashscope-intl.        │  │
│  │                     │  │  aliyuncs.com/          │  │
│  │ API Key:            │  │  compatible-mode/v1     │  │
│  │ OPENROUTER_API_KEY  │  │                          │  │
│  └─────────────────────┘  │  API Key:                │  │
│                            │  DASHSCOPE_API_KEY       │  │
│                            └──────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Provider Detection Strategy

The service uses model name prefixes to route requests:
- Models starting with "openai/" → OpenRouter API
- Models starting with "alibaba/" → DashScope API
- Default (no prefix or unknown) → OpenRouter API (backward compatibility)

## Components and Interfaces

### 1. Enhanced EmbeddingService

**File**: `backend/app/services/embedding_service.py`

**New Constants**:
```python
DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
ALIBABA_MODEL_PREFIX = "alibaba/"
OPENAI_MODEL_PREFIX = "openai/"
```

**New Methods**:
```python
def _get_alibaba_client(self) -> OpenAI:
    """Get or create OpenAI client for Alibaba DashScope.
    
    Returns:
        OpenAI client configured for DashScope
        
    Raises:
        ValueError: If DASHSCOPE_API_KEY is not set
    """
    
def _get_provider_for_model(self, model: str) -> str:
    """Determine which provider to use based on model name.
    
    Args:
        model: Model identifier (e.g., "alibaba/text-embedding-v4")
        
    Returns:
        Provider name: "alibaba" or "openrouter"
    """
    
def _get_client_for_model(self, model: str) -> OpenAI:
    """Get appropriate client based on model.
    
    Args:
        model: Model identifier
        
    Returns:
        Configured OpenAI client for the provider
    """
```

**Modified Methods**:
```python
def get_embedding(self, text: str, model: str = None) -> List[float]:
    """Get embedding for a single text.
    
    Now supports multiple providers based on model prefix.
    Routes to OpenRouter for "openai/*" models.
    Routes to DashScope for "alibaba/*" models.
    """
    
def get_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
    """Get embeddings for multiple texts.
    
    Now supports multiple providers with appropriate batch sizes.
    """
    
def get_embedding_dimension(self, model: str = None) -> int:
    """Get embedding dimension for a model.
    
    Extended to support Alibaba models:
    - alibaba/text-embedding-v4: 1024 (default), 768, 512, 256, 128, 64
    """
```

### 2. Database Schema

**No changes required**. The existing `CourseSettings.default_embedding_model` column (String(255)) already supports storing "alibaba/text-embedding-v4".

### 3. API Endpoints

**File**: `backend/app/routers/course_settings.py`

**No changes required**. The existing endpoints already support updating `default_embedding_model` with any string value.

### 4. Frontend Integration

**File**: `frontend/src/app/teacher/courses/[id]/settings/page.tsx` (or equivalent)

**Changes**: Add "alibaba/text-embedding-v4" to the embedding model dropdown options.

```typescript
const embeddingModels = [
  { value: "openai/text-embedding-3-small", label: "OpenAI text-embedding-3-small (1536 dim)" },
  { value: "openai/text-embedding-3-large", label: "OpenAI text-embedding-3-large (3072 dim)" },
  { value: "openai/text-embedding-ada-002", label: "OpenAI text-embedding-ada-002 (1536 dim)" },
  { value: "alibaba/text-embedding-v4", label: "Alibaba text-embedding-v4 (1024 dim)" },
];
```

## Data Models

### CourseSettings Model

**Existing Schema** (no changes):
```python
class CourseSettings(Base):
    __tablename__ = "course_settings"
    
    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), unique=True)
    default_embedding_model = Column(String(255), default="openai/text-embedding-3-small")
    # ... other fields
```

### Document Model

**Existing Schema** (no changes):
```python
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    embedding_model = Column(String(255), nullable=True)  # Stores which model was used
    # ... other fields
```

The `embedding_model` field already tracks which model was used for each document, enabling dimension compatibility checks.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Provider Routing Based on Model Prefix

*For any* model name, when generating embeddings, the service should route to DashScope API if the model starts with "alibaba/", and to OpenRouter API otherwise.

**Validates: Requirements 3.1, 1.3, 3.2, 3.3**

### Property 2: Embedding Model Persistence

*For any* course, when the embedding model is updated to a specific value, retrieving the course settings should return that same embedding model value.

**Validates: Requirements 1.2**

### Property 3: Consistent Embedding Format

*For any* text and any supported embedding model (OpenRouter or Alibaba), the returned embedding should be a list of floating-point numbers with the correct dimension for that model.

**Validates: Requirements 3.4**

### Property 4: Document Embedding Model Tracking

*For any* document, after generating embeddings with a specific model, the document's embedding_model field should store that model identifier.

**Validates: Requirements 4.3**

### Property 5: Batch Processing Consistency

*For any* list of texts and any supported embedding model, calling get_embeddings should return the same number of embedding vectors as input texts, maintaining order.

**Validates: Requirements 5.1, 5.4**

### Property 6: Batch Size Constraints

*For any* batch of texts larger than the provider's batch size limit, the service should split the batch into multiple API calls and combine results correctly.

**Validates: Requirements 5.2**

### Property 7: Backward Compatibility with Existing Settings

*For any* existing course with a configured embedding model, loading the course settings should return the original model without modification.

**Validates: Requirements 6.2**

### Property 8: Graceful Degradation Without Alibaba Key

*For any* OpenRouter embedding model, when DASHSCOPE_API_KEY is not configured, embedding generation should succeed using OpenRouter.

**Validates: Requirements 2.2, 6.4**

## Error Handling

### Error Categories

1. **Configuration Errors**
   - Missing API keys (OPENROUTER_API_KEY or DASHSCOPE_API_KEY)
   - Invalid API key format (detected by API response)

2. **API Errors**
   - Rate limiting (429 status code)
   - Authentication failures (401 status code)
   - Invalid requests (400 status code)
   - Server errors (500+ status codes)

3. **Network Errors**
   - Connection timeouts
   - DNS resolution failures
   - Network unreachable

### Error Handling Strategy

**Missing API Key**:
```python
if not self._dashscope_api_key:
    raise ValueError(
        "DASHSCOPE_API_KEY environment variable is required "
        "for Alibaba embedding models. Please configure the API key "
        "or use an OpenRouter model instead."
    )
```

**API Errors**:
```python
try:
    response = client.embeddings.create(...)
except OpenAIError as e:
    # Log with provider context
    logger.error(
        f"Embedding API error from {provider}: {e}",
        extra={"model": model, "provider": provider}
    )
    raise
```

**Provider-Specific Error Messages**:
- Include provider name in all error messages
- Include model name for debugging
- Provide actionable guidance (e.g., "check API key", "verify rate limits")

### Logging Strategy

**Success Logging**:
```python
logger.info(
    f"Generated embeddings using {provider}",
    extra={
        "model": model,
        "provider": provider,
        "text_count": len(texts),
        "dimension": dimension
    }
)
```

**Error Logging**:
```python
logger.error(
    f"Failed to generate embeddings from {provider}",
    extra={
        "model": model,
        "provider": provider,
        "error_type": type(e).__name__,
        "error_message": str(e)
    }
)
```

## Testing Strategy

### Dual Testing Approach

This feature requires both unit tests and property-based tests to ensure comprehensive coverage:

**Unit Tests**: Verify specific examples, edge cases, and error conditions
- Test specific model routing (e.g., "alibaba/text-embedding-v4" → DashScope)
- Test missing API key error messages
- Test default embedding model selection
- Test dimension lookup for known models
- Test error logging with correct context

**Property Tests**: Verify universal properties across all inputs
- Test routing logic for any model name pattern
- Test embedding format consistency across providers
- Test batch processing for any list size
- Test settings persistence for any model value
- Test graceful degradation for any OpenRouter model

### Property-Based Testing Configuration

**Framework**: Use `pytest` with `hypothesis` library for Python property-based testing

**Test Configuration**:
- Minimum 100 iterations per property test
- Each test tagged with feature name and property number
- Tag format: `# Feature: alibaba-embedding-integration, Property N: [property text]`

**Example Property Test Structure**:
```python
from hypothesis import given, strategies as st
import pytest

@pytest.mark.property_test
@given(model_name=st.text(min_size=1))
def test_provider_routing_property(model_name):
    """
    Feature: alibaba-embedding-integration, Property 1: Provider Routing
    
    For any model name, routing should be deterministic based on prefix.
    """
    service = EmbeddingService()
    provider = service._get_provider_for_model(model_name)
    
    if model_name.startswith("alibaba/"):
        assert provider == "alibaba"
    else:
        assert provider == "openrouter"
```

### Test Coverage Requirements

**Unit Test Coverage**:
- All error conditions (missing keys, invalid responses)
- All edge cases (empty text, single text, large batches)
- All provider-specific logic (DashScope vs OpenRouter)
- Configuration and initialization
- Dimension lookup for all supported models

**Property Test Coverage**:
- Provider routing for all model patterns
- Embedding format for all providers
- Batch processing for all batch sizes
- Settings persistence for all model values
- Graceful degradation scenarios

### Integration Testing

**Test Scenarios**:
1. End-to-end embedding generation with Alibaba model
2. End-to-end embedding generation with OpenRouter model
3. Course settings update and retrieval
4. Document embedding with model tracking
5. Batch processing with mixed text lengths
6. Error handling with missing API keys
7. Error handling with invalid API keys

**Mock Strategy**:
- Mock external API calls for unit tests
- Use real API calls for integration tests (with test API keys)
- Mock network errors for error handling tests

## Implementation Notes

### API Compatibility

Alibaba's DashScope provides an OpenAI-compatible API endpoint, which means:
- We can use the same OpenAI SDK for both providers
- Request/response formats are identical
- Only the base URL and API key differ

This significantly simplifies the implementation as we don't need a separate client library.

### Dimension Support

Alibaba text-embedding-v4 supports multiple dimensions:
- 2048, 1536, 1024 (default), 768, 512, 256, 128, 64

For this initial implementation, we'll use the default 1024 dimensions. Future enhancements could allow dimension selection in course settings.

### Rate Limiting

**OpenRouter**: 100 texts per batch
**Alibaba DashScope**: 10 texts per batch

The service must respect these limits when processing batches.

### Regional Endpoints

Alibaba provides different endpoints for different regions:
- International: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- China (Beijing): `https://dashscope.aliyuncs.com/compatible-mode/v1`

For this implementation, we'll use the international endpoint. Regional selection could be added as a future enhancement.

### Environment Variables

**Required for OpenRouter** (existing):
- `OPENROUTER_API_KEY`

**Required for Alibaba** (new):
- `DASHSCOPE_API_KEY`

Both keys should be documented in `.env.example`.

### Migration Strategy

**No database migration required**. The existing schema already supports storing any embedding model name.

**No data migration required**. Existing documents with OpenRouter embeddings continue to work. New documents can use either provider.

**Configuration migration**: Add `DASHSCOPE_API_KEY` to environment variables for deployments that want to use Alibaba models.

## Security Considerations

1. **API Key Storage**: Both API keys stored as environment variables, never in code or database
2. **API Key Validation**: Keys validated by API calls, not stored or logged
3. **Error Messages**: Error messages should not expose API key values
4. **Logging**: Ensure API keys are not logged in debug or error logs

## Performance Considerations

1. **Batch Size**: Respect provider-specific batch limits to avoid API errors
2. **Connection Pooling**: Reuse OpenAI client instances (already implemented via singleton pattern)
3. **Timeout Handling**: Set appropriate timeouts for API calls
4. **Retry Logic**: Consider implementing retry logic for transient failures (future enhancement)

## Future Enhancements

1. **Dimension Selection**: Allow users to select embedding dimensions in course settings
2. **Regional Endpoint Selection**: Allow selection of Alibaba regional endpoints
3. **Additional Providers**: Framework supports adding more providers (e.g., Cohere, Voyage AI)
4. **Retry Logic**: Implement exponential backoff for transient failures
5. **Caching**: Cache embeddings for frequently used texts
6. **Monitoring**: Add metrics for API call success rates, latencies, and costs
