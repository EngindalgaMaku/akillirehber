# Design Document: RAGAS Reranker Metadata

## Overview

This design adds reranker metadata support to the RAGAS evaluation system, enabling tracking and reporting of reranker usage in evaluation results. The implementation follows a minimal, backward-compatible approach that extends existing data structures without breaking current functionality.

## Architecture

### Data Flow

```
Chat Endpoint (Backend)
    ↓ (includes reranker info if used)
RAGAS Evaluation Request
    ↓ (stores metadata)
RAGAS Service Processing
    ↓ (includes metadata in response)
RAGAS Evaluation Response
    ↓ (displays reranker status)
Frontend Display
```

### Key Principles

1. **Backward Compatibility**: All new fields are optional
2. **Minimal Changes**: Extend existing structures, don't replace them
3. **Clear Tracking**: Metadata flows through entire evaluation pipeline
4. **User Visibility**: Frontend clearly shows reranker usage

## Components and Interfaces

### 1. RAGAS Service Schema Updates

**File**: `ragas_service/main.py`

```python
class EvaluationInput(BaseModel):
    """Input for single question evaluation."""
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: List[str]
    evaluation_model: Optional[str] = None
    
    # NEW: Reranker metadata
    reranker_provider: Optional[str] = None  # "cohere", "alibaba", or None
    reranker_model: Optional[str] = None     # Model name or None


class EvaluationOutput(BaseModel):
    """Output metrics for single question evaluation."""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None
    error: Optional[str] = None
    
    # NEW: Reranker metadata
    reranker_used: bool = False
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None


class BatchEvaluationOutput(BaseModel):
    """Output for batch evaluation."""
    results: List[EvaluationOutput]
    total_processed: int
    total_errors: int
    
    # NEW: Reranker statistics
    reranker_usage: Dict[str, int] = {}  # {"cohere": 5, "alibaba": 3, "none": 2}
```

### 2. RAGAS Evaluation Logic Updates

**Function**: `evaluate_with_ragas()`

```python
def evaluate_with_ragas(input_data: EvaluationInput) -> EvaluationOutput:
    """Evaluate using RAGAS library with configured LLM provider."""
    # ... existing evaluation logic ...
    
    # After evaluation, add reranker metadata to result
    result = EvaluationOutput(
        faithfulness=safe_float(df['faithfulness'].iloc[0]),
        answer_relevancy=safe_float(df['answer_relevancy'].iloc[0]),
        context_precision=safe_float(df['context_precision'].iloc[0]),
        context_recall=safe_float(df['context_recall'].iloc[0]),
        answer_correctness=safe_float(df['answer_correctness'].iloc[0]),
        
        # Add reranker metadata
        reranker_used=bool(input_data.reranker_provider),
        reranker_provider=input_data.reranker_provider,
        reranker_model=input_data.reranker_model
    )
    
    # Log reranker usage
    if input_data.reranker_provider:
        logger.info(
            "[RAGAS] Evaluation with reranker: %s/%s",
            input_data.reranker_provider,
            input_data.reranker_model or "default"
        )
    
    return result
```

**Function**: `evaluate_simple()`

```python
def evaluate_simple(input_data: EvaluationInput) -> EvaluationOutput:
    """Simple heuristic-based evaluation as fallback."""
    # ... existing evaluation logic ...
    
    return EvaluationOutput(
        faithfulness=min(faithfulness, 1.0),
        answer_relevancy=min(relevancy, 1.0),
        context_precision=min(precision, 1.0),
        context_recall=min(recall, 1.0),
        answer_correctness=min(correctness, 1.0),
        
        # Add reranker metadata
        reranker_used=bool(input_data.reranker_provider),
        reranker_provider=input_data.reranker_provider,
        reranker_model=input_data.reranker_model
    )
```

**Function**: `evaluate_batch()`

```python
async def evaluate_batch(input_data: BatchEvaluationInput):
    """Evaluate a batch of questions."""
    results = []
    errors = 0
    reranker_usage = {}
    
    for item in input_data.items:
        result = await evaluate_question(item)
        results.append(result)
        
        if result.error:
            errors += 1
        
        # Track reranker usage
        provider = result.reranker_provider or "none"
        reranker_usage[provider] = reranker_usage.get(provider, 0) + 1
    
    return BatchEvaluationOutput(
        results=results,
        total_processed=len(results),
        total_errors=errors,
        reranker_usage=reranker_usage
    )
```

### 3. Backend Integration

**File**: `backend/app/routers/ragas.py` (or wherever RAGAS is called)

When calling RAGAS evaluation, include reranker metadata:

```python
# Get course settings
settings = get_or_create_settings(db, course_id)

# Prepare RAGAS evaluation request
evaluation_request = {
    "question": test_case.question,
    "ground_truth": test_case.ground_truth,
    "generated_answer": generated_answer,
    "retrieved_contexts": contexts,
    "evaluation_model": ragas_model,
    
    # Include reranker metadata if enabled
    "reranker_provider": settings.reranker_provider if settings.enable_reranker else None,
    "reranker_model": settings.reranker_model if settings.enable_reranker else None
}

# Call RAGAS service
response = requests.post(
    f"{RAGAS_SERVICE_URL}/evaluate",
    json=evaluation_request
)
```

### 4. Database Schema Updates

**File**: `backend/app/models/db_models.py`

Add reranker metadata to test run results:

```python
class TestRun(Base):
    """Model for storing RAGAS test runs."""
    __tablename__ = "test_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    # ... existing fields ...
    
    # NEW: Reranker metadata
    reranker_used = Column(Boolean, default=False, nullable=False)
    reranker_provider = Column(String(50), nullable=True)
    reranker_model = Column(String(100), nullable=True)
```

**Migration**: Create Alembic migration to add new columns.

### 5. Frontend Display Updates

**File**: `frontend/src/app/dashboard/courses/[id]/components/ragas-tab.tsx`

Display reranker information in test results:

```typescript
interface TestResult {
  // ... existing fields ...
  reranker_used: boolean;
  reranker_provider?: string;
  reranker_model?: string;
}

// In the results display component
{result.reranker_used && (
  <div className="flex items-center gap-2 text-xs text-amber-600 bg-amber-50 px-2 py-1 rounded">
    <Sparkles className="w-3 h-3" />
    <span>
      Reranked: {result.reranker_provider}/{result.reranker_model || 'default'}
    </span>
  </div>
)}

// In batch results summary
{batchResults.reranker_usage && (
  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
    <h4 className="font-medium text-blue-900 mb-2">Reranker Usage</h4>
    <div className="space-y-1 text-sm text-blue-800">
      {Object.entries(batchResults.reranker_usage).map(([provider, count]) => (
        <div key={provider}>
          {provider === 'none' ? 'No reranker' : provider}: {count} tests
        </div>
      ))}
    </div>
  </div>
)}
```

## Data Models

### EvaluationInput (Extended)

```python
{
    "question": "What is RAG?",
    "ground_truth": "RAG is Retrieval-Augmented Generation...",
    "generated_answer": "RAG stands for...",
    "retrieved_contexts": ["Context 1", "Context 2"],
    "evaluation_model": "openai/gpt-4o-mini",
    "reranker_provider": "cohere",  # NEW
    "reranker_model": "rerank-multilingual-v3.0"  # NEW
}
```

### EvaluationOutput (Extended)

```python
{
    "faithfulness": 0.95,
    "answer_relevancy": 0.88,
    "context_precision": 0.92,
    "context_recall": 0.85,
    "answer_correctness": 0.90,
    "error": null,
    "reranker_used": true,  # NEW
    "reranker_provider": "cohere",  # NEW
    "reranker_model": "rerank-multilingual-v3.0"  # NEW
}
```

### BatchEvaluationOutput (Extended)

```python
{
    "results": [...],
    "total_processed": 10,
    "total_errors": 0,
    "reranker_usage": {  # NEW
        "cohere": 7,
        "alibaba": 2,
        "none": 1
    }
}
```

## Error Handling

### Invalid Reranker Metadata

```python
# In evaluate_with_ragas()
if input_data.reranker_provider:
    valid_providers = ["cohere", "alibaba"]
    if input_data.reranker_provider not in valid_providers:
        logger.warning(
            "Invalid reranker provider: %s. Valid: %s",
            input_data.reranker_provider,
            valid_providers
        )
        # Continue with evaluation, but log warning
```

### Missing Metadata

```python
# Default to None if not provided (backward compatibility)
reranker_provider = input_data.reranker_provider or None
reranker_model = input_data.reranker_model or None
```

## Testing Strategy

### Unit Tests

1. **Test metadata passthrough**:
   - Input with reranker metadata → Output includes metadata
   - Input without reranker metadata → Output has reranker_used=False

2. **Test validation**:
   - Valid provider → Accepted
   - Invalid provider → Warning logged, evaluation continues
   - Missing provider → Defaults to None

3. **Test batch statistics**:
   - Mixed reranker usage → Correct counts in summary
   - All same provider → Single entry in summary
   - No reranker → "none" entry in summary

### Integration Tests

1. **End-to-end flow**:
   - Chat with reranker → RAGAS receives metadata → Frontend displays status
   - Chat without reranker → RAGAS receives null → Frontend shows no reranker

2. **Backward compatibility**:
   - Old API calls (no metadata) → Still work
   - Old database records → Migration handles gracefully

## Migration Strategy

### Phase 1: RAGAS Service Update
1. Update schemas (backward compatible)
2. Update evaluation functions
3. Deploy RAGAS service
4. Test with and without metadata

### Phase 2: Backend Integration
1. Create database migration
2. Update RAGAS API calls to include metadata
3. Deploy backend
4. Verify metadata flows correctly

### Phase 3: Frontend Display
1. Update TypeScript interfaces
2. Add reranker status display
3. Add batch statistics display
4. Deploy frontend

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system.*

### Property 1: Metadata Preservation
*For any* evaluation request with reranker metadata, the response should include the same reranker_provider and reranker_model values.
**Validates: Requirements 1.1, 2.2, 2.3**

### Property 2: Reranker Used Flag Consistency
*For any* evaluation response, reranker_used should be true if and only if reranker_provider is not None.
**Validates: Requirements 2.2**

### Property 3: Backward Compatibility
*For any* evaluation request without reranker metadata, the evaluation should complete successfully with reranker_used=False.
**Validates: Requirements 4.1, 4.2, 4.3**

### Property 4: Batch Statistics Accuracy
*For any* batch evaluation, the sum of reranker_usage counts should equal total_processed.
**Validates: Requirements 3.4**

## Performance Considerations

- Metadata adds ~50 bytes per evaluation (negligible)
- No additional API calls or processing
- Database migration is non-blocking (nullable columns)
- Frontend rendering impact is minimal (conditional display)

## Security Considerations

- Metadata is informational only (no security impact)
- No sensitive data in metadata
- Validation prevents injection attacks (limited to known providers)

## References

- RAGAS Documentation: https://docs.ragas.io/
- Reranker Integration Spec: `.kiro/specs/reranker-integration/`
- Pydantic Documentation: https://docs.pydantic.dev/
