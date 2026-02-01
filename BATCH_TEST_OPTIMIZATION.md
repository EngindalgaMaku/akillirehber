# Batch Test Performance Optimization

## Date: February 1, 2026

## Problem
- 100-question batch tests were taking ~120 minutes (2 hours)
- Sequential processing was stable but very slow
- Each test requires 19 API calls (13 LLM + 6 embedding)

## Optimizations Implemented

### 1. Embedding Cache
**What it does:**
- Caches embeddings for questions to avoid redundant API calls
- Uses cache key: `{embedding_model}:{question_text}`
- Shared across all tests in a batch

**Benefits:**
- Reduces embedding API calls for repeated or similar questions
- Saves ~0.5-1 second per cached embedding
- No quality impact - same embeddings are used

**Implementation:**
```python
embedding_cache = {}  # {text: embedding_vector}

# Check cache before generating embedding
cache_key = f"{course_settings.default_embedding_model}:{question}"
if cache_key in embedding_cache:
    query_vector = embedding_cache[cache_key]  # Cache hit
else:
    query_vector = embedding_service.get_embedding(...)
    embedding_cache[cache_key] = query_vector  # Store for future use
```

### 2. Small Batch Parallelism (3 Workers)
**What it does:**
- Processes 2-3 tests simultaneously instead of 1 at a time
- Uses ThreadPoolExecutor with max_workers=3

**Benefits:**
- 2-3x speedup compared to sequential processing
- More stable than 10 workers (which caused missing metrics)
- Balanced trade-off between speed and stability

**Why 3 workers?**
- 1 worker (sequential): Stable but slow (120 min)
- 10 workers: Fast but unstable (missing metrics, W&B conflicts)
- 3 workers: Sweet spot - stable AND faster

**Implementation:**
```python
import concurrent.futures

PARALLEL_WORKERS = 3
with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
    future_to_idx = {
        executor.submit(process_single_test, idx, test_case): idx
        for idx, test_case in enumerate(data.test_cases)
    }
    
    for future in concurrent.futures.as_completed(future_to_idx):
        result_data = future.result()
        # Process result...
```

## Expected Performance Improvement

### Before (Sequential):
- 100 tests = ~120 minutes
- ~72 seconds per test
- 0 parallelism

### After (3 Workers + Cache):
- **Estimated: 40-60 minutes** for 100 tests
- ~24-36 seconds per test (with cache hits)
- 2-3x speedup

### Breakdown:
1. **Parallelism (3 workers)**: 2-3x faster
2. **Embedding cache**: Additional 10-20% speedup for repeated questions
3. **Combined**: ~2.5-3x total speedup

## Stability Maintained

✅ **All existing features preserved:**
- Retry mechanism for missing metrics (2 attempts)
- Low score detection (faithfulness < 50%, relevancy < 40%)
- Critical metric validation (reject if missing after retries)
- W&B logging and tracking
- Thread-safe database operations

✅ **No quality degradation:**
- Same RAGAS metrics
- Same retry logic
- Same validation rules

## Testing Recommendations

1. **Start with small test** (10-20 questions) to verify stability
2. **Monitor logs** for:
   - Cache hit/miss rates
   - Missing metrics (should be rare)
   - W&B conflicts (should not occur)
3. **Compare results** with previous sequential runs to ensure quality
4. **Scale up** to 100 questions once verified

## Rollback Plan

If issues occur, revert to sequential processing:

```python
# Change this line:
PARALLEL_WORKERS = 3

# To this:
PARALLEL_WORKERS = 1  # Back to sequential
```

Or remove parallelism entirely and process in a simple loop.

## Files Modified

- `backend/app/routers/ragas.py` (lines ~2165-2550)
  - Added embedding cache
  - Changed from sequential to 3-worker parallel processing
  - Added cache statistics logging

## Next Steps

1. Test with 10-question batch
2. Verify all metrics are captured
3. Check W&B logging works correctly
4. Scale to 100 questions
5. Measure actual performance improvement
6. Adjust PARALLEL_WORKERS if needed (can try 2 or 4)

## Notes

- Cache is per-batch (not persistent across batches)
- Cache size logged at end: `Embedding cache size: X entries`
- Thread-safe with separate DB sessions per worker
- Async-compatible with proper await/yield patterns
