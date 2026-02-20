# VoyageAI Rate Limit Fix

## Problem
The embedding service was encountering frequent rate limit errors from VoyageAI API:
```
Embedding failed: VoyageAI rate limit exceeded. Please try again later.
```

## Root Cause
The original implementation had several issues:
1. **Batch size too large**: 32 texts per batch was overwhelming VoyageAI's rate limits
2. **Insufficient retry attempts**: Only 5 retries with 3-second base delay
3. **No jitter in backoff**: All retries used deterministic delays, causing synchronized retries
4. **Minimal delays between batches**: Only 1-2 seconds between batch requests

## Solution

### 1. Reduced Batch Size
- **Before**: `VOYAGE_BATCH_SIZE = 32`
- **After**: `VOYAGE_BATCH_SIZE = 10`
- **Impact**: Significantly reduces the number of API calls per batch, staying within rate limits

### 2. Enhanced Retry Logic
- **Increased max retries**: From 5 to 8 attempts
- **Increased base delay**: From 3 seconds to 5 seconds
- **Added jitter**: Random delay between 0.5-2.0s (single) or 1.0-3.0s (batch) added to exponential backoff
- **Better error messages**: More informative logging with batch numbers and retry counts

### 3. Improved Batch Delays
- **Before**: 1.0-2.0 seconds between batches
- **After**: 2.0-4.0 seconds between batches
- **Impact**: Gives VoyageAI more time to reset rate limits between batches

### 4. Added Random Import
- Added `import random` at module level to support jitter in retry delays

## Changes Made

### File: `backend/app/services/embedding_service.py`

#### Import Addition
```python
import random  # Added for jitter in retry delays
```

#### Configuration Update
```python
VOYAGE_BATCH_SIZE = 10  # Further reduced to prevent rate limiting (was 32)
```

#### Single Text Embedding (`get_embedding`)
- Increased `max_retries` from 5 to 8
- Increased `base_delay` from 3 to 5 seconds
- Added jitter: `delay = base_delay * (2 ** attempt) + random.uniform(0.5, 2.0)`
- Enhanced logging with more detailed error messages

#### Batch Text Embedding (`get_embeddings`)
- Increased `max_retries` from 5 to 8
- Increased `base_delay` from 3 to 5 seconds
- Increased batch delay from 1.0-2.0s to 2.0-4.0s
- Added jitter: `delay = base_delay * (2 ** attempt) + random.uniform(1.0, 3.0)`
- Added debug logging for batch delays
- Enhanced error messages with batch numbers

## Benefits

1. **Reduced Rate Limit Errors**: Smaller batch size and longer delays significantly reduce the likelihood of hitting rate limits
2. **Better Recovery**: More retry attempts with exponential backoff give more chances to succeed
3. **Avoids Synchronization**: Jitter prevents multiple requests from retrying simultaneously
4. **Better Observability**: Enhanced logging helps diagnose issues and track retry behavior
5. **Graceful Degradation**: System continues to work even under heavy load

## Usage Recommendations

### For High-Volume Processing
- Consider using alternative embedding providers (OpenRouter, Cohere, Jina AI) for large batches
- Implement caching to avoid re-embedding identical texts
- Process documents in smaller chunks with delays between chunks

### For Optimal Performance
- Monitor logs for rate limit warnings
- Adjust batch size based on your VoyageAI plan limits
- Use the `embedding_cache` service to cache embeddings

### Configuration Options

You can further tune the behavior by modifying these constants in `embedding_service.py`:

```python
VOYAGE_BATCH_SIZE = 10  # Reduce further (e.g., 5) if still hitting limits
# In the VoyageAI sections:
max_retries = 8  # Increase for more retries
base_delay = 5  # Increase for longer initial delays
```

## Monitoring

Watch for these log messages:
- `VoyageAI rate limit hit, retrying in X seconds` - Indicates rate limit was hit
- `VoyageAI: Waiting X seconds before batch N/M` - Shows delays between batches
- `VoyageAI: Successfully processed batch N/M with N texts` - Confirms successful processing

## Testing

To test the changes:
1. Process a document with VoyageAI embeddings
2. Monitor logs for retry behavior
3. Verify that embeddings complete successfully
4. Check for rate limit errors (should be significantly reduced)

## Related Files

- `backend/app/services/embedding_service.py` - Main embedding service with VoyageAI support
- `backend/app/services/embedding_cache.py` - Caching service to reduce API calls
- `backend/app/services/embedding_provider.py` - Provider abstraction with fallback support

## Future Improvements

1. **Dynamic Rate Limiting**: Implement token bucket algorithm for client-side rate limiting
2. **Provider Fallback**: Automatically switch to alternative providers when VoyageAI is rate-limited
3. **Adaptive Batch Size**: Dynamically adjust batch size based on success/failure rate
4. **Metrics Collection**: Track rate limit hits and retry statistics
5. **Configuration via Environment Variables**: Make retry parameters configurable

## Support

If you continue to experience rate limit issues:
1. Check your VoyageAI account limits and upgrade if needed
2. Consider using a different embedding provider
3. Implement caching to reduce API calls
4. Contact VoyageAI support for higher rate limits
