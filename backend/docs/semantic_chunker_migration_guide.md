# Semantic Chunker Migration Guide

## Overview

This guide helps you migrate from the previous semantic chunker implementation to the enhanced version with multi-language support, Q&A detection, and quality metrics.

## Breaking Changes

**None** - The enhanced semantic chunker is fully backward compatible.

## New Features

### 1. Multi-Language Support
- Automatic language detection (Turkish, English, Mixed)
- Language-specific sentence tokenization
- Turkish abbreviation handling (30+ abbreviations)

### 2. Q&A Detection
- Automatic question detection
- Answer pairing with semantic similarity
- Q&A pairs kept together in chunks

### 3. Adaptive Threshold
- Automatic threshold calculation based on text characteristics
- Considers vocabulary diversity and sentence length
- No manual tuning required

### 4. Embedding Caching
- In-memory caching with TTL
- Reduces API calls by >70%
- Automatic cache eviction

### 5. Quality Metrics
- Semantic coherence scores
- Inter-chunk similarity
- Merge/split recommendations

## Migration Steps

### Step 1: Update API Calls

**Before (still works):**
```json
{
  "text": "Your text...",
  "strategy": "semantic",
  "similarity_threshold": 0.5
}
```

**After (recommended):**
```json
{
  "text": "Your text...",
  "strategy": "semantic",
  "enable_qa_detection": true,
  "enable_adaptive_threshold": true,
  "enable_cache": true
}
```

### Step 2: Handle New Response Fields

New optional fields in response:
- `detected_language`: Detected text language
- `adaptive_threshold_used`: Calculated threshold value
- `processing_time_ms`: Processing duration
- `fallback_used`: Fallback strategy if used
- `warning_message`: Any warnings

### Step 3: Enable Quality Metrics (Optional)

Add `include_quality_metrics: true` to get:
- Per-chunk quality scores
- Overall quality report
- Actionable recommendations

### Step 4: Update Error Handling

New error handling provides:
- Automatic fallback to simpler strategies
- Detailed error messages
- Warning notifications

## Code Migration

### Python SDK

**Before:**
```python
chunker = SemanticChunker()
chunks = chunker.chunk(text, similarity_threshold=0.5)
```

**After:**
```python
chunker = SemanticChunker(
    use_provider_manager=True,
    enable_cache=True,
    enable_qa_detection=True,
    enable_adaptive_threshold=True
)
chunks = chunker.chunk(text)  # threshold auto-calculated
```

### Using Error Handling Wrapper

```python
from app.services.chunker import chunk_with_error_handling

result = chunk_with_error_handling(
    text=text,
    strategy=ChunkingStrategy.SEMANTIC,
    enable_qa_detection=True
)

if result.success:
    chunks = result.chunks
else:
    print(f"Error: {result.error}")
    print(f"Fallback used: {result.fallback_used}")
```

## Backward Compatibility

All existing code continues to work:
- Default parameters unchanged
- Response format compatible
- No required changes

## Feature Flags

Disable new features if needed:
```json
{
  "enable_qa_detection": false,
  "enable_adaptive_threshold": false,
  "enable_cache": false
}
```

## Testing Migration

1. Run existing tests - should pass unchanged
2. Add tests for new features
3. Verify Turkish text handling
4. Check quality metrics accuracy

## Support

For issues or questions:
- Check the user guide
- Review error messages
- Enable debug logging
