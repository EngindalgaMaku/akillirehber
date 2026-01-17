# Semantic Chunker User Guide

## Overview

The Enhanced Semantic Chunker provides intelligent text chunking with support for:
- Multi-language processing (Turkish and English)
- Q&A pair detection and preservation
- Adaptive similarity thresholds
- Embedding caching for performance
- Quality metrics and recommendations

## Quick Start

### Basic Usage

```python
from app.services.chunker import SemanticChunker

chunker = SemanticChunker()
chunks = chunker.chunk("Your text here...")
```

### API Endpoint

```bash
curl -X POST "http://localhost:8000/api/chunk" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text to chunk...",
    "strategy": "semantic"
  }'
```

## Configuration Options

### Feature Flags

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_qa_detection` | `true` | Detect and keep Q&A pairs together |
| `enable_adaptive_threshold` | `true` | Calculate threshold based on text |
| `enable_cache` | `true` | Cache embeddings for performance |
| `include_quality_metrics` | `false` | Include quality analysis in response |

### Chunking Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `chunk_size` | 500 | 10-10000 | Target chunk size in characters |
| `overlap` | 50 | 0+ | Overlap between chunks |
| `similarity_threshold` | auto | 0.0-1.0 | Similarity threshold (auto if null) |
| `min_chunk_size` | 150 | 10+ | Minimum characters per chunk |
| `max_chunk_size` | 2000 | up to 10000 | Maximum characters per chunk |
| `buffer_size` | 1 | 0-5 | Context sentences for similarity |

## Examples

### Turkish Document Processing

```json
{
  "text": "Türkiye'nin başkenti Ankara'da toplantı yapıldı. Dr. Ahmet konuşma yaptı.",
  "strategy": "semantic",
  "enable_qa_detection": true
}
```

### With Quality Metrics

```json
{
  "text": "Your document text...",
  "strategy": "semantic",
  "include_quality_metrics": true
}
```

Response includes:
- `quality_metrics`: Per-chunk coherence scores
- `quality_report`: Overall quality analysis
- `recommendations`: Actionable suggestions

### Custom Threshold

```json
{
  "text": "Your text...",
  "strategy": "semantic",
  "similarity_threshold": 0.6,
  "enable_adaptive_threshold": false
}
```

## Quality Metrics

### Semantic Coherence
- Measures how related sentences within a chunk are
- Range: 0.0 (unrelated) to 1.0 (highly related)
- Target: > 0.5 for good quality

### Inter-Chunk Similarity
- Measures connection between consecutive chunks
- High similarity (> 0.8) suggests chunks should be merged

### Recommendations
The system provides:
- Merge recommendations for highly similar chunks
- Split recommendations for low-coherence chunks
- Overall quality score

## Troubleshooting

### Common Issues

1. **Empty chunks**: Increase `min_chunk_size`
2. **Chunks too large**: Decrease `max_chunk_size`
3. **Q&A pairs split**: Enable `enable_qa_detection`
4. **Poor coherence**: Try different `similarity_threshold`

### Error Handling

The chunker automatically falls back to simpler strategies:
1. Semantic → Sentence-based → Fixed-size

Check `fallback_used` and `warning_message` in response.

## Performance Tips

1. Enable caching for repeated texts
2. Use batch processing for multiple documents
3. Adjust `buffer_size` based on document type
4. Monitor cache hit rate via health endpoint
