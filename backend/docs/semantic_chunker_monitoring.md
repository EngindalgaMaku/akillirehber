# Semantic Chunker Monitoring Guide

## Overview

This guide covers monitoring and metrics for the enhanced semantic chunker.

## Health Check Endpoint

### Endpoint
```
GET /health/embedding-providers
```

### Response
```json
{
  "status": "healthy",
  "providers": {
    "openrouter": {
      "available": true,
      "status": "healthy"
    },
    "openai": {
      "available": true,
      "status": "healthy"
    }
  },
  "cache": {
    "enabled": true,
    "hit_rate": 0.75,
    "entry_count": 150
  }
}
```

## Key Metrics

### Processing Metrics
- `chunking_duration_seconds`: Time to process text
- `chunks_created_total`: Total chunks created
- `fallback_used_total`: Fallback strategy usage count

### Cache Metrics
- `cache_hit_rate`: Percentage of cache hits
- `cache_entries`: Current cache size
- `cache_evictions_total`: Evicted entries count

### Provider Metrics
- `provider_requests_total`: API calls per provider
- `provider_errors_total`: Errors per provider
- `provider_latency_seconds`: Response time per provider

## Prometheus Configuration

### Example prometheus.yml
```yaml
scrape_configs:
  - job_name: 'semantic-chunker'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Custom Metrics (Add to application)
```python
from prometheus_client import Counter, Histogram, Gauge

# Processing metrics
chunking_duration = Histogram(
    'chunking_duration_seconds',
    'Time spent chunking text',
    ['strategy', 'language']
)

chunks_created = Counter(
    'chunks_created_total',
    'Total chunks created',
    ['strategy']
)

# Cache metrics
cache_hit_rate = Gauge(
    'cache_hit_rate',
    'Current cache hit rate'
)

cache_entries = Gauge(
    'cache_entries',
    'Current number of cache entries'
)

# Provider metrics
provider_requests = Counter(
    'provider_requests_total',
    'Total provider API requests',
    ['provider', 'status']
)
```

## Grafana Dashboard

### Recommended Panels

1. **Processing Overview**
   - Chunking requests per minute
   - Average processing time
   - Strategy distribution

2. **Cache Performance**
   - Hit rate over time
   - Cache size
   - Eviction rate

3. **Provider Health**
   - Request success rate
   - Latency percentiles
   - Error rate by provider

4. **Quality Metrics**
   - Average coherence score
   - Low-quality chunk percentage
   - Recommendation frequency

### Example Dashboard JSON
```json
{
  "title": "Semantic Chunker Dashboard",
  "panels": [
    {
      "title": "Requests per Minute",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(chunks_created_total[1m])"
        }
      ]
    },
    {
      "title": "Cache Hit Rate",
      "type": "gauge",
      "targets": [
        {
          "expr": "cache_hit_rate"
        }
      ]
    }
  ]
}
```

## Alerting Rules

### Critical Alerts
```yaml
groups:
  - name: semantic-chunker
    rules:
      - alert: HighErrorRate
        expr: rate(provider_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate in semantic chunker

      - alert: LowCacheHitRate
        expr: cache_hit_rate < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: Cache hit rate below 50%

      - alert: HighLatency
        expr: histogram_quantile(0.95, chunking_duration_seconds) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: 95th percentile latency above 10s
```

## Logging

### Structured Logging
The chunker uses structured logging with context:
```python
{
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "INFO",
    "message": "Chunking completed",
    "context": {
        "text_length": 5000,
        "strategy": "semantic",
        "chunks_created": 10,
        "processing_time_ms": 250,
        "language": "turkish",
        "cache_hit": true
    }
}
```

### Log Levels
- `ERROR`: Chunking failures, provider errors
- `WARNING`: Fallback usage, low quality chunks
- `INFO`: Successful operations, metrics
- `DEBUG`: Detailed processing steps

## Best Practices

1. **Monitor cache hit rate** - Should be >70%
2. **Track fallback usage** - Indicates provider issues
3. **Alert on high latency** - May indicate API issues
4. **Review quality metrics** - Identify problematic documents
5. **Check provider health** - Ensure redundancy works
