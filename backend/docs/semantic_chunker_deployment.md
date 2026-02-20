# Semantic Chunker Deployment Plan

## Pre-Deployment Checklist

### Code Quality
- [ ] All 209+ tests passing
- [ ] No critical linting errors
- [ ] Code review completed
- [ ] Documentation updated

### Environment
- [ ] API keys configured (OPENROUTER_API_KEY or OPENAI_API_KEY)
- [ ] NLTK data downloaded (punkt_tab)
- [ ] Dependencies installed
- [ ] Memory limits configured

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks pass
- [ ] Manual testing completed

## Deployment Steps

### Step 1: Prepare Environment
```bash
# Set environment variables
export OPENROUTER_API_KEY="your-key"
export OPENAI_API_KEY="your-key"  # Optional fallback

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt_tab')"
```

### Step 2: Run Tests
```bash
# Run all semantic chunker tests
pytest tests/test_semantic_chunker_enhancement.py \
       tests/test_semantic_chunker_integration.py \
       tests/test_error_handling.py \
       tests/test_chunk_quality.py \
       tests/test_qa_detector.py \
       tests/test_embedding_provider.py \
       tests/test_phase6_integration.py \
       tests/test_performance_benchmarks.py \
       -v
```

### Step 3: Deploy Application
```bash
# Using Docker
docker build -t semantic-chunker .
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  semantic-chunker

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Step 4: Verify Deployment
```bash
# Health check
curl http://localhost:8000/health/embedding-providers

# Test chunking
curl -X POST http://localhost:8000/api/chunk \
  -H "Content-Type: application/json" \
  -d '{"text": "Test sentence.", "strategy": "semantic"}'
```

## Gradual Rollout Strategy

### Phase 1: Shadow Mode (Week 1)
- Deploy alongside existing system
- Route 0% traffic to new system
- Compare results in background
- Monitor for errors

### Phase 2: Canary Release (Week 2)
- Route 5% traffic to new system
- Monitor metrics closely
- Compare quality scores
- Gather user feedback

### Phase 3: Gradual Increase (Week 3-4)
- Increase to 25%, then 50%, then 75%
- Monitor at each stage
- Address any issues
- Document learnings

### Phase 4: Full Rollout (Week 5)
- Route 100% traffic
- Decommission old system
- Update documentation
- Announce completion

## Feature Flags

Enable/disable features without redeployment:

```python
# Environment variables
ENABLE_QA_DETECTION=true
ENABLE_ADAPTIVE_THRESHOLD=true
ENABLE_CACHE=true
```

Or via API:
```json
{
  "enable_qa_detection": false,
  "enable_adaptive_threshold": false
}
```

## Rollback Plan

### Immediate Rollback
```bash
# Revert to previous version
docker run -p 8000:8000 semantic-chunker:previous

# Or disable new features
export ENABLE_QA_DETECTION=false
export ENABLE_ADAPTIVE_THRESHOLD=false
```

### Rollback Triggers
- Error rate > 5%
- Latency > 10s (95th percentile)
- Quality score < 0.5 average
- User complaints

### Rollback Steps
1. Switch traffic to old system
2. Investigate root cause
3. Fix issues
4. Re-test thoroughly
5. Re-deploy with fixes

## Monitoring During Deployment

### Key Metrics to Watch
- Request success rate (target: >99%)
- Average latency (target: <5s for 5K chars)
- Cache hit rate (target: >70%)
- Fallback usage (target: <5%)
- Quality scores (target: >0.5 average)

### Alerts
- Set up alerts for anomalies
- Have on-call engineer ready
- Prepare communication channels

## Post-Deployment

### Verification
- [ ] All endpoints responding
- [ ] Metrics being collected
- [ ] Logs being generated
- [ ] Alerts configured

### Documentation
- [ ] Update API documentation
- [ ] Update user guides
- [ ] Notify stakeholders
- [ ] Update changelog

### Cleanup
- [ ] Remove old code (after stabilization)
- [ ] Archive old documentation
- [ ] Update CI/CD pipelines
