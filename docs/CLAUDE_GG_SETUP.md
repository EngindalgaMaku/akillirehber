# Claude.gg API Integration Guide

## Overview

Claude.gg provides access to Anthropic's Claude models through an OpenAI-compatible API. This integration allows you to use Claude models for RAGAS evaluation in your RAG system.

## Available Models

- `claude-opus-4-5` - Most capable (default)
- `claude-sonnet-4-5` - Balanced performance
- `claude-sonnet-4` - Fast and efficient
- `claude-3-7-sonnet` - Previous generation
- `claude-haiku-4-5` - Fastest model

## Setup Instructions

### 1. Add API Key to Environment

Add your Claude.gg API key to your `.env` file:

```bash
# Claude.gg API Configuration
CLAUDEGG_API_KEY=sk-542ca4c71cf0451a9b92b584eeba6520
```

### 2. Configure Provider (Optional)

By default, the system will auto-select the best available provider. To force Claude.gg:

```bash
# Force Claude.gg as the RAGAS provider
RAGAS_PROVIDER=claudegg

# Optional: Override the default model
RAGAS_MODEL=claude-sonnet-4-5
```

### 3. Restart Services

```bash
docker-compose down
docker-compose up -d
```

## Usage

### Check Provider Status

```bash
curl http://localhost:8001/providers
```

Response:
```json
{
  "providers": [
    {
      "name": "claudegg",
      "available": true,
      "is_free": false,
      "default_model": "claude-sonnet-4",
      "priority": 2
    }
  ],
  "current": {
    "provider": "claudegg",
    "model": "claude-sonnet-4",
    "is_free": false
  }
}
```

### Change Provider at Runtime

```bash
curl -X POST http://localhost:8001/settings \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "claudegg",
    "model": "claude-opus-4-5"
  }'
```

### Evaluate with Specific Model

You can override the model per request:

```bash
curl -X POST http://localhost:8001/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "ground_truth": "Machine learning is...",
    "generated_answer": "ML is...",
    "retrieved_contexts": ["Context 1", "Context 2"],
    "evaluation_model": "claude-opus-4-5"
  }'
```

## API Endpoints

### Anthropic Compatible
- `POST /v1/messages` - Chat completions
- `GET /v1/models` - List available models
- `POST /v1/messages/count_tokens` - Token counting

### OpenAI Compatible
- `POST /v1/chat/completions` - Chat completions
- `GET /v1/models` - List available models

## Priority System

The system automatically selects providers based on priority:

1. **OpenRouter** (priority 1) - Best quality, has embeddings
2. **Claude.gg** (priority 2) - High quality Claude models
3. **Groq** (priority 3) - Free tier, fast inference
4. **OpenAI** (priority 4) - Fallback option

## Important Notes

### Embeddings
⚠️ **Claude.gg does NOT provide embedding models.** The system will automatically use OpenRouter or OpenAI for embeddings, even when using Claude.gg for LLM evaluation.

### Authentication
Claude.gg uses OpenAI-compatible authentication:
- Header: `Authorization: Bearer YOUR_API_KEY`
- Or: `x-api-key: YOUR_API_KEY`

### Base URL
```
https://app.claude.gg/v1
```

## Troubleshooting

### Provider Not Available
```bash
# Check if API key is set
curl http://localhost:8001/providers

# Check health endpoint
curl http://localhost:8001/health
```

### Evaluation Errors
If evaluation fails with Claude.gg, the system will automatically fall back to:
1. Other available providers
2. Simple heuristic evaluation

### View Logs
```bash
docker-compose logs ragas-service
```

## Cost Considerations

Claude.gg is a paid service. Monitor your usage through their dashboard:
- Dashboard: https://app.claude.gg/

## Support

For Claude.gg specific issues:
- Website: https://app.claude.gg/
- API Documentation: Check their API docs for rate limits and pricing

For integration issues:
- Check RAGAS service logs
- Verify API key is correct
- Ensure network connectivity to app.claude.gg
