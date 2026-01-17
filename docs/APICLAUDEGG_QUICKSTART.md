# API Claude.gg Quick Start

Get started with API Claude.gg provider in 5 minutes.

## What is API Claude.gg?

API Claude.gg is a multi-model API provider that gives you access to:
- **GPT-5** and **O3** models (latest OpenAI)
- **Grok** models (xAI)
- **DeepSeek** models (reasoning and chat)
- **Gemini** models (Google)

All through a single API endpoint with one API key.

## Setup Steps

### 1. Add API Key to .env

```bash
# Same key as Claude.gg
APICLAUDEGG_API_KEY=sk-542ca4c71cf0451a9b92b584eeba6520
```

### 2. Restart RAGAS Service

```bash
docker-compose restart ragas
```

### 3. Verify Setup

```bash
# Check if provider is available
curl http://localhost:8001/providers | jq '.providers[] | select(.name=="apiclaudegg")'
```

Expected output:
```json
{
  "name": "apiclaudegg",
  "available": true,
  "is_free": false,
  "default_model": "gpt-5",
  "priority": 3
}
```

### 4. Test Evaluation

```bash
python test_apiclaudegg.py
```

Expected output:
```
✅ API Claude.gg API is working!
✅ Evaluation Metrics:
  faithfulness: 0.6667
  answer_relevancy: 1.0000
  context_precision: 0.5000
  context_recall: 1.0000
  answer_correctness: 0.8422
```

## Usage

### Via Frontend

1. Go to **Dashboard → RAGAS Değerlendirme**
2. Click **Settings** icon (⚙️)
3. Select **apiclaudegg** from dropdown
4. Choose model (default: gpt-5)
5. Click **Kaydet** (Save)

### Via API

```bash
# Set provider
curl -X POST http://localhost:8001/settings \
  -H "Content-Type: application/json" \
  -d '{"provider": "apiclaudegg", "model": "gpt-5"}'

# Run evaluation
curl -X POST http://localhost:8001/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is AI?",
    "ground_truth": "AI is artificial intelligence...",
    "generated_answer": "AI refers to...",
    "retrieved_contexts": ["Context 1", "Context 2"],
    "evaluation_model": "gpt-5"
  }'
```

## Popular Models

### For Best Quality
- `gpt-5` - Latest GPT-5 (default)
- `gpt-o3` - OpenAI O3 reasoning model
- `deepseek-r1` - DeepSeek reasoning model

### For Speed
- `gpt-5-mini` - Faster, cheaper GPT-5
- `gemini-2.0-flash` - Fast Gemini model
- `grok-3-mini` - Compact Grok model

### For Specific Tasks
- `deepseek-chat` - Conversational tasks
- `gemini-3-pro` - Complex reasoning
- `grok-4` - Latest xAI model

## Troubleshooting

### Provider Not Available

```bash
# Check environment variable
echo $APICLAUDEGG_API_KEY

# Restart service
docker-compose restart ragas

# Check logs
docker-compose logs ragas | grep apiclaudegg
```

### Evaluation Fails

Try a different model:
```bash
curl -X POST http://localhost:8001/settings \
  -H "Content-Type: application/json" \
  -d '{"provider": "apiclaudegg", "model": "gpt-4o"}'
```

## Next Steps

- Read [APICLAUDEGG_SETUP.md](APICLAUDEGG_SETUP.md) for detailed configuration
- Explore all available models
- Compare results across different models
- Set up batch evaluations

## Support

- API Documentation: https://api.claude.gg/
- Test Script: `test_apiclaudegg.py`
- Service Logs: `docker-compose logs ragas`
