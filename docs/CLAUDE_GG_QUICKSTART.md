# Claude.gg Quick Start Guide

## ✅ What Was Added

Claude.gg API has been successfully integrated as a new LLM provider for RAGAS evaluation. You can now use Anthropic's Claude models (Opus, Sonnet, Haiku) for evaluating your RAG system.

## 🚀 Quick Setup (3 Steps)

### 1. Add API Key to `.env`

```bash
# Add this line to your .env file
CLAUDEGG_API_KEY=sk-542ca4c71cf0451a9b92b584eeba6520
```

### 2. Restart Services

```bash
docker-compose down
docker-compose up -d
```

### 3. Verify Installation

```bash
# Check if Claude.gg is available
curl http://localhost:8001/providers

# Or run the test script
python test_claude_gg.py
```

## 📋 Available Models

- `claude-opus-4-5` - Most capable (default)
- `claude-sonnet-4-5` - Balanced (recommended)
- `claude-sonnet-4` - Fast
- `claude-3-7-sonnet` - Previous gen
- `claude-haiku-4-5` - Fastest

## 🎯 Usage Examples

### Option 1: Set as Default Provider

```bash
# In .env file
RAGAS_PROVIDER=claudegg
RAGAS_MODEL=claude-sonnet-4-5
```

### Option 2: Change at Runtime

```bash
curl -X POST http://localhost:8001/settings \
  -H "Content-Type: application/json" \
  -d '{"provider": "claudegg", "model": "claude-opus-4-5"}'
```

### Option 3: Per-Request Override

```bash
curl -X POST http://localhost:8001/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is AI?",
    "ground_truth": "AI is...",
    "generated_answer": "Artificial Intelligence...",
    "retrieved_contexts": ["Context 1"],
    "evaluation_model": "claude-opus-4-5"
  }'
```

## 🔍 Check Status

```bash
# Health check
curl http://localhost:8001/health

# List all providers
curl http://localhost:8001/providers

# Current settings
curl http://localhost:8001/settings
```

## ⚠️ Important Notes

1. **Embeddings**: Claude.gg doesn't provide embeddings. The system automatically uses OpenRouter or OpenAI for embeddings.

2. **Priority**: Claude.gg has priority 2 (after OpenRouter). To force it, set `RAGAS_PROVIDER=claudegg`.

3. **Cost**: Claude.gg is a paid service. Monitor usage at https://app.claude.gg/

## 📚 Full Documentation

See `CLAUDE_GG_SETUP.md` for detailed documentation including:
- API endpoints
- Troubleshooting
- Advanced configuration
- Cost considerations

## 🧪 Test Script

Run the included test script to verify everything works:

```bash
python test_claude_gg.py
```

This will:
- ✅ Check RAGAS service health
- ✅ List available providers
- ✅ Test Claude.gg API directly
- ✅ Set Claude.gg as provider
- ✅ Run a sample evaluation

## 🎉 That's It!

Your system now supports Claude models for RAGAS evaluation. The integration is complete and ready to use!
