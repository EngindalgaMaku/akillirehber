# API Claude.gg Setup Guide

This guide explains how to set up and use the API Claude.gg provider for RAGAS evaluations.

## Overview

API Claude.gg (`https://api.claude.gg/v1`) is a multi-model API provider that offers access to various state-of-the-art language models including:

- **GPT Models**: gpt-o3, o3, o3-mini, gpt-5.1, gpt-5, gpt-5-mini, gpt-5-nano, gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-nano
- **Grok Models**: grok-4, grok-3-mini, grok-3-mini-beta, grok-2
- **DeepSeek Models**: deepseek-r1, deepseek-v3, deepseek-chat
- **Gemini Models**: gemini-3-pro, gemini-2.5-flash, gemini-2.0-flash, gemini-2.0-flash-lite, gemini, gemini-lite

## Configuration

### 1. API Key

The API key is the same as Claude.gg:
```
sk-542ca4c71cf0451a9b92b584eeba6520
```

### 2. Environment Variables

Add to your `.env` file:
```bash
APICLAUDEGG_API_KEY=sk-542ca4c71cf0451a9b92b584eeba6520
```

### 3. Docker Compose

The API key is already configured in all docker-compose files:
- `docker-compose.yml`
- `docker-compose.prod.yml`
- `docker-compose.coolify.yml`

## Provider Priority

API Claude.gg has priority 3 in the auto-selection order:

1. **OpenRouter** (priority 1) - Best quality, most reliable
2. **Claude.gg** (priority 2) - High-quality Claude models
3. **API Claude.gg** (priority 3) - Multi-model access (GPT-5, O3, Grok, DeepSeek, Gemini)
4. **Groq** (priority 4) - Free but lower quality
5. **OpenAI** (priority 5) - Fallback option

## Default Model

The default model for API Claude.gg is **gpt-5**, which provides excellent performance across various tasks.

## Usage

### Via Frontend

1. Go to **Dashboard → RAGAS Değerlendirme**
2. Click the **Settings** icon
3. Select **apiclaudegg** from the provider dropdown
4. Choose your preferred model (default: gpt-5)
5. Click **Save**

### Via API

Set the provider programmatically:

```bash
curl -X POST http://localhost:8001/settings \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "apiclaudegg",
    "model": "gpt-5"
  }'
```

### Test Evaluation

Run a test evaluation:

```bash
python test_apiclaudegg.py
```

## Available Models

### GPT Models
- `gpt-o3` - Latest OpenAI O3 model
- `o3` - Alias for gpt-o3
- `o3-mini` - Smaller O3 variant
- `gpt-5.1` - GPT-5.1 model
- `gpt-5` - **Default** - GPT-5 model
- `gpt-5-mini` - Smaller GPT-5 variant
- `gpt-5-nano` - Smallest GPT-5 variant
- `gpt-4o` - GPT-4 Optimized
- `gpt-4o-mini` - Smaller GPT-4o
- `gpt-4.1` - GPT-4.1 model
- `gpt-4.1-nano` - Smallest GPT-4.1

### Grok Models
- `grok-4` - Latest Grok model
- `grok-3-mini` - Smaller Grok 3
- `grok-3-mini-beta` - Beta version
- `grok-2` - Grok 2 model

### DeepSeek Models
- `deepseek-r1` - DeepSeek R1 reasoning model
- `deepseek-v3` - DeepSeek V3
- `deepseek-chat` - DeepSeek Chat model

### Gemini Models
- `gemini-3-pro` - Gemini 3 Pro
- `gemini-2.5-flash` - Gemini 2.5 Flash
- `gemini-2.0-flash` - Gemini 2.0 Flash
- `gemini-2.0-flash-lite` - Lighter version
- `gemini` - Standard Gemini
- `gemini-lite` - Lite version

## Testing

### Quick Test

```bash
# Test the API directly
curl -X POST https://api.claude.gg/v1/chat/completions \
  -H "Authorization: Bearer sk-542ca4c71cf0451a9b92b584eeba6520" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Full Integration Test

```bash
# Run the comprehensive test script
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

## Troubleshooting

### Provider Not Available

If API Claude.gg doesn't appear in the provider list:

1. Check that the API key is set in `.env`:
   ```bash
   echo $APICLAUDEGG_API_KEY
   ```

2. Restart the RAGAS service:
   ```bash
   docker-compose restart ragas
   ```

3. Check the service logs:
   ```bash
   docker-compose logs ragas
   ```

### Evaluation Errors

If evaluations fail:

1. Verify the API key is valid
2. Check the model name is correct
3. Try a different model (e.g., `gpt-4o` instead of `gpt-5`)
4. Check RAGAS service logs for detailed error messages

### API Rate Limits

If you encounter rate limits:

1. Switch to a different provider temporarily
2. Use a smaller model (e.g., `gpt-5-mini` instead of `gpt-5`)
3. Add delays between evaluations

## Best Practices

1. **Model Selection**: Use `gpt-5` for best quality, `gpt-5-mini` for faster/cheaper evaluations
2. **Batch Evaluations**: For large test sets, consider using batch evaluation endpoints
3. **Error Handling**: Always check for errors in evaluation results
4. **Monitoring**: Monitor API usage and costs through the API Claude.gg dashboard

## Comparison with Other Providers

| Provider | Pros | Cons | Best For |
|----------|------|------|----------|
| **OpenRouter** | Highest quality, most reliable | Paid | Production evaluations |
| **Claude.gg** | Excellent Claude models | Limited to Claude | Claude-specific tasks |
| **API Claude.gg** | Multi-model access, GPT-5/O3 | Paid | Diverse model testing |
| **Groq** | Free, fast | Lower quality | Development/testing |
| **OpenAI** | Direct access | Paid, rate limits | Fallback option |

## Support

For issues or questions:
- Check the [API Claude.gg documentation](https://api.claude.gg/)
- Review the test script: `test_apiclaudegg.py`
- Check RAGAS service logs: `docker-compose logs ragas`
