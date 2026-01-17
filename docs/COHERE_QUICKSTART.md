# Cohere Embedding Models - Quick Start Guide

## What is Cohere?

Cohere provides state-of-the-art multilingual embedding models that excel at understanding text in multiple languages. Their models are optimized for semantic search, classification, and clustering tasks.

## Available Models

### 1. embed-multilingual-v3.0 (1024 dimensions)
- **Best for:** High-quality multilingual embeddings
- **Languages:** 100+ languages supported
- **Use case:** Production applications requiring best accuracy
- **Dimensions:** 1024

### 2. embed-multilingual-light-v3.0 (384 dimensions)
- **Best for:** Fast processing with good quality
- **Languages:** 100+ languages supported
- **Use case:** Applications prioritizing speed and lower storage
- **Dimensions:** 384

## Setup

### Step 1: Get API Key

1. Visit [Cohere Dashboard](https://dashboard.cohere.com/api-keys)
2. Sign up or log in
3. Create a new API key
4. Copy the key (starts with your API key format)

### Step 2: Configure Environment

Add to your `.env` file:

```bash
COHERE_API_KEY=your-cohere-api-key-here
```

### Step 3: Install Dependencies

The Cohere package is already included in `requirements.txt`:

```bash
pip install -r backend/requirements.txt
```

Or install manually:

```bash
pip install cohere>=5.0.0
```

### Step 4: Restart Backend

If the backend is running, restart it to load the new API key:

```bash
# Stop the backend (Ctrl+C)
# Start it again
uvicorn app.main:app --reload
```

## Usage

### In Course Settings

1. Navigate to your course
2. Go to **Settings** tab
3. Find **Embedding Model** dropdown
4. Select one of:
   - `Cohere embed-multilingual-v3.0 (1024 dim)`
   - `Cohere embed-multilingual-light-v3.0 (384 dim)`
5. Click **Save Settings**

### In Document Processing

1. Upload a document
2. In the **Processing** tab
3. Select Cohere model from **Embedding Model** dropdown
4. Click **Process Document**

### In Semantic Similarity Testing

1. Go to **Semantic Similarity** page
2. Select Cohere model from **Embedding Model** dropdown
3. Enter your test texts
4. Click **Test Similarity**

## When to Use Cohere

### ✅ Use Cohere When:

- Working with **multilingual content** (100+ languages)
- Need **high-quality embeddings** for semantic search
- Want **consistent performance** across languages
- Building **production applications** with diverse language needs
- Need **smaller embeddings** (light model: 384 dim vs 1536 dim)

### ❌ Consider Alternatives When:

- Only working with **English content** (OpenAI might be better)
- Need **very large embeddings** (OpenAI 3-large: 3072 dim)
- Want **lowest cost** (check pricing comparison)
- Already have **OpenRouter credits**

## Model Comparison

| Model | Dimensions | Languages | Best For |
|-------|-----------|-----------|----------|
| **Cohere multilingual-v3.0** | 1024 | 100+ | Multilingual production |
| **Cohere multilingual-light** | 384 | 100+ | Fast multilingual processing |
| **OpenAI text-embedding-3-small** | 1536 | English+ | General purpose |
| **OpenAI text-embedding-3-large** | 3072 | English+ | Highest quality |
| **Alibaba text-embedding-v4** | 1024 | Chinese+ | Chinese content |

## Pricing

Check current pricing at [Cohere Pricing](https://cohere.com/pricing)

**Typical costs (as of Jan 2026):**
- embed-multilingual-v3.0: ~$0.10 per 1M tokens
- embed-multilingual-light-v3.0: ~$0.10 per 1M tokens

**Free tier:** Cohere offers a free trial with limited usage

## Performance Tips

### 1. Choose the Right Model

```
High accuracy needed? → embed-multilingual-v3.0
Speed/storage priority? → embed-multilingual-light-v3.0
```

### 2. Batch Processing

The system automatically batches requests (up to 96 texts per call) for optimal performance.

### 3. Storage Considerations

```
Light model (384 dim) = 75% less storage than OpenAI small (1536 dim)
Standard model (1024 dim) = 33% less storage than OpenAI small
```

### 4. Language Support

Cohere models work well across all languages without special configuration:
- English, Spanish, French, German, Italian, Portuguese
- Chinese, Japanese, Korean
- Arabic, Hebrew, Hindi, Bengali
- And 90+ more languages

## Troubleshooting

### Error: "COHERE_API_KEY environment variable is required"

**Solution:** Add your API key to `.env` file and restart the backend.

### Error: "Cohere package is not installed"

**Solution:** Install the package:
```bash
pip install cohere>=5.0.0
```

### Error: "Invalid API key"

**Solution:** 
1. Check your API key in `.env` file
2. Verify it's correct at [Cohere Dashboard](https://dashboard.cohere.com/api-keys)
3. Make sure there are no extra spaces or quotes

### Error: "Rate limit exceeded"

**Solution:**
1. Wait a few minutes and try again
2. Check your usage at Cohere Dashboard
3. Consider upgrading your plan if needed

### Embeddings seem incorrect

**Solution:**
1. Verify you're using the correct model name
2. Check that the API key is valid
3. Try the semantic similarity test page to verify

## Testing Your Setup

### Quick Test

1. Go to **Semantic Similarity** page
2. Select `Cohere embed-multilingual-v3.0`
3. Enter test texts:
   - Text 1: "Hello, how are you?"
   - Text 2: "Hi, how are you doing?"
   - Text 3: "The weather is nice today"
4. Click **Test Similarity**
5. Verify that Text 1 and Text 2 have high similarity (>0.8)

### Multilingual Test

Test with different languages:
- English: "Hello, world!"
- Spanish: "¡Hola, mundo!"
- French: "Bonjour, monde!"
- Chinese: "你好，世界！"

All should generate valid embeddings.

## API Usage (Advanced)

If you're using the API directly:

```python
from app.services.embedding_service import get_embedding_service

service = get_embedding_service()

# Single embedding
embedding = service.get_embedding(
    "Your text here",
    model="cohere/embed-multilingual-v3.0"
)

# Batch embeddings (automatically batched in groups of 96)
embeddings = service.get_embeddings(
    ["Text 1", "Text 2", "Text 3", ...],
    model="cohere/embed-multilingual-light-v3.0"
)
```

## Best Practices

1. **Use the light model** for development and testing
2. **Use the standard model** for production
3. **Batch your requests** when processing multiple texts
4. **Monitor your usage** at Cohere Dashboard
5. **Test with your actual content** before committing to a model
6. **Consider language distribution** in your content when choosing models

## Support

- **Cohere Documentation:** https://docs.cohere.com/
- **Cohere Discord:** https://discord.gg/cohere
- **API Status:** https://status.cohere.com/

## Next Steps

1. ✅ Set up your API key
2. ✅ Test with semantic similarity
3. ✅ Process a document with Cohere embeddings
4. ✅ Compare results with other providers
5. ✅ Choose the best model for your use case

## Summary

Cohere's multilingual embedding models are now fully integrated into the system. They offer excellent multilingual support, competitive pricing, and easy switching between providers. Start with the light model for testing, then upgrade to the standard model for production use.

Happy embedding! 🚀
