# Weaviate Reranker Research

## Overview

Weaviate supports native reranking through its reranker modules. This document outlines the configuration requirements and supported providers.

## Supported Reranker Providers

Weaviate supports the following reranker modules:

1. **reranker-cohere** - Uses Cohere's rerank API
2. **reranker-transformers** - Uses local transformer models
3. **reranker-voyageai** - Uses VoyageAI's rerank API
4. **reranker-jinaai** - Uses JinaAI's rerank API

## Configuration Requirements

### Docker Compose Configuration

To enable reranker modules in Weaviate, you need to:

1. Add the module to `ENABLE_MODULES` environment variable
2. Provide the necessary API keys for external providers

Example for Cohere reranker:

```yaml
weaviate:
  environment:
    ENABLE_MODULES: 'reranker-cohere'
    COHERE_APIKEY: ${COHERE_API_KEY}
```

### Collection Configuration

When creating a collection, you can configure the reranker module:

```python
collection_config = {
    "class": "Document",
    "moduleConfig": {
        "reranker-cohere": {
            "model": "rerank-multilingual-v3.0"
        }
    }
}
```

### Query-Time Reranking

Reranking happens during query execution:

```python
response = collection.query.hybrid(
    query=query,
    vector=query_vector,
    alpha=0.5,
    limit=100,  # Initial retrieval
    rerank={
        "property": "content",
        "query": query
    },
    return_metadata=MetadataQuery(score=True, rerank_score=True)
).with_limit(10)  # Final top-k after reranking
```

## Limitations

1. **Requires Documents in Weaviate**: The reranker module only works with documents already stored in Weaviate. It cannot rerank external documents.

2. **Module Must Be Enabled**: The reranker module must be enabled in Weaviate's configuration before use.

3. **API Key Required**: For external providers (Cohere, VoyageAI, JinaAI), API keys must be configured in Weaviate's environment.

4. **Query-Time Only**: Reranking happens during query execution, not as a separate step.

## Implementation Approach

For our use case, we have two options:

### Option 1: Query-Time Reranking (Recommended for Weaviate-stored documents)

Use Weaviate's native reranker during the hybrid search query itself. This is most efficient when documents are already in Weaviate.

**Pros:**
- Lower latency (single API call)
- No need to transfer documents
- Leverages Weaviate's optimization

**Cons:**
- Requires module configuration
- Only works for documents in Weaviate
- Less flexible than external reranking

### Option 2: External Reranking (Current Implementation)

Retrieve documents from Weaviate, then rerank them using external APIs (Cohere, Alibaba). This is what we've already implemented.

**Pros:**
- Works with any documents
- More flexible
- Provider-agnostic

**Cons:**
- Higher latency (two API calls)
- Data transfer overhead

## Recommendation

For the `_rerank_weaviate` method, we should:

1. **Document the limitation**: Clearly state that Weaviate reranking requires documents to be in Weaviate
2. **Implement query-time reranking**: Modify the search query to include reranking
3. **Provide fallback**: If Weaviate reranking is not available, fall back to external reranking

## Implementation Notes

The `_rerank_weaviate` method should:

1. Check if the reranker module is enabled in Weaviate
2. If enabled, perform a new hybrid search with reranking enabled
3. If not enabled, raise an error or fall back to external reranking
4. Return results in the same format as other rerankers

## References

- [Weaviate Reranker Modules](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/reranker)
- [Cohere Reranker Module](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/reranker-cohere)
- [Reranking in Queries](https://weaviate.io/developers/weaviate/search/rerank)
