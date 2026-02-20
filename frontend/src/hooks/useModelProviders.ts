import { useState, useEffect, useCallback } from "react";
import { api } from "@/lib/api";

export interface ModelProviders {
  llm: Record<string, string[]>;
  embedding: string[];
}

export function useModelProviders() {
  const [providers, setProviders] = useState<ModelProviders>({
    llm: {},
    embedding: []
  });

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadProviders = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [llmProviders] = await Promise.all([
        api.getLLMProviders()
      ]);

      // Embedding models - hardcoded list (same as course settings)
      const embeddingModels = [
        "openai/text-embedding-3-small",
        "openai/text-embedding-3-large",
        "openai/text-embedding-ada-002",
        "alibaba/text-embedding-v4",
        "cohere/embed-multilingual-v3.0",
        "cohere/embed-multilingual-light-v3.0",
        "jina/jina-embeddings-v2",
        "jina/jina-embeddings-v3",
        "qwen/qwen3-embedding-8b",
        "ollama/bge-m3",
        "ollama/bge-small-en-v1.5",
        "ollama/nomic-embed-text-v1.5",
        "ollama/nomic-embed-text-v1",
        "voyage/voyage-4-large",
        "voyage/voyage-3-large",
        "voyage/voyage-3-lite",
        "voyage/voyage-2"
      ];

      setProviders({
        llm: llmProviders,
        embedding: embeddingModels
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load providers");
      console.error("Failed to load model providers:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProviders();
  }, [loadProviders]);

  const getLLMModels = useCallback((provider: string) => {
    return providers.llm[provider] || [];
  }, [providers.llm]);

  const getEmbeddingModels = useCallback(() => {
    return providers.embedding;
  }, [providers.embedding]);

  const getLLMProviders = useCallback(() => {
    return Object.keys(providers.llm);
  }, [providers.llm]);

  const isModelAvailable = useCallback((provider: string, model: string) => {
    return getLLMModels(provider).includes(model);
  }, [getLLMModels]);

  const isEmbeddingModelAvailable = useCallback((model: string) => {
    return getEmbeddingModels().includes(model);
  }, [getEmbeddingModels]);

  return {
    providers,
    isLoading,
    error,
    loadProviders,
    getLLMModels,
    getEmbeddingModels,
    getLLMProviders,
    isModelAvailable,
    isEmbeddingModelAvailable
  };
}
