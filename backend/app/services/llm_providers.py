"""
LLM Provider Configurations

This module contains the configuration for various LLM providers including
their base URLs, environment variable keys, and available models.
"""

LLM_PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "models": [
            # Küçük/Hızlı Modeller
            "openai/gpt-4o-mini",
            "anthropic/claude-3-haiku",
            "google/gemini-flash-1.5",
            "meta-llama/llama-3.2-3b-instruct",
            "microsoft/phi-3-mini-128k-instruct",
            # Orta Seviye Modeller
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.1-8b-instruct",
            # Büyük Modeller
            "meta-llama/llama-3.3-70b-instruct:free",
            "meta-llama/llama-3.1-70b-instruct",
            "anthropic/claude-3-opus",
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "mistralai/mistral-small-3.1-24b-instruct:free"
        ]
    },
    "claudegg": {
        "base_url": "https://claude.gg/v1",
        "env_key": "CLAUDEGG_API_KEY",
        "models": [
            # Claude Modelleri
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-sonnet-4",
            "claude-3-7-sonnet",
            "claude-haiku-4-5"
        ]
    },
    "apiclaudegg": {
        "base_url": "https://api.claude.gg/v1",
        "env_key": "APICLAUDEGG_API_KEY",
        "models": [
            # GPT Modelleri
            "gpt-o3",
            "o3",
            "o3-mini",
            "gpt-5.1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-nano",
            # Grok Modelleri
            "grok-4",
            "grok-3-mini",
            "grok-3-mini-beta",
            "grok-2",
            # DeepSeek Modelleri
            "deepseek-r1",
            "deepseek-v3",
            "deepseek-chat",
            # Gemini Modelleri
            "gemini-3-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini",
            "gemini-lite"
        ]
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "models": [
            # Test Edilmiş Çalışan Model
            "llama-3.1-8b-instant",
            # Diğer Mevcut Modeller (test edilmeli)
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "llama3-70b-8192"
        ]
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "models": [
            # Küçük/Ekonomik
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            # Büyük/Güçlü
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4"
        ]
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "models": [
            # Genel Amaçlı
            "deepseek-chat",
            # Kod Odaklı
            "deepseek-coder"
        ]
    },
    "cohere": {
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "env_key": "COHERE_API_KEY",
        "models": [
            # Test Edilmiş Çalışan Modeller
            "command-r-08-2024",
            "command-r-plus-08-2024",
            "command-a-03-2025",
            # Diğer Modeller
            "command-r-plus"
        ]
    },
    "alibaba": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "models": [
            # Küçük/Hızlı Modeller
            "qwen-turbo",
            "qwen-flash",
            "qwen3-8b",
            "qwen3-14b",
            # Orta Seviye
            "qwen-plus",
            "qwen3-32b",
            # Büyük/Güçlü
            "qwen-max",
            "qwen3-max"
        ]
    },
    "zai": {
        "base_url": "https://api.z.ai/api/anthropic/v1/messages",
        "env_key": "ZAI_API_KEY",
        "default_headers": {
            "anthropic-version": "2023-06-01"
        },
        "models": [
            "glm-4.7"
        ]
    }
}
