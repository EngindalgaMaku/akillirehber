"""
LLM Service

This module provides a unified interface for interacting with various LLM
providers. It handles client initialization, API calls, and error handling.
"""

import os
from typing import List, Dict
from openai import OpenAI

from .llm_providers import LLM_PROVIDERS


# Custom Exceptions
class LLMProviderError(Exception):
    """Raised when there's an issue with the LLM provider configuration."""
    pass


class LLMConfigurationError(Exception):
    """Raised when there's a configuration error."""
    pass


class LLMAPIError(Exception):
    """Raised when there's an error calling the LLM API."""
    pass


class LLMService:
    """
    Service class for interacting with various LLM providers.
    
    This class provides a unified interface for making requests to different
    LLM providers using the OpenAI client library.
    """
    
    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize the LLM service.
        
        Args:
            provider: The LLM provider name (e.g., 'openai', 'groq')
            model: The model name to use
            temperature: Temperature for response generation (0.0-1.0)
            max_tokens: Maximum tokens in the response
            
        Raises:
            LLMProviderError: If the provider is not supported
            LLMConfigurationError: If the provider configuration is invalid
        """
        if provider not in LLM_PROVIDERS:
            raise LLMProviderError(
                f"Provider '{provider}' is not supported. "
                f"Available providers: {list(LLM_PROVIDERS.keys())}"
            )
        
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_config = LLM_PROVIDERS[provider]
        
        # Validate configuration
        self.validate_provider_config(provider)
        
        # Validate model - skip for openrouter to allow any model
        if provider != "openrouter":
            if model not in self.provider_config["models"]:
                raise LLMConfigurationError(
                    f"Model '{model}' is not available for provider "
                    f"'{provider}'. "
                    f"Available models: {self.provider_config['models']}"
                )
        
        self.client = self._get_client()
    
    def _get_client(self) -> OpenAI:
        """
        Create and return an OpenAI client configured for the provider.
        
        Returns:
            Configured OpenAI client instance
            
        Raises:
            LLMConfigurationError: If API key is not found
        """
        api_key = os.getenv(self.provider_config["env_key"])
        
        if not api_key:
            env_key = self.provider_config['env_key']
            raise LLMConfigurationError(
                f"API key not found for provider '{self.provider}'. "
                f"Please set the environment variable: {env_key}"
            )
        
        # Add OpenRouter specific headers if using OpenRouter
        default_headers = {}
        if "openrouter" in self.provider_config["base_url"]:
            default_headers = {
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "RAG System"
            }
        
        return OpenAI(
            api_key=api_key,
            base_url=self.provider_config["base_url"],
            default_headers=default_headers
        )
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The generated response text
            
        Raises:
            LLMAPIError: If there's an error calling the API
        """
        try:
            # Merge default parameters with provided kwargs
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens)
            }
            
            # Add any additional kwargs
            for key, value in kwargs.items():
                if key not in ["temperature", "max_tokens"]:
                    params[key] = value
            
            response = self.client.chat.completions.create(**params)
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e)
            # Check for OpenRouter privacy policy error
            if "data policy" in error_str.lower() or "privacy" in error_str.lower():
                error_msg = (
                    f"OpenRouter gizlilik ayarları hatası: Bu model için "
                    f"https://openrouter.ai/settings/privacy adresinden "
                    f"veri politikasını kabul etmeniz gerekiyor. "
                    f"Model: {self.model}"
                )
            else:
                error_msg = (
                    f"Error calling {self.provider} API with model "
                    f"{self.model}: {error_str}"
                )
            raise LLMAPIError(error_msg)
    
    @staticmethod
    def get_available_models(provider: str) -> List[str]:
        """
        Get the list of available models for a provider.
        
        Args:
            provider: The provider name
            
        Returns:
            List of available model names
            
        Raises:
            LLMProviderError: If the provider is not supported
        """
        if provider not in LLM_PROVIDERS:
            raise LLMProviderError(
                f"Provider '{provider}' is not supported. "
                f"Available providers: {list(LLM_PROVIDERS.keys())}"
            )
        
        return LLM_PROVIDERS[provider]["models"]
    
    @staticmethod
    def validate_provider_config(provider: str) -> bool:
        """
        Validate that the provider configuration is complete and API key
        exists.
        
        Args:
            provider: The provider name
            
        Returns:
            True if configuration is valid
            
        Raises:
            LLMProviderError: If the provider is not supported
            LLMConfigurationError: If the API key is not found
        """
        if provider not in LLM_PROVIDERS:
            raise LLMProviderError(
                f"Provider '{provider}' is not supported. "
                f"Available providers: {list(LLM_PROVIDERS.keys())}"
            )
        
        provider_config = LLM_PROVIDERS[provider]
        api_key = os.getenv(provider_config["env_key"])
        
        if not api_key:
            env_key = provider_config['env_key']
            raise LLMConfigurationError(
                f"API key not found for provider '{provider}'. "
                f"Please set the environment variable: {env_key}"
            )
        
        return True


def get_llm_service(
    provider: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> LLMService:
    """
    Factory function to create and return an LLMService instance.
    
    Args:
        provider: The LLM provider name
        model: The model name to use
        temperature: Temperature for response generation (0.0-1.0)
        max_tokens: Maximum tokens in the response
        
    Returns:
        Configured LLMService instance
        
    Raises:
        LLMProviderError: If the provider is not supported
        LLMConfigurationError: If the configuration is invalid
    """
    return LLMService(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )