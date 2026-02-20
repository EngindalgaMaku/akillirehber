"""
LLM Service

This module provides a unified interface for interacting with various LLM
providers. It handles client initialization, API calls, and error handling.
"""

import os
import logging
import time
from typing import List, Dict
from openai import OpenAI
import requests

from .llm_providers import LLM_PROVIDERS

logger = logging.getLogger(__name__)


# Custom Exceptions
class LLMProviderError(Exception):
    """Raised when there's an issue with LLM provider configuration."""
    pass


class LLMConfigurationError(Exception):
    """Raised when there's a configuration error."""
    pass


class LLMAPIError(Exception):
    """Raised when there's an error calling LLM API."""
    pass


class LLMService:
    """
    Service class for interacting with various LLM providers.

    This class provides a unified interface for making requests to different
    LLM providers using OpenAI client library.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize LLM service.

        Args:
            provider: The LLM provider name (e.g., 'openai', 'groq')
            model: The model name to use
            temperature: Temperature for response generation (0.0-1.0)
            max_tokens: Maximum tokens in response

        Raises:
            LLMProviderError: If provider is not supported
            LLMConfigurationError: If provider configuration is invalid
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
        Create and return an OpenAI client configured for provider.

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
                f"Please set environment variable: {env_key}"
            )

        # Add provider-specific default headers and OpenRouter
        # specific headers if using OpenRouter
        default_headers = dict(
            self.provider_config.get("default_headers", {}) or {}
        )
        if "openrouter" in self.provider_config["base_url"]:
            default_headers = {
                **default_headers,
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "RAG System",
            }

        return OpenAI(
            api_key=api_key,
            base_url=self.provider_config["base_url"],
            default_headers=default_headers
        )

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate a response from LLM with retry mechanism.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Initial delay between retries in seconds
            (default: 1.0)
            **kwargs: Additional parameters to pass to API

        Returns:
            The generated response text

        Raises:
            LLMAPIError: If there's an error calling API after all retries
        """
        # Handle z.ai separately (Anthropic-compatible API)
        if self.provider == "zai":
            return self._generate_response_zai(
                messages, max_retries, retry_delay, **kwargs
            )

        for attempt in range(max_retries + 1):
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

                if params.get("stream"):
                    chunks = []
                    for chunk in response:
                        if not hasattr(chunk, 'choices') or not chunk.choices:
                            logger.warning(
                                "Streaming chunk missing choices attribute"
                            )
                            continue
                        delta = getattr(
                            chunk.choices[0],
                            "delta",
                            None,
                        )
                        if delta is not None:
                            content = getattr(delta, "content", None)
                            if content is not None:
                                chunks.append(content)
                    return "".join(chunks)

                # Check if response has choices before accessing it
                if (not hasattr(response, 'choices') or
                        response.choices is None):
                    raise LLMAPIError(
                        f"API returned invalid response (no choices). "
                        f"Provider: {self.provider}, Model: {self.model}"
                    )

                if len(response.choices) == 0:
                    raise LLMAPIError(
                        f"API returned empty choices array. "
                        f"Provider: {self.provider}, Model: {self.model}"
                    )

                # Check if message exists
                message = response.choices[0].message
                if message is None:
                    raise LLMAPIError(
                        f"API returned None message. "
                        f"Provider: {self.provider}, Model: {self.model}"
                    )

                content = message.content or ""
                if not content.strip():
                    if attempt == max_retries:
                        raise LLMAPIError(
                            "LLM returned an empty response after retries. "
                            f"Provider: {self.provider}, Model: {self.model}"
                        )
                    logger.warning(
                        "LLM returned empty response (attempt %d/%d). "
                        "Retrying in %ss...",
                        attempt + 1,
                        max_retries + 1,
                        retry_delay * (attempt + 1),
                    )
                    time.sleep(retry_delay * (attempt + 1))
                    continue

                return content

            except LLMAPIError as e:
                error_str = str(e)
                lower = error_str.lower()

                if "data policy" in lower or "privacy" in lower:
                    raise

                retryable_api_error = any(
                    keyword in lower
                    for keyword in [
                        "timeout",
                        "connection",
                        "rate limit",
                        "too many requests",
                        "overloaded",
                        "temporarily",
                        "server error",
                        "invalid response",
                        "empty choices",
                        "none message",
                        "503",
                        "502",
                        "500",
                        "429",
                    ]
                )

                if not retryable_api_error or attempt == max_retries:
                    raise

                logger.warning(
                    "LLM API error (attempt %d/%d): %s. Retrying in %ss...",
                    attempt + 1,
                    max_retries + 1,
                    error_str,
                    retry_delay * (attempt + 1),
                )
                time.sleep(retry_delay * (attempt + 1))
                continue
            except Exception as e:
                error_str = str(e)

                # Check if error is retryable
                lower = error_str.lower()
                is_retryable = any(
                    keyword in lower
                    for keyword in [
                        "timeout",
                        "connection",
                        "rate limit",
                        "too many requests",
                        "overloaded",
                        "temporarily",
                        "server error",
                        "503",
                        "502",
                        "500",
                        "429",
                    ]
                )

                # Check for OpenRouter privacy policy error (not retryable)
                if "data policy" in lower or "privacy" in lower:
                    error_msg = (
                        "OpenRouter gizlilik ayarları hatası: Bu model için "
                        "https://openrouter.ai/settings/privacy adresinden "
                        "veri politikasını kabul etmeniz gerekiyor. "
                        f"Model: {self.model}"
                    )
                    raise LLMAPIError(error_msg)

                # If not retryable or last attempt, raise error
                if not is_retryable or attempt == max_retries:
                    error_msg = (
                        f"Error calling {self.provider} API with model "
                        f"{self.model}: {error_str}"
                    )
                    raise LLMAPIError(error_msg)

                # Log retry attempt
                logger.warning(
                    f"LLM API call failed (attempt {attempt + 1}/"
                    f"{max_retries + 1}): {error_str}. "
                    f"Retrying in {retry_delay * (attempt + 1)}s..."
                )

                # Exponential backoff
                time.sleep(retry_delay * (attempt + 1))

    def _generate_response_zai(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate a response from z.ai API (Anthropic-compatible format).

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Initial delay between retries in seconds
            (default: 1.0)
            **kwargs: Additional parameters to pass to API

        Returns:
            The generated response text

        Raises:
            LLMAPIError: If there's an error calling API after all retries
        """
        api_key = os.getenv(self.provider_config["env_key"])

        if not api_key:
            env_key = self.provider_config['env_key']
            raise LLMConfigurationError(
                f"API key not found for provider '{self.provider}'. "
                f"Please set environment variable: {env_key}"
            )

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }

        # z.ai API doesn't support 'system' role, only 'user' and
        # 'assistant'. Filter out system messages and prepend their
        # content to first user message
        filtered_messages = []
        system_content = ""

        for msg in messages:
            if msg.get("role") == "system":
                # Accumulate system messages
                if system_content:
                    system_content += "\n\n"
                system_content += msg.get("content", "")
            elif msg.get("role") in ["user", "assistant"]:
                filtered_messages.append(msg)

        # If there was system content, prepend it to first user message
        if system_content and filtered_messages:
            # Find the first user message
            for i, msg in enumerate(filtered_messages):
                if msg.get("role") == "user":
                    # Prepend system content to the user message
                    filtered_messages[i] = {
                        "role": "user",
                        "content": (
                            f"{system_content}\n\n"
                            f"{msg.get('content', '')}"
                        )
                    }
                    break

        for attempt in range(max_retries + 1):
            try:
                data = {
                    "model": self.model,
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "messages": filtered_messages
                }

                # Add temperature if provided
                if "temperature" in kwargs:
                    data["temperature"] = kwargs["temperature"]
                elif self.temperature is not None:
                    data["temperature"] = self.temperature

                response = requests.post(
                    self.provider_config["base_url"],
                    headers=headers,
                    json=data,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    # Anthropic format: result["content"][0]["text"]
                    if "content" in result and len(result["content"]) > 0:
                        text = result["content"][0].get("text", "")
                        if text and text.strip():
                            return text
                        raise LLMAPIError("z.ai returned empty response")
                    else:
                        raise LLMAPIError(
                            f"Invalid response format from z.ai API. "
                            f"Provider: {self.provider}, Model: {self.model}"
                        )
                else:
                    error_msg = (
                        f"z.ai API returned status {response.status_code}: "
                        f"{response.text}"
                    )
                    raise LLMAPIError(error_msg)

            except LLMAPIError as e:
                error_str = str(e)
                lower = error_str.lower()

                non_retryable_4xx = (
                    "status 4" in lower
                    and "429" not in lower
                )

                if non_retryable_4xx or attempt == max_retries:
                    raise

                logger.warning(
                    "z.ai API error (attempt %d/%d): %s. Retrying in %ss...",
                    attempt + 1,
                    max_retries + 1,
                    error_str,
                    retry_delay * (attempt + 1),
                )
                time.sleep(retry_delay * (attempt + 1))
                continue
            except Exception as e:
                error_str = str(e)

                lower = error_str.lower()
                is_retryable = any(
                    keyword in lower
                    for keyword in [
                        "timeout",
                        "connection",
                        "rate limit",
                        "too many requests",
                        "overloaded",
                        "temporarily",
                        "server error",
                        "503",
                        "502",
                        "500",
                        "429",
                    ]
                )

                # If not retryable or last attempt, raise error
                if not is_retryable or attempt == max_retries:
                    error_msg = (
                        f"Error calling {self.provider} API with model "
                        f"{self.model}: {error_str}"
                    )
                    raise LLMAPIError(error_msg)

                # Log retry attempt
                logger.warning(
                    f"LLM API call failed (attempt {attempt + 1}/"
                    f"{max_retries + 1}): {error_str}. "
                    f"Retrying in {retry_delay * (attempt + 1)}s..."
                )

                # Exponential backoff
                time.sleep(retry_delay * (attempt + 1))

    @staticmethod
    def get_available_models(provider: str) -> List[str]:
        """
        Get list of available models for a provider.

        Args:
            provider: The provider name

        Returns:
            List of available model names

        Raises:
            LLMProviderError: If provider is not supported
        """
        if provider not in LLM_PROVIDERS:
            available_providers = list(LLM_PROVIDERS.keys())
            raise LLMProviderError(
                f"Provider '{provider}' is not supported. "
                f"Available providers: {available_providers}"
            )

        return LLM_PROVIDERS[provider]["models"]

    @staticmethod
    def validate_provider_config(provider: str) -> bool:
        """
        Validate that provider configuration is complete and API key
        exists.

        Args:
            provider: The provider name

        Returns:
            True if configuration is valid

        Raises:
            LLMProviderError: If provider is not supported
            LLMConfigurationError: If API key is not found
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
                f"Please set environment variable: {env_key}"
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
        max_tokens: Maximum tokens in response

    Returns:
        Configured LLMService instance

    Raises:
        LLMProviderError: If provider is not supported
        LLMConfigurationError: If configuration is invalid
    """
    return LLMService(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
