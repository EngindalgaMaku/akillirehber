"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env
    )

    # Database
    database_url: str = "postgresql://raguser:ragpassword@localhost:5432/ragchatbot"

    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: Optional[str] = None

    # RAGAS Service
    ragas_url: str = "http://localhost:8001"

    # JWT Authentication
    secret_key: str = "your-super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15  # Short-lived access token (15 min)
    refresh_token_expire_days: int = 30    # Long-lived refresh token (30 days)

    # Application
    environment: str = "development"
    debug: bool = True

    # OpenAI (optional)
    openai_api_key: Optional[str] = None

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    # SMTP Email Configuration (optional)
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from_email: Optional[str] = None
    smtp_use_tls: bool = True


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
