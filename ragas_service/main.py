"""RAGAS Evaluation Service - FastAPI application for RAG evaluation metrics."""

import math
import os
import logging
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVICE_NAME = "RAGAS Evaluation Service"

# Check Voyage AI availability at startup
try:
    from langchain_voyageai import VoyageAIEmbeddings
    VOYAGE_AVAILABLE = True
    logger.info("[STARTUP] langchain_voyageai is available")
except ImportError as e:
    VOYAGE_AVAILABLE = False
    logger.warning(f"[STARTUP] langchain_voyageai NOT available: {e}")

# Check VOYAGE_API_KEY at startup
voyage_key_present = bool(os.getenv("VOYAGE_API_KEY"))
logger.info(f"[STARTUP] VOYAGE_API_KEY present: {voyage_key_present}")

app = FastAPI(
    title=SERVICE_NAME,
    description="Service for evaluating RAG systems using RAGAS metrics",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== In-Memory Settings ====================

# Runtime settings (can be changed via API)
RUNTIME_SETTINGS = {
    "provider": None,  # None = auto-select
    "model": None,     # None = use provider default
}


# ==================== Schemas ====================

class EvaluationInput(BaseModel):
    """Input for single question evaluation."""
    question: str
    ground_truth: str  # Primary ground truth (backward compatibility)
    ground_truths: Optional[List[str]] = None  # Alternative ground truths
    generated_answer: str
    retrieved_contexts: List[str]
    evaluation_model: Optional[str] = None
    
    # Embedding settings (to match course's embedding model)
    embedding_provider: Optional[str] = None  # "voyage", "openai", "openrouter", etc.
    embedding_model: Optional[str] = None     # Model name
    
    # Reranker metadata (optional, for tracking)
    reranker_provider: Optional[str] = None  # "cohere", "alibaba", or None
    reranker_model: Optional[str] = None     # Model name or None


class RagasSettings(BaseModel):
    """RAGAS evaluation settings."""
    provider: Optional[str] = None
    model: Optional[str] = None


class EvaluationOutput(BaseModel):
    """Output metrics for single question evaluation."""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None
    error: Optional[str] = None
    
    # Reranker metadata (tracking)
    reranker_used: bool = False
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None


class BatchEvaluationInput(BaseModel):
    """Input for batch evaluation."""
    items: List[EvaluationInput]


class BatchEvaluationOutput(BaseModel):
    """Output for batch evaluation."""
    results: List[EvaluationOutput]
    total_processed: int
    total_errors: int
    
    # Reranker usage statistics
    reranker_usage: dict = {}  # {"cohere": 5, "alibaba": 3, "none": 2}


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    ragas_available: bool
    llm_provider: str
    voyage_available: bool = False
    voyage_api_key_present: bool = False


# ==================== LLM Provider Configurations ====================

LLM_PROVIDERS_CONFIG = {
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "embedding_model": "openai/text-embedding-3-small",
        "is_free": False,
        "priority": 1,  # Highest priority (best quality)
        "models": [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3-haiku",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-opus",
            "glm-4.7",
        ],
    },
    "claudegg": {
        "env_key": "CLAUDEGG_API_KEY",
        "base_url": "https://claude.gg/v1",
        "default_model": "claude-sonnet-4-5",
        "embedding_model": None,  # Claude.gg doesn't provide embeddings
        "is_free": False,
        "priority": 2,  # High quality Claude models
        "models": [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-sonnet-4",
            "claude-3-7-sonnet",
            "claude-haiku-4-5",
        ],
    },
    "apiclaudegg": {
        "env_key": "APICLAUDEGG_API_KEY",
        "base_url": "https://api.claude.gg/v1",
        "default_model": "gpt-5",
        "embedding_model": None,  # api.claude.gg doesn't provide embeddings
        "is_free": False,
        "priority": 3,  # Multi-model provider (GPT-5, O3, Grok, DeepSeek, Gemini)
        "models": [
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
            "grok-4",
            "grok-3-mini",
            "grok-3-mini-beta",
            "grok-2",
            "deepseek-r1",
            "deepseek-v3",
            "deepseek-chat",
            "gemini-3-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini",
            "gemini-lite",
        ],
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-8b-instant",
        "embedding_model": None,  # Groq doesn't have embeddings
        "is_free": False,
        "priority": 4,
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
        ],
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
        "default_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "is_free": False,
        "priority": 5,
        "models": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "embedding_model": None,
        "is_free": False,
        "priority": 6,
        "models": [
            "deepseek-chat",
            "deepseek-coder",
        ],
    },
    "alibaba": {
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-turbo",
        "embedding_model": None,
        "is_free": False,
        "priority": 7,
        "models": [
            "qwen-turbo",
            "qwen-flash",
            "qwen3-8b",
            "qwen3-14b",
            "qwen-plus",
            "qwen3-32b",
            "qwen-max",
            "qwen3-max",
        ],
    },
}


# ==================== RAGAS Integration ====================

RAGAS_AVAILABLE = False


def get_llm_config():
    """Get LLM configuration based on settings or available API keys.

    Priority:
    1. Runtime settings (set via API)
    2. Environment variables (RAGAS_PROVIDER, RAGAS_MODEL)
    3. Auto-select based on available keys (free providers first)
    """
    # Check runtime settings first
    runtime_provider = RUNTIME_SETTINGS.get("provider")
    runtime_model = RUNTIME_SETTINGS.get("model")

    # Then check environment variables
    env_provider = os.getenv("RAGAS_PROVIDER", "").lower()
    env_model = os.getenv("RAGAS_MODEL")

    # Use runtime settings if set, otherwise env vars
    preferred_provider = runtime_provider or env_provider or None
    custom_model = runtime_model or env_model or None

    # If preferred provider is set, try it first
    if preferred_provider and preferred_provider in LLM_PROVIDERS_CONFIG:
        config = LLM_PROVIDERS_CONFIG[preferred_provider]
        api_key = os.getenv(config["env_key"])
        if api_key:
            result = {
                "provider": preferred_provider,
                "api_key": api_key,
                "base_url": config["base_url"],
                "model": custom_model or config["default_model"],
                "embedding_model": config["embedding_model"],
                "is_free": config["is_free"],
            }
            # Add default_headers if provider has them
            if "default_headers" in config:
                result["default_headers"] = config["default_headers"]
            return result

    # Auto-select based on priority (free providers first)
    sorted_providers = sorted(
        LLM_PROVIDERS_CONFIG.items(),
        key=lambda x: x[1]["priority"]
    )

    for provider_name, config in sorted_providers:
        api_key = os.getenv(config["env_key"])
        if api_key:
            result = {
                "provider": provider_name,
                "api_key": api_key,
                "base_url": config["base_url"],
                "model": custom_model or config["default_model"],
                "embedding_model": config["embedding_model"],
                "is_free": config["is_free"],
            }
            # Add default_headers if provider has them
            if "default_headers" in config:
                result["default_headers"] = config["default_headers"]
            return result

    return None


def init_ragas():
    """Initialize RAGAS metrics."""
    global RAGAS_AVAILABLE
    try:
        # Just check if ragas is importable
        import ragas  # noqa: F401
        RAGAS_AVAILABLE = True
        logger.info("RAGAS metrics initialized successfully")
        return True
    except ImportError as e:
        logger.warning("RAGAS metrics not available: %s", e)
        RAGAS_AVAILABLE = False
        return False


# Try to initialize on startup
init_ragas()


def safe_float(value, default=None):
    """Convert value to float, handling NaN and None."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f):
            logger.warning("[SAFE_FLOAT] NaN detected, returning default: %s", default)
            return default
        if math.isinf(f):
            logger.warning("[SAFE_FLOAT] Inf detected, returning default: %s", default)
            return default
        return f
    except (ValueError, TypeError) as e:
        logger.warning("[SAFE_FLOAT] Conversion error for value %s: %s", value, e)
        return default


class VoyageDirectEmbeddings:
    """Direct Voyage AI embeddings via REST API.
    
    Avoids langchain_voyageai's HuggingFace tokenizer download
    which causes permission errors in Docker containers.
    Implements the same interface RAGAS expects (embed_query, embed_documents).
    """
    
    def __init__(self, api_key: str, model: str = "voyage-3"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.voyageai.com/v1/embeddings"
    
    def _call_api(self, texts: list, input_type: str = None) -> list:
        import httpx
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self.model,
        }
        if input_type:
            payload["input_type"] = input_type
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
    
    def embed_query(self, text: str) -> list:
        return self._call_api([text], input_type="query")[0]
    
    def embed_text(self, text: str) -> list:
        return self.embed_query(text)
    
    def embed_documents(self, texts: list) -> list:
        if not texts:
            return []
        # Voyage API max 128 texts per request
        all_embeddings = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self._call_api(batch, input_type="document"))
        return all_embeddings


def get_embeddings(provider: Optional[str] = None, model: Optional[str] = None):
    """Get embeddings instance based on provider and model.
    
    Args:
        provider: Embedding provider ("voyage", "openai", "openrouter", etc.)
        model: Embedding model name
    
    If provider/model not specified, falls back to OpenRouter text-embedding-3-small.
    """
    from langchain_openai import OpenAIEmbeddings
    
    logger.info(f"[EMBEDDINGS] Requested provider: {provider}, model: {model}")
    
    # Voyage AI embeddings
    if provider == "voyage":
        voyage_key = os.getenv("VOYAGE_API_KEY")
        logger.info(f"[EMBEDDINGS] Voyage provider requested, API key present: {bool(voyage_key)}, VOYAGE_AVAILABLE: {VOYAGE_AVAILABLE}")
        if voyage_key:
            # Use direct API wrapper to avoid HuggingFace tokenizer download issues
            embedding_model = model or "voyage-3"
            if embedding_model.startswith("voyage/"):
                embedding_model = embedding_model.replace("voyage/", "", 1)
            logger.info(f"[EMBEDDINGS] Using Voyage AI (direct API): {embedding_model}")
            return VoyageDirectEmbeddings(
                api_key=voyage_key,
                model=embedding_model,
            )
    
    # OpenAI embeddings (direct)
    if provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            embedding_model = model or "text-embedding-3-small"
            logger.info(f"[EMBEDDINGS] Using OpenAI: {embedding_model}")
            return OpenAIEmbeddings(
                api_key=openai_key,
                model=embedding_model
            )
    
    # OpenRouter embeddings
    if provider == "openrouter" or provider is None:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            embedding_model = model or "openai/text-embedding-3-small"
            logger.info(f"[EMBEDDINGS] Using OpenRouter: {embedding_model}")
            return OpenAIEmbeddings(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                model=embedding_model,
                default_headers={
                    "HTTP-Referer": "http://localhost:8001",
                    "X-Title": SERVICE_NAME,
                }
            )

    # Final fallback to OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logger.info("[EMBEDDINGS] Fallback to OpenAI text-embedding-3-small")
        return OpenAIEmbeddings(
            api_key=openai_key,
            model="text-embedding-3-small"
        )

    logger.warning("No embedding API key available")
    return None


def evaluate_with_ragas(input_data: EvaluationInput) -> EvaluationOutput:
    """Evaluate using RAGAS library with configured LLM provider."""
    llm_config = get_llm_config()
    if not llm_config:
        return EvaluationOutput(error="No LLM API key configured")

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        )
        from langchain_openai import ChatOpenAI

        # DEBUG: Log the evaluation_model received from request
        logger.info(
            "[RAGAS DEBUG] Başlatılıyor - "
            "evaluation_model: %s, llm_config model: %s",
            input_data.evaluation_model,
            llm_config['model']
        )

        # Use evaluation_model from input if provided, otherwise use config
        # If evaluation_model contains provider prefix (e.g., "apiclaudegg/gpt-5.1"), extract just the model name
        model_to_use = input_data.evaluation_model or llm_config["model"]
        if model_to_use and "/" in model_to_use:
            # Extract model name from "provider/model" format
            model_to_use = model_to_use.split("/", 1)[1]
            logger.info(
                "[RAGAS DEBUG] Extracted model name from provider/model format: %s",
                model_to_use
            )
        
        logger.info(
            "[RAGAS DEBUG] Kullanılacak model: %s (provider: %s)",
            model_to_use,
            llm_config['provider']
        )
        
        # Configure LLM from config (use current provider settings)
        # Note: Most providers (OpenRouter, Claude.gg, etc.) don't support n>1
        # We explicitly set n=1 to avoid "returned 1 generations instead of 3" warnings
        llm_kwargs = {
            "model": model_to_use,
            "api_key": llm_config["api_key"],
            "temperature": 0,
            "n": 1,  # Explicitly set to 1 to avoid multi-generation warnings
        }

        if llm_config["base_url"]:
            llm_kwargs["base_url"] = llm_config["base_url"]
            
        # Add default headers from config if available
        if "default_headers" in llm_config:
            llm_kwargs["default_headers"] = llm_config["default_headers"]
        # Legacy: Add headers for OpenRouter and Claude.gg if not already set
        elif llm_config["provider"] in ["openrouter", "claudegg"]:
            llm_kwargs["default_headers"] = {
                "HTTP-Referer": "http://localhost:8001",
                "X-Title": SERVICE_NAME,
            }

        llm = ChatOpenAI(**llm_kwargs)
        
        # Get embeddings using course's embedding settings
        embeddings = get_embeddings(
            provider=input_data.embedding_provider,
            model=input_data.embedding_model
        )

        if not embeddings:
            return EvaluationOutput(
                error="No embeddings API key (need OPENROUTER or OPENAI)"
            )

        logger.info("[RAGAS DEBUG] Answer Relevancy metriği hazırlandı (Orijinal RAGAS)")

        # Prepare dataset for RAGAS
        # RAGAS uses "reference" column (single string)
        # If multiple ground truths, find best match using embedding similarity
        best_reference = input_data.ground_truth
        
        if input_data.ground_truths and len(input_data.ground_truths) > 1:
            logger.info(
                "[RAGAS ALT GT] Finding best match from %d ground truths",
                len(input_data.ground_truths)
            )
            # Find best matching ground truth using embedding similarity
            answer_embedding = embeddings.embed_query(input_data.generated_answer)
            
            best_score = -1
            for gt in input_data.ground_truths:
                gt_embedding = embeddings.embed_query(gt)
                # Cosine similarity
                import numpy as np
                similarity = np.dot(answer_embedding, gt_embedding) / (
                    np.linalg.norm(answer_embedding) * np.linalg.norm(gt_embedding)
                )
                if similarity > best_score:
                    best_score = similarity
                    best_reference = gt
            
            logger.info(
                "[RAGAS ALT GT] Best match score: %.4f, Reference: %s...",
                best_score,
                best_reference[:50]
            )
        
        data = {
            "question": [input_data.question],
            "answer": [input_data.generated_answer],
            "contexts": [input_data.retrieved_contexts],
            "reference": [best_reference],  # Best matching GT
        }

        dataset = Dataset.from_dict(data)
        
        # DEBUG: Log input data details
        logger.info(
            "[RAGAS DEBUG] Input detayları - "
            "Soru uzunluğu: %d, Cevap uzunluğu: %d, Context sayısı: %d",
            len(input_data.question),
            len(input_data.generated_answer),
            len(input_data.retrieved_contexts)
        )

        # Run evaluation
        logger.info("[RAGAS DEBUG] Değerlendirme başlatılıyor...")
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            ],
            llm=llm,
            embeddings=embeddings,
        )
        logger.info("[RAGAS DEBUG] Değerlendirme tamamlandı")

        # Parse and log the result
        parsed_result = _parse_ragas_result(result, input_data)
        
        # Log reranker usage if present
        if input_data.reranker_provider:
            logger.info(
                "[RAGAS] Reranker kullanıldı: %s/%s",
                input_data.reranker_provider,
                input_data.reranker_model or "default"
            )
        
        # DEBUG: Detailed metric logging
        logger.info(
            "[RAGAS SONUÇLAR] ========================================\n"
            "Soru: %s\n"
            "Cevap Uzunluğu: %d karakter\n"
            "Context Sayısı: %d\n"
            "--- METRİKLER ---\n"
            "Faithfulness (Sadakat): %s\n"
            "Answer Relevancy (Cevap İlgisi): %s\n"
            "Context Precision (Bağlam Kesinliği): %s\n"
            "Context Recall (Bağlam Hatırlama): %s\n"
            "Answer Correctness (Cevap Doğruluğu): %s\n"
            "Reranker: %s\n"
            "========================================",
            input_data.question[:100],
            len(input_data.generated_answer),
            len(input_data.retrieved_contexts),
            f"{parsed_result.faithfulness:.3f}" if parsed_result.faithfulness else "N/A",
            f"{parsed_result.answer_relevancy:.3f}" if parsed_result.answer_relevancy else "N/A",
            f"{parsed_result.context_precision:.3f}" if parsed_result.context_precision else "N/A",
            f"{parsed_result.context_recall:.3f}" if parsed_result.context_recall else "N/A",
            f"{parsed_result.answer_correctness:.3f}" if parsed_result.answer_correctness else "N/A",
            f"{input_data.reranker_provider}/{input_data.reranker_model}" if input_data.reranker_provider else "none"
        )
        
        return parsed_result

    except Exception as e:
        logger.error("[RAGAS ERROR] Değerlendirme hatası: %s", e, exc_info=True)
        return EvaluationOutput(error=str(e))


def _parse_ragas_result(result, input_data: EvaluationInput) -> EvaluationOutput:
    """Parse RAGAS evaluation result into EvaluationOutput."""
    # DEBUG: Log raw result structure
    logger.info("[RAGAS RAW RESULT] Type: %s", type(result))
    
    # Determine reranker metadata
    reranker_used = bool(input_data.reranker_provider)
    reranker_provider = input_data.reranker_provider
    reranker_model = input_data.reranker_model
    
    if hasattr(result, 'to_pandas'):
        df = result.to_pandas()
        logger.info("[RAGAS RAW RESULT] DataFrame columns: %s", df.columns.tolist())
        logger.info("[RAGAS RAW RESULT] DataFrame values: %s", df.iloc[0].to_dict())
        
        return EvaluationOutput(
            faithfulness=safe_float(
                df['faithfulness'].iloc[0]
            ) if 'faithfulness' in df else None,
            answer_relevancy=safe_float(
                df['answer_relevancy'].iloc[0]
            ) if 'answer_relevancy' in df else None,
            context_precision=safe_float(
                df['context_precision'].iloc[0]
            ) if 'context_precision' in df else None,
            context_recall=safe_float(
                df['context_recall'].iloc[0]
            ) if 'context_recall' in df else None,
            answer_correctness=safe_float(
                df['answer_correctness'].iloc[0]
            ) if 'answer_correctness' in df else None,
            reranker_used=reranker_used,
            reranker_provider=reranker_provider,
            reranker_model=reranker_model,
        )
    if isinstance(result, dict):
        logger.info("[RAGAS RAW RESULT] Dict keys: %s", result.keys())
        logger.info("[RAGAS RAW RESULT] Dict values: %s", result)
        
        return EvaluationOutput(
            faithfulness=safe_float(result.get("faithfulness")),
            answer_relevancy=safe_float(result.get("answer_relevancy")),
            context_precision=safe_float(result.get("context_precision")),
            context_recall=safe_float(result.get("context_recall")),
            answer_correctness=safe_float(result.get("answer_correctness")),
            reranker_used=reranker_used,
            reranker_provider=reranker_provider,
            reranker_model=reranker_model,
        )
    # Try accessing as attributes
    logger.info("[RAGAS RAW RESULT] Attributes: faithfulness=%s, answer_relevancy=%s", 
                getattr(result, 'faithfulness', 'N/A'),
                getattr(result, 'answer_relevancy', 'N/A'))
    
    return EvaluationOutput(
        faithfulness=safe_float(getattr(result, 'faithfulness', None)),
        answer_relevancy=safe_float(getattr(result, 'answer_relevancy', None)),
        context_precision=safe_float(getattr(result, 'context_precision', None)),
        context_recall=safe_float(getattr(result, 'context_recall', None)),
        answer_correctness=safe_float(getattr(result, 'answer_correctness', None)),
        reranker_used=reranker_used,
        reranker_provider=reranker_provider,
        reranker_model=reranker_model,
    )



# ==================== Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    llm_config = get_llm_config()
    provider_info = llm_config["provider"] if llm_config else "none"
    if llm_config and llm_config.get("is_free"):
        provider_info += " (free)"
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        ragas_available=RAGAS_AVAILABLE,
        llm_provider=provider_info
    )


@app.post("/evaluate", response_model=EvaluationOutput)
async def evaluate_question(input_data: EvaluationInput):
    """Evaluate a single question using RAGAS metrics."""
    import time as _time

    llm_config = get_llm_config()

    # Try RAGAS with retry on failure (rate limits, transient errors)
    if RAGAS_AVAILABLE and (llm_config or input_data.evaluation_model):
        if input_data.evaluation_model:
            logger.info(
                "Using RAGAS with custom model: %s",
                input_data.evaluation_model
            )
        else:
            free_info = " (FREE)" if llm_config.get("is_free") else ""
            logger.info(
                "Using RAGAS with %s (%s)%s",
                llm_config['provider'],
                llm_config['model'],
                free_info
            )

        max_ragas_retries = 3
        for attempt in range(max_ragas_retries):
            result = evaluate_with_ragas(input_data)
            if not result.error:
                return result
            logger.warning(
                "RAGAS attempt %d/%d failed: %s",
                attempt + 1, max_ragas_retries, result.error
            )
            if attempt < max_ragas_retries - 1:
                _time.sleep(2 * (attempt + 1))  # Backoff: 2s, 4s

        logger.error(
            "RAGAS failed after %d attempts. Returning null metrics (no fallback).",
            max_ragas_retries
        )
        return EvaluationOutput(
            error=f"RAGAS evaluation failed after {max_ragas_retries} attempts",
            reranker_used=bool(input_data.reranker_provider),
            reranker_provider=input_data.reranker_provider,
            reranker_model=input_data.reranker_model,
        )

    # No RAGAS available and no LLM configured
    logger.error("No RAGAS or LLM available for evaluation")
    return EvaluationOutput(error="No evaluation method available")


@app.post("/evaluate-batch", response_model=BatchEvaluationOutput)
async def evaluate_batch(input_data: BatchEvaluationInput):
    """Evaluate a batch of questions."""
    results = []
    errors = 0
    reranker_usage = {}

    for item in input_data.items:
        result = await evaluate_question(item)
        results.append(result)
        if result.error:
            errors += 1
        
        # Track reranker usage
        provider = result.reranker_provider or "none"
        reranker_usage[provider] = reranker_usage.get(provider, 0) + 1

    return BatchEvaluationOutput(
        results=results,
        total_processed=len(results),
        total_errors=errors,
        reranker_usage=reranker_usage
    )


@app.get("/providers")
async def list_providers():
    """List available LLM providers and their status."""
    providers = []
    for name, config in LLM_PROVIDERS_CONFIG.items():
        api_key = os.getenv(config["env_key"])
        providers.append({
            "name": name,
            "available": bool(api_key),
            "is_free": config["is_free"],
            "default_model": config["default_model"],
            "models": config.get("models", [config["default_model"]]),
            "priority": config["priority"],
        })
    return {
        "providers": sorted(providers, key=lambda x: x["priority"]),
        "current": get_llm_config(),
    }


@app.get("/settings")
async def get_settings():
    """Get current RAGAS settings."""
    llm_config = get_llm_config()
    return {
        "provider": RUNTIME_SETTINGS.get("provider"),
        "model": RUNTIME_SETTINGS.get("model"),
        "current_provider": llm_config["provider"] if llm_config else None,
        "current_model": llm_config["model"] if llm_config else None,
        "is_free": llm_config["is_free"] if llm_config else False,
    }


@app.post("/settings")
async def update_settings(settings: RagasSettings):
    """Update RAGAS settings."""
    if settings.provider is not None:
        # Validate provider
        if settings.provider and settings.provider not in LLM_PROVIDERS_CONFIG:
            valid = list(LLM_PROVIDERS_CONFIG.keys())
            return {"error": f"Invalid provider. Valid: {valid}"}
        RUNTIME_SETTINGS["provider"] = settings.provider or None

    if settings.model is not None:
        RUNTIME_SETTINGS["model"] = settings.model or None

    llm_config = get_llm_config()
    return {
        "message": "Settings updated",
        "provider": RUNTIME_SETTINGS.get("provider"),
        "model": RUNTIME_SETTINGS.get("model"),
        "current_provider": llm_config["provider"] if llm_config else None,
        "current_model": llm_config["model"] if llm_config else None,
        "is_free": llm_config["is_free"] if llm_config else False,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    llm_config = get_llm_config()
    provider_info = llm_config["provider"] if llm_config else "none"
    if llm_config and llm_config.get("is_free"):
        provider_info += " (free)"
    return {
        "service": SERVICE_NAME,
        "version": "1.0.0",
        "ragas_available": RAGAS_AVAILABLE,
        "llm_provider": provider_info,
        "docs": "/docs"
    }


# ==================== Test Generation Endpoint ====================

class TestGenerationRequest(BaseModel):
    """Request for generating test questions from documents."""
    documents: List[str]  # List of document contents
    persona: str  # Student persona description
    test_size: int = 50
    distributions: dict = {
        "simple": 0.4,
        "reasoning": 0.4,
        "multi_context": 0.2
    }
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


class TestGenerationResponse(BaseModel):
    """Response with generated test questions."""
    questions: List[dict]
    total_generated: int
    persona_used: str
    llm_used: str


@app.post("/generate-testset", response_model=TestGenerationResponse)
async def generate_testset(request: TestGenerationRequest):
    """Generate test questions from documents using RAGAS TestsetGenerator.
    
    Uses Bloom taxonomy with different evolution types:
    - simple: Basic factual questions
    - reasoning: Questions requiring logical reasoning
    - multi_context: Questions requiring multiple document contexts
    """
    if not RAGAS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="RAGAS library not available"
        )
    
    llm_config = get_llm_config()
    if not llm_config:
        raise HTTPException(
            status_code=503,
            detail="No LLM configured for test generation"
        )
    
    try:
        # RAGAS 0.2.15 - Direct LLM usage (no wrapper needed)
        from ragas.testset import TestsetGenerator
        from langchain_openai import ChatOpenAI
        from langchain_core.documents import Document
        
        # Use provided LLM or get from config
        model_to_use = request.llm_model or llm_config["model"]
        provider_to_use = request.llm_provider or llm_config["provider"]
        
        logger.info(
            "[TEST GEN] Starting generation with %s documents, persona: %s",
            len(request.documents),
            request.persona[:50]
        )
        
        # Configure LLM
        llm_kwargs = {
            "model": model_to_use,
            "api_key": llm_config["api_key"],
            "temperature": 0.7,  # More creative for questions
        }
        
        if llm_config["base_url"]:
            llm_kwargs["base_url"] = llm_config["base_url"]
        
        # Add default headers from config if available
        if "default_headers" in llm_config:
            llm_kwargs["default_headers"] = llm_config["default_headers"]
        # Legacy: Add headers for OpenRouter and Claude.gg if not already set
        elif provider_to_use in ["openrouter", "claudegg"]:
            llm_kwargs["default_headers"] = {
                "HTTP-Referer": "http://localhost:8001",
                "X-Title": SERVICE_NAME,
            }
        
        generator_llm = ChatOpenAI(**llm_kwargs)
        critic_llm = ChatOpenAI(**llm_kwargs)  # Same LLM for critic
        
        # Get embeddings
        embeddings_obj = get_embeddings()
        if not embeddings_obj:
            raise HTTPException(
                status_code=503,
                detail="No embeddings configured"
            )
        
        # Create TestsetGenerator with proper embeddings
        logger.info("[TEST GEN] Creating TestsetGenerator (with embeddings)...")
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        
        # RAGAS needs proper wrappers after all!
        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            LangchainEmbeddingsWrapper(embeddings_obj)  # WRAP embeddings!
        )
        
        # Convert documents with BETTER chunking for RAGAS
        logger.info("[TEST GEN] Converting %d documents...", len(request.documents))
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Combine all documents into one text
        full_text = "\n\n".join(request.documents)
        
        # Use RecursiveCharacterTextSplitter (BETTER for RAGAS!)
        # 1500 chars = ~300-400 tokens, enough context for questions
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks = more context
            chunk_overlap=200,  # More overlap = better coherence
            length_function=len,
        )
        
        langchain_docs = splitter.create_documents(
            texts=[full_text],
            metadatas=[{"source": "uploaded_pdf"}]
        )
        
        logger.info("[TEST GEN] Created %d chunks from documents", len(langchain_docs))
        
        # Debug: Log first chunk
        if langchain_docs:
            logger.info(
                "[TEST GEN DEBUG] First chunk (%d chars): %s...",
                len(langchain_docs[0].page_content),
                langchain_docs[0].page_content[:200]
            )
        
        # RAGAS 0.4.3 - Custom transforms WITHOUT HeadlineSplitter
        logger.info("[TEST GEN] Generating %d questions (v0.4.3)...", request.test_size)
        
        from ragas.testset.transforms import (
            EmbeddingExtractor,
            KeyphrasesExtractor,
            SummaryExtractor,
        )
        
        # Create custom transforms list (skip HeadlineSplitter!)
        custom_transforms = [
            SummaryExtractor(llm=generator_llm),
            KeyphrasesExtractor(llm=generator_llm),
            EmbeddingExtractor(embeddings=embeddings_obj),
        ]
        
        logger.info("[TEST GEN] Using %d custom transforms", len(custom_transforms))
        
        # Generate with custom transforms
        testset = generator.generate_with_langchain_docs(
            langchain_docs,
            testset_size=request.test_size,
            transforms=custom_transforms,
        )
        
        logger.info("[TEST GEN] Generation complete! Converting...")
        
        # Convert to pandas DataFrame
        df = testset.to_pandas()
        
        # Extract questions from DataFrame
        questions = []
        for _, row in df.iterrows():
            questions.append({
                "question": row.get("user_input", row.get("question", "")),
                "ground_truth": row.get("reference", row.get("ground_truth", "")),
                "contexts": row.get("reference_contexts", row.get("contexts", [])),
            })
        
        logger.info("[TEST GEN] Generated %d questions", len(questions))
        
        return TestGenerationResponse(
            questions=questions,
            total_generated=len(questions),
            persona_used=request.persona,
            llm_used=f"{provider_to_use}/{model_to_use}"
        )
        
    except ImportError as e:
        logger.error("[TEST GEN] Import error: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Required library not available: {e}"
        )
    except Exception as e:
        logger.error("[TEST GEN] Generation error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Test generation failed: {e}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
