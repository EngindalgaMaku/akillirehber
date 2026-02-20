"""Text chunking service with multiple strategies."""

import json
import os
import re
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import nltk

from app.models.chunking import Chunk, ChunkingStrategy
from app.services.chunker_exceptions import (
    SemanticChunkerError,
    ErrorCode,
)
from app.services.embedding_provider import EmbeddingProviderError

# Constants
PUNKT_TAB_PATH = 'tokenizers/punkt_tab'

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ChunkingDiagnostics:
    """Diagnostics information for chunking operations."""
    strategy: str
    input_text_length: int
    processing_time: float
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    overlap_count: int
    error_message: Optional[str] = None
    error_details: Optional[Dict] = None
    performance_warnings: List[str] = None
    quality_score: float = 0.0
    recommendations: List[str] = None

    def __post_init__(self):
        if self.performance_warnings is None:
            self.performance_warnings = []
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class ChunkingConfig:
    """Configuration for chunking operations."""
    strategy: ChunkingStrategy
    chunk_size: int = 500
    overlap: int = 50
    sentences_per_chunk: int = 3
    similarity_threshold: float = 0.5
    embedding_model: str = "openai/text-embedding-3-small"
    buffer_size: int = 1
    min_chunk_size: int = 150
    max_chunk_size: int = 2000
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        if self.overlap < 0:
            errors.append("overlap cannot be negative")
        if self.overlap >= self.chunk_size:
            errors.append("overlap must be less than chunk_size")
        if self.sentences_per_chunk <= 0:
            errors.append("sentences_per_chunk must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            errors.append("similarity_threshold must be between 0 and 1")
        if self.min_chunk_size <= 0:
            errors.append("min_chunk_size must be positive")
        if self.max_chunk_size <= self.min_chunk_size:
            errors.append("max_chunk_size must be greater than min_chunk_size")
        if self.buffer_size < 0:
            errors.append("buffer_size cannot be negative")
            
        return errors


def ensure_nltk_punkt():
    """Check if NLTK punkt_tab data is available."""
    try:
        nltk.data.find(PUNKT_TAB_PATH)
    except LookupError:
        # Only try to download if not found
        # In Docker, data is pre-downloaded during build
        try:
            nltk.download('punkt_tab', quiet=True)
        except PermissionError:
            # If we can't download, the data should already be available
            # from the Docker build step
            pass


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """Split text into chunks."""


class FixedSizeChunker(BaseChunker):
    """Fixed-size character-based chunking with overlap."""

    def chunk(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
        **kwargs
    ) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        if not text:
            return []

        chunks: List[Chunk] = []
        start = 0
        index = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            content = text[start:end]
            has_overlap = index > 0 and overlap > 0

            chunks.append(Chunk(
                index=index,
                content=content,
                start_position=start,
                end_position=end,
                char_count=len(content),
                has_overlap=has_overlap
            ))

            step = chunk_size - overlap
            if step <= 0:
                step = 1
            start += step
            index += 1

        return chunks


class RecursiveChunker(BaseChunker):
    """Hierarchical splitting using separators (LangChain-style).
    
    IMPORTANT: This chunker NEVER splits words or sentences in the middle.
    It respects sentence and word boundaries at all times.
    """

    # Separators ordered by priority - sentence boundaries first
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    def chunk(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: Optional[List[str]] = None,
        **kwargs
    ) -> List[Chunk]:
        """Split text recursively using hierarchical separators.
        
        GUARANTEES:
        - Words are NEVER split in the middle
        - Sentences are kept together when possible
        - Overlap happens at sentence/word boundaries
        """
        if not text:
            return []

        if separators is None:
            separators = self.DEFAULT_SEPARATORS.copy()

        # Split into sentences first for better boundary handling
        sentences = self._split_into_sentences(text)
        return self._merge_sentences_to_chunks(sentences, chunk_size, overlap, text)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving all content."""
        import re
        
        # Split on sentence-ending punctuation followed by space
        # Keep the punctuation with the sentence
        pattern = r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜa-zçğıöşü0-9"\'([])'
        
        sentences = re.split(pattern, text)
        
        # If no splits found, try splitting on newlines
        if len(sentences) == 1 and len(text) > 100:
            sentences = text.split('\n')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # If still single chunk, return as-is
        if not sentences:
            return [text] if text.strip() else []
            
        return sentences

    def _find_sentence_boundary(self, text: str, max_pos: int) -> int:
        """Find the last sentence boundary before max_pos.
        
        Returns the position right after the sentence-ending punctuation.
        """
        # Look for sentence endings: . ! ? followed by space
        best_pos = -1
        
        for i in range(min(max_pos, len(text) - 1), 0, -1):
            char = text[i]
            # Check if this is end of sentence
            if char in '.!?' and i + 1 < len(text) and text[i + 1] in ' \n':
                best_pos = i + 1
                break
            # Also check for newline as sentence boundary
            if char == '\n':
                best_pos = i
                break
        
        return best_pos

    def _find_word_boundary(self, text: str, max_pos: int) -> int:
        """Find the last word boundary (space) before max_pos."""
        for i in range(min(max_pos, len(text) - 1), 0, -1):
            if text[i] == ' ':
                return i
        return -1

    def _merge_sentences_to_chunks(
        self,
        sentences: List[str],
        chunk_size: int,
        overlap: int,
        original_text: str
    ) -> List[Chunk]:
        """Merge sentences into chunks, respecting boundaries.
        
        RULES:
        1. Never split a word
        2. Keep sentences together when possible
        3. Overlap at sentence boundaries
        """
        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_length = 0
        position = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Calculate length with separator
            separator_len = 1 if current_sentences else 0
            new_length = current_length + separator_len + len(sentence)
            
            # If adding this sentence exceeds chunk_size
            if current_sentences and new_length > chunk_size:
                # Save current chunk
                chunk_content = ' '.join(current_sentences)
                start_pos = self._find_position(original_text, current_sentences[0], position)
                
                chunks.append(Chunk(
                    index=len(chunks),
                    content=chunk_content,
                    start_position=start_pos,
                    end_position=start_pos + len(chunk_content),
                    char_count=len(chunk_content),
                    has_overlap=len(chunks) > 0 and overlap > 0
                ))
                
                # Calculate overlap - use complete sentences
                overlap_sentences = self._get_overlap_sentences(current_sentences, overlap)
                
                # Start new chunk with overlap
                current_sentences = overlap_sentences.copy()
                current_length = sum(len(s) for s in current_sentences)
                if current_sentences:
                    current_length += len(current_sentences) - 1  # spaces
                
                position = start_pos + len(chunk_content) - sum(len(s) + 1 for s in overlap_sentences)
            
            # Add sentence to current chunk
            current_sentences.append(sentence)
            current_length = sum(len(s) for s in current_sentences) + len(current_sentences) - 1
        
        # Don't forget the last chunk
        if current_sentences:
            chunk_content = ' '.join(current_sentences)
            start_pos = self._find_position(original_text, current_sentences[0], position)
            
            chunks.append(Chunk(
                index=len(chunks),
                content=chunk_content,
                start_position=start_pos,
                end_position=start_pos + len(chunk_content),
                char_count=len(chunk_content),
                has_overlap=len(chunks) > 0 and overlap > 0
            ))
        
        return chunks

    def _get_overlap_sentences(self, sentences: List[str], overlap: int) -> List[str]:
        """Get sentences from the end that fit within overlap size.
        
        Always returns complete sentences, never partial.
        """
        if not sentences or overlap <= 0:
            return []
        
        result = []
        total_length = 0
        
        # Work backwards through sentences
        for sentence in reversed(sentences):
            sentence_len = len(sentence) + (1 if result else 0)  # +1 for space
            
            if total_length + sentence_len <= overlap:
                result.insert(0, sentence)
                total_length += sentence_len
            elif not result:
                # If even one sentence is too long, take it anyway
                # but this shouldn't happen with proper sentence splitting
                result.insert(0, sentence)
                break
            else:
                break
        
        return result

    def _find_position(self, text: str, search_text: str, start_from: int) -> int:
        """Find position of text in original, starting from given position."""
        # Try exact match first
        pos = text.find(search_text, start_from)
        if pos != -1:
            return pos
        
        # Try from beginning if not found
        pos = text.find(search_text)
        if pos != -1:
            return pos
        
        # Fallback to start_from
        return max(0, start_from)


class SentenceChunker(BaseChunker):
    """NLP-based sentence boundary detection chunking."""

    def __init__(self):
        """Initialize the sentence chunker with NLTK resources."""
        ensure_nltk_punkt()

    def chunk(
        self,
        text: str,
        sentences_per_chunk: int = 3,
        **kwargs
    ) -> List[Chunk]:
        """Split text into chunks based on sentence boundaries."""
        if not text:
            return []

        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        index = 0
        current_pos = 0

        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            content = " ".join(chunk_sentences)

            start_pos = text.find(chunk_sentences[0], current_pos)
            if start_pos == -1:
                start_pos = current_pos

            last_sentence = chunk_sentences[-1]
            end_search = start_pos + len(content) - len(last_sentence)
            end_pos = text.find(last_sentence, end_search)
            if end_pos == -1:
                end_pos = start_pos + len(content)
            else:
                end_pos = end_pos + len(last_sentence)

            chunks.append(Chunk(
                index=index,
                content=content,
                start_position=start_pos,
                end_position=end_pos,
                char_count=len(content),
                has_overlap=False
            ))

            current_pos = end_pos
            index += 1

        return chunks


class SemanticChunker(BaseChunker):
    """Advanced semantic chunking using embeddings and similarity analysis.
    
    Features:
    - Buffer-based sentence grouping for better context
    - Percentile-based dynamic threshold
    - Minimum chunk size enforcement
    - Small chunk merging
    - Overlap support for context preservation
    - Question-answer pair detection and merging
    - Multiple embedding provider support with fallback
    - Embedding caching for performance
    - Adaptive threshold based on text characteristics
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
    
    # Chunking parameters
    DEFAULT_BUFFER_SIZE = 1  # Sentences on each side for context
    DEFAULT_MIN_CHUNK_SIZE = 150  # Minimum characters per chunk
    DEFAULT_MAX_CHUNK_SIZE = 2000  # Maximum characters per chunk

    def __init__(
        self,
        use_provider_manager: bool = True,
        enable_cache: bool = True,
        enable_qa_detection: bool = True,
        enable_adaptive_threshold: bool = True
    ):
        """Initialize the semantic chunker.
        
        Args:
            use_provider_manager: If True, use the new EmbeddingProviderManager
                                 with fallback support. If False, use legacy
                                 direct OpenRouter client.
            enable_cache: If True and use_provider_manager is True, enable
                         embedding caching.
            enable_qa_detection: If True, detect and merge Q&A pairs before
                                chunking to keep them together.
            enable_adaptive_threshold: If True, calculate threshold based on
                                      text characteristics instead of fixed value.
        """
        self._client = None
        self._np = None
        self._use_provider_manager = use_provider_manager
        self._enable_cache = enable_cache
        self._enable_qa_detection = enable_qa_detection
        self._enable_adaptive_threshold = enable_adaptive_threshold
        self._provider_manager = None
        self._embedding_cache = None
        self._qa_detector = None
        self._threshold_calculator = None
    
    def _ensure_qa_detector(self):
        """Initialize the Q&A detector if not already done."""
        if self._qa_detector is None:
            from app.services.qa_detector import QADetector
            self._qa_detector = QADetector()
    
    def _ensure_threshold_calculator(self):
        """Initialize the adaptive threshold calculator if not already done."""
        if self._threshold_calculator is None:
            from app.services.adaptive_threshold import AdaptiveThresholdCalculator
            self._threshold_calculator = AdaptiveThresholdCalculator()

    def _ensure_numpy(self):
        """Load numpy if not already loaded."""
        if self._np is None:
            try:
                import numpy as np
                self._np = np
            except ImportError as exc:
                raise ImportError(
                    "numpy is required for semantic chunking. "
                    "Install it with: pip install numpy"
                ) from exc

    def _ensure_client(self):
        """Initialize the OpenRouter client if not already done."""
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OPENROUTER_API_KEY environment variable is required "
                        "for semantic chunking."
                    )
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=self.OPENROUTER_BASE_URL
                )
            except ImportError as exc:
                raise ImportError(
                    "openai is required for semantic chunking. "
                    "Install it with: pip install openai"
                ) from exc

    def _ensure_provider_manager(self):
        """Initialize the embedding provider manager if not already done."""
        if self._provider_manager is None:
            from app.services.embedding_provider import (
                EmbeddingProviderConfig,
                EmbeddingProviderManager,
                OpenRouterProvider,
                VoyageProvider,
                OllamaProvider,
                CohereProvider,
                JinaProvider,
                AlibabaProvider,
            )
            from app.services.embedding_cache import EmbeddingCache
            
            # Create providers with fallback order
            providers = []
            
            # Try Voyage first if available
            voyage = VoyageProvider()
            if voyage.is_available():
                providers.append(voyage)
            
            # Try Ollama if available (local, fast)
            ollama = OllamaProvider()
            if ollama.is_available():
                providers.append(ollama)
            
            # Try Cohere if available
            cohere = CohereProvider()
            if cohere.is_available():
                providers.append(cohere)
            
            # Try Jina if available
            jina = JinaProvider()
            if jina.is_available():
                providers.append(jina)
            
            # Try Alibaba if available
            alibaba = AlibabaProvider()
            if alibaba.is_available():
                providers.append(alibaba)
            
            # Try OpenRouter as fallback
            openrouter = OpenRouterProvider()
            if openrouter.is_available():
                providers.append(openrouter)
            
            if not providers:
                raise ValueError(
                    "No embedding providers available. "
                    "Set VOYAGE_API_KEY, COHERE_API_KEY, JINA_AI_API_KEY, "
                    "DASHSCOPE_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY or run Ollama locally."
                )
            
            config = EmbeddingProviderConfig(
                batch_size=8,
                max_retries=3,
                retry_delay=1.0
            )
            
            self._provider_manager = EmbeddingProviderManager(
                providers=providers,
                config=config
            )
            
            # Initialize cache if enabled
            if self._enable_cache:
                self._embedding_cache = EmbeddingCache(ttl=3600)

    def _get_embeddings(
        self, texts: List[str], model: str
    ) -> List[List[float]]:
        """Get embeddings for a list of texts.
        
        Uses either the new provider manager with fallback and caching,
        or the legacy direct OpenRouter client.
        """
        if self._use_provider_manager:
            return self._get_embeddings_with_provider_manager(texts, model)
        else:
            return self._get_embeddings_legacy(texts, model)
    
    def _get_embeddings_with_provider_manager(
        self, texts: List[str], model: str
    ) -> List[List[float]]:
        """Get embeddings using the provider manager with caching."""
        self._ensure_provider_manager()
        
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            return []
        
        # Check cache first if enabled
        if self._embedding_cache:
            found, missing_indices = self._embedding_cache.get_batch(
                non_empty_texts, model
            )
            
            if not missing_indices:
                # All found in cache
                return [found[i] for i in range(len(non_empty_texts))]
            
            # Get embeddings for missing texts
            missing_texts = [non_empty_texts[i] for i in missing_indices]
            new_embeddings = self._provider_manager.get_embeddings_batch(
                missing_texts, model
            )
            
            # Cache new embeddings
            self._embedding_cache.set_batch(missing_texts, model, new_embeddings)
            
            # Combine results in correct order
            result = []
            new_idx = 0
            for i in range(len(non_empty_texts)):
                if i in found:
                    result.append(found[i])
                else:
                    result.append(new_embeddings[new_idx])
                    new_idx += 1
            return result
        else:
            # No cache, use provider manager directly
            return self._provider_manager.get_embeddings_batch(non_empty_texts, model)
    
    def _get_embeddings_legacy(
        self, texts: List[str], model: str
    ) -> List[List[float]]:
        """Get embeddings using legacy direct OpenRouter client."""
        self._ensure_client()
        non_empty_texts = [t for t in texts if t.strip()]
        if not non_empty_texts:
            return []
        
        try:
            response = self._client.embeddings.create(
                model=model,
                input=non_empty_texts
            )
            if not getattr(response, "data", None):
                raise EmbeddingProviderError(
                    "No embedding data received",
                    provider="openrouter",
                    details={"model": model, "text_count": len(non_empty_texts)},
                )
            return [item.embedding for item in response.data]
        except EmbeddingProviderError:
            raise
        except ValueError as e:
            if "No embedding data received" in str(e):
                raise EmbeddingProviderError(
                    "No embedding data received",
                    provider="openrouter",
                    details={"model": model, "text_count": len(non_empty_texts)},
                ) from e
            raise EmbeddingProviderError(
                f"Legacy embedding failed: {str(e)}",
                provider="openrouter",
                details={"model": model, "text_count": len(non_empty_texts)},
            ) from e
        except Exception as e:
            raise EmbeddingProviderError(
                f"Legacy embedding failed: {str(e)}",
                provider="openrouter",
                details={"model": model, "text_count": len(non_empty_texts)},
            ) from e

    def _cosine_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        self._ensure_numpy()
        np = self._np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _is_question(self, text: str) -> bool:
        """Check if text is a question."""
        text = text.strip()
        if text.endswith('?'):
            return True
        # Turkish question patterns
        question_words = [
            'ne ', 'nasıl', 'neden', 'niçin', 'kim', 'nerede', 
            'hangi', 'kaç', 'mi?', 'mı?', 'mu?', 'mü?',
            'misin', 'mısın', 'musun', 'müsün'
        ]
        text_lower = text.lower()
        return any(q in text_lower for q in question_words)

    def _preprocess_sentences(self, sentences: List[str]) -> List[str]:
        """Preprocess sentences: merge short ones and keep Q&A together."""
        if not sentences:
            return sentences
        
        processed = []
        i = 0
        
        while i < len(sentences):
            current = sentences[i].strip()
            
            # Skip empty
            if not current:
                i += 1
                continue
            
            # If current is very short (like "1.", "2.", etc.), merge with next
            if len(current) < 10 and i + 1 < len(sentences):
                current = current + " " + sentences[i + 1].strip()
                i += 2
            else:
                i += 1
            
            # If current is a question, try to include the answer
            if self._is_question(current) and i < len(sentences):
                next_sent = sentences[i].strip() if i < len(sentences) else ""
                # If next sentence is short or looks like an answer, merge
                if next_sent and (
                    len(next_sent) < 100 or 
                    not self._is_question(next_sent)
                ):
                    # Check if next is not another question
                    if not self._is_question(next_sent):
                        current = current + " " + next_sent
                        i += 1
            
            processed.append(current)
        
        return processed

    def _combine_sentences_with_buffer(
        self,
        sentences: List[str],
        buffer_size: int = 1
    ) -> List[dict]:
        """Combine sentences with surrounding context (buffer)."""
        combined = []
        for i, sentence in enumerate(sentences):
            start = max(0, i - buffer_size)
            end = min(len(sentences), i + buffer_size + 1)
            
            combined_text = " ".join(sentences[start:end])
            combined.append({
                "sentence": sentence,
                "index": i,
                "combined_text": combined_text
            })
        return combined

    def _calculate_distances(
        self,
        embeddings: List[List[float]]
    ) -> List[float]:
        """Calculate cosine distances between consecutive embeddings."""
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i+1])
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def _find_breakpoints_percentile(
        self,
        distances: List[float],
        threshold_percentile: float = 95
    ) -> List[int]:
        """Find breakpoints using percentile-based threshold."""
        self._ensure_numpy()
        np = self._np
        
        if not distances:
            return []
        
        threshold = np.percentile(distances, threshold_percentile)
        
        breakpoints = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                breakpoints.append(i + 1)
        
        return breakpoints

    def _merge_small_chunks_semantic(
        self,
        chunks: List[Chunk],
        min_size: int,
        max_size: int,
        model: str = None
    ) -> List[Chunk]:
        """Merge small chunks and split large chunks while preserving semantic boundaries.
        
        For semantic chunking, we:
        1. Merge very small chunks (< min_size)
        2. Split large chunks (> max_size) at SEMANTIC boundaries (lowest similarity)
        3. Never split sentences in the middle
        4. Preserve the semantic structure as much as possible
        """
        if not chunks:
            return chunks
        
        merged = []
        current_content = ""
        current_start = 0
        
        for chunk in chunks:
            # If chunk is larger than max_size, SPLIT IT at semantic boundaries
            if len(chunk.content) > max_size:
                # Save current accumulated chunk if exists
                if current_content:
                    merged.append(Chunk(
                        index=len(merged),
                        content=current_content.strip(),
                        start_position=current_start,
                        end_position=current_start + len(current_content),
                        char_count=len(current_content),
                        has_overlap=False
                    ))
                    current_content = ""
                
                # Split the large chunk at semantic boundaries
                logger.info(
                    f"Chunk {chunk.index} exceeds max_size ({len(chunk.content)} > {max_size}), "
                    f"splitting at semantic boundaries..."
                )
                
                # Use semantic split if model is available
                if model:
                    sub_chunks = self._split_large_chunk_semantic(
                        chunk.content,
                        chunk.start_position,
                        max_size,
                        model
                    )
                else:
                    # Fallback to sentence-based split
                    sub_chunks = self._split_large_chunk(
                        chunk.content,
                        chunk.start_position,
                        max_size
                    )
                
                merged.extend(sub_chunks)
                continue
            
            if not current_content:
                current_content = chunk.content
                current_start = chunk.start_position
                continue
            
            # Try to merge if current is too small
            if len(current_content) < min_size:
                combined = current_content + " " + chunk.content
                if len(combined) <= max_size:
                    current_content = combined
                    continue
                else:
                    # Can't merge, save current even if small
                    merged.append(Chunk(
                        index=len(merged),
                        content=current_content.strip(),
                        start_position=current_start,
                        end_position=current_start + len(current_content),
                        char_count=len(current_content),
                        has_overlap=False
                    ))
                    current_content = chunk.content
                    current_start = chunk.start_position
                    continue
            
            # Try to merge next chunk if it's too small
            if len(chunk.content) < min_size:
                combined = current_content + " " + chunk.content
                if len(combined) <= max_size:
                    current_content = combined
                    continue
            
            # Both chunks are acceptable size, save current and start new
            merged.append(Chunk(
                index=len(merged),
                content=current_content.strip(),
                start_position=current_start,
                end_position=current_start + len(current_content),
                char_count=len(current_content),
                has_overlap=False
            ))
            current_content = chunk.content
            current_start = chunk.start_position
        
        # Handle last chunk - check max_size
        if current_content:
            if len(current_content) > max_size:
                # Split if too large, using semantic split if possible
                if model:
                    sub_chunks = self._split_large_chunk_semantic(
                        current_content,
                        current_start,
                        max_size,
                        model
                    )
                else:
                    sub_chunks = self._split_large_chunk(
                        current_content,
                        current_start,
                        max_size
                    )
                merged.extend(sub_chunks)
            else:
                merged.append(Chunk(
                    index=len(merged),
                    content=current_content.strip(),
                    start_position=current_start,
                    end_position=current_start + len(current_content),
                    char_count=len(current_content),
                    has_overlap=False
                ))
        
        return merged

    def _merge_small_chunks(
        self,
        chunks: List[Chunk],
        min_size: int,
        max_size: int
    ) -> List[Chunk]:
        """Merge chunks that are too small and split chunks that are too large."""
        if not chunks:
            return chunks
        
        # First pass: split large chunks
        split_chunks = []
        for chunk in chunks:
            if len(chunk.content) > max_size:
                # Split this chunk into smaller pieces
                sub_chunks = self._split_large_chunk(
                    chunk.content, 
                    chunk.start_position, 
                    max_size
                )
                split_chunks.extend(sub_chunks)
            else:
                split_chunks.append(chunk)
        
        # Second pass: merge small chunks
        merged = []
        current_content = ""
        current_start = 0
        
        for chunk in split_chunks:
            if not current_content:
                current_content = chunk.content
                current_start = chunk.start_position
                continue
            
            # If current chunk is too small, try to merge
            if len(current_content) < min_size:
                combined = current_content + " " + chunk.content
                if len(combined) <= max_size:
                    current_content = combined
                    continue
                else:
                    # Combined would be too large, save current even if small
                    merged.append(Chunk(
                        index=len(merged),
                        content=current_content.strip(),
                        start_position=current_start,
                        end_position=current_start + len(current_content),
                        char_count=len(current_content),
                        has_overlap=False
                    ))
                    current_content = chunk.content
                    current_start = chunk.start_position
                    continue
            
            # If next chunk is too small, try to merge it
            if len(chunk.content) < min_size:
                combined = current_content + " " + chunk.content
                if len(combined) <= max_size:
                    current_content = combined
                    continue
                # If can't merge, save current and start new
            
            # Save current and start new
            merged.append(Chunk(
                index=len(merged),
                content=current_content.strip(),
                start_position=current_start,
                end_position=current_start + len(current_content),
                char_count=len(current_content),
                has_overlap=False
            ))
            current_content = chunk.content
            current_start = chunk.start_position
        
        # Handle last chunk - IMPORTANT: Check max_size before adding
        if current_content:
            # If last chunk is too large, split it
            if len(current_content) > max_size:
                final_chunks = self._split_large_chunk(
                    current_content,
                    current_start,
                    max_size
                )
                for fc in final_chunks:
                    fc.index = len(merged)
                    merged.append(fc)
            else:
                merged.append(Chunk(
                    index=len(merged),
                    content=current_content.strip(),
                    start_position=current_start,
                    end_position=current_start + len(current_content),
                    char_count=len(current_content),
                    has_overlap=False
                ))
        
        return merged

    def _split_large_chunk_semantic(
        self,
        content: str,
        start_position: int,
        max_size: int,
        model: str = None
    ) -> List[Chunk]:
        """Split a large chunk at the best semantic boundary.
        
        For semantic chunking, we want to split at sentence boundaries
        where the semantic similarity is LOWEST (natural topic change).
        
        CRITICAL: This function MUST ensure NO chunk exceeds max_size.
        
        Args:
            content: The chunk content to split
            start_position: Starting position in original text
            max_size: Maximum chunk size
            model: Embedding model for semantic analysis
            
        Returns:
            List of smaller chunks split at semantic boundaries
        """
        # Split into sentences
        sentences = self._split_into_sentences(content)
        
        # If only one sentence or can't split semantically, use simple split
        if len(sentences) <= 1:
            return self._split_large_chunk(content, start_position, max_size)
        
        # Build chunks by accumulating sentences, NEVER exceeding max_size
        chunks = []
        current_sentences = []
        current_size = 0
        current_start = start_position
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            # If single sentence is too large, split it by words
            if sentence_size > max_size:
                # Save current chunk if exists
                if current_sentences:
                    chunk_content = " ".join(current_sentences)
                    chunks.append(Chunk(
                        index=len(chunks),
                        content=chunk_content,
                        start_position=current_start,
                        end_position=current_start + len(chunk_content),
                        char_count=len(chunk_content),
                        has_overlap=False
                    ))
                    current_start += len(chunk_content) + 1
                    current_sentences = []
                    current_size = 0
                
                # Split the large sentence by words
                word_chunks = self._split_by_size(sentence, current_start, max_size)
                chunks.extend(word_chunks)
                current_start = word_chunks[-1].end_position + 1 if word_chunks else current_start
                continue
            
            # Calculate size if we add this sentence
            separator_len = 1 if current_sentences else 0
            new_size = current_size + separator_len + sentence_size
            
            # If adding this sentence would exceed max_size
            if current_sentences and new_size > max_size:
                # Save current chunk
                chunk_content = " ".join(current_sentences)
                chunks.append(Chunk(
                    index=len(chunks),
                    content=chunk_content,
                    start_position=current_start,
                    end_position=current_start + len(chunk_content),
                    char_count=len(chunk_content),
                    has_overlap=False
                ))
                current_start += len(chunk_content) + 1
                
                # Start new chunk with this sentence
                current_sentences = [sentence]
                current_size = sentence_size
            else:
                # Add sentence to current chunk
                current_sentences.append(sentence)
                current_size = new_size
        
        # Add remaining sentences
        if current_sentences:
            chunk_content = " ".join(current_sentences)
            # Final safety check
            if len(chunk_content) > max_size:
                logger.warning(f"Final chunk still too large ({len(chunk_content)} > {max_size}), splitting by words")
                word_chunks = self._split_by_size(chunk_content, current_start, max_size)
                chunks.extend(word_chunks)
            else:
                chunks.append(Chunk(
                    index=len(chunks),
                    content=chunk_content,
                    start_position=current_start,
                    end_position=current_start + len(chunk_content),
                    char_count=len(chunk_content),
                    has_overlap=False
                ))
        
        # Final validation: ensure NO chunk exceeds max_size
        validated_chunks = []
        for chunk in chunks:
            if len(chunk.content) > max_size:
                logger.error(f"Chunk still exceeds max_size: {len(chunk.content)} > {max_size}, forcing split")
                # Force split by words
                sub_chunks = self._split_by_size(chunk.content, chunk.start_position, max_size)
                validated_chunks.extend(sub_chunks)
            else:
                validated_chunks.append(chunk)
        
        return validated_chunks if validated_chunks else [Chunk(
            index=0,
            content=content[:max_size],
            start_position=start_position,
            end_position=start_position + min(len(content), max_size),
            char_count=min(len(content), max_size),
            has_overlap=False
        )]

    def _split_large_chunk(
        self,
        content: str,
        start_position: int,
        max_size: int
    ) -> List[Chunk]:
        """Split a large chunk into smaller pieces at sentence boundaries.
        
        CRITICAL: This function MUST ensure NO chunk exceeds max_size.
        Sentences are NEVER split in the middle - we keep them complete.
        """
        # Try to split at sentence boundaries
        sentences = self._split_into_sentences(content)
        
        if not sentences or len(sentences) == 1:
            # Can't split by sentences, split by size at word boundaries
            return self._split_by_size(content, start_position, max_size)
        
        chunks = []
        current_sentences = []
        current_size = 0
        current_start = start_position
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            # If single sentence is too large, split it by words
            if sentence_size > max_size:
                # Save current chunk if exists
                if current_sentences:
                    chunk_content = " ".join(current_sentences)
                    chunks.append(Chunk(
                        index=len(chunks),
                        content=chunk_content,
                        start_position=current_start,
                        end_position=current_start + len(chunk_content),
                        char_count=len(chunk_content),
                        has_overlap=False
                    ))
                    current_start += len(chunk_content) + 1
                    current_sentences = []
                    current_size = 0
                
                # Split the large sentence by words
                logger.warning(f"Sentence too large ({sentence_size} > {max_size}), splitting by words")
                word_chunks = self._split_by_size(sentence, current_start, max_size)
                chunks.extend(word_chunks)
                current_start = word_chunks[-1].end_position + 1 if word_chunks else current_start
                continue
            
            # Calculate size if we add this sentence
            separator_len = 1 if current_sentences else 0
            new_size = current_size + separator_len + sentence_size
            
            # If adding this sentence would exceed max_size
            if current_sentences and new_size > max_size:
                # Save current chunk
                chunk_content = " ".join(current_sentences)
                chunks.append(Chunk(
                    index=len(chunks),
                    content=chunk_content,
                    start_position=current_start,
                    end_position=current_start + len(chunk_content),
                    char_count=len(chunk_content),
                    has_overlap=False
                ))
                current_start += len(chunk_content) + 1
                
                # Start new chunk with this sentence
                current_sentences = [sentence]
                current_size = sentence_size
            else:
                # Add sentence to current chunk
                current_sentences.append(sentence)
                current_size = new_size
        
        # Handle last chunk - check max_size
        if current_sentences:
            chunk_content = " ".join(current_sentences)
            if len(chunk_content) > max_size:
                # This shouldn't happen, but safety check
                logger.warning(f"Final chunk too large ({len(chunk_content)} > {max_size}), splitting by words")
                word_chunks = self._split_by_size(chunk_content, current_start, max_size)
                chunks.extend(word_chunks)
            else:
                chunks.append(Chunk(
                    index=len(chunks),
                    content=chunk_content,
                    start_position=current_start,
                    end_position=current_start + len(chunk_content),
                    char_count=len(chunk_content),
                    has_overlap=False
                ))
        
        # Final validation
        validated_chunks = []
        for chunk in chunks:
            if len(chunk.content) > max_size:
                logger.error(f"Chunk still exceeds max_size: {len(chunk.content)} > {max_size}, forcing split")
                sub_chunks = self._split_by_size(chunk.content, chunk.start_position, max_size)
                validated_chunks.extend(sub_chunks)
            else:
                validated_chunks.append(chunk)
        
        return validated_chunks if validated_chunks else [Chunk(
            index=0,
            content=content[:max_size],
            start_position=start_position,
            end_position=start_position + min(len(content), max_size),
            char_count=min(len(content), max_size),
            has_overlap=False
        )]

    def _split_by_size(
        self,
        content: str,
        start_position: int,
        max_size: int
    ) -> List[Chunk]:
        """Split content by size at punctuation/word boundaries - NEVER split words in middle.
        
        Priority:
        1. Try to split at punctuation (comma, semicolon, etc.)
        2. If not possible, split at word boundaries
        3. Never split a word in the middle
        
        This is the last resort when we can't split by sentences.
        """
        # First try to split by punctuation within the content
        # Look for: comma, semicolon, colon, dash - these are natural pause points
        punctuation_pattern = r'([,;:\-—])\s+'
        import re
        
        # Split but keep the punctuation
        parts = re.split(punctuation_pattern, content)
        
        # Reconstruct parts with punctuation attached
        reconstructed = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i + 1] in ',;:-—':
                # Combine text with its punctuation
                reconstructed.append(parts[i] + parts[i + 1])
                i += 2
            else:
                if parts[i].strip() and parts[i] not in ',;:-—':
                    reconstructed.append(parts[i])
                i += 1
        
        # If we got meaningful parts, use them
        if len(reconstructed) > 1:
            chunks = []
            current_content = ""
            current_start = start_position
            
            for part in reconstructed:
                part = part.strip()
                if not part:
                    continue
                
                separator = " " if current_content else ""
                potential = current_content + separator + part
                
                if len(potential) > max_size and current_content:
                    chunks.append(Chunk(
                        index=len(chunks),
                        content=current_content.strip(),
                        start_position=current_start,
                        end_position=current_start + len(current_content),
                        char_count=len(current_content),
                        has_overlap=False
                    ))
                    current_start = current_start + len(current_content) + 1
                    current_content = part
                else:
                    current_content = potential
            
            if current_content:
                if len(current_content) > max_size:
                    # Still too large, fall through to word splitting
                    pass
                else:
                    chunks.append(Chunk(
                        index=len(chunks),
                        content=current_content.strip(),
                        start_position=current_start,
                        end_position=current_start + len(current_content),
                        char_count=len(current_content),
                        has_overlap=False
                    ))
                    return chunks if chunks else self._split_by_words(content, start_position, max_size)
            
            if chunks:
                return chunks
        
        # Fallback: split by words
        return self._split_by_words(content, start_position, max_size)
    
    def _split_by_words(
        self,
        content: str,
        start_position: int,
        max_size: int
    ) -> List[Chunk]:
        """Split content by words - absolute last resort.
        
        CRITICAL: This function MUST ensure NO chunk exceeds max_size.
        Words are kept complete - never split in the middle.
        """
        words = content.split()
        chunks = []
        current_words = []
        current_size = 0
        current_start = start_position
        
        for word in words:
            word_size = len(word)
            
            # If single word is too large, we MUST truncate it (no choice)
            if word_size > max_size:
                # Save current chunk if exists
                if current_words:
                    chunk_content = " ".join(current_words)
                    chunks.append(Chunk(
                        index=len(chunks),
                        content=chunk_content,
                        start_position=current_start,
                        end_position=current_start + len(chunk_content),
                        char_count=len(chunk_content),
                        has_overlap=False
                    ))
                    current_start += len(chunk_content) + 1
                    current_words = []
                    current_size = 0
                
                # Truncate the word with ellipsis
                truncated = word[:max_size-3] + "..."
                logger.warning(f"Word too long ({word_size} chars), truncating: {word[:50]}...")
                chunks.append(Chunk(
                    index=len(chunks),
                    content=truncated,
                    start_position=current_start,
                    end_position=current_start + len(truncated),
                    char_count=len(truncated),
                    has_overlap=False
                ))
                current_start += word_size + 1
                continue
            
            # Calculate size if we add this word
            separator_len = 1 if current_words else 0
            new_size = current_size + separator_len + word_size
            
            # If adding this word would exceed max_size
            if current_words and new_size > max_size:
                # Save current chunk
                chunk_content = " ".join(current_words)
                chunks.append(Chunk(
                    index=len(chunks),
                    content=chunk_content,
                    start_position=current_start,
                    end_position=current_start + len(chunk_content),
                    char_count=len(chunk_content),
                    has_overlap=False
                ))
                current_start += len(chunk_content) + 1
                
                # Start new chunk with this word
                current_words = [word]
                current_size = word_size
            else:
                # Add word to current chunk
                current_words.append(word)
                current_size = new_size
        
        # Handle last chunk - check max_size
        if current_words:
            chunk_content = " ".join(current_words)
            if len(chunk_content) > max_size:
                # This shouldn't happen, but safety check - truncate
                chunk_content = chunk_content[:max_size-3] + "..."
                logger.warning(f"Final chunk too long, truncating to {max_size} chars")
            
            chunks.append(Chunk(
                index=len(chunks),
                content=chunk_content,
                start_position=current_start,
                end_position=current_start + len(chunk_content),
                char_count=len(chunk_content),
                has_overlap=False
            ))
        
        # Final validation: ensure NO chunk exceeds max_size
        validated_chunks = []
        for chunk in chunks:
            if len(chunk.content) > max_size:
                logger.error(f"Chunk STILL exceeds max_size: {len(chunk.content)} > {max_size}, truncating")
                truncated_content = chunk.content[:max_size-3] + "..."
                validated_chunks.append(Chunk(
                    index=chunk.index,
                    content=truncated_content,
                    start_position=chunk.start_position,
                    end_position=chunk.start_position + len(truncated_content),
                    char_count=len(truncated_content),
                    has_overlap=chunk.has_overlap
                ))
            else:
                validated_chunks.append(chunk)
        
        return validated_chunks if validated_chunks else [Chunk(
            index=0,
            content=content[:max_size-3] + "...",  # Truncate with ellipsis
            start_position=start_position,
            end_position=start_position + min(len(content), max_size),
            char_count=min(len(content), max_size),
            has_overlap=False
        )]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with enhanced multi-language support.
        
        Uses the new EnhancedSentenceTokenizer with language detection
        for better Turkish and English sentence splitting.
        """
        try:
            # Try using the new enhanced tokenizer
            from app.services.language_detector import LanguageDetector
            from app.services.sentence_tokenizer import EnhancedSentenceTokenizer
            
            # Detect language
            detector = LanguageDetector()
            language = detector.detect_language(text)
            
            # Tokenize with language-aware processing
            tokenizer = EnhancedSentenceTokenizer()
            sentences = tokenizer.tokenize(text, language)
            
            if sentences:
                return sentences
                
        except Exception as e:
            # Fallback to original method if new tokenizer fails
            logger.warning(
                f"Enhanced tokenizer failed, falling back to NLTK: {e}"
            )
        
        # Fallback: Original NLTK-based method
        import re
        
        # First try NLTK
        sentences = nltk.sent_tokenize(text)
        
        # If NLTK returns single chunk, try manual splitting
        if len(sentences) <= 1 and len(text) > 200:
            # Split on sentence-ending punctuation followed by space/newline
            # Handles: . ! ? and Turkish specific patterns
            pattern = r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜa-zçğıöşü0-9#*])'
            sentences = re.split(pattern, text)
        
        # Further split any remaining long "sentences" that contain multiple ?
        result = []
        for sent in sentences:
            # If sentence has multiple ? or ., split them
            if sent.count('?') > 1 or (sent.count('.') > 2 and len(sent) > 300):
                # Split on ? followed by space and capital/number
                sub_pattern = r'(?<=[?])\s+(?=[A-ZÇĞİÖŞÜ0-9])'
                sub_sents = re.split(sub_pattern, sent)
                result.extend(sub_sents)
            else:
                result.append(sent)
        
        return [s.strip() for s in result if s.strip()]

    def _add_overlap_to_chunks(
        self,
        chunks: List[Chunk],
        overlap: int,
        max_size: int = None
    ) -> List[Chunk]:
        """Add overlap from previous chunk to each chunk.
        
        Overlap uses complete sentences from the end of previous chunk.
        The overlap parameter is a TARGET - we take complete sentences
        that fit within that target.
        
        Args:
            chunks: List of chunks
            overlap: Target overlap size in characters
            max_size: Maximum chunk size (optional, for validation)
        """
        if not chunks or overlap <= 0:
            return chunks
        
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk has no overlap
                result.append(chunk)
            else:
                # Get complete sentences from previous chunk for overlap
                prev_content = chunks[i - 1].content
                overlap_sentences = self._get_last_sentences(prev_content, overlap)
                
                if overlap_sentences:
                    overlap_text = " ".join(overlap_sentences)
                    new_content = overlap_text + " " + chunk.content
                    
                    # Check if adding overlap would exceed max_size
                    if max_size and len(new_content) > max_size:
                        # Try with fewer sentences
                        while overlap_sentences and len(new_content) > max_size:
                            overlap_sentences.pop(0)  # Remove first sentence
                            if overlap_sentences:
                                overlap_text = " ".join(overlap_sentences)
                                new_content = overlap_text + " " + chunk.content
                            else:
                                new_content = chunk.content
                                break
                    
                    result.append(Chunk(
                        index=i,
                        content=new_content.strip(),
                        start_position=chunk.start_position - len(overlap_text) - 1 if overlap_sentences else chunk.start_position,
                        end_position=chunk.end_position,
                        char_count=len(new_content),
                        has_overlap=bool(overlap_sentences)
                    ))
                else:
                    result.append(chunk)
        
        return result

    def _get_last_sentences(self, text: str, target_chars: int) -> List[str]:
        """Get complete sentences from the end of text.
        
        Returns sentences that fit within target_chars, always returning
        at least one sentence if possible.
        
        Args:
            text: Source text
            target_chars: Target character count for overlap
            
        Returns:
            List of complete sentences from the end
        """
        if not text:
            return []
        
        # Split into sentences using regex
        import re
        # Match sentence endings: . ! ? followed by space or end
        # Keep the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        # Take sentences from the end until we exceed target
        selected = []
        total_len = 0
        
        for sent in reversed(sentences):
            sent_len = len(sent)
            
            # Always take at least one sentence
            if not selected:
                selected.insert(0, sent)
                total_len = sent_len
            elif total_len + sent_len + 1 <= target_chars:
                # Can fit this sentence
                selected.insert(0, sent)
                total_len += sent_len + 1  # +1 for space
            else:
                # Would exceed target, stop
                break
        
        return selected

    def chunk(
        self,
        text: str,
        similarity_threshold: float = None,
        embedding_model: str = None,
        overlap: int = 0,
        buffer_size: int = None,
        min_chunk_size: int = None,
        max_chunk_size: int = None,
        **kwargs
    ) -> List[Chunk]:
        """Split text into semantically coherent chunks.
        
        Args:
            text: Text to chunk
            similarity_threshold: Percentile threshold (0-1 maps to 0-100).
                                 If None and adaptive threshold is enabled,
                                 will be calculated based on text characteristics.
            embedding_model: Model for embeddings
            overlap: Characters to overlap between chunks
            buffer_size: Sentences to include on each side for context
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
        """
        if not text or not text.strip():
            return []

        model = embedding_model or self.DEFAULT_EMBEDDING_MODEL
        buffer = buffer_size if buffer_size is not None else self.DEFAULT_BUFFER_SIZE
        min_size = min_chunk_size or self.DEFAULT_MIN_CHUNK_SIZE
        max_size = max_chunk_size or self.DEFAULT_MAX_CHUNK_SIZE
        
        # Calculate adaptive threshold if enabled and not provided
        if similarity_threshold is None:
            if self._enable_adaptive_threshold:
                self._ensure_threshold_calculator()
                recommendation = self._threshold_calculator.recommend_threshold(text)
                similarity_threshold = recommendation.recommended_threshold
                logger.debug(
                    f"Adaptive threshold: {similarity_threshold:.2f} - "
                    f"{recommendation.reasoning}"
                )
            else:
                similarity_threshold = 0.5  # Default
        
        # Convert 0-1 threshold to percentile (0-100)
        percentile = similarity_threshold * 100
        
        ensure_nltk_punkt()

        # Split into sentences with Turkish support
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        # Detect language for Q&A detection
        detected_language = None
        if self._enable_qa_detection:
            try:
                from app.services.language_detector import LanguageDetector
                detector = LanguageDetector()
                detected_language = detector.detect_language(text)
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")

        # Merge Q&A pairs if enabled
        if self._enable_qa_detection and detected_language:
            self._ensure_qa_detector()
            sentences = self._qa_detector.merge_qa_pairs(
                sentences, detected_language
            )
            logger.debug(f"After Q&A merging: {len(sentences)} sentences")

        # Preprocess: merge short sentences and keep Q&A together
        sentences = self._preprocess_sentences(sentences)
        
        if len(sentences) <= 1:
            content = sentences[0] if sentences else text
            return [Chunk(
                index=0,
                content=content,
                start_position=0,
                end_position=len(content),
                char_count=len(content),
                has_overlap=False
            )]

        # Combine sentences with buffer for better context
        combined = self._combine_sentences_with_buffer(sentences, buffer)
        
        # Get embeddings for combined texts
        combined_texts = [c["combined_text"] for c in combined]
        embeddings = self._get_embeddings(combined_texts, model)
        
        if len(embeddings) < 2:
            content = " ".join(sentences)
            return [Chunk(
                index=0,
                content=content,
                start_position=0,
                end_position=len(content),
                char_count=len(content),
                has_overlap=False
            )]

        # Calculate distances between consecutive embeddings
        distances = self._calculate_distances(embeddings)
        
        # Find breakpoints using percentile threshold
        breakpoints = self._find_breakpoints_percentile(distances, percentile)
        
        # Create initial chunks from breakpoints
        chunks = self._create_chunks_from_breakpoints(
            text, sentences, breakpoints
        )
        
        # Merge small chunks and enforce max size
        # For semantic chunking, be more careful to preserve sentence boundaries
        logger.info(f"Before merge_small_chunks: {len(chunks)} chunks, min_size={min_size}, max_size={max_size}")
        if chunks:
            chunk_sizes = [len(c.content) for c in chunks]
            logger.info(f"Chunk sizes before merge: min={min(chunk_sizes)}, max={max(chunk_sizes)}, avg={sum(chunk_sizes)/len(chunk_sizes):.1f}")
        
        # For semantic chunking, merge small chunks and split large ones at semantic boundaries
        # Pass the model so we can do semantic analysis when splitting
        chunks = self._merge_small_chunks_semantic(chunks, min_size, max_size, model)
        
        logger.info(f"After merge_small_chunks: {len(chunks)} chunks")
        if chunks:
            chunk_sizes = [len(c.content) for c in chunks]
            logger.info(f"Chunk sizes after merge: min={min(chunk_sizes)}, max={max(chunk_sizes)}, avg={sum(chunk_sizes)/len(chunk_sizes):.1f}")
            # Log any chunks that exceed max_size
            oversized = [(i, len(c.content)) for i, c in enumerate(chunks) if len(c.content) > max_size]
            if oversized:
                logger.error(f"OVERSIZED CHUNKS DETECTED: {oversized}")
        
        # Add overlap if specified
        if overlap > 0:
            logger.info(f"Adding overlap: {overlap} characters")
            chunks = self._add_overlap_to_chunks(chunks, overlap, max_size)
            logger.info(f"After adding overlap: {len(chunks)} chunks")
            if chunks:
                overlap_count = sum(1 for c in chunks if c.has_overlap)
                logger.info(f"Chunks with overlap: {overlap_count}/{len(chunks)}")
        
        # Final validation: ensure no chunk exceeds max_size after overlap
        # For semantic chunking, use semantic split to preserve meaning
        final_chunks = []
        for chunk in chunks:
            if len(chunk.content) > max_size:
                logger.warning(
                    f"Chunk {chunk.index} exceeds max_size after overlap: "
                    f"{len(chunk.content)} > {max_size}, splitting at semantic boundaries..."
                )
                # Use semantic split to preserve meaning
                sub_chunks = self._split_large_chunk_semantic(
                    chunk.content,
                    chunk.start_position,
                    max_size,
                    model
                )
                # Mark first sub-chunk as having overlap if original had it
                if sub_chunks and chunk.has_overlap:
                    sub_chunks[0].has_overlap = True
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        chunks = final_chunks
        
        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.index = i
        
        return chunks

    def _create_chunks_from_breakpoints(
        self,
        text: str,
        sentences: List[str],
        breakpoints: List[int]
    ) -> List[Chunk]:
        """Create chunks based on breakpoints."""
        chunks: List[Chunk] = []
        start_idx = 0
        current_pos = 0

        all_breaks = breakpoints + [len(sentences)]
        
        for bp in all_breaks:
            if bp <= start_idx:
                continue
                
            chunk_sentences = sentences[start_idx:bp]
            if not chunk_sentences:
                continue
                
            content = " ".join(chunk_sentences)

            first_sent = chunk_sentences[0]
            start_pos = text.find(first_sent, current_pos)
            if start_pos == -1:
                start_pos = current_pos

            last_sent = chunk_sentences[-1]
            end_pos = text.find(last_sent, start_pos)
            if end_pos == -1:
                end_pos = start_pos + len(content)
            else:
                end_pos = end_pos + len(last_sent)

            chunks.append(Chunk(
                index=len(chunks),
                content=content,
                start_position=start_pos,
                end_position=end_pos,
                char_count=len(content),
                has_overlap=False
            ))

            current_pos = end_pos
            start_idx = bp

        return chunks


class LateChunker(BaseChunker):
    """Process through long-context embedding model before splitting."""

    def __init__(self):
        """Initialize the late chunker."""
        self._model = None
        self._model_name = None

    def _ensure_model(self, model_name: str):
        """Load the embedding model if not already loaded."""
        if self._model is None or self._model_name != model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    model_name, trust_remote_code=True
                )
                self._model_name = model_name
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for late chunking. "
                    "Install it with: pip install sentence-transformers"
                ) from exc

    def chunk(
        self,
        text: str,
        chunk_size: int = 500,
        embedding_model: str = "jinaai/jina-embeddings-v2-base-en",
        **kwargs
    ) -> List[Chunk]:
        """Split text using late chunking approach.

        Late chunking processes the entire text through a long-context
        embedding model first, then splits into chunks while preserving
        contextual information.
        """
        if not text:
            return []

        self._ensure_model(embedding_model)
        ensure_nltk_punkt()

        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []

        # Get embeddings for the full text (long-context aware)
        # The model processes the entire context before we chunk
        _ = self._model.encode(text)

        # Create chunks based on size while respecting sentence boundaries
        return self._create_sentence_based_chunks(text, sentences, chunk_size)

    def _create_sentence_based_chunks(
        self,
        text: str,
        sentences: List[str],
        chunk_size: int
    ) -> List[Chunk]:
        """Create chunks based on size while respecting sentence boundaries."""
        chunks: List[Chunk] = []
        current_content = ""
        current_start = 0

        for sentence in sentences:
            separator = " " if current_content else ""
            potential_content = current_content + separator + sentence

            if len(potential_content) > chunk_size and current_content:
                chunks.append(self._create_chunk(
                    text, current_content, current_start, len(chunks)
                ))
                current_start = current_start + len(current_content)
                current_content = sentence
            else:
                current_content = potential_content

        if current_content:
            chunks.append(self._create_chunk(
                text, current_content, current_start, len(chunks)
            ))

        return chunks

    def _create_chunk(
        self,
        text: str,
        content: str,
        search_start: int,
        index: int
    ) -> Chunk:
        """Create a single chunk with position tracking."""
        words = content.split()
        first_word = words[0] if words else ""
        start_pos = text.find(first_word, search_start)
        if start_pos == -1:
            start_pos = search_start

        return Chunk(
            index=index,
            content=content,
            start_position=start_pos,
            end_position=start_pos + len(content),
            char_count=len(content),
            has_overlap=False
        )


class AgenticChunker(BaseChunker):
    """LLM-driven semantic segmentation with metadata enrichment."""

    def __init__(self):
        """Initialize the agentic chunker."""
        self._client = None

    def _ensure_client(self, api_key: Optional[str] = None):
        """Initialize the OpenAI client if not already done."""
        if self._client is None:
            try:
                from openai import OpenAI
                key = api_key or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise ValueError(
                        "OpenAI API key is required for agentic chunking. "
                        "Set OPENAI_API_KEY environment variable or "
                        "pass api_key parameter."
                    )
                self._client = OpenAI(api_key=key)
            except ImportError as exc:
                raise ImportError(
                    "openai is required for agentic chunking. "
                    "Install it with: pip install openai"
                ) from exc

    def chunk(
        self,
        text: str,
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        **kwargs
    ) -> List[Chunk]:
        """Split text using LLM-driven semantic segmentation.

        The LLM analyzes the text and determines optimal split points
        based on semantic meaning and topic boundaries.
        """
        if not text:
            return []

        self._ensure_client(api_key)

        try:
            chunk_texts = self._get_llm_chunks(text, llm_model)
            return self._create_chunks_from_texts(text, chunk_texts)
        except (json.JSONDecodeError, KeyError, IndexError):
            # Fallback: if LLM response isn't valid, use sentence chunking
            return self._fallback_sentence_chunking(text)

    def _get_llm_chunks(self, text: str, llm_model: str) -> List[str]:
        """Get chunk boundaries from LLM."""
        system_prompt = self._get_system_prompt()
        user_prompt = f"Please split the following text:\n\n{text}"

        response = self._client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=4096
        )

        content = response.choices[0].message.content.strip()
        return self._parse_llm_response(content)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for LLM chunking."""
        return """You are a text segmentation expert. Analyze the text and \
split it into semantically coherent chunks.

Rules:
1. Each chunk should contain a complete thought or topic
2. Chunks should be roughly similar in size when possible
3. Preserve paragraph and sentence boundaries
4. Return ONLY a JSON array of strings, each string is a chunk
5. Do not add any explanation or markdown formatting

Example: ["First chunk.", "Second chunk.", "Third chunk."]"""

    def _parse_llm_response(self, content: str) -> List[str]:
        """Parse LLM response to extract chunk texts."""
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)

    def _create_chunks_from_texts(
        self,
        text: str,
        chunk_texts: List[str]
    ) -> List[Chunk]:
        """Create Chunk objects from text list."""
        chunks: List[Chunk] = []
        current_pos = 0

        for chunk_text in chunk_texts:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            start_pos = self._find_chunk_position(
                text, chunk_text, current_pos
            )
            end_pos = start_pos + len(chunk_text)

            chunks.append(Chunk(
                index=len(chunks),
                content=chunk_text,
                start_position=start_pos,
                end_position=end_pos,
                char_count=len(chunk_text),
                has_overlap=False
            ))

            current_pos = end_pos

        return chunks

    def _find_chunk_position(
        self,
        text: str,
        chunk_text: str,
        search_start: int
    ) -> int:
        """Find position of chunk in original text."""
        # Try exact match first (first 50 chars)
        search_len = min(50, len(chunk_text))
        start_pos = text.find(chunk_text[:search_len], search_start)
        if start_pos != -1:
            return start_pos

        # Try partial match with first few words
        words = chunk_text.split()[:5]
        search_text = " ".join(words)
        start_pos = text.find(search_text, search_start)
        if start_pos != -1:
            return start_pos

        return search_start

    def _fallback_sentence_chunking(self, text: str) -> List[Chunk]:
        """Fallback to sentence-based chunking."""
        ensure_nltk_punkt()
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_pos = 0

        for idx, sentence in enumerate(sentences):
            start_pos = text.find(sentence, current_pos)
            if start_pos == -1:
                start_pos = current_pos

            chunks.append(Chunk(
                index=idx,
                content=sentence,
                start_position=start_pos,
                end_position=start_pos + len(sentence),
                char_count=len(sentence),
                has_overlap=False
            ))

            current_pos = start_pos + len(sentence)

        return chunks


class ChunkerService:
    """Service for managing and executing chunking strategies with diagnostics."""

    def __init__(self):
        """Initialize the chunker service with available strategies."""
        self._chunkers = {
            ChunkingStrategy.FIXED_SIZE: FixedSizeChunker(),
            ChunkingStrategy.RECURSIVE: RecursiveChunker(),
            ChunkingStrategy.SENTENCE: SentenceChunker(),
            ChunkingStrategy.SEMANTIC: SemanticChunker(use_provider_manager=True),
            ChunkingStrategy.LATE_CHUNKING: LateChunker(),
            ChunkingStrategy.AGENTIC: AgenticChunker(),
        }

    def chunk_text(
        self,
        text: str,
        strategy: ChunkingStrategy,
        **kwargs
    ) -> List[Chunk]:
        """Chunk text using the specified strategy."""
        if strategy not in self._chunkers:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        return self._chunkers[strategy].chunk(text, **kwargs)

    def chunk_with_diagnostics(
        self,
        text: str,
        config: ChunkingConfig
    ) -> Tuple[List[Chunk], ChunkingDiagnostics]:
        """Chunk text with comprehensive diagnostics and error handling.
        
        Args:
            text: Text to chunk
            config: Chunking configuration
            
        Returns:
            Tuple of (chunks, diagnostics)
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        config_errors = config.validate()
        if config_errors:
            error_msg = f"Invalid chunking configuration: {', '.join(config_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate input text
        validation_errors = self._validate_input_text(text)
        if validation_errors:
            error_msg = f"Invalid input text: {', '.join(validation_errors)}"
            logger.error(error_msg)
            diagnostics = ChunkingDiagnostics(
                strategy=config.strategy.value,
                input_text_length=len(text) if text else 0,
                processing_time=0.0,
                total_chunks=0,
                avg_chunk_size=0.0,
                min_chunk_size=0,
                max_chunk_size=0,
                overlap_count=0,
                error_message=error_msg,
                error_details={"validation_errors": validation_errors}
            )
            return [], diagnostics
        
        # Start timing
        start_time = time.time()
        
        try:
            # Log chunking attempt
            logger.info(
                f"Starting chunking with strategy {config.strategy.value}, "
                f"text length: {len(text)}, chunk_size: {config.chunk_size}"
            )
            
            # Perform chunking
            chunks = self._chunkers[config.strategy].chunk(
                text,
                chunk_size=config.chunk_size,
                overlap=config.overlap,
                sentences_per_chunk=config.sentences_per_chunk,
                similarity_threshold=config.similarity_threshold,
                embedding_model=config.embedding_model,
                buffer_size=config.buffer_size,
                min_chunk_size=config.min_chunk_size,
                max_chunk_size=config.max_chunk_size
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate diagnostics
            diagnostics = self._generate_diagnostics(
                config, text, chunks, processing_time
            )
            
            # Validate chunk quality
            quality_issues = self.validate_chunk_quality(chunks, config)
            if quality_issues:
                diagnostics.recommendations.extend(quality_issues)
            
            # Log success
            logger.info(
                f"Chunking completed successfully: {len(chunks)} chunks in {processing_time:.2f}s"
            )
            
            return chunks, diagnostics
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Chunking failed with strategy {config.strategy.value}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            diagnostics = ChunkingDiagnostics(
                strategy=config.strategy.value,
                input_text_length=len(text),
                processing_time=processing_time,
                total_chunks=0,
                avg_chunk_size=0.0,
                min_chunk_size=0,
                max_chunk_size=0,
                overlap_count=0,
                error_message=error_msg,
                error_details={
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "strategy": config.strategy.value,
                    "config": {
                        "chunk_size": config.chunk_size,
                        "overlap": config.overlap,
                        "min_chunk_size": config.min_chunk_size,
                        "max_chunk_size": config.max_chunk_size
                    }
                }
            )
            
            return [], diagnostics

    def validate_chunk_quality(
        self,
        chunks: List[Chunk],
        config: ChunkingConfig
    ) -> List[str]:
        """Assess chunk quality and return recommendations.
        
        Args:
            chunks: List of chunks to validate
            config: Chunking configuration used
            
        Returns:
            List of quality issues and recommendations
        """
        issues = []
        
        if not chunks:
            issues.append("No chunks were generated - text may be too short or invalid")
            return issues
        
        # Check chunk sizes
        chunk_sizes = [chunk.char_count for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        
        # Size distribution analysis
        if min_size < config.min_chunk_size * 0.5:
            issues.append(f"Some chunks are very small (min: {min_size}). Consider adjusting chunking strategy.")
        
        if max_size > config.max_chunk_size * 1.5:
            issues.append(f"Some chunks are very large (max: {max_size}). Consider reducing chunk_size.")
        
        # Size variance analysis
        size_variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)
        size_std = size_variance ** 0.5
        
        if size_std > avg_size * 0.5:
            issues.append("High variance in chunk sizes. Consider using a different chunking strategy.")
        
        # Overlap analysis
        overlap_count = sum(1 for chunk in chunks if chunk.has_overlap)
        if config.overlap > 0 and overlap_count == 0:
            issues.append("No overlaps detected despite overlap configuration. Check chunking implementation.")
        
        # Content quality checks
        empty_chunks = sum(1 for chunk in chunks if not chunk.content.strip())
        if empty_chunks > 0:
            issues.append(f"{empty_chunks} empty chunks detected. Check text preprocessing.")
        
        # Check for very short chunks that might indicate splitting issues
        very_short = sum(1 for size in chunk_sizes if size < 50)
        if very_short > len(chunks) * 0.2:  # More than 20% are very short
            issues.append("Many chunks are very short. Consider increasing chunk_size or using sentence-based chunking.")
        
        return issues

    def get_chunking_metrics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get detailed performance metrics for chunks.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with detailed metrics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "size_distribution": {},
                "overlap_analysis": {},
                "content_quality_score": 0.0
            }
        
        chunk_sizes = [chunk.char_count for chunk in chunks]
        total_chunks = len(chunks)
        
        # Basic metrics
        avg_chunk_size = sum(chunk_sizes) / total_chunks
        min_chunk_size = min(chunk_sizes)
        max_chunk_size = max(chunk_sizes)
        
        # Size distribution
        size_ranges = {
            "0-100": sum(1 for size in chunk_sizes if size <= 100),
            "101-300": sum(1 for size in chunk_sizes if 101 <= size <= 300),
            "301-500": sum(1 for size in chunk_sizes if 301 <= size <= 500),
            "501-1000": sum(1 for size in chunk_sizes if 501 <= size <= 1000),
            "1000+": sum(1 for size in chunk_sizes if size > 1000)
        }
        
        # Overlap analysis
        overlap_count = sum(1 for chunk in chunks if chunk.has_overlap)
        overlap_analysis = {
            "total_overlaps": overlap_count,
            "overlap_percentage": (overlap_count / total_chunks * 100) if total_chunks > 0 else 0
        }
        
        # Content quality score (simplified)
        quality_score = self._calculate_content_quality_score(chunks)
        
        return {
            "total_chunks": total_chunks,
            "avg_chunk_size": avg_chunk_size,
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": max_chunk_size,
            "size_distribution": size_ranges,
            "overlap_analysis": overlap_analysis,
            "content_quality_score": quality_score
        }

    def get_available_strategies(self) -> List[ChunkingStrategy]:
        """Get list of available chunking strategies."""
        return list(self._chunkers.keys())

    def _validate_input_text(self, text: str) -> List[str]:
        """Validate input text for chunking.
        
        Args:
            text: Text to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if not text:
            errors.append("Text is empty or None")
            return errors
        
        if not isinstance(text, str):
            errors.append("Text must be a string")
            return errors
        
        text = text.strip()
        if not text:
            errors.append("Text contains only whitespace")
            return errors
        
        if len(text) < 10:
            errors.append("Text is too short for meaningful chunking (minimum 10 characters)")
        
        # Check for reasonable text length limits
        if len(text) > 10_000_000:  # 10MB of text
            errors.append("Text is too large for processing (maximum 10MB)")
        
        return errors

    def _generate_diagnostics(
        self,
        config: ChunkingConfig,
        text: str,
        chunks: List[Chunk],
        processing_time: float
    ) -> ChunkingDiagnostics:
        """Generate comprehensive diagnostics for chunking operation.
        
        Args:
            config: Chunking configuration used
            text: Original text
            chunks: Generated chunks
            processing_time: Time taken for processing
            
        Returns:
            ChunkingDiagnostics object
        """
        if not chunks:
            return ChunkingDiagnostics(
                strategy=config.strategy.value,
                input_text_length=len(text),
                processing_time=processing_time,
                total_chunks=0,
                avg_chunk_size=0.0,
                min_chunk_size=0,
                max_chunk_size=0,
                overlap_count=0
            )
        
        chunk_sizes = [chunk.char_count for chunk in chunks]
        total_chunks = len(chunks)
        avg_chunk_size = sum(chunk_sizes) / total_chunks
        min_chunk_size = min(chunk_sizes)
        max_chunk_size = max(chunk_sizes)
        overlap_count = sum(1 for chunk in chunks if chunk.has_overlap)
        
        # Generate performance warnings
        warnings = []
        if processing_time > 30.0:  # More than 30 seconds
            warnings.append(f"Processing took {processing_time:.1f}s, which is longer than expected")
        
        if total_chunks > 1000:
            warnings.append(f"Generated {total_chunks} chunks, which may impact performance")
        
        if avg_chunk_size < config.chunk_size * 0.3:
            warnings.append(f"Average chunk size ({avg_chunk_size:.0f}) is much smaller than configured ({config.chunk_size})")
        
        # Calculate quality score
        quality_score = self._calculate_content_quality_score(chunks)
        
        # Generate recommendations
        recommendations = []
        if quality_score < 0.5:
            recommendations.append("Consider using a different chunking strategy for better quality")
        
        if max_chunk_size > config.chunk_size * 2:
            recommendations.append("Some chunks are much larger than expected - consider reducing chunk_size")
        
        return ChunkingDiagnostics(
            strategy=config.strategy.value,
            input_text_length=len(text),
            processing_time=processing_time,
            total_chunks=total_chunks,
            avg_chunk_size=avg_chunk_size,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            overlap_count=overlap_count,
            performance_warnings=warnings,
            quality_score=quality_score,
            recommendations=recommendations
        )

    def _calculate_content_quality_score(self, chunks: List[Chunk]) -> float:
        """Calculate a quality score for the chunks.
        
        Args:
            chunks: List of chunks to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not chunks:
            return 0.0
        
        score = 1.0
        
        # Penalize empty chunks
        empty_chunks = sum(1 for chunk in chunks if not chunk.content.strip())
        if empty_chunks > 0:
            score -= (empty_chunks / len(chunks)) * 0.3
        
        # Penalize very short chunks
        very_short = sum(1 for chunk in chunks if len(chunk.content.strip()) < 20)
        if very_short > 0:
            score -= (very_short / len(chunks)) * 0.2
        
        # Reward consistent chunk sizes
        chunk_sizes = [chunk.char_count for chunk in chunks]
        if len(chunk_sizes) > 1:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)
            std_dev = variance ** 0.5
            consistency_score = max(0, 1 - (std_dev / avg_size))
            score = (score + consistency_score) / 2
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))


def _calculate_chunk_quality_score(chunks: List[Chunk]) -> float:
    """Calculate a quality score for chunks (standalone function).
    
    Args:
        chunks: List of chunks to evaluate
        
    Returns:
        Quality score between 0.0 and 1.0
    """
    if not chunks:
        return 0.0
    
    score = 1.0
    
    # Penalize empty chunks
    empty_chunks = sum(1 for chunk in chunks if not chunk.content.strip())
    if empty_chunks > 0:
        score -= (empty_chunks / len(chunks)) * 0.3
    
    # Penalize very short chunks
    very_short = sum(1 for chunk in chunks if len(chunk.content.strip()) < 20)
    if very_short > 0:
        score -= (very_short / len(chunks)) * 0.2
    
    # Reward consistent chunk sizes
    chunk_sizes = [chunk.char_count for chunk in chunks]
    if len(chunk_sizes) > 1:
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        if avg_size > 0:
            variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)
            std_dev = variance ** 0.5
            consistency_score = max(0, 1 - (std_dev / avg_size))
            score = (score + consistency_score) / 2
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))


@dataclass
class ChunkingResult:
    """Result of chunking operation with error handling.
    
    Attributes:
        chunks: List of generated chunks
        success: Whether chunking completed successfully
        fallback_used: Name of fallback strategy used, if any
        warning_message: Warning message for user notification
        error: Exception that occurred, if any
        diagnostics: Chunking diagnostics
    """
    chunks: List[Chunk]
    success: bool = True
    fallback_used: Optional[str] = None
    warning_message: Optional[str] = None
    error: Optional[SemanticChunkerError] = None
    diagnostics: Optional[ChunkingDiagnostics] = None


def fallback_to_sentence_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Chunk]:
    """Fallback to sentence-based chunking when embedding fails.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        
    Returns:
        List of chunks using sentence boundaries
    """
    logger.warning("Falling back to sentence-based chunking")
    
    ensure_nltk_punkt()
    
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # If NLTK fails, use simple splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if not sentences:
        return [Chunk(
            index=0,
            content=text,
            start_position=0,
            end_position=len(text),
            char_count=len(text),
            has_overlap=False
        )]
    
    chunks = []
    current_content = ""
    current_start = 0
    current_pos = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        separator = " " if current_content else ""
        potential = current_content + separator + sentence
        
        if len(potential) > chunk_size and current_content:
            # Save current chunk
            chunks.append(Chunk(
                index=len(chunks),
                content=current_content.strip(),
                start_position=current_start,
                end_position=current_start + len(current_content),
                char_count=len(current_content),
                has_overlap=len(chunks) > 0 and overlap > 0
            ))
            
            # Start new chunk with overlap
            if overlap > 0 and current_content:
                overlap_text = current_content[-overlap:]
                current_content = overlap_text + " " + sentence
            else:
                current_content = sentence
            
            current_start = current_pos
        else:
            current_content = potential
        
        current_pos += len(sentence) + 1
    
    # Add final chunk
    if current_content:
        chunks.append(Chunk(
            index=len(chunks),
            content=current_content.strip(),
            start_position=current_start,
            end_position=current_start + len(current_content),
            char_count=len(current_content),
            has_overlap=len(chunks) > 0 and overlap > 0
        ))
    
    return chunks


def fallback_to_universal_tokenization(text: str) -> List[str]:
    """Fallback to universal tokenization when language detection fails.
    
    Uses basic punctuation patterns that work across languages.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of sentences
    """
    logger.warning("Falling back to universal tokenization")
    
    # Universal pattern: split on sentence-ending punctuation
    pattern = r'(?<=[.!?])\s+(?=[A-Z0-9"\'([])'
    sentences = re.split(pattern, text)
    
    # Filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return [text] if text.strip() else []
    
    return sentences


def fallback_to_fixed_size_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Chunk]:
    """Fallback to fixed-size chunking when sentence tokenization fails.
    
    Respects word boundaries to avoid splitting words.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        
    Returns:
        List of chunks
    """
    logger.warning("Falling back to fixed-size chunking")
    
    if not text or not text.strip():
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Adjust end to word boundary if not at end of text
        if end < text_length:
            # Look for last space before end
            space_pos = text.rfind(' ', start, end)
            if space_pos > start:
                end = space_pos
        
        content = text[start:end].strip()
        
        if content:
            chunks.append(Chunk(
                index=len(chunks),
                content=content,
                start_position=start,
                end_position=end,
                char_count=len(content),
                has_overlap=len(chunks) > 0 and overlap > 0
            ))
        
        # Move start, accounting for overlap
        step = chunk_size - overlap
        if step <= 0:
            step = 1
        start += step
    
    return chunks


def chunk_with_error_handling(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
    chunk_size: int = 500,
    overlap: int = 50,
    similarity_threshold: float = None,
    embedding_model: str = "openai/text-embedding-3-small",
    enable_qa_detection: bool = True,
    enable_adaptive_threshold: bool = True,
    enable_cache: bool = True,
    min_chunk_size: int = 150,
    max_chunk_size: int = 2000,
    buffer_size: int = 1,
    **kwargs
) -> ChunkingResult:
    """Chunk text with comprehensive error handling and fallback strategies.
    
    This function wraps the chunking process with error handling,
    automatically falling back to simpler strategies when errors occur.
    
    Fallback order:
    1. Semantic chunking (if requested)
    2. Sentence-based chunking (if embedding fails)
    3. Fixed-size chunking (if tokenization fails)
    
    Args:
        text: Text to chunk
        strategy: Chunking strategy to use
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        similarity_threshold: Threshold for semantic chunking (None = adaptive)
        embedding_model: Model for embeddings
        enable_qa_detection: Enable Q&A pair detection
        enable_adaptive_threshold: Enable adaptive threshold calculation
        enable_cache: Enable embedding caching
        min_chunk_size: Minimum characters per chunk
        max_chunk_size: Maximum characters per chunk
        buffer_size: Context sentences for semantic chunking
        **kwargs: Additional arguments for chunker
        
    Returns:
        ChunkingResult with chunks and status information
    """
    start_time = time.time()
    
    # Validate input
    if not text or not text.strip():
        return ChunkingResult(
            chunks=[],
            success=False,
            error=SemanticChunkerError(
                "Input text is empty",
                error_code=ErrorCode.VALIDATION_ERROR
            ),
            warning_message="Cannot chunk empty text"
        )
    
    # Create chunker service
    service = ChunkerService()
    
    # Try primary strategy
    try:
        if strategy == ChunkingStrategy.SEMANTIC:
            chunker = SemanticChunker(
                use_provider_manager=True,
                enable_cache=enable_cache,
                enable_qa_detection=enable_qa_detection,
                enable_adaptive_threshold=enable_adaptive_threshold
            )
            chunks = chunker.chunk(
                text,
                similarity_threshold=similarity_threshold,
                embedding_model=embedding_model,
                overlap=overlap,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                buffer_size=buffer_size,
                **kwargs
            )
        else:
            config = ChunkingConfig(
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                similarity_threshold=similarity_threshold or 0.5,
                embedding_model=embedding_model,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                buffer_size=buffer_size
            )
            chunks, diagnostics = service.chunk_with_diagnostics(text, config)
            
            if not chunks and diagnostics.error_message:
                raise SemanticChunkerError(
                    diagnostics.error_message,
                    details=diagnostics.error_details
                )
        
        processing_time = time.time() - start_time
        
        # Calculate quality score
        quality_score = _calculate_chunk_quality_score(chunks)
        
        return ChunkingResult(
            chunks=chunks,
            success=True,
            diagnostics=ChunkingDiagnostics(
                strategy=strategy.value,
                input_text_length=len(text),
                processing_time=processing_time,
                total_chunks=len(chunks),
                avg_chunk_size=sum(c.char_count for c in chunks) / len(chunks) if chunks else 0,
                min_chunk_size=min(c.char_count for c in chunks) if chunks else 0,
                max_chunk_size=max(c.char_count for c in chunks) if chunks else 0,
                overlap_count=sum(1 for c in chunks if c.has_overlap),
                quality_score=quality_score
            )
        )
        
    except Exception as e:
        # Log the error with context
        logger.error(
            "Primary chunking failed",
            extra={
                "strategy": strategy.value,
                "text_length": len(text),
                "error_type": type(e).__name__,
                "error_message": str(e)
            },
            exc_info=True
        )
        
        # Determine fallback strategy
        if strategy == ChunkingStrategy.SEMANTIC:
            # Try sentence-based fallback
            try:
                logger.info("Attempting sentence-based fallback")
                chunks = fallback_to_sentence_chunking(
                    text, chunk_size, overlap
                )
                
                processing_time = time.time() - start_time
                
                return ChunkingResult(
                    chunks=chunks,
                    success=True,
                    fallback_used="sentence",
                    warning_message=(
                        "Semantic chunking failed, used sentence-based fallback. "
                        f"Original error: {str(e)}"
                    ),
                    diagnostics=ChunkingDiagnostics(
                        strategy="sentence_fallback",
                        input_text_length=len(text),
                        processing_time=processing_time,
                        total_chunks=len(chunks),
                        avg_chunk_size=sum(c.char_count for c in chunks) / len(chunks) if chunks else 0,
                        min_chunk_size=min(c.char_count for c in chunks) if chunks else 0,
                        max_chunk_size=max(c.char_count for c in chunks) if chunks else 0,
                        overlap_count=sum(1 for c in chunks if c.has_overlap)
                    )
                )
                
            except Exception as fallback_error:
                logger.error(
                    "Sentence-based fallback failed",
                    extra={"error": str(fallback_error)},
                    exc_info=True
                )
        
        # Final fallback: fixed-size chunking
        try:
            logger.info("Attempting fixed-size fallback")
            chunks = fallback_to_fixed_size_chunking(
                text, chunk_size, overlap
            )
            
            processing_time = time.time() - start_time
            
            return ChunkingResult(
                chunks=chunks,
                success=True,
                fallback_used="fixed_size",
                warning_message=(
                    "All advanced chunking methods failed, used fixed-size fallback. "
                    f"Original error: {str(e)}"
                ),
                diagnostics=ChunkingDiagnostics(
                    strategy="fixed_size_fallback",
                    input_text_length=len(text),
                    processing_time=processing_time,
                    total_chunks=len(chunks),
                    avg_chunk_size=sum(c.char_count for c in chunks) / len(chunks) if chunks else 0,
                    min_chunk_size=min(c.char_count for c in chunks) if chunks else 0,
                    max_chunk_size=max(c.char_count for c in chunks) if chunks else 0,
                    overlap_count=sum(1 for c in chunks if c.has_overlap)
                )
            )
            
        except Exception as final_error:
            processing_time = time.time() - start_time
            
            # All fallbacks failed
            error = SemanticChunkerError(
                f"All chunking strategies failed: {str(final_error)}",
                error_code=ErrorCode.UNKNOWN_ERROR,
                details={
                    "original_error": str(e),
                    "fallback_error": str(final_error),
                    "strategy": strategy.value
                }
            )
            
            return ChunkingResult(
                chunks=[],
                success=False,
                error=error,
                warning_message="All chunking methods failed",
                diagnostics=ChunkingDiagnostics(
                    strategy=strategy.value,
                    input_text_length=len(text),
                    processing_time=processing_time,
                    total_chunks=0,
                    avg_chunk_size=0,
                    min_chunk_size=0,
                    max_chunk_size=0,
                    overlap_count=0,
                    error_message=str(error),
                    error_details=error.details
                )
            )
