# Design Document: Semantic Chunker Enhancement

## Overview

Bu design dokümanı, mevcut Semantic Chunker sisteminin Türkçe dil desteği ve performans iyileştirmeleri için detaylı teknik tasarımı içermektedir. Sistem, embedding tabanlı benzerlik analizi kullanarak metni anlamsal olarak tutarlı parçalara bölen gelişmiş bir chunking stratejisidir.

### Goals

1. **Türkçe Dil Desteği**: Türkçe metinler için %95+ doğrulukla sentence splitting
2. **Çoklu Provider Desteği**: OpenRouter, OpenAI, HuggingFace, local models
3. **Performans**: 5000 karakter için <5 saniye işlem süresi
4. **Kalite Metrikleri**: Semantic coherence, inter-chunk similarity, quality reports
5. **Hata Toleransı**: Fallback mekanizmaları ve graceful degradation

### Non-Goals

- Diğer diller için özel optimizasyon (sadece Türkçe ve İngilizce)
- Real-time streaming chunking (batch processing yeterli)
- Distributed chunking (single-node yeterli)

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Semantic Chunker                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Language Detection & Preprocessing            │  │
│  │  - Auto-detect Turkish/English/Mixed                  │  │
│  │  - Normalize text (whitespace, encoding)              │  │
│  │  - Apply language-specific rules                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Enhanced Sentence Tokenization                │  │
│  │  - Turkish abbreviations handling                     │  │
│  │  - Decimal numbers preservation                       │  │
│  │  - Quoted text handling                               │  │
│  │  - URL/Email preservation                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Q&A Detection & Preprocessing                 │  │
│  │  - Comprehensive Turkish question patterns            │  │
│  │  - Answer detection (next 1-3 sentences)              │  │
│  │  - Semantic similarity confirmation                   │  │
│  │  - Q&A pair merging                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Embedding Provider Abstraction                │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │ OpenRouter │  │   OpenAI   │  │ HuggingFace│     │  │
│  │  └────────────┘  └────────────┘  └────────────┘     │  │
│  │  ┌────────────┐  ┌────────────┐                      │  │
│  │  │   Local    │  │   Cache    │                      │  │
│  │  │  Models    │  │  (Redis)   │                      │  │
│  │  └────────────┘  └────────────┘                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Semantic Similarity Analysis                  │  │
│  │  - Batch embedding processing                         │  │
│  │  - Cosine similarity calculation                      │  │
│  │  - Adaptive threshold determination                   │  │
│  │  - Breakpoint detection (percentile-based)            │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Chunk Creation & Optimization                 │  │
│  │  - Create chunks from breakpoints                     │  │
│  │  - Merge small chunks                                 │  │
│  │  - Add overlap (sentence-boundary aware)              │  │
│  │  - Validate chunk quality                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Quality Metrics & Reporting                   │  │
│  │  - Semantic coherence score                           │  │
│  │  - Inter-chunk similarity                             │  │
│  │  - Topic distribution analysis                        │  │
│  │  - Quality report generation                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User Request
    ↓
[Language Detection] → Detect Turkish/English/Mixed
    ↓
[Sentence Tokenization] → Split into sentences (language-aware)
    ↓
[Q&A Detection] → Identify and merge Q&A pairs
    ↓
[Embedding Provider] → Get embeddings (with caching)
    ↓
[Similarity Analysis] → Calculate distances, find breakpoints
    ↓
[Chunk Creation] → Create, merge, optimize chunks
    ↓
[Quality Metrics] → Calculate coherence, similarity, quality
    ↓
Response (chunks + metrics)
```

## Components and Interfaces

### 1. Language Detector

**Purpose**: Automatically detect text language and apply appropriate processing rules.

**Interface**:
```python
class LanguageDetector:
    """Detect language of text and provide language-specific configuration."""
    
    def detect_language(self, text: str) -> Language:
        """Detect primary language of text.
        
        Returns:
            Language enum (TURKISH, ENGLISH, MIXED, UNKNOWN)
        """
        
    def get_language_config(self, language: Language) -> LanguageConfig:
        """Get language-specific configuration.
        
        Returns:
            LanguageConfig with tokenizer, patterns, etc.
        """
```

**Implementation Details**:
- Use `langdetect` library for initial detection
- Fallback to character-based heuristics (Turkish chars: ç, ğ, ı, ö, ş, ü)
- For mixed texts, detect dominant language
- Cache detection results per text hash

### 2. Enhanced Sentence Tokenizer

**Purpose**: Split text into sentences with Turkish and English support.

**Interface**:
```python
class EnhancedSentenceTokenizer:
    """Advanced sentence tokenization with multi-language support."""
    
    # Turkish abbreviations
    TURKISH_ABBREVIATIONS = {
        'Dr.', 'Prof.', 'Doç.', 'Yrd.', 'Öğr.', 'Uzm.',
        'örn.', 'vs.', 'vb.', 'vd.', 'bkz.', 'krş.',
        'A.Ş.', 'Ltd.', 'Şti.', 'Inc.', 'Co.', 'Corp.',
        'No.', 'Tel.', 'Fax.', 'Apt.', 'Cad.', 'Sok.',
        'Mah.', 'İl.', 'İlçe.', 'Köy.', 'Bld.', 'Blv.'
    }
    
    # English abbreviations
    ENGLISH_ABBREVIATIONS = {
        'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.',
        'Inc.', 'Ltd.', 'Corp.', 'Co.', 'etc.', 'e.g.', 'i.e.',
        'vs.', 'vol.', 'no.', 'p.', 'pp.', 'ed.', 'eds.'
    }
    
    def tokenize(
        self, 
        text: str, 
        language: Language
    ) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Input text
            language: Detected language
            
        Returns:
            List of sentences
        """
        
    def _handle_abbreviations(
        self, 
        text: str, 
        abbreviations: Set[str]
    ) -> str:
        """Replace abbreviations with placeholders before splitting."""
        
    def _handle_decimals(self, text: str) -> str:
        """Protect decimal numbers from splitting."""
        
    def _handle_quoted_text(self, text: str) -> str:
        """Protect quoted text from splitting."""
        
    def _handle_urls_emails(self, text: str) -> str:
        """Protect URLs and emails from splitting."""
```

**Implementation Strategy**:
1. **Preprocessing**: Replace special patterns with placeholders
2. **Splitting**: Use regex patterns for sentence boundaries
3. **Postprocessing**: Restore original patterns
4. **Validation**: Check for common splitting errors

**Turkish Sentence Splitting Pattern**:
```python
# Pattern for Turkish sentence boundaries
TURKISH_SENTENCE_PATTERN = r'''
    (?<=[.!?])              # After sentence-ending punctuation
    \s+                     # Followed by whitespace
    (?=[A-ZÇĞİÖŞÜ0-9"'(])  # Before capital letter, number, or quote
'''
```

### 3. Q&A Detector

**Purpose**: Detect question-answer pairs and keep them together.

**Interface**:
```python
class QADetector:
    """Detect and handle question-answer pairs."""
    
    # Comprehensive Turkish question patterns
    TURKISH_QUESTION_WORDS = {
        # Basic question words
        'ne', 'nasıl', 'neden', 'niçin', 'niye',
        'kim', 'kimi', 'kimin', 'kime', 'kimden',
        'nerede', 'nereye', 'nereden', 'nere',
        'hangi', 'hangisi', 'hangileri',
        'kaç', 'kaçıncı', 'kaçar',
        'ne zaman', 'ne kadar', 'ne için',
        
        # Indirect questions
        'acaba', 'merak ediyorum', 'bilmiyorum',
        'anlamadım', 'anlayamadım',
        
        # Question particles (all forms)
        'mi', 'mı', 'mu', 'mü',
        'midir', 'mıdır', 'mudur', 'müdür',
        'misin', 'mısın', 'musun', 'müsün',
        'miyim', 'mıyım', 'muyum', 'müyüm',
        'miyiz', 'mıyız', 'muyuz', 'müyüz',
        'misiniz', 'mısınız', 'musunuz', 'müsünüz',
        'midirler', 'mıdırlar', 'mudurlar', 'müdürler'
    }
    
    def is_question(self, sentence: str, language: Language) -> bool:
        """Check if sentence is a question."""
        
    def find_answer(
        self, 
        question: str, 
        following_sentences: List[str],
        embeddings: Optional[List[List[float]]] = None
    ) -> Optional[str]:
        """Find answer to question in following sentences.
        
        Uses semantic similarity if embeddings provided.
        """
        
    def merge_qa_pairs(
        self, 
        sentences: List[str],
        language: Language
    ) -> List[str]:
        """Merge Q&A pairs into single sentences."""
```

**Q&A Detection Algorithm**:
1. **Question Detection**: Pattern matching + ending with '?'
2. **Answer Search**: Check next 1-3 sentences
3. **Semantic Confirmation**: Calculate similarity between Q and A
4. **Merging**: Combine if similarity > threshold (0.6)

### 4. Embedding Provider Abstraction

**Purpose**: Support multiple embedding providers with fallback.

**Interface**:
```python
class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def get_embeddings(
        self, 
        texts: List[str],
        model: str
    ) -> List[List[float]]:
        """Get embeddings for texts."""
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        
    @abstractmethod
    def get_cost_estimate(self, text_count: int) -> float:
        """Estimate cost for embedding texts."""


class OpenRouterProvider(EmbeddingProvider):
    """OpenRouter API provider."""
    
    DEFAULT_MODEL = "openai/text-embedding-3-small"
    
    def get_embeddings(
        self, 
        texts: List[str],
        model: str = DEFAULT_MODEL
    ) -> List[List[float]]:
        """Get embeddings via OpenRouter API."""


class OpenAIProvider(EmbeddingProvider):
    """Direct OpenAI API provider."""
    
    DEFAULT_MODEL = "text-embedding-3-small"
    
    def get_embeddings(
        self, 
        texts: List[str],
        model: str = DEFAULT_MODEL
    ) -> List[List[float]]:
        """Get embeddings via OpenAI API."""


class HuggingFaceProvider(EmbeddingProvider):
    """HuggingFace models provider."""
    
    TURKISH_MODELS = [
        "dbmdz/bert-base-turkish-cased",
        "loodos/bert-turkish-base",
        "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    ]
    
    def get_embeddings(
        self, 
        texts: List[str],
        model: str
    ) -> List[List[float]]:
        """Get embeddings via HuggingFace."""


class LocalModelProvider(EmbeddingProvider):
    """Local sentence-transformers models."""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        
    def get_embeddings(
        self, 
        texts: List[str],
        model: str = None
    ) -> List[List[float]]:
        """Get embeddings using local model."""


class EmbeddingProviderManager:
    """Manage multiple embedding providers with fallback."""
    
    def __init__(
        self,
        providers: List[EmbeddingProvider],
        cache: Optional[EmbeddingCache] = None
    ):
        self.providers = providers
        self.cache = cache
        
    def get_embeddings(
        self, 
        texts: List[str],
        model: str,
        language: Language
    ) -> List[List[float]]:
        """Get embeddings with fallback and caching.
        
        1. Check cache first
        2. Try primary provider
        3. Fallback to secondary providers
        4. Cache results
        """
```

**Provider Selection Strategy**:
```python
def select_provider(
    language: Language,
    text_length: int,
    config: ProviderConfig
) -> EmbeddingProvider:
    """Select best provider based on context."""
    
    if language == Language.TURKISH:
        # Prefer Turkish-optimized models
        if config.prefer_local:
            return LocalModelProvider("dbmdz/bert-base-turkish-cased")
        else:
            return HuggingFaceProvider()
    
    elif text_length > 10000:
        # For large texts, prefer local to avoid API costs
        return LocalModelProvider("all-MiniLM-L6-v2")
    
    else:
        # Default to OpenRouter
        return OpenRouterProvider()
```

### 5. Embedding Cache

**Purpose**: Cache embeddings to reduce API calls and improve performance.

**Interface**:
```python
class EmbeddingCache:
    """Cache for embedding vectors."""
    
    def __init__(
        self,
        backend: str = "memory",  # "memory" or "redis"
        ttl: int = 3600  # 1 hour
    ):
        self.backend = backend
        self.ttl = ttl
        
    def get(
        self, 
        text: str, 
        model: str
    ) -> Optional[List[float]]:
        """Get cached embedding."""
        
    def set(
        self, 
        text: str, 
        model: str, 
        embedding: List[float]
    ):
        """Cache embedding."""
        
    def get_batch(
        self, 
        texts: List[str], 
        model: str
    ) -> Dict[str, List[float]]:
        """Get multiple cached embeddings."""
        
    def set_batch(
        self, 
        embeddings: Dict[str, List[float]], 
        model: str
    ):
        """Cache multiple embeddings."""
        
    def get_stats(self) -> CacheStats:
        """Get cache statistics (hit rate, size, etc.)."""
```

**Cache Key Strategy**:
```python
def generate_cache_key(text: str, model: str) -> str:
    """Generate cache key for text and model."""
    # Use hash to keep keys short
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    model_hash = hashlib.sha256(model.encode()).hexdigest()[:8]
    return f"emb:{model_hash}:{text_hash}"
```

### 6. Adaptive Threshold Calculator

**Purpose**: Automatically determine optimal similarity threshold based on text characteristics.

**Interface**:
```python
class AdaptiveThresholdCalculator:
    """Calculate optimal similarity threshold for text."""
    
    def analyze_text(self, text: str) -> TextCharacteristics:
        """Analyze text characteristics.
        
        Returns:
            TextCharacteristics with:
            - length: character count
            - sentence_count: number of sentences
            - vocabulary_diversity: unique words / total words
            - avg_sentence_length: average sentence length
            - topic_coherence: estimated topic coherence
        """
        
    def calculate_threshold(
        self, 
        characteristics: TextCharacteristics,
        base_threshold: float = 0.5
    ) -> float:
        """Calculate optimal threshold.
        
        Algorithm:
        - High diversity → Lower threshold (more splits)
        - Low diversity → Higher threshold (fewer splits)
        - Long sentences → Lower threshold
        - Short sentences → Higher threshold
        """
        
    def recommend_threshold(
        self, 
        text: str
    ) -> ThresholdRecommendation:
        """Provide threshold recommendation with explanation."""
```

**Threshold Calculation Algorithm**:
```python
def calculate_adaptive_threshold(
    vocabulary_diversity: float,
    avg_sentence_length: float,
    base_threshold: float = 0.5
) -> float:
    """
    Adaptive threshold formula:
    
    threshold = base_threshold * diversity_factor * length_factor
    
    diversity_factor:
    - High diversity (>0.7): 0.8 (lower threshold, more splits)
    - Medium diversity (0.4-0.7): 1.0 (keep base)
    - Low diversity (<0.4): 1.2 (higher threshold, fewer splits)
    
    length_factor:
    - Long sentences (>100 chars): 0.9
    - Medium sentences (50-100 chars): 1.0
    - Short sentences (<50 chars): 1.1
    """
    
    # Calculate diversity factor
    if vocabulary_diversity > 0.7:
        diversity_factor = 0.8
    elif vocabulary_diversity > 0.4:
        diversity_factor = 1.0
    else:
        diversity_factor = 1.2
    
    # Calculate length factor
    if avg_sentence_length > 100:
        length_factor = 0.9
    elif avg_sentence_length > 50:
        length_factor = 1.0
    else:
        length_factor = 1.1
    
    # Calculate final threshold
    threshold = base_threshold * diversity_factor * length_factor
    
    # Clamp to reasonable range
    return max(0.3, min(0.9, threshold))
```

## Data Models

### Language Enum
```python
class Language(str, Enum):
    """Supported languages."""
    TURKISH = "tr"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"
```

### LanguageConfig
```python
@dataclass
class LanguageConfig:
    """Language-specific configuration."""
    language: Language
    abbreviations: Set[str]
    question_patterns: Set[str]
    sentence_pattern: str
    tokenizer: Optional[Any] = None
```

### TextCharacteristics
```python
@dataclass
class TextCharacteristics:
    """Text analysis results."""
    length: int
    sentence_count: int
    vocabulary_diversity: float  # unique words / total words
    avg_sentence_length: float
    topic_coherence: float  # 0-1
    has_questions: bool
    question_count: int
```

### ThresholdRecommendation
```python
@dataclass
class ThresholdRecommendation:
    """Threshold recommendation with explanation."""
    recommended_threshold: float
    confidence: float  # 0-1
    reasoning: str
    text_characteristics: TextCharacteristics
```

### ChunkQualityMetrics
```python
@dataclass
class ChunkQualityMetrics:
    """Quality metrics for a chunk."""
    chunk_index: int
    semantic_coherence: float  # 0-1
    sentence_count: int
    avg_sentence_similarity: float  # 0-1
    topic_consistency: float  # 0-1
    has_questions: bool
    has_qa_pairs: bool
```

### QualityReport
```python
@dataclass
class QualityReport:
    """Overall quality report for chunking."""
    total_chunks: int
    avg_coherence: float
    min_coherence: float
    max_coherence: float
    chunks_below_threshold: List[int]  # chunk indices
    inter_chunk_similarities: List[float]
    merge_recommendations: List[Tuple[int, int]]  # (chunk1, chunk2)
    split_recommendations: List[int]  # chunk indices
    overall_quality_score: float  # 0-1
    recommendations: List[str]
```

### EmbeddingProviderConfig
```python
@dataclass
class EmbeddingProviderConfig:
    """Configuration for embedding providers."""
    primary_provider: str  # "openrouter", "openai", "huggingface", "local"
    fallback_providers: List[str]
    turkish_model: str = "dbmdz/bert-base-turkish-cased"
    english_model: str = "openai/text-embedding-3-small"
    prefer_local: bool = False
    enable_cache: bool = True
    cache_backend: str = "memory"  # "memory" or "redis"
    cache_ttl: int = 3600
    batch_size: int = 32
    max_retries: int = 3
    retry_delay: float = 1.0
```

### CacheStats
```python
@dataclass
class CacheStats:
    """Cache statistics."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    total_size_bytes: int
    entry_count: int
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property Reflection

Prework analizinde toplam 60 acceptance criteria incelendi. Bunlardan:
- **Property tests**: 35 adet (tüm input'lar için geçerli kurallar)
- **Example tests**: 20 adet (spesifik durumlar, konfigürasyon kontrolleri)
- **Edge cases**: 5 adet (hata durumları, özel senaryolar)
- **Meta-tests**: 5 adet (test coverage kontrolleri - implement edilmeyecek)

**Redundancy Analysis**:

1. **Sentence Splitting Properties** (3.2, 3.3, 3.4, 3.5) → Tek property'de birleştirilebilir: "Sentence splitting preserves special patterns"
2. **Provider Configuration** (4.1, 4.3, 5.1, 5.3) → Example testler, ayrı tutulacak
3. **Turkish Question Detection** (1.2, 9.1) → Aynı property, birleştirilecek
4. **Q&A Pairing** (1.5, 9.2, 9.3, 9.4) → Tek comprehensive property'de birleştirilebilir
5. **Threshold Calculation** (8.2, 8.3) → Tek property: "Adaptive threshold responds to text diversity"

**Final Property Count**: 25 unique properties

### Correctness Properties

#### Property 1: Turkish Abbreviation Preservation
*For any* Turkish text containing abbreviations (Dr., Prof., örn., vs., vb., etc.), sentence splitting SHALL NOT split at abbreviation points, and the abbreviation SHALL remain part of its sentence.

**Validates: Requirements 1.1, 3.2**

**Test Strategy**: Generate random Turkish texts with various abbreviations, verify no splits occur at abbreviation points.

#### Property 2: Turkish Character Encoding Preservation
*For any* Turkish text containing special characters (ç, ğ, ı, ö, ş, ü, Ç, Ğ, İ, Ö, Ş, Ü), the chunking process SHALL preserve all characters without corruption or loss.

**Validates: Requirements 1.3**

**Test Strategy**: Generate random Turkish texts with all special characters, verify character-by-character equality after chunking.

#### Property 3: Turkish Punctuation Handling
*For any* Turkish text with punctuation patterns (ellipsis, dashes, quotes), sentence splitting SHALL handle them correctly without inappropriate splits.

**Validates: Requirements 1.4**

**Test Strategy**: Generate texts with various punctuation patterns, verify splits only occur at true sentence boundaries.

#### Property 4: Comprehensive Turkish Question Detection
*For any* Turkish question (using any question word or particle), the system SHALL correctly identify it as a question.

**Validates: Requirements 1.2, 9.1**

**Test Strategy**: Generate random Turkish questions using all question patterns, verify 100% detection rate.

#### Property 5: Q&A Pair Preservation
*For any* detected question-answer pair where semantic similarity > 0.6, both SHALL be kept in the same chunk.

**Validates: Requirements 1.5, 9.2, 9.3, 9.4**

**Test Strategy**: Generate random Q&A pairs, verify they remain together after chunking.

#### Property 6: Language Detection Accuracy
*For any* text in Turkish or English, language detection SHALL correctly identify the primary language with >90% accuracy.

**Validates: Requirements 2.1**

**Test Strategy**: Generate random texts in Turkish and English, verify detection accuracy.

#### Property 7: Mixed Language Handling
*For any* mixed-language text, the system SHALL apply appropriate rules for each language segment without cross-contamination.

**Validates: Requirements 2.2, 2.3**

**Test Strategy**: Generate mixed Turkish-English texts, verify each segment is processed with correct rules.

#### Property 8: Sentence Boundary Accuracy
*For any* Turkish or English text, sentence boundary detection SHALL maintain >95% accuracy compared to human annotation.

**Validates: Requirements 2.5**

**Test Strategy**: Use benchmark datasets with human-annotated sentence boundaries, measure accuracy.

#### Property 9: Special Pattern Preservation
*For any* text containing decimal numbers (1.5, 2.3), URLs, or email addresses, sentence splitting SHALL NOT split these patterns.

**Validates: Requirements 3.3, 3.5**

**Test Strategy**: Generate texts with decimals, URLs, emails, verify they remain intact.

#### Property 10: Quoted Text Preservation
*For any* text with quoted segments, sentence splitting SHALL NOT split within quotes.

**Validates: Requirements 3.4**

**Test Strategy**: Generate texts with various quote styles, verify no splits within quotes.

#### Property 11: Turkish Model Preference
*For any* Turkish text, when Turkish-optimized models are available, the system SHALL prefer them over generic models.

**Validates: Requirements 4.5, 5.2**

**Test Strategy**: Process Turkish texts, verify Turkish models are selected when available.

#### Property 12: Embedding Cache Effectiveness
*For any* repeated text segment, the second request SHALL retrieve embeddings from cache, not make a new API call.

**Validates: Requirements 6.2**

**Test Strategy**: Process same text twice, verify second request uses cache (measure API calls).

#### Property 13: Processing Time Performance
*For any* text up to 5000 characters, processing SHALL complete within 5 seconds.

**Validates: Requirements 6.5**

**Test Strategy**: Generate random texts of various sizes up to 5000 chars, measure processing time.

#### Property 14: Semantic Coherence Calculation
*For any* generated chunk, a semantic coherence score (0-1) SHALL be calculated based on intra-chunk sentence similarity.

**Validates: Requirements 7.1**

**Test Strategy**: Generate chunks, verify coherence scores are calculated and within valid range.

#### Property 15: Inter-Chunk Similarity Measurement
*For any* pair of consecutive chunks, inter-chunk similarity SHALL be measured and reported.

**Validates: Requirements 7.2**

**Test Strategy**: Generate multiple chunks, verify similarity is calculated for all consecutive pairs.

#### Property 16: Low Coherence Detection
*For any* chunk with semantic coherence < 0.5, the system SHALL flag it as low-quality.

**Validates: Requirements 7.4**

**Test Strategy**: Generate intentionally incoherent chunks, verify they are flagged.

#### Property 17: Text Characteristics Analysis
*For any* input text, the system SHALL analyze and report vocabulary diversity, sentence count, and average sentence length.

**Validates: Requirements 8.1**

**Test Strategy**: Generate texts with known characteristics, verify analysis accuracy.

#### Property 18: Adaptive Threshold Response to Diversity
*For any* text with high vocabulary diversity (>0.7), the adaptive threshold SHALL be lower than base threshold; for low diversity (<0.4), it SHALL be higher.

**Validates: Requirements 8.2, 8.3**

**Test Strategy**: Generate high and low diversity texts, verify threshold adjustments.

#### Property 19: Answer Detection After Question
*For any* detected question, the system SHALL analyze the next 1-3 sentences to identify potential answers.

**Validates: Requirements 9.2**

**Test Strategy**: Generate Q&A sequences, verify answer detection attempts.

#### Property 20: Semantic Q&A Confirmation
*For any* question-answer candidate pair, semantic similarity SHALL be used to confirm the pairing (threshold > 0.6).

**Validates: Requirements 9.3**

**Test Strategy**: Generate Q&A pairs with varying similarity, verify confirmation logic.

#### Property 21: Batch Embedding Processing
*For any* list of sentences, embeddings SHALL be requested in batches rather than individually, reducing API calls by >80%.

**Validates: Requirements 6.1**

**Test Strategy**: Process texts with multiple sentences, count API calls, verify batching.

#### Property 22: Provider Fallback on Failure
*For any* embedding provider failure, the system SHALL automatically fallback to the next available provider within 2 seconds.

**Validates: Requirements 4.2, 11.1**

**Test Strategy**: Simulate provider failures, verify fallback occurs and timing.

#### Property 23: Error Logging Completeness
*For any* error during processing, the system SHALL log the error with full context (text length, provider, model, error message).

**Validates: Requirements 11.2**

**Test Strategy**: Trigger various errors, verify log completeness.

#### Property 24: Fallback Notification
*For any* fallback to sentence-based chunking, the system SHALL notify the user with a warning message.

**Validates: Requirements 11.3**

**Test Strategy**: Trigger fallback scenarios, verify user notification.

#### Property 25: Exponential Backoff Implementation
*For any* API retry, the system SHALL implement exponential backoff with delays: 1s, 2s, 4s (max 3 retries).

**Validates: Requirements 11.4**

**Test Strategy**: Simulate API failures, measure retry delays, verify exponential pattern.

## Error Handling

### Error Categories

1. **Input Validation Errors**
   - Empty or whitespace-only text
   - Text too large (>10MB)
   - Invalid configuration parameters
   - **Handling**: Return 422 with descriptive error message

2. **Language Detection Errors**
   - Unable to detect language
   - **Handling**: Fallback to universal tokenization, log warning

3. **Embedding Provider Errors**
   - API key missing or invalid
   - API rate limit exceeded
   - API timeout or network error
   - Model not available
   - **Handling**: Fallback to next provider, implement retry with exponential backoff

4. **Sentence Tokenization Errors**
   - NLTK data not available
   - Regex pattern errors
   - **Handling**: Fallback to simple splitting, log error

5. **Cache Errors**
   - Redis connection failed
   - Cache corruption
   - **Handling**: Fallback to no-cache mode, log error

6. **Quality Calculation Errors**
   - Insufficient data for metrics
   - Numerical errors in calculations
   - **Handling**: Return partial metrics, log warning

### Error Handling Strategy

```python
class SemanticChunkerError(Exception):
    """Base exception for semantic chunker errors."""
    pass

class LanguageDetectionError(SemanticChunkerError):
    """Language detection failed."""
    pass

class EmbeddingProviderError(SemanticChunkerError):
    """Embedding provider error."""
    pass

class SentenceTokenizationError(SemanticChunkerError):
    """Sentence tokenization error."""
    pass

class CacheError(SemanticChunkerError):
    """Cache operation error."""
    pass


def chunk_with_error_handling(
    text: str,
    config: SemanticChunkerConfig
) -> Tuple[List[Chunk], Optional[str]]:
    """Chunk text with comprehensive error handling.
    
    Returns:
        (chunks, error_message)
        - If successful: (chunks, None)
        - If fallback: (chunks, warning_message)
        - If failed: ([], error_message)
    """
    try:
        # Try semantic chunking
        return semantic_chunk(text, config), None
        
    except EmbeddingProviderError as e:
        logger.warning(f"Embedding provider failed: {e}, falling back to sentence chunking")
        # Fallback to sentence-based chunking
        chunks = sentence_chunk(text, config)
        warning = "Semantic chunking unavailable, used sentence-based chunking"
        return chunks, warning
        
    except LanguageDetectionError as e:
        logger.warning(f"Language detection failed: {e}, using universal tokenization")
        # Continue with universal tokenization
        return semantic_chunk_universal(text, config), None
        
    except SentenceTokenizationError as e:
        logger.error(f"Sentence tokenization failed: {e}")
        # Fallback to fixed-size chunking
        chunks = fixed_size_chunk(text, config)
        warning = "Sentence tokenization failed, used fixed-size chunking"
        return chunks, warning
        
    except Exception as e:
        logger.error(f"Unexpected error in semantic chunking: {e}", exc_info=True)
        return [], f"Chunking failed: {str(e)}"
```

### Retry Logic with Exponential Backoff

```python
def get_embeddings_with_retry(
    provider: EmbeddingProvider,
    texts: List[str],
    model: str,
    max_retries: int = 3
) -> List[List[float]]:
    """Get embeddings with exponential backoff retry."""
    
    for attempt in range(max_retries):
        try:
            return provider.get_embeddings(texts, model)
            
        except (APIError, TimeoutError, NetworkError) as e:
            if attempt == max_retries - 1:
                raise EmbeddingProviderError(f"Failed after {max_retries} attempts: {e}")
            
            # Exponential backoff: 1s, 2s, 4s
            delay = 2 ** attempt
            logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s")
            time.sleep(delay)
```

## Testing Strategy

### Unit Tests

**Purpose**: Test individual components in isolation.

**Coverage**:
1. **Language Detection**
   - Test Turkish text detection
   - Test English text detection
   - Test mixed text detection
   - Test edge cases (very short text, numbers only)

2. **Sentence Tokenization**
   - Test abbreviation handling
   - Test decimal number preservation
   - Test quoted text handling
   - Test URL/email preservation
   - Test Turkish-specific patterns

3. **Q&A Detection**
   - Test Turkish question patterns
   - Test answer detection
   - Test Q&A pairing logic
   - Test nested questions

4. **Embedding Providers**
   - Test each provider individually
   - Test provider fallback
   - Test cache operations
   - Test batch processing

5. **Adaptive Threshold**
   - Test threshold calculation
   - Test text analysis
   - Test recommendations

6. **Quality Metrics**
   - Test coherence calculation
   - Test similarity measurement
   - Test report generation

### Property-Based Tests

**Purpose**: Verify universal properties across all inputs.

**Framework**: Hypothesis (Python)

**Key Properties** (from Correctness Properties section):
- Property 1-25 as defined above

**Example Property Test**:
```python
from hypothesis import given, strategies as st

@given(
    text=st.text(
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll'),
            whitelist_characters='çğıöşüÇĞİÖŞÜ'
        ),
        min_size=100,
        max_size=1000
    )
)
def test_turkish_character_preservation(text):
    """Property: Turkish characters are preserved."""
    chunker = SemanticChunker()
    chunks = chunker.chunk(text)
    
    # Reconstruct text from chunks
    reconstructed = ''.join(chunk.content for chunk in chunks)
    
    # Verify all Turkish characters are preserved
    for char in 'çğıöşüÇĞİÖŞÜ':
        assert text.count(char) == reconstructed.count(char)
```

### Integration Tests

**Purpose**: Test component interactions and end-to-end flows.

**Scenarios**:
1. **Full Chunking Pipeline**
   - Input: Real Turkish document
   - Verify: Correct chunks, quality metrics, no errors

2. **Provider Fallback**
   - Simulate primary provider failure
   - Verify: Fallback occurs, chunks still generated

3. **Cache Integration**
   - Process same text twice
   - Verify: Second request uses cache

4. **Mixed Language Processing**
   - Input: Turkish-English mixed text
   - Verify: Correct language detection and processing

### Performance Benchmarks

**Purpose**: Measure and track performance metrics.

**Benchmarks**:
1. **Processing Speed**
   - Text sizes: 1K, 5K, 10K, 50K characters
   - Measure: Time to complete chunking
   - Target: <5s for 5K characters

2. **API Call Reduction**
   - Measure: API calls with/without batching
   - Target: >80% reduction with batching

3. **Cache Hit Rate**
   - Measure: Cache hits vs misses
   - Target: >70% hit rate for repeated content

4. **Memory Usage**
   - Measure: Peak memory during processing
   - Target: <500MB for 50K character text

### Test Data

**Turkish Test Corpus**:
- News articles (formal Turkish)
- Social media posts (informal Turkish)
- Academic papers (technical Turkish)
- Q&A forums (conversational Turkish)
- Mixed Turkish-English content

**English Test Corpus**:
- News articles
- Technical documentation
- Conversational text
- Mixed content

**Synthetic Test Data**:
- Generated texts with specific patterns
- Edge cases (very short, very long, special characters)
- Adversarial examples (difficult to chunk correctly)

## Deployment Considerations

### Configuration

**Environment Variables**:
```bash
# Embedding Providers
OPENROUTER_API_KEY=xxx
OPENAI_API_KEY=xxx
HUGGINGFACE_API_KEY=xxx

# Provider Configuration
EMBEDDING_PRIMARY_PROVIDER=openrouter
EMBEDDING_FALLBACK_PROVIDERS=openai,huggingface,local
EMBEDDING_TURKISH_MODEL=dbmdz/bert-base-turkish-cased
EMBEDDING_ENGLISH_MODEL=openai/text-embedding-3-small

# Cache Configuration
EMBEDDING_CACHE_BACKEND=redis  # or "memory"
EMBEDDING_CACHE_TTL=3600
REDIS_URL=redis://localhost:6379

# Performance Configuration
EMBEDDING_BATCH_SIZE=32
EMBEDDING_MAX_RETRIES=3
EMBEDDING_RETRY_DELAY=1.0

# Feature Flags
ENABLE_ADAPTIVE_THRESHOLD=true
ENABLE_QA_DETECTION=true
ENABLE_LANGUAGE_DETECTION=true
```

### Monitoring

**Metrics to Track**:
1. **Performance Metrics**
   - Processing time (p50, p95, p99)
   - API call count
   - Cache hit rate
   - Memory usage

2. **Quality Metrics**
   - Average chunk coherence
   - Sentence boundary accuracy
   - Language detection accuracy

3. **Error Metrics**
   - Error rate by type
   - Fallback frequency
   - Retry count

4. **Business Metrics**
   - API cost per request
   - Throughput (requests/second)
   - User satisfaction

**Monitoring Tools**:
- Prometheus for metrics collection
- Grafana for visualization
- Sentry for error tracking
- Custom dashboards for quality metrics

### Scaling Considerations

1. **Horizontal Scaling**
   - Stateless design allows multiple instances
   - Load balancer distributes requests
   - Shared Redis cache across instances

2. **Vertical Scaling**
   - Increase memory for larger cache
   - More CPU for faster processing
   - GPU for local model inference

3. **Caching Strategy**
   - Redis cluster for distributed cache
   - Cache warming for common texts
   - TTL optimization based on usage patterns

4. **Rate Limiting**
   - Per-user rate limits
   - Per-provider rate limits
   - Graceful degradation under load

## Migration Plan

### Phase 1: Foundation (Week 1-2)

1. **Language Detection**
   - Implement LanguageDetector
   - Add Turkish character detection
   - Unit tests

2. **Enhanced Sentence Tokenization**
   - Implement EnhancedSentenceTokenizer
   - Add abbreviation handling
   - Add decimal/URL/email preservation
   - Unit tests

### Phase 2: Embedding Infrastructure (Week 3-4)

1. **Provider Abstraction**
   - Implement EmbeddingProvider interface
   - Implement OpenRouterProvider (existing)
   - Implement OpenAIProvider
   - Implement HuggingFaceProvider
   - Implement LocalModelProvider

2. **Provider Manager**
   - Implement EmbeddingProviderManager
   - Add fallback logic
   - Add retry with exponential backoff

3. **Embedding Cache**
   - Implement EmbeddingCache
   - Add memory backend
   - Add Redis backend
   - Unit tests

### Phase 3: Q&A and Adaptive Features (Week 5-6)

1. **Q&A Detection**
   - Implement QADetector
   - Add comprehensive Turkish question patterns
   - Add answer detection logic
   - Unit tests

2. **Adaptive Threshold**
   - Implement AdaptiveThresholdCalculator
   - Add text analysis
   - Add threshold calculation
   - Unit tests

### Phase 4: Quality Metrics (Week 7-8)

1. **Chunk Quality Metrics**
   - Implement semantic coherence calculation
   - Implement inter-chunk similarity
   - Implement topic distribution analysis

2. **Quality Reporting**
   - Implement QualityReport generation
   - Add recommendations engine
   - Unit tests

### Phase 5: Integration and Testing (Week 9-10)

1. **Integration**
   - Integrate all components into SemanticChunker
   - Update API endpoints
   - Update documentation

2. **Testing**
   - Property-based tests
   - Integration tests
   - Performance benchmarks
   - User acceptance testing

### Phase 6: Deployment and Monitoring (Week 11-12)

1. **Deployment**
   - Deploy to staging
   - Performance testing
   - Deploy to production (gradual rollout)

2. **Monitoring**
   - Set up metrics collection
   - Set up dashboards
   - Set up alerts
   - Monitor and optimize

## Success Criteria

1. **Functional Requirements**
   - ✅ Turkish sentence splitting accuracy >95%
   - ✅ Turkish question detection accuracy >95%
   - ✅ Q&A pair preservation rate >90%
   - ✅ Language detection accuracy >90%
   - ✅ All 25 correctness properties pass

2. **Performance Requirements**
   - ✅ Processing time <5s for 5K characters
   - ✅ API call reduction >80% with batching
   - ✅ Cache hit rate >70%
   - ✅ Memory usage <500MB for 50K characters

3. **Quality Requirements**
   - ✅ Average chunk coherence >0.8
   - ✅ Inter-chunk similarity properly measured
   - ✅ Quality reports generated with actionable recommendations

4. **Reliability Requirements**
   - ✅ Fallback mechanisms work correctly
   - ✅ Error handling covers all scenarios
   - ✅ System remains available during provider failures

5. **User Satisfaction**
   - ✅ User feedback rating >4.5/5
   - ✅ Reduced complaints about Turkish text handling
   - ✅ Improved RAG system performance metrics
