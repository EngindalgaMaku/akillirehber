# Requirements Document: Semantic Chunker Enhancement

## Introduction

Bu doküman, mevcut Semantic Chunker sisteminin detaylı analizini ve Türkçe dil desteği ile genel iyileştirmeler için gereksinimleri içermektedir. Semantic chunker, embedding tabanlı benzerlik analizi kullanarak metni anlamsal olarak tutarlı parçalara bölen gelişmiş bir chunking stratejisidir.

## Glossary

- **Semantic_Chunker**: Embedding ve benzerlik analizi kullanarak metni anlamsal olarak tutarlı parçalara bölen sistem
- **Embedding**: Metinlerin vektör uzayında sayısal temsili
- **Cosine_Similarity**: İki vektör arasındaki benzerliği ölçen metrik (0-1 arası)
- **Buffer**: Cümlelerin bağlam için etrafına eklenen ek cümle sayısı
- **Breakpoint**: Chunk'ların bölüneceği nokta
- **Percentile_Threshold**: Benzerlik mesafelerinin yüzdelik dilim eşiği
- **NLTK**: Natural Language Toolkit - doğal dil işleme kütüphanesi
- **OpenRouter**: Embedding API sağlayıcısı
- **Q&A_Pair**: Soru-cevap çifti

## Mevcut Durum Analizi

### Güçlü Yönler

1. **Buffer-based Sentence Grouping**: Cümleler etrafındaki bağlamla birlikte işleniyor
2. **Dynamic Threshold**: Percentile tabanlı dinamik eşik belirleme
3. **Minimum Chunk Size Enforcement**: Çok küçük chunk'ların birleştirilmesi
4. **Overlap Support**: Chunk'lar arası bağlam korunması
5. **Q&A Detection**: Soru-cevap çiftlerinin birlikte tutulması

### Zayıf Yönler ve İyileştirme Alanları

#### 1. Türkçe Dil Desteği Eksiklikleri

**Mevcut Durum:**
- NLTK'nın `sent_tokenize` fonksiyonu Türkçe için optimize edilmemiş
- Türkçe soru kalıpları sınırlı (`mi?`, `mı?`, `mu?`, `mü?` gibi)
- Türkçe'ye özgü cümle sonu işaretleri tam desteklenmiyor
- Türkçe karakterler (ç, ğ, ı, ö, ş, ü) için özel işlem yok

**Sorunlar:**
```python
# Mevcut kod:
question_words = [
    'ne ', 'nasıl', 'neden', 'niçin', 'kim', 'nerede', 
    'hangi', 'kaç', 'mi?', 'mı?', 'mu?', 'mü?',
    'misin', 'mısın', 'musun', 'müsün'
]
```
Bu liste eksik ve sınırlı. Türkçe'de çok daha fazla soru kalıbı var.

#### 2. Sentence Splitting Problemleri

**Mevcut Durum:**
```python
def _split_into_sentences(self, text: str) -> List[str]:
    # First try NLTK
    sentences = nltk.sent_tokenize(text)
    
    # If NLTK returns single chunk, try manual splitting
    if len(sentences) <= 1 and len(text) > 200:
        pattern = r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜa-zçğıöşü0-9#*])'
        sentences = re.split(pattern, text)
```

**Sorunlar:**
- Kısaltmalar (Dr., Prof., vb.) yanlış bölünebiliyor
- Sayısal ifadeler (1.5, 2.3) yanlış bölünebiliyor
- Türkçe özel durumlar (örn., vs., vb.) eksik
- Tırnak içi cümleler yanlış işlenebiliyor

#### 3. Embedding Model Bağımlılığı

**Mevcut Durum:**
- Sadece OpenRouter API kullanılıyor
- Tek bir embedding modeli: `openai/text-embedding-3-small`
- API anahtarı zorunlu
- Offline çalışma imkanı yok

**Sorunlar:**
- Türkçe için optimize edilmiş embedding modelleri kullanılamıyor
- Maliyet ve hız optimizasyonu yok
- Fallback mekanizması yok

#### 4. Performans ve Ölçeklenebilirlik

**Mevcut Durum:**
- Her cümle için embedding alınıyor
- Batch işleme sınırlı
- Cache mekanizması yok

**Sorunlar:**
- Uzun metinlerde yavaş
- API rate limit sorunları
- Gereksiz API çağrıları

#### 5. Chunk Quality Metrics

**Mevcut Durum:**
- Temel metrikler var (min, max, avg size)
- Semantic coherence ölçümü yok
- Chunk kalitesi için detaylı analiz yok

## Requirements

### Requirement 1: Gelişmiş Türkçe Dil Desteği

**User Story:** Bir kullanıcı olarak, Türkçe metinlerin doğru ve anlamsal olarak tutarlı şekilde bölünmesini istiyorum, böylece RAG sisteminde daha iyi sonuçlar alabilirim.

#### Acceptance Criteria

1. THE Semantic_Chunker SHALL support comprehensive Turkish sentence detection patterns including abbreviations (Dr., Prof., örn., vs., vb., etc.)
2. WHEN processing Turkish text, THE Semantic_Chunker SHALL correctly identify Turkish question patterns including all grammatical variations (mi/mı/mu/mü with all conjugations)
3. THE Semantic_Chunker SHALL preserve Turkish character encoding (ç, ğ, ı, ö, ş, ü, Ç, Ğ, İ, Ö, Ş, Ü) throughout the chunking process
4. WHEN encountering Turkish-specific punctuation patterns, THE Semantic_Chunker SHALL handle them correctly without splitting inappropriately
5. THE Semantic_Chunker SHALL detect and keep together Turkish Q&A pairs using comprehensive question word patterns

### Requirement 2: Çok Dilli Sentence Tokenization

**User Story:** Bir sistem yöneticisi olarak, hem Türkçe hem İngilizce metinleri aynı kalitede işleyebilmek istiyorum, böylece çok dilli içerikler için tek bir sistem kullanabilirim.

#### Acceptance Criteria

1. THE Semantic_Chunker SHALL automatically detect text language (Turkish, English, or mixed)
2. WHEN processing mixed-language text, THE Semantic_Chunker SHALL apply appropriate sentence splitting rules for each language segment
3. THE Semantic_Chunker SHALL use language-specific sentence tokenizers when available
4. WHEN language detection fails, THE Semantic_Chunker SHALL fallback to a universal sentence splitting approach
5. THE Semantic_Chunker SHALL maintain sentence boundary accuracy above 95% for both Turkish and English texts

### Requirement 3: Gelişmiş Sentence Splitting

**User Story:** Bir geliştirici olarak, kısaltmalar, sayılar ve özel karakterler içeren metinlerin doğru bölünmesini istiyorum, böylece chunk kalitesi artacak.

#### Acceptance Criteria

1. THE Semantic_Chunker SHALL maintain a comprehensive list of common abbreviations for Turkish and English
2. WHEN encountering abbreviations (Dr., Prof., vs., etc.), THE Semantic_Chunker SHALL NOT split sentences at these points
3. WHEN encountering decimal numbers (1.5, 2.3, etc.), THE Semantic_Chunker SHALL NOT split at decimal points
4. THE Semantic_Chunker SHALL correctly handle quoted text and NOT split sentences within quotes
5. WHEN encountering URLs or email addresses, THE Semantic_Chunker SHALL treat them as single units

### Requirement 4: Esnek Embedding Provider Desteği

**User Story:** Bir sistem yöneticisi olarak, farklı embedding sağlayıcıları kullanabilmek istiyorum, böylece maliyet, hız ve kalite arasında optimizasyon yapabilirim.

#### Acceptance Criteria

1. THE Semantic_Chunker SHALL support multiple embedding providers (OpenRouter, OpenAI, HuggingFace, local models)
2. WHEN an embedding provider fails, THE Semantic_Chunker SHALL automatically fallback to the next available provider
3. THE Semantic_Chunker SHALL allow configuration of provider priority and selection strategy
4. THE Semantic_Chunker SHALL support offline operation using local embedding models
5. WHEN using Turkish text, THE Semantic_Chunker SHALL prefer Turkish-optimized embedding models when available

### Requirement 5: Türkçe-Optimized Embedding Models

**User Story:** Bir veri bilimcisi olarak, Türkçe için optimize edilmiş embedding modellerini kullanabilmek istiyorum, böylece Türkçe metinler için daha iyi semantic similarity elde edebilirim.

#### Acceptance Criteria

1. THE Semantic_Chunker SHALL support Turkish-specific embedding models (e.g., dbmdz/bert-base-turkish-cased, loodos/bert-turkish-base)
2. WHEN processing Turkish text, THE Semantic_Chunker SHALL automatically select Turkish-optimized models if configured
3. THE Semantic_Chunker SHALL allow manual override of embedding model selection
4. THE Semantic_Chunker SHALL cache embedding model instances to avoid repeated loading
5. THE Semantic_Chunker SHALL provide performance metrics comparing different embedding models

### Requirement 6: Performans Optimizasyonu

**User Story:** Bir sistem yöneticisi olarak, büyük metinleri hızlı ve verimli şekilde işleyebilmek istiyorum, böylece sistem ölçeklenebilir olacak.

#### Acceptance Criteria

1. THE Semantic_Chunker SHALL implement batch embedding processing to reduce API calls
2. THE Semantic_Chunker SHALL cache embeddings for frequently processed text segments
3. WHEN processing large texts (>10000 characters), THE Semantic_Chunker SHALL use streaming or chunked processing
4. THE Semantic_Chunker SHALL implement rate limiting and retry logic for API calls
5. THE Semantic_Chunker SHALL complete processing within 5 seconds for texts up to 5000 characters

### Requirement 7: Gelişmiş Chunk Quality Metrics

**User Story:** Bir veri bilimcisi olarak, chunk kalitesini ölçebilmek istiyorum, böylece chunking stratejisini optimize edebilirim.

#### Acceptance Criteria

1. THE Semantic_Chunker SHALL calculate semantic coherence score for each chunk
2. THE Semantic_Chunker SHALL measure inter-chunk similarity to detect potential merge opportunities
3. THE Semantic_Chunker SHALL provide topic distribution analysis across chunks
4. THE Semantic_Chunker SHALL detect and report chunks with low semantic coherence
5. THE Semantic_Chunker SHALL generate quality reports with actionable recommendations

### Requirement 8: Adaptive Threshold Mechanism

**User Story:** Bir kullanıcı olarak, farklı metin türleri için otomatik olarak optimize edilmiş threshold değerleri kullanmak istiyorum, böylece manuel ayar yapmama gerek kalmayacak.

#### Acceptance Criteria

1. THE Semantic_Chunker SHALL analyze text characteristics (length, sentence count, vocabulary diversity) before chunking
2. WHEN text has high semantic diversity, THE Semantic_Chunker SHALL automatically lower the similarity threshold
3. WHEN text has low semantic diversity, THE Semantic_Chunker SHALL automatically raise the similarity threshold
4. THE Semantic_Chunker SHALL provide threshold recommendation based on text analysis
5. THE Semantic_Chunker SHALL allow manual override of automatic threshold selection

### Requirement 9: Gelişmiş Q&A Detection

**User Story:** Bir kullanıcı olarak, soru-cevap çiftlerinin her zaman birlikte tutulmasını istiyorum, böylece RAG sisteminde bağlam kaybı olmayacak.

#### Acceptance Criteria

1. THE Semantic_Chunker SHALL detect Turkish questions using comprehensive pattern matching (all question words and grammatical forms)
2. WHEN a question is detected, THE Semantic_Chunker SHALL analyze the next 1-3 sentences to identify the answer
3. THE Semantic_Chunker SHALL use semantic similarity to confirm Q&A pairing
4. WHEN Q&A pair is confirmed, THE Semantic_Chunker SHALL keep them in the same chunk
5. THE Semantic_Chunker SHALL handle nested questions (questions within answers) correctly

### Requirement 10: Configurable Preprocessing

**User Story:** Bir geliştirici olarak, metin ön işleme adımlarını yapılandırabilmek istiyorum, böylece farklı metin türleri için optimize edebilirim.

#### Acceptance Criteria

1. THE Semantic_Chunker SHALL allow configuration of sentence merging rules (minimum length, maximum length)
2. THE Semantic_Chunker SHALL support custom preprocessing functions for text normalization
3. THE Semantic_Chunker SHALL allow disabling specific preprocessing steps
4. THE Semantic_Chunker SHALL provide preprocessing diagnostics showing what transformations were applied
5. THE Semantic_Chunker SHALL validate preprocessing configuration before chunking

### Requirement 11: Error Handling ve Fallback

**User Story:** Bir sistem yöneticisi olarak, embedding API'sinin başarısız olması durumunda sistemin çalışmaya devam etmesini istiyorum, böylece kullanıcı deneyimi kesintiye uğramayacak.

#### Acceptance Criteria

1. WHEN embedding API fails, THE Semantic_Chunker SHALL fallback to sentence-based chunking
2. THE Semantic_Chunker SHALL log all errors with detailed context for debugging
3. WHEN fallback is triggered, THE Semantic_Chunker SHALL notify the user with a warning message
4. THE Semantic_Chunker SHALL implement exponential backoff for API retries
5. THE Semantic_Chunker SHALL provide health check endpoint for monitoring embedding service availability

### Requirement 12: Comprehensive Testing

**User Story:** Bir geliştirici olarak, semantic chunker'ın Türkçe ve İngilizce metinler için doğru çalıştığından emin olmak istiyorum, böylece production'da sorun yaşamayacağım.

#### Acceptance Criteria

1. THE Semantic_Chunker SHALL have unit tests covering all Turkish-specific functionality
2. THE Semantic_Chunker SHALL have integration tests with real Turkish and English texts
3. THE Semantic_Chunker SHALL have property-based tests for sentence splitting accuracy
4. THE Semantic_Chunker SHALL have performance benchmarks for different text sizes
5. THE Semantic_Chunker SHALL have tests comparing chunk quality across different configurations

## Non-Functional Requirements

### Performance

- Semantic chunking SHALL complete within 5 seconds for texts up to 5000 characters
- Embedding cache hit rate SHALL be above 70% for repeated content
- API retry logic SHALL not exceed 3 attempts per request

### Reliability

- System SHALL maintain 99.9% uptime with fallback mechanisms
- Fallback to sentence-based chunking SHALL occur within 2 seconds of embedding failure

### Scalability

- System SHALL handle concurrent requests from multiple users
- Batch processing SHALL support up to 100 texts simultaneously

### Maintainability

- Code SHALL follow PEP 8 style guidelines
- All functions SHALL have comprehensive docstrings
- Configuration SHALL be externalized and environment-based

### Security

- API keys SHALL be stored securely in environment variables
- Embedding cache SHALL not store sensitive information
- Rate limiting SHALL prevent API abuse

## Success Metrics

1. **Türkçe Sentence Splitting Accuracy**: >95%
2. **Chunk Semantic Coherence Score**: >0.8
3. **Processing Speed**: <5s for 5000 characters
4. **API Cost Reduction**: >30% through caching and batching
5. **User Satisfaction**: >4.5/5 in feedback surveys
