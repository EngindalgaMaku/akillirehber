# Design Document

## Overview

Bu tasarım, AkıllıRehber uygulamasında ders bazlı doküman işleme pipeline'ını ve Weaviate entegrasyonunu tanımlar. Sistem, öğretmenlerin dokümanları chunk'lamasına, embedding oluşturmasına ve öğrencilerin bu içeriklerle RAG tabanlı sohbet etmesine olanak tanır.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                        │
│  ┌──────────┬──────────┬──────────┬──────────┐                  │
│  │Dokümanlar│  İşleme  │  Sohbet  │  Ayarlar │                  │
│  └──────────┴──────────┴──────────┴──────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Document   │  │  Chunker    │  │  Embedding  │              │
│  │  Service    │  │  Service    │  │  Service    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Weaviate   │  │    Chat     │  │   Course    │              │
│  │  Service    │  │  Service    │  │  Service    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
          │                   │                    │
          ▼                   ▼                    ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │PostgreSQL│        │ Weaviate │        │OpenRouter│
    │(metadata)│        │(vectors) │        │  (LLM)   │
    └──────────┘        └──────────┘        └──────────┘
```

## Components and Interfaces

### 1. Frontend Components

#### CourseDetailPage
Ana ders detay sayfası, tab yapısını yönetir.

```typescript
interface CourseDetailPageProps {
  courseId: number;
}

type TabType = "documents" | "processing" | "chat" | "settings";
```

#### DocumentsTab
Doküman yükleme, listeleme ve silme işlemleri.

```typescript
interface DocumentWithStatus {
  id: number;
  filename: string;
  file_size: number;
  uploaded_at: string;
  chunk_count: number;
  embedding_status: "pending" | "processing" | "completed" | "error";
}
```

#### ProcessingTab
Chunking ve Embedding işlemleri için iki bölümlü arayüz.

```typescript
interface ChunkingOptions {
  strategy: "recursive" | "semantic";
  chunk_size?: number;
  overlap?: number;
  similarity_threshold?: number;
  min_chunk_size?: number;
  max_chunk_size?: number;
  embedding_model?: string;
}

interface EmbeddingOptions {
  model: string;
}
```

#### ChatTab
RAG tabanlı sohbet arayüzü.

```typescript
interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: ChunkReference[];
}

interface ChunkReference {
  document_name: string;
  chunk_index: number;
  content_preview: string;
  score: number;
}
```

### 2. Backend Services

#### WeaviateService
Weaviate ile tüm etkileşimleri yönetir.

```python
class WeaviateService:
    def __init__(self, url: str):
        """Weaviate bağlantısını başlatır."""
    
    def ensure_collection(self, course_id: int) -> str:
        """Ders için collection oluşturur veya var olanı döner."""
    
    def store_chunks(
        self, 
        course_id: int, 
        document_id: int, 
        chunks: List[ChunkWithEmbedding]
    ) -> List[str]:
        """Chunk'ları vektörleriyle birlikte Weaviate'e kaydeder."""
    
    def delete_by_document(self, course_id: int, document_id: int) -> int:
        """Bir dokümana ait tüm vektörleri siler."""
    
    def delete_by_course(self, course_id: int) -> int:
        """Bir derse ait tüm vektörleri siler."""
    
    def hybrid_search(
        self,
        course_id: int,
        query: str,
        query_vector: List[float],
        alpha: float = 0.5,
        limit: int = 5
    ) -> List[SearchResult]:
        """Hybrid search yapar (vector + keyword)."""
    
    def vector_search(
        self,
        course_id: int,
        query_vector: List[float],
        limit: int = 5
    ) -> List[SearchResult]:
        """Sadece vektör araması yapar."""
    
    def keyword_search(
        self,
        course_id: int,
        query: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """Sadece BM25 kelime araması yapar."""
```

#### EmbeddingService
OpenRouter API ile embedding oluşturur.

```python
class EmbeddingService:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openai/text-embedding-3-small"
    
    def get_embeddings(
        self, 
        texts: List[str], 
        model: str = None
    ) -> List[List[float]]:
        """Metinler için embedding vektörleri oluşturur."""
    
    def get_embedding(self, text: str, model: str = None) -> List[float]:
        """Tek metin için embedding oluşturur."""
```

#### ChatService
RAG tabanlı sohbet işlemlerini yönetir.

```python
class ChatService:
    def __init__(
        self, 
        weaviate_service: WeaviateService,
        embedding_service: EmbeddingService
    ):
        """Servisleri başlatır."""
    
    def chat(
        self,
        course_id: int,
        message: str,
        chat_history: List[ChatMessage],
        search_type: str = "hybrid",
        alpha: float = 0.5,
        top_k: int = 5
    ) -> ChatResponse:
        """RAG tabanlı sohbet yanıtı üretir."""
```

## Data Models

### PostgreSQL Models

#### Course Settings (Yeni)
```python
class CourseSettings(Base):
    __tablename__ = "course_settings"
    
    id: int
    course_id: int  # FK to courses
    default_chunk_strategy: str = "recursive"
    default_embedding_model: str = "openai/text-embedding-3-small"
    search_alpha: float = 0.5  # 0=keyword, 1=vector
    search_top_k: int = 5
    created_at: datetime
    updated_at: datetime
```

#### Document (Güncelleme)
```python
class Document(Base):
    # Mevcut alanlar...
    embedding_status: str = "pending"  # pending, processing, completed, error
    embedding_model: str = None
    embedded_at: datetime = None
    vector_count: int = 0
```

### Weaviate Schema

Her ders için ayrı collection:
```
Collection: Course_{course_id}
Properties:
  - chunk_id: int (PostgreSQL chunk ID)
  - document_id: int
  - content: text (BM25 için indexlenir)
  - chunk_index: int
Vector: embedding (model'e göre boyut değişir)
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system.*


### Property 1: Document Upload Persistence
*For any* valid document file uploaded by a teacher, the document SHALL appear in the document list with correct metadata (filename, size, upload date).
**Validates: Requirements 2.1**

### Property 2: Cascade Deletion
*For any* document that is deleted, all associated chunks in PostgreSQL and all vectors in Weaviate SHALL be removed.
**Validates: Requirements 2.3**

### Property 3: Chunking Persistence
*For any* document that is chunked, the resulting chunks SHALL be saved to PostgreSQL and the chunk count SHALL match the number of chunks created.
**Validates: Requirements 3.5**

### Property 4: Chunk Clearing
*For any* document with chunks, after clearing, the chunk count SHALL be zero and no chunks SHALL exist in PostgreSQL for that document.
**Validates: Requirements 3.6**

### Property 5: Embedding Enablement
*For any* document, the embedding section SHALL be enabled if and only if the document has at least one chunk.
**Validates: Requirements 4.1**

### Property 6: Embedding Storage
*For any* set of chunks that are embedded, the vectors SHALL be stored in Weaviate with both vector data and original text content.
**Validates: Requirements 4.3, 4.4**

### Property 7: Vector Clearing
*For any* document with vectors in Weaviate, after clearing vectors, no vectors SHALL exist in Weaviate for that document.
**Validates: Requirements 4.5**

### Property 8: Collection Idempotence
*For any* course, calling ensure_collection multiple times SHALL always succeed and return the same collection name.
**Validates: Requirements 5.2**

### Property 9: Vector Search Retrieval
*For any* stored chunk with embedding, a vector search with the same embedding SHALL return that chunk with high similarity score.
**Validates: Requirements 5.4**

### Property 10: Keyword Search Retrieval
*For any* stored chunk with text content, a keyword search with words from that content SHALL return that chunk.
**Validates: Requirements 5.5**

### Property 11: Hybrid Search Combination
*For any* query, hybrid search results SHALL be influenced by both vector similarity and keyword matching based on alpha parameter.
**Validates: Requirements 5.6**

### Property 12: Chat History Accumulation
*For any* chat session, each new message SHALL be added to the history and the history length SHALL increase by one.
**Validates: Requirements 6.6**

### Property 13: Settings Round-Trip
*For any* course settings that are saved, loading the settings SHALL return the same values that were saved.
**Validates: Requirements 7.5**

## Error Handling

### Weaviate Connection Errors
- Retry with exponential backoff (3 attempts)
- Log error and return user-friendly message
- Mark embedding status as "error"

### OpenRouter API Errors
- Handle rate limiting with retry
- Handle invalid API key with clear error message
- Handle model not found errors

### File Processing Errors
- Validate file type before processing
- Handle corrupted files gracefully
- Limit file size (configurable, default 10MB)

## Testing Strategy

### Unit Tests
- Test each service method in isolation
- Mock external dependencies (Weaviate, OpenRouter)
- Test error handling paths

### Property-Based Tests
- Use Hypothesis for Python property tests
- Test data persistence properties
- Test search result properties

### Integration Tests
- Test full pipeline: upload → chunk → embed → search
- Test with real Weaviate instance (Docker)
- Test chat flow end-to-end
