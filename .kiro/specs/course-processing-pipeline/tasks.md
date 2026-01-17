# Implementation Plan: Course Processing Pipeline

## Overview

Bu plan, ders bazlı doküman işleme pipeline'ını ve Weaviate entegrasyonunu adım adım implement eder. Backend servisleri önce, ardından frontend bileşenleri geliştirilecek.

## Tasks

- [x] 1. Weaviate Service Oluşturma
  - [x] 1.1 WeaviateService sınıfını oluştur
    - Weaviate client bağlantısı
    - ensure_collection metodu (ders için collection oluşturma)
    - _Requirements: 5.1, 5.2_
  - [x] 1.2 CRUD metodlarını implement et
    - store_chunks: Chunk'ları vektörleriyle kaydet
    - delete_by_document: Dokümana ait vektörleri sil
    - delete_by_course: Derse ait tüm vektörleri sil
    - get_document_vectors: Dokümana ait vektör sayısını getir
    - _Requirements: 5.3, 5.7_
  - [x] 1.3 Search metodlarını implement et
    - vector_search: Sadece vektör araması
    - keyword_search: BM25 kelime araması
    - hybrid_search: Kombine arama
    - _Requirements: 5.4, 5.5, 5.6_

- [x] 2. Embedding Service Oluşturma
  - [x] 2.1 EmbeddingService sınıfını oluştur
    - OpenRouter API bağlantısı
    - get_embeddings: Çoklu metin için embedding
    - get_embedding: Tek metin için embedding
    - _Requirements: 4.3_
  - [x] 2.2 Batch embedding desteği ekle
    - Büyük chunk listelerini batch'lere böl
    - Rate limiting ve retry mekanizması
    - _Requirements: 4.3, 4.7_

- [x] 3. Database Model Güncellemeleri
  - [x] 3.1 Document modeline yeni alanlar ekle
    - embedding_status: pending/processing/completed/error
    - embedding_model: Kullanılan model
    - embedded_at: Embedding tarihi
    - vector_count: Weaviate'deki vektör sayısı
    - _Requirements: 4.6_
  - [x] 3.2 CourseSettings modelini oluştur
    - default_chunk_strategy
    - default_embedding_model
    - search_alpha
    - search_top_k
    - Alembic migration
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 4. Backend API Endpoints
  - [x] 4.1 Embedding endpoints oluştur
    - POST /api/documents/{id}/embed - Embedding başlat
    - DELETE /api/documents/{id}/vectors - Vektörleri sil
    - GET /api/documents/{id}/vectors/count - Vektör sayısı
    - _Requirements: 4.3, 4.5, 4.6_
  - [x] 4.2 Course settings endpoints oluştur
    - GET /api/courses/{id}/settings - Ayarları getir
    - PUT /api/courses/{id}/settings - Ayarları güncelle
    - _Requirements: 7.5_
  - [x] 4.3 Chat endpoint oluştur
    - POST /api/courses/{id}/chat - RAG sohbet
    - _Requirements: 6.1, 6.2, 6.3_

- [ ] 5. Checkpoint - Backend Testleri
  - Weaviate bağlantısını test et
  - Embedding oluşturmayı test et
  - Search fonksiyonlarını test et
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Frontend - Tab Yapısı
  - [x] 6.1 CourseDetailPage'i tab yapısına dönüştür
    - 4 tab: Dokümanlar, İşleme, Sohbet, Ayarlar
    - Tab state yönetimi
    - URL'de tab parametresi
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  - [x] 6.2 DocumentsTab bileşenini oluştur
    - Doküman listesi (filename, size, date, chunk_count, embedding_status)
    - Yükleme butonu (sadece teacher)
    - Silme butonu (sadece teacher)
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

- [x] 7. Frontend - İşleme Sekmesi
  - [x] 7.1 ProcessingTab bileşenini oluştur
    - Doküman seçimi
    - İki bölüm: Chunking ve Embedding
    - _Requirements: 3.1_
  - [x] 7.2 Chunking bölümünü implement et
    - Strateji seçimi (Recursive, Semantic)
    - Strateji bazlı parametre alanları
    - Chunk'la butonu
    - Chunk listesi (sayfalama, expand/collapse)
    - Temizle butonu
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_
  - [x] 7.3 Embedding bölümünü implement et
    - Model seçimi
    - Embed et butonu (chunk varsa aktif)
    - İlerleme göstergesi
    - Vektörleri temizle butonu
    - _Requirements: 4.1, 4.2, 4.3, 4.5, 4.6, 4.7_

- [x] 8. Frontend - Sohbet Sekmesi
  - [x] 8.1 ChatTab bileşenini oluştur
    - Mesaj listesi
    - Input alanı
    - Gönder butonu
    - _Requirements: 6.4_
  - [x] 8.2 RAG entegrasyonunu implement et
    - Mesaj gönderme
    - Kaynak referansları gösterme
    - Boş sonuç mesajı
    - Geçmişi temizle butonu
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 9. Frontend - Ayarlar Sekmesi
  - [x] 9.1 SettingsTab bileşenini oluştur
    - Default chunking strategy
    - Default embedding model
    - Search alpha (slider 0-1)
    - Top-k retrieval count
    - Kaydet butonu
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 10. Final Checkpoint
  - Tüm sekmeleri test et
  - Full pipeline test: upload → chunk → embed → chat
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Weaviate v4 Python client kullanılacak (weaviate-client>=4.0.0)
- OpenRouter API key .env'de OPENROUTER_API_KEY olarak tanımlı
- Her ders için ayrı Weaviate collection: Course_{course_id}
- Embedding model default: openai/text-embedding-3-small
