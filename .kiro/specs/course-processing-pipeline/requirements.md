# Requirements Document

## Introduction

Bu özellik, AkıllıRehber uygulamasında ders bazlı doküman işleme pipeline'ını ve Weaviate entegrasyonunu kapsar. Öğretmenler derslerine doküman yükleyebilir, chunk'layabilir, embedding oluşturabilir ve öğrenciler bu içeriklerle sohbet edebilir.

## Glossary

- **Course_Page**: Ders detay sayfası, 4 sekmeden oluşur
- **Document**: Derse yüklenen PDF, MD, DOCX veya TXT dosyası
- **Chunk**: Dokümanın parçalanmış metin bölümü
- **Embedding**: Chunk'ın vektör temsili
- **Weaviate_Service**: Weaviate veritabanına bağlanan Python servisi
- **Hybrid_Search**: Vektör ve kelime bazlı aramanın kombinasyonu
- **Processing_Pipeline**: Chunking → Embedding → Weaviate kayıt akışı

## Requirements

### Requirement 1: Ders Detay Sayfası Yapısı

**User Story:** As a teacher, I want to see a well-organized course detail page with tabs, so that I can manage documents, processing, chat and settings separately.

#### Acceptance Criteria

1. WHEN a user navigates to a course detail page, THE Course_Page SHALL display 4 tabs: Dokümanlar, İşleme, Sohbet, Ayarlar
2. WHEN a user clicks on a tab, THE Course_Page SHALL show the corresponding content without page reload
3. THE Course_Page SHALL remember the last active tab during the session
4. WHEN the page loads, THE Course_Page SHALL default to the Dokümanlar tab

### Requirement 2: Dokümanlar Sekmesi

**User Story:** As a teacher, I want to upload, view and delete documents in my course, so that I can manage course materials.

#### Acceptance Criteria

1. WHEN a teacher uploads a document, THE System SHALL save it to PostgreSQL and display it in the list
2. WHEN viewing documents, THE System SHALL show filename, size, upload date, chunk count and embedding status
3. WHEN a teacher deletes a document, THE System SHALL remove it from PostgreSQL, its chunks, and Weaviate vectors
4. THE System SHALL support PDF, MD, DOCX and TXT file formats
5. WHEN a student views the documents tab, THE System SHALL show documents in read-only mode without upload/delete options

### Requirement 3: İşleme Sekmesi - Chunking Aşaması

**User Story:** As a teacher, I want to chunk documents with different strategies, so that I can prepare them for embedding.

#### Acceptance Criteria

1. WHEN a teacher selects a document for chunking, THE System SHALL show chunking options (strategy, parameters)
2. THE System SHALL support only Recursive and Semantic chunking strategies
3. WHEN Recursive strategy is selected, THE System SHALL show chunk_size and overlap parameters
4. WHEN Semantic strategy is selected, THE System SHALL show similarity_threshold, min_chunk_size, max_chunk_size, overlap and embedding_model parameters
5. WHEN chunking is executed, THE System SHALL save chunks to PostgreSQL and display them with pagination
6. WHEN a teacher clicks "Temizle", THE System SHALL delete all chunks for that document
7. THE System SHALL show chunk content fully expandable, not truncated

### Requirement 4: İşleme Sekmesi - Embedding Aşaması

**User Story:** As a teacher, I want to create embeddings for chunks and store them in Weaviate, so that they can be searched semantically.

#### Acceptance Criteria

1. WHEN a document has chunks, THE System SHALL enable the embedding section
2. WHEN a teacher selects embedding options, THE System SHALL show model selection (OpenRouter models)
3. WHEN embedding is executed, THE System SHALL generate vectors using OpenRouter API and store them in Weaviate
4. THE Weaviate_Service SHALL store both vector embeddings and original text for hybrid search
5. WHEN a teacher clicks "Vektörleri Temizle", THE System SHALL delete all vectors for that document from Weaviate
6. THE System SHALL show embedding status (pending, processing, completed, error) for each document
7. WHEN embedding fails, THE System SHALL show error message and allow retry

### Requirement 5: Weaviate Service

**User Story:** As a system, I want to connect to Weaviate and perform CRUD operations, so that vectors can be stored and searched.

#### Acceptance Criteria

1. THE Weaviate_Service SHALL connect to Weaviate using the configured URL
2. THE Weaviate_Service SHALL create a collection for each course if not exists
3. WHEN storing vectors, THE Weaviate_Service SHALL include chunk_id, document_id, course_id, content and vector
4. THE Weaviate_Service SHALL support vector search (semantic)
5. THE Weaviate_Service SHALL support keyword search (BM25)
6. THE Weaviate_Service SHALL support hybrid search combining vector and keyword
7. WHEN deleting, THE Weaviate_Service SHALL remove vectors by document_id or course_id

### Requirement 6: Sohbet Sekmesi

**User Story:** As a student, I want to chat with course materials using RAG, so that I can learn from the documents.

#### Acceptance Criteria

1. WHEN a user sends a message, THE System SHALL perform hybrid search on Weaviate
2. THE System SHALL retrieve top-k relevant chunks based on the query
3. THE System SHALL send retrieved chunks as context to LLM (OpenRouter)
4. THE System SHALL display LLM response with source references
5. WHEN no relevant chunks are found, THE System SHALL inform the user
6. THE System SHALL maintain chat history during the session
7. THE System SHALL allow clearing chat history

### Requirement 7: Ayarlar Sekmesi

**User Story:** As a teacher, I want to configure course-specific settings, so that I can customize the RAG behavior.

#### Acceptance Criteria

1. THE System SHALL allow setting default chunking strategy for the course
2. THE System SHALL allow setting default embedding model for the course
3. THE System SHALL allow setting hybrid search alpha parameter (0=keyword, 1=vector)
4. THE System SHALL allow setting top-k retrieval count
5. WHEN settings are saved, THE System SHALL persist them to the database
