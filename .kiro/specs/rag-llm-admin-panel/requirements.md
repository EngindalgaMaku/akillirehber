# Requirements Document

## Introduction

Lise düzeyinde eğitim ortamlarında kullanılmak üzere RAG (Retrieval-Augmented Generation) tabanlı bir chatbot sistemi. Öğretmenler ders materyalleri yükleyebilir, öğrenciler bu materyaller üzerinden sorular sorabilir. Sistem Docker konteynerlerinde çalışacak, PostgreSQL ve Weaviate veritabanları kullanacak.

## Glossary

- **RAG_Chatbot**: Retrieval-Augmented Generation tabanlı soru-cevap sistemi
- **Teacher_User**: Ders oluşturabilen ve döküman yükleyebilen kullanıcı rolü
- **Student_User**: Derslere erişip soru sorabilen kullanıcı rolü
- **Course**: Öğretmen tarafından oluşturulan, dökümanlar içeren ders birimi
- **Document**: Derse yüklenen PDF, Markdown veya Word dosyası
- **Chunk**: Dökümanın işlenmiş metin parçası
- **Embedding**: Metin parçasının vektör temsili
- **Weaviate_DB**: Vektör veritabanı servisi
- **PostgreSQL_DB**: İlişkisel veritabanı servisi
- **FastAPI_Backend**: Python tabanlı REST API servisi
- **Next.js_Frontend**: React tabanlı modern web arayüzü

## Requirements

### Requirement 1: Docker Altyapısı

**User Story:** As a developer, I want all services to run in Docker containers, so that the system is portable and easy to deploy.

#### Acceptance Criteria

1. THE System SHALL have a docker-compose.yml file orchestrating all services
2. THE Backend SHALL run in a Docker container with Python 3.12+
3. THE Frontend SHALL run in a Docker container with Node.js 20+
4. THE PostgreSQL_DB SHALL run in a Docker container with version 16+
5. THE Weaviate_DB SHALL run in a Docker container with latest stable version
6. WHEN docker-compose up is executed, THE System SHALL start all services with proper networking
7. THE System SHALL use environment variables for configuration

### Requirement 2: Kullanıcı Yönetimi

**User Story:** As a user, I want to authenticate with my role, so that I can access appropriate features.

#### Acceptance Criteria

1. THE System SHALL support two user roles: Teacher_User and Student_User
2. WHEN a Teacher_User logs in, THE System SHALL display course management interface
3. WHEN a Student_User logs in, THE System SHALL display available courses and chat interface
4. THE System SHALL store user credentials securely in PostgreSQL_DB
5. THE System SHALL use JWT tokens for authentication
6. IF invalid credentials are provided, THEN THE System SHALL return an authentication error

### Requirement 3: Ders Yönetimi

**User Story:** As a teacher, I want to create and manage courses, so that I can organize educational content.

#### Acceptance Criteria

1. WHEN a Teacher_User creates a course, THE System SHALL store course metadata in PostgreSQL_DB
2. THE Course SHALL have a name, description, and creation date
3. WHEN a Teacher_User views courses, THE System SHALL list all courses created by that teacher
4. WHEN a Student_User views courses, THE System SHALL list all available courses
5. THE System SHALL allow Teacher_User to edit and delete their courses
6. IF a course is deleted, THEN THE System SHALL remove all associated documents and embeddings

### Requirement 4: Döküman Yükleme

**User Story:** As a teacher, I want to upload documents to courses, so that students can learn from them.

#### Acceptance Criteria

1. THE System SHALL support document upload via drag-and-drop or file picker
2. THE System SHALL support the following file formats: PDF, Markdown (.md), Word (.docx)
3. WHEN a document is uploaded, THE System SHALL extract text content
4. WHEN a document is uploaded, THE System SHALL store document metadata in PostgreSQL_DB
5. IF an unsupported file format is uploaded, THEN THE System SHALL display an error message
6. WHEN text is extracted, THE System SHALL display document metadata (file name, size, character count)

### Requirement 5: Metin Parçalama (Chunking)

**User Story:** As a teacher, I want uploaded documents to be chunked, so that they can be processed for RAG.

#### Acceptance Criteria

1. WHEN a document is processed, THE System SHALL chunk the text using configurable strategies
2. THE System SHALL support the following chunking strategies:
   - **Fixed-size**: Character-based chunking with configurable chunk size and overlap
   - **Recursive**: Hierarchical splitting using separators (paragraphs → sentences → words)
   - **Sentence-based**: Split by sentence boundaries using NLP
3. WHEN chunking is performed, THE System SHALL display chunk statistics (count, average size, min/max size)
4. THE System SHALL store chunks in PostgreSQL_DB with references to source document
5. WHEN a user adjusts chunk parameters, THE System SHALL re-process and update results
6. IF the input text is empty, THEN THE System SHALL display a validation message

### Requirement 6: Chunk Görselleştirme

**User Story:** As a teacher, I want to visualize chunking results, so that I can verify document processing.

#### Acceptance Criteria

1. WHEN chunks are generated, THE System SHALL display each chunk as a separate card with index number
2. WHEN a user hovers over a chunk card, THE System SHALL highlight the corresponding text
3. THE System SHALL display character count for each individual chunk
4. WHEN chunks have overlap, THE System SHALL visually indicate overlapping portions

### Requirement 7: API Entegrasyonu

**User Story:** As a developer, I want clean APIs for all operations, so that frontend and backend communicate efficiently.

#### Acceptance Criteria

1. THE FastAPI_Backend SHALL expose RESTful endpoints for all operations
2. THE FastAPI_Backend SHALL use Pydantic models for request/response validation
3. THE FastAPI_Backend SHALL return appropriate HTTP status codes
4. THE FastAPI_Backend SHALL include OpenAPI documentation
5. IF an error occurs, THEN THE FastAPI_Backend SHALL return descriptive error messages

### Requirement 8: Veritabanı Entegrasyonu

**User Story:** As a developer, I want proper database integration, so that data is persisted reliably.

#### Acceptance Criteria

1. THE System SHALL use PostgreSQL_DB for relational data (users, courses, documents, chunks)
2. THE System SHALL use Weaviate_DB for vector embeddings (future phase)
3. THE System SHALL use SQLAlchemy ORM for PostgreSQL operations
4. THE System SHALL implement database migrations using Alembic
5. WHEN the system starts, THE System SHALL verify database connections

