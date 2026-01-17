# Implementation Plan: RAG-Based Educational Chatbot System

## Overview

Bu plan, lise düzeyinde eğitim ortamlarında kullanılmak üzere RAG tabanlı chatbot sisteminin Docker altyapısı ve chunking modülünü implement eder.

## Tasks

- [x] 1. Backend Proje Yapısı ve Temel Kurulum
  - FastAPI projesi oluştur (Python 3.12+)
  - `backend/` klasör yapısını kur
  - `requirements.txt` ve `pyproject.toml` dosyalarını oluştur
  - Temel FastAPI app ve CORS ayarları
  - _Requirements: 1.2, 7.1_

- [x] 2. Chunking Data Models
  - [x] 2.1 Pydantic modelleri oluştur (`backend/app/models/chunking.py`)
    - ChunkingStrategy enum (fixed_size, recursive, sentence)
    - ChunkRequest, Chunk, ChunkStats, ChunkResponse modelleri
    - Validation rules (overlap < chunk_size)
    - _Requirements: 5.1, 5.5_

- [x] 3. Basic Chunking Strategies
  - [x] 3.1 FixedSizeChunker implementasyonu
  - [x] 3.2 RecursiveChunker implementasyonu
  - [x] 3.3 SentenceChunker implementasyonu
  - _Requirements: 5.2_

- [x] 4. Advanced Chunking Strategies
  - [x] 4.1 SemanticChunker implementasyonu
  - [x] 4.2 LateChunker implementasyonu
  - [x] 4.3 AgenticChunker implementasyonu
  - _Requirements: 5.2_

- [x] 5. Chunking API Endpoint
  - [x] 5.1 POST /api/chunk endpoint oluştur
  - [x] 5.2 Property test: Statistics accuracy
  - [x] 5.3 Property test: Character count accuracy
  - [x] 5.4 Property test: Error response consistency
  - _Requirements: 5.3, 7.1_

- [x] 6. Document Processing API
  - [x] 6.1 POST /api/upload endpoint oluştur
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 7. Checkpoint - Backend Tests
  - Tüm backend testlerini çalıştır
  - API endpoint'lerini manuel test et

- [x] 8. Docker Altyapısı - Backend
  - [x] 8.1 Backend Dockerfile oluştur
    - Python 3.12 base image
    - Multi-stage build for optimization
    - Non-root user for security
    - _Requirements: 1.2_

  - [x] 8.2 Backend için .dockerignore oluştur
    - __pycache__, .pytest_cache, .hypothesis
    - Virtual environment folders
    - _Requirements: 1.2_

- [x] 9. Docker Altyapısı - Veritabanları
  - [x] 9.1 PostgreSQL Docker yapılandırması
    - PostgreSQL 16 Alpine image
    - Volume for data persistence
    - Environment variables
    - _Requirements: 1.4, 8.1_

  - [x] 9.2 Weaviate Docker yapılandırması
    - Weaviate 1.27.x image
    - Volume for data persistence
    - Environment variables
    - _Requirements: 1.5, 8.2_

- [x] 10. Docker Compose Orchestration
  - [x] 10.1 docker-compose.yml oluştur
    - Backend, PostgreSQL, Weaviate servisleri
    - Network configuration
    - Volume definitions
    - Health checks
    - _Requirements: 1.1, 1.6_

  - [x] 10.2 .env.example dosyası oluştur
    - Database credentials
    - Secret keys
    - Service URLs
    - _Requirements: 1.7_

- [x] 11. Backend Database Integration
  - [x] 11.1 SQLAlchemy database setup
    - Database connection configuration
    - Session management
    - _Requirements: 8.3_

  - [x] 11.2 User model oluştur
    - UserRole enum (teacher, student)
    - User SQLAlchemy model
    - _Requirements: 2.1, 2.4_

  - [x] 11.3 Course model oluştur
    - Course SQLAlchemy model
    - Teacher relationship
    - _Requirements: 3.1, 3.2_

  - [x] 11.4 Document model oluştur
    - Document SQLAlchemy model
    - Course relationship
    - _Requirements: 4.4_

  - [x] 11.5 Chunk model oluştur
    - Chunk SQLAlchemy model
    - Document relationship
    - _Requirements: 5.4_

- [x] 12. Alembic Migrations
  - [x] 12.1 Alembic kurulumu ve yapılandırması
    - alembic.ini configuration
    - Migration environment setup
    - _Requirements: 8.4_

  - [x] 12.2 Initial migration oluştur
    - Users, Courses, Documents, Chunks tables
    - _Requirements: 8.4_

- [x] 13. Authentication System
  - [x] 13.1 Auth service oluştur
    - Password hashing (bcrypt)
    - JWT token generation/validation
    - _Requirements: 2.5_

  - [x] 13.2 Auth API endpoints
    - POST /api/auth/register
    - POST /api/auth/login
    - GET /api/auth/me
    - _Requirements: 2.1, 2.5, 2.6_

  - [ ]* 13.3 Property test: Authentication consistency
    - **Property 6: Authentication Consistency**
    - **Validates: Requirements 2.5, 2.6**

- [x] 14. Course Management API
  - [x] 14.1 Course service oluştur
    - CRUD operations
    - Teacher authorization
    - _Requirements: 3.1, 3.3, 3.4, 3.5_

  - [x] 14.2 Course API endpoints
    - GET/POST /api/courses
    - GET/PUT/DELETE /api/courses/{id}
    - _Requirements: 3.1, 3.3, 3.4, 3.5_

  - [ ]* 14.3 Property test: Role-based access
    - **Property 7: Role-Based Access**
    - **Validates: Requirements 2.2, 2.3, 3.5**

- [x] 15. Document Management API
  - [x] 15.1 Document service güncelle
    - Database integration
    - Course association
    - _Requirements: 4.4_

  - [x] 15.2 Document API endpoints güncelle
    - GET/POST /api/courses/{id}/documents
    - DELETE /api/documents/{id}
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [x] 15.3 Document processing with chunking
    - POST /api/documents/{id}/process
    - Chunk storage in database
    - _Requirements: 5.1, 5.4_

- [x] 16. Checkpoint - Backend with Database
  - Docker compose ile backend ve veritabanlarını başlat
  - Database migration'ları çalıştır
  - API endpoint'lerini test et
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Frontend Proje Yapısı
  - [x] 17.1 Next.js 15.x projesi oluştur
    - App Router structure
    - TypeScript configuration
    - _Requirements: 1.3_

  - [x] 17.2 Frontend Dockerfile oluştur
    - Node.js 20 base image
    - Multi-stage build
    - _Requirements: 1.3_

  - [x] 17.3 Tailwind CSS ve shadcn/ui kurulumu
    - Modern UI components
    - _Requirements: 1.3_

- [x] 18. Frontend Authentication
  - [x] 18.1 Auth context ve provider
    - JWT token management
    - User state management
    - _Requirements: 2.1, 2.5_

  - [x] 18.2 Login page
    - Email/password form
    - Role-based redirect
    - _Requirements: 2.2, 2.3_

- [x] 19. Frontend Dashboard
  - [x] 19.1 Teacher dashboard
    - Course list
    - Create course button
    - _Requirements: 2.2, 3.3_

  - [x] 19.2 Student dashboard
    - Available courses list
    - _Requirements: 2.3, 3.4_

- [x] 20. Frontend Course Management
  - [x] 20.1 Course list component
    - Course cards
    - Edit/delete actions (teacher)
    - _Requirements: 3.3, 3.4, 3.5_

  - [x] 20.2 Course detail page
    - Document list
    - Upload button (teacher)
    - _Requirements: 4.1_

- [x] 21. Frontend Document Upload
  - [x] 21.1 Document upload component
    - Drag-and-drop zone
    - File picker
    - Progress indicator
    - _Requirements: 4.1, 4.6_

  - [x] 21.2 Document processing view
    - Chunking configuration
    - Process button
    - _Requirements: 5.1, 5.5_

- [x] 22. Frontend Chunking Visualization
  - [x] 22.1 Chunk results component
    - Chunk cards grid
    - Statistics display
    - _Requirements: 5.3, 6.1, 6.3_

  - [x] 22.2 Text highlight component
    - Hover interaction
    - Overlap indication
    - _Requirements: 6.2, 6.4_

- [x] 23. Docker Compose - Full Stack
  - [x] 23.1 docker-compose.yml güncelle
    - Frontend service ekle
    - All services networking
    - _Requirements: 1.1, 1.3, 1.6_

- [ ] 24. Final Checkpoint
  - docker-compose up ile tüm sistemi başlat
  - End-to-end test: Teacher login → Course create → Document upload → Chunking
  - End-to-end test: Student login → View courses
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based tests
- Backend önce tamamlanacak, sonra frontend
- Docker altyapısı her aşamada test edilecek
- Property testleri Hypothesis kütüphanesi ile yazılacak
- Bu faz sadece chunking'i kapsıyor, embedding ve chat sonraki fazlarda eklenecek

