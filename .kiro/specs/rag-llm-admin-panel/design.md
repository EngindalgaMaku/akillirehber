# Design Document: RAG-Based Educational Chatbot System

## Overview

Bu tasarım, lise düzeyinde eğitim ortamlarında kullanılmak üzere RAG tabanlı bir chatbot sistemi oluşturmayı hedefler. Sistem Docker konteynerlerinde çalışacak ve mikroservis mimarisi kullanacaktır.

### Sistem Bileşenleri

| Servis | Teknoloji | Port | Açıklama |
|--------|-----------|------|----------|
| Frontend | Next.js 15.x | 3000 | Modern React arayüzü |
| Backend | FastAPI 0.115.x | 8000 | REST API servisi |
| PostgreSQL | PostgreSQL 16 | 5432 | İlişkisel veritabanı |
| Weaviate | Weaviate 1.27.x | 8080 | Vektör veritabanı |

## Architecture

```mermaid
graph TB
    subgraph Docker["Docker Environment"]
        subgraph Frontend["Frontend Container"]
            NextJS[Next.js 15.x App]
            UI[React Components]
        end
        
        subgraph Backend["Backend Container"]
            FastAPI[FastAPI Server]
            Auth[Auth Service]
            Course[Course Service]
            Chunker[Chunking Service]
            DocProc[Document Processor]
        end
        
        subgraph Databases["Database Containers"]
            PostgreSQL[(PostgreSQL 16)]
            Weaviate[(Weaviate)]
        end
    end
    
    NextJS --> FastAPI
    FastAPI --> PostgreSQL
    FastAPI --> Weaviate
    Auth --> PostgreSQL
    Course --> PostgreSQL
    Chunker --> PostgreSQL
    DocProc --> Chunker
```

### Proje Yapısı

```
project-root/
├── docker-compose.yml
├── .env.example
│
├── frontend/                    # Next.js Application
│   ├── Dockerfile
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   ├── login/
│   │   │   ├── dashboard/
│   │   │   │   ├── teacher/
│   │   │   │   └── student/
│   │   │   └── courses/
│   │   │       └── [id]/
│   │   ├── components/
│   │   │   ├── ui/
│   │   │   ├── layout/
│   │   │   ├── auth/
│   │   │   ├── course/
│   │   │   └── chunking/
│   │   ├── lib/
│   │   │   ├── api.ts
│   │   │   └── auth.ts
│   │   └── types/
│   ├── package.json
│   └── next.config.js
│
├── backend/                     # FastAPI Application
│   ├── Dockerfile
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── routers/
│   │   │   ├── auth.py
│   │   │   ├── courses.py
│   │   │   ├── documents.py
│   │   │   └── chunking.py
│   │   ├── services/
│   │   │   ├── auth_service.py
│   │   │   ├── course_service.py
│   │   │   ├── document_processor.py
│   │   │   └── chunker.py
│   │   └── models/
│   │       ├── user.py
│   │       ├── course.py
│   │       ├── document.py
│   │       └── chunking.py
│   ├── alembic/
│   │   └── versions/
│   ├── alembic.ini
│   ├── requirements.txt
│   └── pyproject.toml
│
└── README.md
```

## Components and Interfaces

### Docker Configuration

#### docker-compose.yml

```yaml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/ragchatbot
      - WEAVIATE_URL=http://weaviate:8080
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - postgres
      - weaviate

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=ragchatbot
    volumes:
      - postgres_data:/var/lib/postgresql/data

  weaviate:
    image: semitechnologies/weaviate:1.27.0
    ports:
      - "8080:8080"
    environment:
      - QUERY_DEFAULTS_LIMIT=25
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
      - DEFAULT_VECTORIZER_MODULE=none
      - CLUSTER_HOSTNAME=node1
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  postgres_data:
  weaviate_data:
```

### Backend Components

#### Database Models (SQLAlchemy)

```python
# models/user.py
from sqlalchemy import Column, Integer, String, Enum, DateTime
from sqlalchemy.orm import relationship
import enum

class UserRole(str, enum.Enum):
    TEACHER = "teacher"
    STUDENT = "student"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(Enum(UserRole), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    courses = relationship("Course", back_populates="teacher")

# models/course.py
class Course(Base):
    __tablename__ = "courses"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    teacher_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    teacher = relationship("User", back_populates="courses")
    documents = relationship("Document", back_populates="course")

# models/document.py
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer)
    char_count = Column(Integer)
    course_id = Column(Integer, ForeignKey("courses.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    course = relationship("Course", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document")

# models/chunk.py
class Chunk(Base):
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    index = Column(Integer, nullable=False)
    start_position = Column(Integer)
    end_position = Column(Integer)
    char_count = Column(Integer)
    document_id = Column(Integer, ForeignKey("documents.id"))
    
    document = relationship("Document", back_populates="chunks")
```

#### API Endpoints

```python
# Auth endpoints
POST /api/auth/register    # Kullanıcı kaydı
POST /api/auth/login       # Giriş (JWT token döner)
GET  /api/auth/me          # Mevcut kullanıcı bilgisi

# Course endpoints
GET    /api/courses        # Ders listesi
POST   /api/courses        # Yeni ders oluştur (Teacher)
GET    /api/courses/{id}   # Ders detayı
PUT    /api/courses/{id}   # Ders güncelle (Teacher)
DELETE /api/courses/{id}   # Ders sil (Teacher)

# Document endpoints
GET    /api/courses/{id}/documents     # Ders dökümanları
POST   /api/courses/{id}/documents     # Döküman yükle
DELETE /api/documents/{id}             # Döküman sil

# Chunking endpoints
POST   /api/chunk                      # Metin parçala (test)
GET    /api/documents/{id}/chunks      # Döküman chunk'ları
POST   /api/documents/{id}/process     # Dökümanı işle
```

### Frontend Components

#### Authentication Flow

```typescript
// lib/auth.ts
interface AuthContext {
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  isTeacher: boolean;
  isStudent: boolean;
}
```

#### Page Structure

```typescript
// app/layout.tsx - Root layout with auth provider
// app/login/page.tsx - Login page
// app/dashboard/teacher/page.tsx - Teacher dashboard
// app/dashboard/student/page.tsx - Student dashboard
// app/courses/[id]/page.tsx - Course detail with documents
// app/courses/[id]/documents/[docId]/page.tsx - Document chunking view
```

## Data Models

### Pydantic Models (API)

```python
# schemas/user.py
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: UserRole

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: UserRole

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# schemas/course.py
class CourseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None

class CourseResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    teacher_id: int
    created_at: datetime
    document_count: int

# schemas/document.py
class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    char_count: int
    chunk_count: int
    created_at: datetime

# schemas/chunk.py (mevcut modeller korunacak)
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system.*

### Property 1: Chunking Produces Valid Output

*For any* non-empty text and any valid chunking strategy, the chunker SHALL produce a non-empty list of chunks where each chunk contains non-empty content.

**Validates: Requirements 5.1, 5.2**

### Property 2: Chunk Coverage Invariant

*For any* text that is chunked, the concatenation of all chunks (accounting for overlap) SHALL cover the entire original text without losing any characters.

**Validates: Requirements 5.1, 5.2**

### Property 3: Statistics Accuracy

*For any* set of generated chunks, the reported statistics (total_chunks, avg_chunk_size, min_chunk_size, max_chunk_size) SHALL be mathematically correct based on the actual chunk data.

**Validates: Requirements 5.3**

### Property 4: Character Count Accuracy

*For any* individual chunk, the reported char_count SHALL equal the actual length of the chunk content.

**Validates: Requirements 6.3**

### Property 5: Parameter Validation

*For any* chunk request where overlap >= chunk_size, the system SHALL reject the request with a validation error.

**Validates: Requirements 5.5, 5.6**

### Property 6: Authentication Consistency

*For any* protected endpoint, requests without valid JWT token SHALL return 401 Unauthorized.

**Validates: Requirements 2.5, 2.6**

### Property 7: Role-Based Access

*For any* teacher-only operation (course create/edit/delete), requests from Student_User SHALL return 403 Forbidden.

**Validates: Requirements 2.2, 2.3, 3.5**

## Error Handling

### HTTP Status Codes

| Status | Kullanım |
|--------|----------|
| 200 | Başarılı GET/PUT |
| 201 | Başarılı POST (kaynak oluşturma) |
| 204 | Başarılı DELETE |
| 400 | Geçersiz istek |
| 401 | Kimlik doğrulama hatası |
| 403 | Yetkilendirme hatası |
| 404 | Kaynak bulunamadı |
| 422 | Validasyon hatası |
| 500 | Sunucu hatası |

## Testing Strategy

### Unit Tests

- Her chunking stratejisi için testler
- Authentication ve authorization testleri
- Database model testleri
- API endpoint testleri

### Property-Based Tests

Library: **Hypothesis** (Python)

| Property | Test |
|----------|------|
| Property 1-5 | Chunking testleri (mevcut) |
| Property 6 | Auth token validation |
| Property 7 | Role-based access control |

### Integration Tests

- Docker container başlatma
- Database bağlantıları
- Frontend-backend iletişimi
- End-to-end kullanıcı akışları

