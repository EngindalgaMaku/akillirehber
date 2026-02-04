# ChromaDB Benchmark - Quick Start

## ⚠️ EXPERIMENTAL FEATURE

Bu özellik Weaviate ile ChromaDB'yi karşılaştırmak için eklendi. **Production kullanımı önerilmez.**

## Hızlı Başlangıç

### 1. Container'ları Başlat

```bash
docker-compose up -d
```

ChromaDB otomatik olarak `localhost:8081` portunda başlayacak.

### 2. Migration Çalıştır

```bash
cd backend
alembic upgrade head
```

### 3. Test Course Oluştur

Frontend'den normal şekilde bir course oluşturun.

### 4. Vector Store Seçimi

**Seçenek A: Database'den Manuel**
```sql
-- Course ID 1 için ChromaDB kullan
UPDATE course_settings 
SET vector_store = 'chromadb' 
WHERE course_id = 1;
```

**Seçenek B: Python'dan**
```python
from app.models.db_models import CourseSettings

settings = db.query(CourseSettings).filter_by(course_id=1).first()
settings.vector_store = "chromadb"
db.commit()
```

### 5. Document Yükle

Frontend'den normal şekilde document yükleyin. Sistem otomatik olarak ChromaDB'yi kullanacak.

### 6. Test Et

- Chat yapın
- RAGAS testleri çalıştırın
- Performansı gözlemleyin

## Karşılaştırma

### Aynı Course'u Her İki DB'de Test Etmek

```sql
-- 1. Weaviate ile test et
UPDATE course_settings SET vector_store = 'weaviate' WHERE course_id = 1;
-- Document yükle, RAGAS çalıştır

-- 2. ChromaDB ile test et
UPDATE course_settings SET vector_store = 'chromadb' WHERE course_id = 1;
-- Aynı document'i yükle, RAGAS çalıştır

-- 3. Sonuçları karşılaştır
```

## Geri Alma

```bash
# 1. Tüm course'ları Weaviate'e çevir
docker exec -it rag-postgres psql -U raguser -d ragchatbot -c "UPDATE course_settings SET vector_store = 'weaviate';"

# 2. ChromaDB container'ını durdur
docker-compose stop chromadb

# 3. Migration'ı geri al (opsiyonel)
cd backend
alembic downgrade -1
```

## Varsayılan Davranış

- Yeni course'lar **otomatik olarak Weaviate** kullanır
- Mevcut course'lar etkilenmez
- `vector_store` field'ı boşsa Weaviate kullanılır

## Notlar

- **Weaviate**: Hybrid search (vector + BM25 keyword, alpha=0.5)
- **ChromaDB**: Pure vector search (sadece embedding)
- **Karşılaştırma**: Hybrid search ne kadar değer katıyor?
- Performans farkları course boyutuna göre değişir

## Daha Fazla Bilgi

Detaylı dokümantasyon: `backend/docs/CHROMADB_BENCHMARK.md`
