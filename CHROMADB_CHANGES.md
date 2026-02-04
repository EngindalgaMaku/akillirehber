# ChromaDB Integration - Değişiklik Listesi

## ✅ Eklenen Dosyalar (Yeni)

### Backend Services
1. `backend/app/services/vector_store_interface.py` - Abstract interface
2. `backend/app/services/weaviate_adapter.py` - Weaviate wrapper
3. `backend/app/services/chromadb_service.py` - ChromaDB implementation
4. `backend/app/services/vector_store_factory.py` - Factory pattern

### Database Migration
5. `backend/alembic/versions/20260203_230511_add_vector_store_to_course_settings.py`

### Documentation
6. `backend/docs/CHROMADB_BENCHMARK.md` - Detaylı dokümantasyon
7. `CHROMADB_QUICKSTART.md` - Hızlı başlangıç
8. `CHROMADB_CHANGES.md` - Bu dosya

## 📝 Değiştirilen Dosyalar (Minimal)

### Docker
1. `docker-compose.yml`
   - ChromaDB container eklendi
   - ChromaDB volume eklendi
   - Backend'e CHROMADB_URL env variable eklendi

### Configuration
2. `backend/app/config.py`
   - `chromadb_url` field eklendi

### Dependencies
3. `backend/requirements.txt`
   - `chromadb>=0.5.23` eklendi

## ❌ Değiştirilmeyen Dosyalar (Korunan)

- `backend/app/services/weaviate_service.py` - **HİÇ DEĞİŞMEDİ**
- `backend/app/routers/*` - **HİÇ DEĞİŞMEDİ**
- `backend/app/models/db_models.py` - **HİÇ DEĞİŞMEDİ** (sadece migration ile DB'ye field eklendi)
- Tüm frontend dosyaları - **HİÇ DEĞİŞMEDİ**

## 🔄 Geri Alma Adımları

### Hızlı Geri Alma (Sadece Disable Et)
```bash
# ChromaDB'yi kullanmayı bırak
docker exec -it rag-postgres psql -U raguser -d ragchatbot -c "UPDATE course_settings SET vector_store = 'weaviate';"
docker-compose stop chromadb
```

### Tam Geri Alma (Tüm Değişiklikleri Kaldır)
```bash
# 1. Migration'ı geri al
cd backend
alembic downgrade -1

# 2. Yeni dosyaları sil
rm backend/app/services/vector_store_interface.py
rm backend/app/services/weaviate_adapter.py
rm backend/app/services/chromadb_service.py
rm backend/app/services/vector_store_factory.py
rm backend/alembic/versions/20260203_230511_add_vector_store_to_course_settings.py
rm backend/docs/CHROMADB_BENCHMARK.md
rm CHROMADB_QUICKSTART.md
rm CHROMADB_CHANGES.md

# 3. docker-compose.yml'i geri al (git kullanıyorsanız)
git checkout docker-compose.yml

# 4. config.py'yi geri al
git checkout backend/app/config.py

# 5. requirements.txt'i geri al
git checkout backend/requirements.txt

# 6. Container'ı kaldır
docker-compose stop chromadb
docker-compose rm -f chromadb
docker volume rm rag-chromadb-data
```

## 🎯 Kullanım Senaryoları

### Senaryo 1: Sadece Test (Önerilen)
- Tek bir test course'u oluştur
- ChromaDB'ye geçir
- Benchmark yap
- Sonra Weaviate'e geri dön

### Senaryo 2: Karşılaştırmalı Test
- Aynı course'u kopyala
- Biri Weaviate, biri ChromaDB kullansın
- RAGAS testlerini çalıştır
- Sonuçları karşılaştır

### Senaryo 3: Production (ÖNERİLMEZ)
- Tüm course'ları ChromaDB'ye geçir
- Risk: Yeni teknoloji, az test edildi

## 📊 Beklenen Sonuçlar

### ChromaDB Avantajları
- Daha basit setup
- Daha az RAM kullanımı
- Daha hızlı startup

### Weaviate Avantajları
- Native hybrid search (daha hızlı)
- Daha gelişmiş filtering
- Production-proven
- Daha iyi documentation

## 🔒 Güvenlik

- Hiçbir mevcut özellik bozulmadı
- Varsayılan davranış değişmedi (Weaviate)
- Backward compatible
- Rollback kolay

## 📞 Destek

Sorun olursa:
1. `docker-compose logs chromadb` - ChromaDB logları
2. `docker-compose logs backend` - Backend logları
3. Hızlı geri alma komutlarını çalıştır
