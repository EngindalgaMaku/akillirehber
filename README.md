# 🎓 AkıllıRehber - Eğitim İçin RAG Sistemi

AkıllıRehber, eğitim kurumları için geliştirilmiş, yapay zeka destekli akıllı soru-cevap sistemidir. Retrieval-Augmented Generation (RAG) teknolojisi ile öğrencilere ders materyalleri hakkında doğru ve bağlama uygun yanıtlar sunar.

## 🔄 Localhost → Coolify Veri Senkronizasyonu

Localhost'taki verilerinizi Coolify ortamına kolayca aktarmak için hazır araçlar:

- **[⚡ Hızlı Başlangıç](./SYNC_QUICKSTART.md)** - 5 dakikada başlayın
- **[📖 Detaylı Rehber](./COOLIFY_SYNC_GUIDE.md)** - Tüm seçenekler ve sorun giderme
- **[📊 Özet](./SYNC_SUMMARY.md)** - Karşılaştırma ve öneriler

### Tek Komutla Aktarım

```bash
# Linux/Mac
./sync-to-coolify.sh user@coolify-server.com

# Windows PowerShell
.\sync-to-coolify.ps1 -CoolifyHost "user@coolify-server.com"
```

**Ne aktarılır:**
- ✅ PostgreSQL veritabanı (kullanıcılar, kurslar, ayarlar)
- ✅ Weaviate vektör veritabanı (embeddings, dökümanlar)
- ✅ Otomatik yedekleme ve restore talimatları

---

## ✨ Özellikler

### 🎯 Temel Özellikler
- **Çoklu Doküman Desteği**: PDF, DOCX, TXT formatlarında ders materyali yükleme
- **Akıllı Chunking**: Recursive ve Semantic chunking stratejileri ile optimal metin bölümleme
- **Hibrit Arama**: BM25 ve vektör aramasını birleştiren gelişmiş retrieval sistemi
- **Reranker Entegrasyonu**: Cohere ve Alibaba reranker'ları ile arama sonuçlarını iyileştirme
- **Çoklu LLM Desteği**: OpenRouter, Claude.gg, Groq, OpenAI entegrasyonu
- **RAGAS Değerlendirme**: Faithfulness, relevancy, precision, recall metrikleri ile sistem performans analizi

### 👥 Kullanıcı Rolleri
- **Öğretmen Paneli**: Ders oluşturma, materyal yükleme, sistem ayarları, performans raporları
- **Öğrenci Paneli**: Ders seçimi, soru sorma, kaynak görüntüleme, sohbet geçmişi
- **Admin Paneli**: Kullanıcı yönetimi, sistem konfigürasyonu

### 📊 Değerlendirme Metrikleri
- **Faithfulness**: Yanıtın kaynak metinlere sadakati
- **Answer Relevancy**: Yanıtın soruyla ilgisi
- **Context Precision**: Getirilen bağlamın hassasiyeti
- **Context Recall**: Getirilen bağlamın kapsamı
- **ROUGE-N**: N-gram tabanlı benzerlik
- **BERTScore**: Semantik benzerlik

### 📦 Veri Seti

RAGAS ve ROUGE/BERTScore testlerinde kullanılan Bloom veri seti ekip tarafından hazırlanmıştır:

https://github.com/EngindalgaMaku/Bilisim-Teknolojileri-Bloom-Dataset

## 🏗️ Mimari

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend   │────▶│   Backend    │────▶│   Weaviate   │
│  (Next.js)   │     │  (FastAPI)   │     │   (Vector)   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ├──────────────┐
                            ▼              ▼
                     ┌──────────┐   ┌──────────┐
                     │  RAGAS   │   │PostgreSQL│
                     │ Service  │   │    DB    │
                     └──────────┘   └──────────┘
                            │
                     ┌──────┴──────┐
                     ▼             ▼
               ┌─────────┐   ┌─────────┐
               │OpenRouter│   │Claude.gg│
               └─────────┘   └─────────┘
```

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Docker & Docker Compose
- Python 3.11+ (yerel geliştirme için)
- Node.js 18+ (frontend geliştirme için)

### Kurulum

1. **Projeyi klonlayın**
```bash
git clone https://github.com/your-username/akilli-rehber.git
cd akilli-rehber
```

2. **Ortam değişkenlerini ayarlayın**
```bash
cp .env.example .env
```

`.env` dosyasını düzenleyin ve API anahtarlarınızı ekleyin:
```bash
# Zorunlu
OPENROUTER_API_KEY=sk-or-your-key
SECRET_KEY=your-secret-key-change-in-production

# Opsiyonel
CLAUDEGG_API_KEY=your-claude-gg-key
COHERE_API_KEY=your-cohere-key
DASHSCOPE_API_KEY=your-dashscope-key
```

3. **Servisleri başlatın**
```bash
# Geliştirme ortamı (frontend yerel)
docker-compose up -d

# Üretim ortamı (tüm servisler Docker'da)
docker-compose -f docker-compose.prod.yml up -d
```

4. **Uygulamaya erişin**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Dokümantasyonu**: http://localhost:8000/docs
- **RAGAS Service**: http://localhost:8001

### İlk Kullanım

1. http://localhost:3000/register adresinden kayıt olun
2. Öğretmen hesabı ile giriş yapın
3. Yeni bir ders oluşturun
4. Ders materyallerini yükleyin (PDF, DOCX, TXT)
5. Öğrenci hesabı ile giriş yaparak dersi seçin ve soru sorun

## 🔧 Konfigürasyon

### Chunking Stratejileri

**Recursive Chunking**
- Chunk boyutu: 500 karakter
- Overlap: 50 karakter
- Cümle bütünlüğünü korur
- Türkçe karakter desteği

**Semantic Chunking**
- Dinamik eşik belirleme (95. percentile)
- Buffer tabanlı embedding (buffer_size=1)
- Soru-cevap çifti tespiti
- Min/max boyut kontrolü (150-2000 karakter)

### Hibrit Arama

Alpha parametresi ile BM25 ve vektör aramasını dengeleyin:
- `alpha=0.0`: Sadece BM25 (anahtar kelime)
- `alpha=0.5`: Dengeli hibrit arama
- `alpha=1.0`: Sadece vektör araması (semantik)

### Embedding Modelleri

| Sağlayıcı | Model | Boyut | API Key |
|-----------|-------|-------|---------|
| OpenRouter | text-embedding-3-small | 1536 | OPENROUTER_API_KEY |
| OpenRouter | text-embedding-3-large | 3072 | OPENROUTER_API_KEY |
| Alibaba | text-embedding-v4 | 1024 | DASHSCOPE_API_KEY |
| Cohere | embed-multilingual-v3.0 | 1024 | COHERE_API_KEY |

### Reranker Modelleri

| Sağlayıcı | Model | Dil Desteği |
|-----------|-------|-------------|
| Cohere | rerank-multilingual-v3.0 | 100+ dil |
| Alibaba | gte-rerank-hybrid | Çok dilli |

## 🛠️ Geliştirme

### Backend Geliştirme

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend Geliştirme

```bash
cd frontend
npm install
npm run dev
```

### Test Çalıştırma

```bash
# Backend testleri
cd backend
pytest

# Belirli bir test dosyası
pytest tests/test_chunking.py -v
```

## 📁 Proje Yapısı

```
akilli-rehber/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── models/         # Veritabanı modelleri
│   │   ├── routers/        # API endpoint'leri
│   │   ├── services/       # İş mantığı
│   │   └── main.py         # Uygulama giriş noktası
│   ├── alembic/            # Veritabanı migration'ları
│   ├── tests/              # Backend testleri
│   └── requirements.txt    # Python bağımlılıkları
├── frontend/               # Next.js frontend
│   └── src/
│       ├── app/           # Sayfalar (App Router)
│       ├── components/    # React bileşenleri
│       └── lib/          # Yardımcı fonksiyonlar
├── ragas_service/         # RAGAS değerlendirme servisi
├── docker/                # Docker konfigürasyonları
├── docs/                  # Dokümantasyon
├── docker-compose.yml     # Geliştirme ortamı
├── docker-compose.prod.yml # Üretim ortamı
└── .env.example          # Örnek ortam değişkenleri
```

## 📊 API Endpoint'leri

### Kimlik Doğrulama
- `POST /auth/register` - Yeni kullanıcı kaydı
- `POST /auth/login` - Kullanıcı girişi
- `GET /auth/me` - Mevcut kullanıcı bilgisi

### Ders Yönetimi
- `GET /courses` - Ders listesi
- `POST /courses` - Yeni ders oluştur
- `GET /courses/{id}` - Ders detayı
- `PUT /courses/{id}` - Ders güncelle
- `DELETE /courses/{id}` - Ders sil

### Doküman Yönetimi
- `POST /courses/{id}/documents` - Doküman yükle
- `GET /courses/{id}/documents` - Doküman listesi
- `DELETE /documents/{id}` - Doküman sil

### Chat
- `POST /courses/{id}/chat` - Soru sor
- `GET /courses/{id}/chat/history` - Sohbet geçmişi
- `DELETE /courses/{id}/chat/history` - Geçmişi temizle

### RAGAS Değerlendirme
- `POST /ragas/evaluate` - Test seti değerlendir
- `GET /ragas/runs` - Değerlendirme geçmişi
- `GET /ragas/runs/{id}` - Değerlendirme detayı

Detaylı API dokümantasyonu için: http://localhost:8000/docs

## 🔍 Sorun Giderme

### RAGAS Servisi Sorunları
```bash
# Logları kontrol edin
docker-compose logs ragas

# Sağlayıcı durumunu kontrol edin
curl http://localhost:8001/providers

# Servisi yeniden başlatın
docker-compose restart ragas
```

### Veritabanı Sorunları
```bash
# Veritabanını sıfırlayın
docker-compose down -v
docker-compose up -d postgres

# Migration'ları çalıştırın
docker-compose exec backend alembic upgrade head
```

### Weaviate Sorunları
```bash
# Weaviate durumunu kontrol edin
curl http://localhost:8080/v1/.well-known/ready

# Weaviate'i sıfırlayın
docker-compose down -v
docker-compose up -d weaviate
```

### Frontend Build Sorunları
```bash
cd frontend
rm -rf .next node_modules
npm install
npm run build
```

## 📚 Dokümantasyon

- [API Dokümantasyonu](http://localhost:8000/docs)
- [RAGAS API Dokümantasyonu](http://localhost:8001/docs)
- [Chunking Stratejileri](docs/chunking-strategies.md)
- [Hibrit Arama Konfigürasyonu](docs/hybrid-search.md)
- [Reranker Kullanımı](docs/reranker-setup.md)

## 🤝 Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'feat: Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- [RAGAS](https://github.com/explodinggradients/ragas) - RAG değerlendirme framework'ü
- [Weaviate](https://weaviate.io/) - Vektör veritabanı
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [Next.js](https://nextjs.org/) - Frontend framework
- [LangChain](https://www.langchain.com/) - LLM orchestration

## 📧 İletişim

Sorularınız için issue açabilir veya [email@example.com](mailto:email@example.com) adresinden iletişime geçebilirsiniz.

---

⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!
