# Custom LLM Test Generation - Kullanım Kılavuzu

## Genel Bakış

Bu sistem, ders içeriğinden (Weaviate chunks) Bloom Taksonomisi uyumlu test soruları otomatik olarak üretir. Dersin kendi LLM ayarlarını kullanarak RAGAS uyumlu sorular oluşturulur.

## Özellikler

- ✅ **Ders İçeriğinden Üretim**: Weaviate'teki mevcut chunks'lardan soru üretir
- ✅ **Dersin LLM Ayarları**: Her dersin kendi LLM provider/model/temperature ayarlarını kullanır
- ✅ **Bloom Taxonomy**: 3 seviyeli Bloom taksonomisi desteği
- ✅ **RAGAS Uyumlu**: Üretilen sorular RAGAS evaluation için hazır
- ✅ **Streaming Progress**: Real-time ilerleme takibi
- ✅ **Ekstra Dependency Yok**: DeepEval, ChromaDB gibi ekstra kütüphane gerektirmez

## Bloom Seviyeleri

### 1. Hatırlama (Remembering) - %40
- Temel tanım ve bilgi soruları
- Örnekler: "X nedir?", "Y'nin özellikleri nelerdir?"

### 2. Uygulama/Analiz (Applying & Analyzing) - %40
- Problem çözme ve karşılaştırma soruları
- Örnekler: "X ve Y'yi karşılaştırın", "Bu durumda hangi çözüm uygulanmalıdır?"

### 3. Değerlendirme/Sentez (Evaluating & Creating) - %20
- Yüksek düzey analiz ve değerlendirme soruları
- Örnekler: "X'in avantaj ve dezavantajlarını değerlendirin"

## API Endpoints

### 1. Status Check
```http
GET /api/test-generation/status
Authorization: Bearer {token}
```

**Response:**
```json
{
  "available": true,
  "message": "Custom LLM-based test generation is ready",
  "custom_generation_available": true
}
```

### 2. Generate from Course (Streaming)
```http
POST /api/test-generation/generate-from-course-stream
Authorization: Bearer {token}
Content-Type: multipart/form-data

test_set_id: 1
total_questions: 50
remembering_ratio: 0.40
applying_analyzing_ratio: 0.40
evaluating_creating_ratio: 0.20
```

**Response:** Server-Sent Events (SSE)
```
data: {"event": "start", "message": "Loading course chunks...", "progress": 0}
data: {"event": "progress", "message": "Generating questions...", "progress": 20}
data: {"event": "progress", "message": "Saving questions...", "progress": 80}
data: {"event": "complete", "saved_count": 50, "progress": 100}
```

## Kullanım

### Frontend'den Kullanım

1. **Test Seti Seçin**
   - Mevcut bir test seti seçin veya yeni oluşturun
   - Her test seti bir derse bağlıdır

2. **Soru Ayarları**
   - Toplam soru sayısı: 10-100 arası
   - Bloom dağılımı: Toplamı %100 olmalı

3. **Üret**
   - "Ders İçeriğinden Üret" butonuna tıklayın
   - İlerleme çubuğu ile takip edin
   - Tamamlandığında test setine yönlendirilirsiniz

### Backend'den Kullanım

```python
from app.services.custom_test_generator import CustomTestGenerator

# Initialize
generator = CustomTestGenerator()

# Generate questions
result = await generator.generate_from_course(
    course=course,  # Course object
    total_questions=50,
    bloom_distribution={
        "remembering": 0.40,
        "applying_analyzing": 0.40,
        "evaluating_creating": 0.20
    }
)

# Result structure
{
    "questions": [
        {
            "question": "...",
            "ground_truth": "...",
            "alternative_ground_truths": [],
            "expected_contexts": ["..."],
            "question_metadata": {
                "bloom_level": "remembering",
                "generated_by": "custom_llm",
                "generated_at": "2026-01-30T00:00:00",
                "chunk_id": "...",
                "llm_provider": "openai",
                "llm_model": "gpt-4"
            }
        }
    ],
    "statistics": {
        "total_generated": 50,
        "requested": 50,
        "bloom_distribution": {...},
        "chunks_used": 100,
        "llm_provider": "openai",
        "llm_model": "gpt-4"
    }
}
```

## Teknik Detaylar

### Soru Üretim Süreci

1. **Chunk Loading**: Weaviate'ten dersin tüm chunks'ları çekilir
2. **Bloom Distribution**: Soru sayıları Bloom seviyelerine göre dağıtılır
3. **LLM Generation**: Her soru için:
   - Farklı bir chunk seçilir (döngüsel)
   - Bloom seviyesine uygun prompt kullanılır
   - Dersin LLM ayarları ile soru üretilir
4. **Parsing**: LLM çıktısı parse edilir (SORU: / CEVAP:)
5. **Validation**: Cevap uzunluğu kontrol edilir (min 10 kelime)
6. **Database Save**: Sorular TestQuestion tablosuna kaydedilir

### Prompt Yapısı

Her Bloom seviyesi için özel prompt şablonları kullanılır:

**Remembering Prompt:**
- Self-contained soru (metne göre kullanma)
- Cevap içerikte geçmeli
- Detaylı cevap (2-3 cümle)

**Applying/Analyzing Prompt:**
- Senaryo veya problem içermeli
- İçeriğe dayalı analiz
- Akıl yürütme içeren cevap (3-4 cümle)

**Evaluating/Creating Prompt:**
- Karşılaştırma veya değerlendirme
- İçeriğe dayalı sentez
- Kapsamlı cevap (4-5 cümle)

### RAGAS Uyumluluğu

Üretilen sorular RAGAS evaluation için hazır formattadır:

```python
{
    "question": "Self-contained soru",
    "ground_truth": "Detaylı cevap",
    "alternative_ground_truths": [],  # Opsiyonel
    "expected_contexts": ["Chunk içeriği"],  # Opsiyonel
    "question_metadata": {
        "bloom_level": "...",
        "generated_by": "custom_llm",
        ...
    }
}
```

## Avantajlar

### DeepEval'a Göre Avantajlar

1. **Ekstra Dependency Yok**: DeepEval, langchain-community, pypdf gerektirmez
2. **Dersin LLM'i Kullanır**: Her ders kendi LLM ayarlarını kullanır
3. **Mevcut Chunks**: Weaviate'teki işlenmiş chunks'ları kullanır
4. **Gerçek Bloom Control**: Prompt seviyesinde Bloom kontrolü
5. **Daha Hızlı**: Ekstra PDF processing yok

### PDF Üretimine Göre Avantajlar

1. **Önceden İşlenmiş**: Chunks zaten chunking stratejisi ile işlenmiş
2. **Kalite Kontrolü**: Chunks quality metrics ile filtrelenmiş
3. **Embedding Ready**: Chunks zaten embed edilmiş
4. **Consistent**: Aynı chunks chat ve evaluation'da kullanılır

## Sınırlamalar

1. **Chunk Dependency**: Ders chunks'ları olmalı (doküman yüklenmiş olmalı)
2. **LLM Dependency**: Dersin LLM ayarları doğru yapılandırılmış olmalı
3. **Chunk Quality**: Chunk kalitesi soru kalitesini etkiler
4. **Token Limits**: Çok uzun chunks kısaltılır (2000 karakter)

## Hata Durumları

### 1. No Chunks Found
```
ValueError: No chunks found for course {course_id}
```
**Çözüm**: Derse doküman yükleyin ve işleyin

### 2. LLM Error
```
Error generating question: API rate limit exceeded
```
**Çözüm**: LLM provider ayarlarını kontrol edin, rate limit bekleyin

### 3. Parse Error
```
Warning: Answer too short: ...
```
**Çözüm**: LLM temperature'ı ayarlayın, prompt'u iyileştirin

## Best Practices

1. **Chunk Quality**: Yüksek kaliteli chunks için semantic chunking kullanın
2. **LLM Selection**: GPT-4 veya Claude gibi güçlü modeller kullanın
3. **Temperature**: 0.7-0.8 arası optimal (çok düşük: tekrarlı, çok yüksek: tutarsız)
4. **Question Count**: İlk denemede 10-20 soru ile test edin
5. **Bloom Distribution**: Varsayılan dağılımı (40/40/20) kullanın

## Monitoring

### Logs

```bash
# Backend logs
docker logs rag-backend -f | grep "custom_test_generator"
```

### Metrics

- `total_generated`: Üretilen soru sayısı
- `chunks_used`: Kullanılan chunk sayısı
- `bloom_distribution`: Bloom seviyelerine göre dağılım
- `llm_provider` / `llm_model`: Kullanılan LLM

## Troubleshooting

### Problem: Sorular çok kısa

**Çözüm:**
- LLM temperature'ı artırın (0.8-0.9)
- Daha güçlü model kullanın (GPT-4)
- Prompt'u kontrol edin

### Problem: Sorular içerikle uyumsuz

**Çözüm:**
- Chunk quality'yi kontrol edin
- Semantic chunking kullanın
- Chunk size'ı optimize edin

### Problem: Üretim çok yavaş

**Çözüm:**
- Daha hızlı LLM provider kullanın
- Soru sayısını azaltın
- Parallel generation ekleyin (gelecek özellik)

## Gelecek Geliştirmeler

- [ ] Parallel question generation
- [ ] Custom prompt templates
- [ ] Question quality scoring
- [ ] Automatic alternative ground truths
- [ ] Multi-language support
- [ ] Question difficulty estimation

## Destek

Sorun yaşarsanız:
1. Backend loglarını kontrol edin
2. LLM ayarlarını doğrulayın
3. Chunk sayısını kontrol edin
4. GitHub'da issue açın

## Changelog

- **v1.0.0** (2026-01-30): İlk sürüm
  - Custom LLM entegrasyonu
  - Bloom taxonomy support
  - Streaming progress tracking
  - RAGAS uyumlu format
