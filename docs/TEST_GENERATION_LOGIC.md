# Test Generation Logic (Custom Bloom-Based)

Bu doküman, projedeki **custom (LLM tabanlı) Bloom taksonomisi** soru üretiminin uçtan uca mantığını açıklar.

Odak: **/api/test-generation/generate-from-course** akışı (Weaviate chunk + dersin LLM ayarları + Bloom dağılımı).

---

## 1) Amaç ve Çıktı Formatı

### Amaç
- Bir dersin içeriğinden (Weaviate’te saklanan **chunk**’lar) otomatik test soruları üretmek.
- Üretilen soruların **Bloom taksonomisine** göre dengeli dağılması.
- Üretilen soruların **RAGAS evaluation** için uygun formatta saklanması.

### Çıktı (DB’ye yazılan alanlar)
Her üretilen soru `TestQuestion` olarak kaydedilir:
- `test_set_id`: Hangi test setine eklendiği
- `question`: Üretilen soru
- `ground_truth`: Beklenen doğru cevap
- `alternative_ground_truths`: (opsiyonel) alternatif cevaplar
- `expected_contexts`: (opsiyonel) sorunun üretildiği/bağlam chunk’ı
- `question_metadata`: Üretim metadata’sı
  - `bloom_level`: Bloom seviyesi
  - `generated_by`: `custom_llm`
  - `generated_at`: ISO timestamp
  - `chunk_id`: kullanılan chunk id
  - `llm_provider`, `llm_model`: dersin ayarlarından

---

## 2) Ana Backend Entry Point

### Router
Dosya: `backend/app/routers/test_generation.py`

#### Endpoint: Generate from Course
- **Method**: `POST`
- **Path**: `/api/test-generation/generate-from-course`
- **Auth**: `Depends(get_current_teacher)` → öğretmen oturumu gerekir
- **Content-Type**: `multipart/form-data` (FastAPI `Form(...)` ile parse eder)

#### Beklenen Form alanları
- `test_set_id` (zorunlu)
- `total_questions` (varsayılan `50`)
- `remembering_ratio` (varsayılan `0.30`)
- `understanding_applying_ratio` (varsayılan `0.40`)
- `analyzing_evaluating_ratio` (varsayılan `0.30`)

> Not: Router toplam oranı kontrol eder. `remembering_ratio + understanding_applying_ratio + analyzing_evaluating_ratio ≈ 1.0` değilse `400` döner.

#### Ana akış
1. `test_set_id` ile `TestSet` bulunur.
2. Kullanıcı erişimi doğrulanır:
   - `verify_course_access(db, test_set.course_id, current_user)`
3. `Course` çekilir ve course.settings kontrol edilir.
4. Bloom oranlarından `bloom_distribution` hazırlanır.
5. `CustomTestGenerator.generate_from_course(...)` çağrılır.
6. Dönen sorular DB’ye `TestQuestion` olarak kaydedilir.
7. JSON response döner:
   - `success`, `test_set_id`, `generated_count`, `saved_count`, `statistics` vs.

---

## 3) Soru Üretim Motoru (CustomTestGenerator)

Dosya: `backend/app/services/custom_test_generator.py`

### Veri Kaynağı
- `WeaviateService` kullanılarak dersin chunk’ları alınır.
- Chunk’lar genelde:
  - `content`
  - `chunk_id`
  gibi alanlar taşır.

### Bloom dağılımı → Soru sayısı
- `total_questions` ile dağılım oranlarına göre **kaç soru** üretileceği hesaplanır.
- Seviye anahtarları:
  - `remembering`
  - `understanding_applying`
  - `analyzing_evaluating`

> Bu anahtarlar hem üretim döngüsünde hem de `question_metadata.bloom_level`’da kullanılır.

### LLM çağrısı
- Dersin ayarlarından LLM konfigürasyonu alınır:
  - `course.settings.llm_provider`
  - `course.settings.llm_model`
  - `course.settings.llm_temperature` (varsayılan 0.7)
- `LLMService` ile `generate_response(messages)` çağrılır.

### Prompt yapısı
`CustomTestGenerator.BLOOM_PROMPTS` içinde her seviye için ayrı prompt template vardır:
- **remembering**: tanım/liste tipi, içerikten doğrudan bilgi
- **understanding_applying**: senaryo/durum, uygulama/akıl yürütme
- **analyzing_evaluating**: karşılaştırma/analiz, çok boyutlu değerlendirme

Tüm prompt’larda ortak hedef:
- “Cevap içerikte geçmeli / içerikten türetilmeli”
- “Self-contained soru”
- “Detaylı cevap (2-5 cümle)”

### Chunk seçimi
- Her soru için chunk seçimi döngüseldir:
  - `chunk = chunks[i % len(chunks)]`
- Context çok uzunsa kısaltılır (ör: 2000 karakter).

### Parse & Validasyon
- LLM çıktısı `SORU:` ve `CEVAP:` pattern’lerine göre parse edilir.
- Parse başarısızsa soru atlanır.

### Sonuç / İstatistik
- Üretilen sorular listesi ve `statistics` döner:
  - `total_generated`, `requested`
  - Bloom seviyelerine göre üretilen sayılar
  - kullanılan LLM bilgisi

---

## 4) Frontend Akışı

### UI
Sayfa: `frontend/src/app/dashboard/ragas/test-sets/generate/page.tsx`

- Ders seçimi → test set listesi yüklenir.
- Test set seçimi / yeni test set oluşturma.
- `total_questions` ve Bloom oranları girilir.
- Oran toplamı `1.0` değilse request gönderilmez.

### API Client
Dosya: `frontend/src/lib/api.ts`

- `api.generateFromCourse(...)` metodu FormData ile request atar.
- Önemli teknik detay:
  - `ApiClient.request()` **FormData** gönderirken `Content-Type: application/json` basmamalı.
  - Aksi halde backend `Form(...)` parse edemez ve `422 Unprocessable Entity` görülebilir.

---

## 5) Sık Görülen Hatalar ve Nedenleri

### 422 Unprocessable Entity
- Sebep: Form alanları gelmiyor / yanlış Content-Type.
- Kontrol:
  - Request body gerçekten `multipart/form-data` mı?
  - Form field isimleri: `test_set_id`, `total_questions`, `remembering_ratio`, `understanding_applying_ratio`, `analyzing_evaluating_ratio`

### 400 Bloom ratios must sum to 1.0
- Sebep: oranların toplamı 1.0 değil.

### 400 Course settings not configured
- Sebep: dersin LLM ayarları eksik.

### 500 Error generating questions
- Sebep: LLM servis hatası / Weaviate chunk yok / parse hataları.

---

## 6) RAGAS ile ilişki

Bu akış **RAGAS servisinden bağımsız** “custom” üretimdir.
- RAGAS tarafındaki `/api/ragas/test-sets/{id}/generate-questions` ayrı bir akıştır.
- Custom akışın amacı, dersin kendi chunk’ları ve dersin LLM ayarlarıyla Bloom kontrollü üretim yapmaktır.

---

## Referans Dosyalar

- Backend router: `backend/app/routers/test_generation.py`
- Generator service: `backend/app/services/custom_test_generator.py`
- Weaviate erişimi: `backend/app/services/weaviate_service.py`
- Frontend UI: `frontend/src/app/dashboard/ragas/test-sets/generate/page.tsx`
- Frontend API: `frontend/src/lib/api.ts`
