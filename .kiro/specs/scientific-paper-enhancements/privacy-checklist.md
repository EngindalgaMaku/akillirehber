# Gizlilik Sistemi - Uygulama Checklist

## 📋 GENEL BAKIŞ

Bu checklist, gizlilik sisteminin tam çalışır halde uygulanması için gereken tüm adımları içerir.

**Toplam Süre:** 4 gün
**Dosya Sayısı:** ~15 dosya
**Test Case Sayısı:** 100+

---

## ✅ GÜN 1: BACKEND - PII DETECTION CORE

### Adım 1.1: Proje Yapısı Oluşturma (30 dk)
- [ ] `backend/app/services/pii_detection.py` dosyası oluştur
- [ ] `backend/app/services/content_safety.py` dosyası oluştur
- [ ] `backend/app/data/` klasörü oluştur
- [ ] `backend/tests/test_pii_detection.py` dosyası oluştur

### Adım 1.2: Türkçe İsim Listesi (30 dk)
- [ ] `backend/app/data/turkish_names.txt` oluştur
- [ ] En yaygın 1000 Türkçe ismi ekle
- [ ] Kaynak: TDK, baby name websites
- [ ] UTF-8 encoding kontrol et

### Adım 1.3: TurkishPIIDetector Sınıfı (2 saat)
- [ ] `PIIType` enum tanımla
- [ ] `PIIMatch` dataclass oluştur
- [ ] `PIIDetectionResult` dataclass oluştur
- [ ] `TurkishPIIDetector` sınıfı oluştur
- [ ] `detect()` ana metodu implement et
- [ ] `_load_resources()` metodu implement et

### Adım 1.4: TC Kimlik Tespiti (1 saat)
- [ ] `_detect_tc_kimlik()` metodu
- [ ] `_validate_tc_kimlik()` algoritması
- [ ] 10. hane kontrolü
- [ ] 11. hane kontrolü
- [ ] Test case'ler yaz

### Adım 1.5: Telefon Tespiti (45 dk)
- [ ] `_detect_phone()` metodu
- [ ] Regex pattern: `0?5\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}`
- [ ] Format normalizasyonu
- [ ] 5XX kontrolü (cep telefonu)
- [ ] Test case'ler

### Adım 1.6: E-posta Tespiti (30 dk)
- [ ] `_detect_email()` metodu
- [ ] Regex pattern
- [ ] Test case'ler

### Adım 1.7: İsim-Soyisim Tespiti (1.5 saat)
- [ ] `_detect_names()` metodu
- [ ] Büyük harf kontrolü
- [ ] Türkçe isim listesi kontrolü
- [ ] Ardışık büyük harf (soyisim) kontrolü
- [ ] Test case'ler

### Adım 1.8: IBAN ve Kredi Kartı (1 saat)
- [ ] `_detect_iban()` metodu
- [ ] `_detect_credit_card()` metodu
- [ ] `_validate_luhn()` algoritması
- [ ] Test case'ler

### Adım 1.9: Maskeleme (45 dk)
- [ ] `_apply_masking()` metodu
- [ ] Sondan başa maskeleme (pozisyon kayması önleme)
- [ ] Test case'ler

### Adım 1.10: Risk Skoru (30 dk)
- [ ] `_calculate_risk_score()` metodu
- [ ] Risk ağırlıkları tanımla
- [ ] Normalizasyon
- [ ] Test case'ler

### Adım 1.11: Unit Testler (1 saat)
- [ ] `test_tc_kimlik_detection()`
- [ ] `test_phone_detection()`
- [ ] `test_email_detection()`
- [ ] `test_name_detection()`
- [ ] `test_no_pii()`
- [ ] `test_multiple_pii()`
- [ ] `test_masking_order()`
- [ ] Tüm testleri çalıştır: `pytest tests/test_pii_detection.py -v`

---

## ✅ GÜN 2: BACKEND - CONTENT SAFETY & INTEGRATION

### Adım 2.1: ContentSafetyFilter Sınıfı (2 saat)
- [ ] `ContentIssueType` enum
- [ ] `ContentSafetyFilter` sınıfı
- [ ] `_load_profanity_list()` metodu
- [ ] `SENSITIVE_TOPICS` dictionary
- [ ] `check()` ana metodu
- [ ] `_contains_profanity()` metodu
- [ ] `_detect_sensitive_topics()` metodu
- [ ] `_is_spam()` metodu
- [ ] `_calculate_risk_level()` metodu
- [ ] `_filter_content()` metodu

### Adım 2.2: Privacy Middleware (2 saat)
- [ ] `backend/app/middleware/privacy_middleware.py` oluştur
- [ ] `PrivacyMiddleware` sınıfı
- [ ] `PROTECTED_ENDPOINTS` listesi
- [ ] `__call__()` metodu
- [ ] PII kontrolü entegrasyonu
- [ ] Content safety kontrolü entegrasyonu
- [ ] `_get_text_field()` metodu
- [ ] `_log_pii_detection()` metodu
- [ ] `_log_content_safety()` metodu

### Adım 2.3: Database Modelleri (1 saat)
- [ ] `backend/app/models/db_models.py` güncelle
- [ ] `PIIDetectionLog` modeli ekle
- [ ] `ContentSafetyLog` modeli ekle
- [ ] Alembic migration oluştur
- [ ] Migration çalıştır

### Adım 2.4: API Endpoints (2 saat)
- [ ] `backend/app/routers/privacy.py` oluştur
- [ ] `DetectRequest` Pydantic model
- [ ] `PIIMatchResponse` Pydantic model
- [ ] `DetectResponse` Pydantic model
- [ ] `POST /api/privacy/detect` endpoint
- [ ] `GET /api/privacy/stats` endpoint (opsiyonel)
- [ ] Router'ı main.py'ye ekle

### Adım 2.5: Integration Testler (1 saat)
- [ ] `backend/tests/test_privacy_integration.py` oluştur
- [ ] Middleware test'leri
- [ ] API endpoint test'leri
- [ ] Database logging test'leri
- [ ] Tüm testleri çalıştır

---

## ✅ GÜN 3: FRONTEND - UI COMPONENTS

### Adım 3.1: Types Tanımları (30 dk)
- [ ] `frontend/src/types/privacy.ts` oluştur
- [ ] `PIIMatch` interface
- [ ] `PIIDetectionResult` interface
- [ ] `ContentSafetyIssue` interface

### Adım 3.2: Privacy Test Sayfası (2 saat)
- [ ] `frontend/src/app/dashboard/privacy-test/page.tsx` oluştur
- [ ] Input textarea
- [ ] Hızlı test butonları
- [ ] Tespit butonu
- [ ] Loading state
- [ ] Sonuç gösterimi
- [ ] Maskelenmiş metin gösterimi

### Adım 3.3: Privacy Warning Component (1 saat)
- [ ] `frontend/src/components/privacy-warning.tsx` oluştur
- [ ] Alert component
- [ ] PII listesi gösterimi
- [ ] Risk skoru gösterimi

### Adım 3.4: Batch Test Sayfası (2 saat)
- [ ] `frontend/src/app/dashboard/privacy-test/batch/page.tsx` oluştur
- [ ] Progress bar
- [ ] Batch test logic
- [ ] Sonuç tablosu
- [ ] Metrik hesaplama
- [ ] Görselleştirme

### Adım 3.5: Admin Dashboard (2 saat)
- [ ] `frontend/src/app/dashboard/privacy-admin/page.tsx` oluştur
- [ ] PII detection logs tablosu
- [ ] Content safety logs tablosu
- [ ] İstatistikler
- [ ] Grafikler (Chart.js veya Recharts)

---

## ✅ GÜN 4: TEST & EVALUATION SYSTEM

### Adım 4.1: Test Dataset Oluşturma (2 saat)
- [ ] `backend/tests/data/pii_test_dataset.json` oluştur
- [ ] 50 positive case
- [ ] 30 negative case
- [ ] 20 edge case
- [ ] Her case için expected_pii tanımla
- [ ] Zorluk seviyeleri ekle

### Adım 4.2: Evaluation Script (2 saat)
- [ ] `backend/tests/evaluate_pii_detection.py` oluştur
- [ ] `EvaluationResult` dataclass
- [ ] `PIIDetectionEvaluator` sınıfı
- [ ] `evaluate()` metodu
- [ ] `_evaluate_single_case()` metodu
- [ ] `_calculate_metrics()` metodu
- [ ] `_calculate_per_type_metrics()` metodu
- [ ] `generate_report()` metodu

### Adım 4.3: Görselleştirmeler (1 saat)
- [ ] `_create_visualizations()` metodu
- [ ] F1 Score by Type bar chart
- [ ] Precision vs Recall chart
- [ ] Confusion Matrix heatmap
- [ ] matplotlib/seaborn kullan

### Adım 4.4: JSON Rapor (30 dk)
- [ ] `_save_json_report()` metodu
- [ ] Overall metrics
- [ ] Confusion matrix
- [ ] Per-type metrics
- [ ] Timestamp

### Adım 4.5: Automated Testing (1 saat)
- [ ] Evaluation script'i çalıştır
- [ ] Sonuçları analiz et
- [ ] Threshold'ları ayarla
- [ ] CI/CD entegrasyonu (opsiyonel)

---

## ✅ FİNAL: ENTEGRASYON VE DOKÜMANTASYON

### Adım 5.1: Sistem Entegrasyonu (1 saat)
- [ ] Middleware'i FastAPI'ye ekle
- [ ] Chat endpoint'ine entegre et
- [ ] RAGAS quick test'e entegre et
- [ ] Tüm endpoint'leri test et

### Adım 5.2: End-to-End Test (1 saat)
- [ ] Frontend'den backend'e tam akış test et
- [ ] PII tespit edildiğinde uyarı gösterilmeli
- [ ] Maskelenmiş metin LLM'e gitmeli
- [ ] Log'lar database'e kaydedilmeli

### Adım 5.3: Performans Testi (30 dk)
- [ ] 100 metin ile latency testi
- [ ] Memory usage ölçümü
- [ ] Throughput hesaplama
- [ ] Bottleneck'leri tespit et

### Adım 5.4: Dokümantasyon (1 saat)
- [ ] README.md güncelle
- [ ] API dokümantasyonu
- [ ] Kullanım örnekleri
- [ ] Bilimsel metrikler açıklaması

### Adım 5.5: Bilimsel Rapor (1 saat)
- [ ] Evaluation sonuçlarını derle
- [ ] Grafikleri hazırla
- [ ] Tablo oluştur
- [ ] Makale taslağı yaz

---

## 📊 BAŞARI KRİTERLERİ

### Teknik Kriterler
- [ ] Precision ≥ 0.85
- [ ] Recall ≥ 0.80
- [ ] F1 Score ≥ 0.82
- [ ] Latency < 100ms (ortalama)
- [ ] False Positive Rate < 0.15

### Fonksiyonel Kriterler
- [ ] TC Kimlik doğru tespit ediliyor
- [ ] Telefon numarası doğru tespit ediliyor
- [ ] E-posta doğru tespit ediliyor
- [ ] İsim-soyisim doğru tespit ediliyor
- [ ] Maskeleme doğru çalışıyor
- [ ] Log'lar kaydediliyor

### Kullanıcı Deneyimi
- [ ] Test sayfası çalışıyor
- [ ] Uyarılar gösteriliyor
- [ ] Maskelenmiş metin okunabilir
- [ ] Hızlı test örnekleri çalışıyor

### Bilimsel Çıktılar
- [ ] 100+ test case
- [ ] Precision/Recall/F1 metrikleri
- [ ] PII türüne göre analiz
- [ ] Confusion matrix
- [ ] 3+ görselleştirme
- [ ] JSON rapor

---

## 🚀 HIZLI BAŞLANGIÇ

Hemen başlamak için:

```bash
# 1. Backend setup
cd backend
pip install -r requirements.txt

# 2. Test dataset oluştur
python scripts/create_test_dataset.py

# 3. PII detector implement et
# backend/app/services/pii_detection.py

# 4. Testleri çalıştır
pytest tests/test_pii_detection.py -v

# 5. Evaluation yap
python tests/evaluate_pii_detection.py

# 6. Frontend setup
cd ../frontend
npm install

# 7. Test sayfasını aç
npm run dev
# http://localhost:3000/dashboard/privacy-test
```

---

## 📈 İLERLEME TAKİBİ

### Gün 1: Backend Core
- [ ] 0/11 adım tamamlandı
- [ ] Tahmini süre: 8 saat
- [ ] Gerçek süre: ___ saat

### Gün 2: Integration
- [ ] 0/5 adım tamamlandı
- [ ] Tahmini süre: 8 saat
- [ ] Gerçek süre: ___ saat

### Gün 3: Frontend
- [ ] 0/5 adım tamamlandı
- [ ] Tahmini süre: 8 saat
- [ ] Gerçek süre: ___ saat

### Gün 4: Testing
- [ ] 0/5 adım tamamlandı
- [ ] Tahmini süre: 6 saat
- [ ] Gerçek süre: ___ saat

### Final: Integration
- [ ] 0/5 adım tamamlandı
- [ ] Tahmini süre: 4 saat
- [ ] Gerçek süre: ___ saat

**TOPLAM:** 0/31 adım tamamlandı

---

## 💡 İPUÇLARI

### Hata Ayıklama
- PII tespit edilmiyor → Regex pattern'leri kontrol et
- False positive çok → Confidence threshold'u artır
- Yavaş çalışıyor → Regex'leri optimize et
- Test fail oluyor → Expected vs Detected karşılaştır

### Optimizasyon
- Türkçe isim listesini cache'le
- Regex compile et (re.compile)
- Batch processing için async kullan
- Database query'leri optimize et

### Bilimsel Değerlendirme
- Test dataset'ini dengeli tut (positive/negative)
- Edge case'leri dahil et
- Zorluk seviyelerini belirt
- Baseline ile karşılaştır

---

## ✅ TAMAMLANDIĞINDA

Sisteminiz:
- ✅ Tam çalışır durumda
- ✅ Bilimsel olarak test edilmiş
- ✅ Precision/Recall/F1 metrikleri mevcut
- ✅ Test sayfası hazır
- ✅ Dokümante edilmiş
- ✅ Makale için hazır

**Başlamaya hazır mısınız?** 🚀

İlk adım: `backend/app/services/pii_detection.py` dosyasını oluşturun!
