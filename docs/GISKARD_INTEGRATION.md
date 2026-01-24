# Giskard Integration for AkıllıRehber

Bu dokümantasyon, Giskard kütüphanesini AkıllıRehber RAG sistemi ile nasıl entegre edeceğinizi ve Alpha modellerini nasıl test edeceğinizi açıklar.

## İçindekiler

1. [Giskard Nedir?](#giskard-nedir)
2. [Kurulum](#kurulum)
3. [Kullanım](#kullanım)
4. [Türkçe Desteği](#türkçe-desteği)
5. [Test Sonuçları](#test-sonuçları)
6. [Sık Karşılaşılan Sorunlar](#sık-karşılaşılan-sorunlar)

---

## Giskard Nedir?

Giskard, RAG sistemlerini test etmek için kullanılan açık kaynaklı bir Python kütüphanesidir. Aşağıdaki testleri sağlar:

- **Halüsinasyon Tespiti**: Modelin alakasız sorulara uydurma cevap verip vermediğini test eder
- **Relevans Testi**: Modelin sorulara doğru ve alakalı cevaplar verip vermediğini kontrol eder
- **Dil Tutarlılığı**: Cevapların Türkçe dilinde olup olmadığını doğrular
- **Kalite Skoru**: Genel sistem performansını puanlar

### Avantajlar

| Özellik | Durum | Önemi |
|---------|--------|--------|
| Kütüphane Yapısı | Python tabanlı | FastAPI ve microservice yapına tam uyumlu |
| Türkçe Uyumu | Orta-Yüksek | Promptları özelleştirerek tam verim alabilirsin |
| Halüsinasyon Testi | Çok Güçlü | "Alakasız soru" testini manuel yapmaktan kurtarır |
| Kurulum | Kolay | `pip install giskard[rag]` ile hızlıca başlanır |

---

## Kurulum

### 1. Gerekli Paketler

Giskard'ın kendisini kurmanıza gerek yoktur. Bu proje, Giskard'ın temel test mantığını kendi içinde uygular.

Ancak testleri çalıştırmak için aşağıdaki paketlere ihtiyacınız var:

```bash
# Backend requirements.txt'ye eklenmiş paketler
pip install openai requests
```

### 2. Ortam Değişkenleri

`.env` dosyanıza API anahtarınızı ekleyin:

```env
ZAI_API_KEY=your_zai_api_key_here
```

---

## Kullanım

### Temel Kullanım

Giskard tester'ını kullanmak için aşağıdaki adımları izleyin:

```python
from app.services.giskard_service import create_giskard_tester
from app.services.llm_service import get_llm_service

# LLM servisi oluştur
llm_service = get_llm_service(
    provider="zai",
    model="glm-4.7",
    temperature=0.7,
    max_tokens=1000
)

# Giskard tester'ı oluştur
tester = create_giskard_tester(
    model_name="glm-4.7",
    provider="zai",
    llm_service=llm_service,
    num_test_questions=10,      # Alakalı soru sayısı
    num_irrelevant_questions=20, # Alakasız soru sayısı
    language="tr"
)

# Test fonksiyonunu tanımla
def rag_function(question: str) -> str:
    # Sizin RAG sisteminizin sorgu fonksiyonu
    return your_rag_system.query(question)

# Testleri çalıştır
results = tester.run_test_suite(
    rag_function=rag_function,
    document_content=dokuman_iceriği
)

# Rapor oluştur ve yazdır
report = tester.generate_report(results)
print(report)
```

### Hızlı Test

Hazır test script'ini kullanabilirsiniz:

```bash
python test_giskard_alpha.py
```

Bu script:
1. Alpha modelini (glm-4.7) yükler
2. Örnek ders notlarını kullanarak test soruları üretir
3. Alakalı ve alakasız soruları test eder
4. Sonuçları JSON dosyasına kaydeder

### Çoklu Model Karşılaştırması

Birden fazla modeli karşılaştırmak için:

```python
from test_giskard_alpha import compare_models

models_to_test = [
    ("glm-4.7", "zai"),
    # Diğer modelleri buraya ekleyin
]

compare_models(models_to_test, num_relevant=10, num_irrelevant=20)
```

---

## Türkçe Desteği

Giskard entegrasyonu tam Türkçe desteği sunar:

### 1. Türkçe Promptlar

Tüm test prompt'ları Türkçe olarak tasarlanmıştır:

```python
SYSTEM_PROMPT = """Sen Türkçe dilinde uzmanlaşmış bir RAG sistemi
test uzmanısın.

GÖREVİN:
- Sadece TÜRKÇE sorular üret
- Soruların akademik düzeyde ve anlaşılır olmasını sağla
- Halüsinasyon tespiti için alakasız sorular üret
...
"""
```

### 2. Dil Kontrolü

Sistem cevapların dilini otomatik kontrol eder:

```python
def _check_language(self, text: str) -> str:
    """Check if text is Turkish, English, or mixed."""
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    text_has_turkish = any(char in text for char in turkish_chars)
    
    if text_has_turkish:
        return "Türkçe"
    else:
        return "İngilizce"
```

### 3. Türkçe Raporlama

Tüm test raporları Türkçe olarak oluşturulur:

```
GENEL SKOR: 85.5%

ALAKALI SORULAR TESTİ:
  - Soru Sayısı: 10
  - Ortalama Skor: 90.0%
  - Başarı Oranı: 90.0%

ALAKASIZ SORULAR TESTİ (HALÜSİNASYON):
  - Soru Sayısı: 20
  - Halüsinasyon Oranı: 15.0%
  - Doğru Reddetme Oranı: 85.0%
```

---

## Test Sonuçları

### Sonuç Yapısı

Test sonuçları aşağıdaki yapıda döner:

```python
{
    "overall_score": 0.855,
    "metrics": {
        "relevant_questions": {
            "count": 10,
            "avg_score": 0.900,
            "success_rate": 0.900
        },
        "irrelevant_questions": {
            "count": 20,
            "avg_score": 0.850,
            "success_rate": 0.850,
            "hallucination_rate": 0.150,
            "correct_refusal_rate": 0.850
        },
        "language_consistency": 0.900,
        "turkish_response_rate": 0.900
    },
    "evaluations": [...],
    "test_questions": {...}
}
```

### Metrikler

| Metrik | Açıklama | İdeal Değer |
|--------|----------|-------------|
| `overall_score` | Genel performans skoru | > 0.80 |
| `hallucination_rate` | Halüsinasyon oranı | < 0.20 |
| `correct_refusal_rate` | Doğru reddetme oranı | > 0.80 |
| `turkish_response_rate` | Türkçe cevap oranı | > 0.90 |
| `success_rate` | Başarı oranı | > 0.80 |

### Sonuçları Kaydetme

Sonuçlar otomatik olarak JSON dosyasına kaydedilir:

```bash
giskard_test_results_glm_4_7.json
```

---

## Sık Karşılaşılan Sorunlar

### 1. API Anahtarı Hatası

**Hata:** `ZAI_API_KEY environment variable is not set!`

**Çözüm:** `.env` dosyanıza API anahtarınızı ekleyin:

```env
ZAI_API_KEY=your_actual_api_key
```

### 2. JSON Parse Hatası

**Hata:** `Failed to parse relevant questions JSON`

**Çözüm:** LLM'in JSON formatında cevap verememesi durumunda sistem otomatik olarak fallback sorular kullanır. Bu normaldir.

### 3. Halüsinasyon Oranı Yüksek

**Sorun:** Model alakasız sorulara cevap veriyor.

**Çözüm:** System prompt'unuzu güçlendirin:

```python
system_prompt = """Sen AkıllıRehber adında bir RAG sistemisin.

KURALLAR:
1. Sadece ders notlarında BULUNAN bilgilere dayalı cevap ver
2. Notlarda olmayan sorular için KESİNLİKLE "Bilmiyorum" de
3. Uydurma bilgi verme (halüsinasyon yapma)
"""
```

### 4. Dil Tutarlılığı Düşük

**Sorun:** Cevaplar İngilizce veya karışık çıkıyor.

**Çözüm:** Prompt'a net dil talimatı ekleyin:

```python
system_prompt = """
...
3. Cevaplarını KESİNLİKLE TÜRKÇE ver
4. İngilizce kelime kullanma (terimler hariç)
"""
```

---

## İleri Kullanım

### Kendi RAG Sisteminizi Entegre Edin

Gerçek RAG sisteminizi test etmek için:

```python
from app.services.giskard_service import create_giskard_tester

# Sizin RAG servisinizi import edin
from app.services.rag_service import RAGService

# RAG servisini başlat
rag_service = RAGService()

# Giskard tester'ı oluştur
tester = create_giskard_tester(
    model_name="glm-4.7",
    provider="zai",
    llm_service=rag_service.llm_service
)

# Test fonksiyonunu tanımla
def rag_function(question: str) -> str:
    # Weaviate'den context al
    context = rag_service.retrieve_context(question)
    
    # LLM ile cevap üret
    answer = rag_service.generate_answer(question, context)
    
    return answer

# Testleri çalıştır
results = tester.run_test_suite(
    rag_function=rag_function,
    document_content=dokuman_iceriği
)
```

### Özel Test Soruları

Kendi test sorularınızı kullanabilirsiniz:

```python
# Test sorularınızı hazırlayın
test_questions = {
    "relevant": [
        {
            "question": "Yapay zeka nedir?",
            "expected_answer": "Ders notlarındaki tanım",
            "topic": "Temel kavramlar"
        }
    ],
    "irrelevant": [
        {
            "question": "Mars'ta yaşam var mı?",
            "expected_answer": "Bilmiyorum",
            "reason": "Konu ders notlarında yok"
        }
    ]
}

# Tester'ı başlat ve test et
tester = create_giskard_tester("glm-4.7", "zai")
results = tester.run_test_suite(
    rag_function=rag_function,
    document_content=SAMPLE_DOCUMENT
)

# Kendi sorularınızı kullanarak değerlendirin
for q in test_questions["relevant"]:
    answer = rag_function(q["question"])
    evaluation = tester.evaluate_response(
        q["question"],
        answer,
        q["expected_answer"],
        "relevant"
    )
```

---

## Özet

Giskard entegrasyonu ile AkıllıRehber sisteminizi şu şekilde test edebilirsiniz:

1. **Kurulum**: `.env` dosyasına API anahtarınızı ekleyin
2. **Test**: `python test_giskard_alpha.py` komutunu çalıştırın
3. **Sonuç**: JSON dosyasını inceleyin ve raporu okuyun
4. **İyileştirme**: Halüsinasyon oranı yüksekse prompt'ları güçlendirin

### Başarı Kriterleri

✅ **Mükemmel**: Genel skor > 90%, Halüsinasyon < 10%  
✅ **İyi**: Genel skor > 80%, Halüsinasyon < 20%  
⚠️ **Geliştirme Gerekiyor**: Genel skor < 80% veya Halüsinasyon > 30%

---

## Ek Kaynaklar

- [Giskard Resmi Dokümantasyon](https://docs.giskard.ai/)
- [RAG Test En İyi Uygulamaları](https://docs.giskard.ai/en/latest/guides/rag/)
- [Proje GitHub](https://github.com/Giskard-AI/giskard)

---

**Not:** Bu entegrasyon, Giskard'ın temel test mantığını kullanır ancak tam kütüphane bağımlılığı gerektirmez. Bu sayede projenize ek bağımlılık eklemeden test yapabilirsiniz.
