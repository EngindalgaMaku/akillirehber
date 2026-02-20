# 🧪 Deney Tekrarlama Betikleri

Bu dizin, AkıllıRehber makalesinde sunulan deneysel sonuçların tekrarlanabilmesi için gerekli betikleri içermektedir.

## Ön Koşullar

1. AkıllıRehber sistemi çalışır durumda olmalıdır (Docker Compose ile):
   ```bash
   docker-compose up -d
   ```

2. Python bağımlılıkları:
   ```bash
   pip install -r requirements.txt
   ```

3. `.env` dosyasında API anahtarları tanımlı olmalıdır:
   ```
   OPENROUTER_API_KEY=sk-or-...
   SECRET_KEY=...
   ```

4. Bloom veri seti yüklenmiş olmalıdır:
   - Veri seti: https://github.com/EngindalgaMaku/Bilisim-Teknolojileri-Bloom-Dataset

## Betikler

### 1. RAGAS Değerlendirmesi (`run_ragas_evaluation.py`)

Bloom veri setindeki 100 soru üzerinde RAGAS metriklerini hesaplar:
- Faithfulness (Sadakat)
- Answer Relevancy (Yanıt İlgililiği)
- Context Precision (Bağlam Hassasiyeti)
- Context Recall (Bağlam Duyarlılığı)
- Answer Correctness (Yanıt Doğruluğu)

```bash
python run_ragas_evaluation.py --course-id 1 --test-set-id 1
```

### 2. ROUGE ve BERTScore Değerlendirmesi (`run_rouge_bertscore.py`)

Aynı veri seti üzerinde metin benzerliği metriklerini hesaplar:
- ROUGE-1, ROUGE-2, ROUGE-L
- BERTScore (Precision, Recall, F1)

```bash
python run_rouge_bertscore.py --course-id 1 --test-set-id 1
```

### 3. RAG vs Direct LLM Karşılaştırması (`run_rag_vs_directllm.py`)

RAG tabanlı yanıtlar ile yalın LLM yanıtlarını karşılaştırır:

```bash
python run_rag_vs_directllm.py --course-id 1 --test-set-id 1
```

### 4. PII Filtreleme Performans Değerlendirmesi (`run_pii_evaluation.py`)

KVKK uyumlu kişisel bilgi filtreleme katmanının precision/recall analizini yapar:
- Katman 1 (Regex): TC kimlik, telefon, e-posta, IBAN, kredi kartı, pasaport
- Katman 2 (Few-Shot Embedding): Şifre, adres, doğum tarihi gibi belirsiz durumlar

```bash
# Sadece regex katmanı (offline, API gerektirmez)
python run_pii_evaluation.py

# Tam test (regex + embedding, API gerektirir, PII filtresi açık olmalı)
python run_pii_evaluation.py --with-api --course-id 1
```

### 5. Tüm Deneyleri Çalıştır (`run_all_experiments.py`)

Yukarıdaki üç deneyi sırasıyla çalıştırır ve sonuçları `results/` dizinine kaydeder:

```bash
python run_all_experiments.py --course-id 1 --test-set-id 1
```

## Çıktılar

Sonuçlar `results/` dizinine JSON ve CSV formatında kaydedilir:
- `results/ragas_results.json` — RAGAS metrikleri
- `results/rouge_bertscore_results.json` — ROUGE ve BERTScore metrikleri
- `results/rag_vs_directllm_results.json` — Karşılaştırma sonuçları
- `results/pii_evaluation_results.json` — PII filtreleme precision/recall
- `results/summary.csv` — Tüm metriklerin özet tablosu

## Deney Konfigürasyonu

Varsayılan konfigürasyon `experiment_config.json` dosyasında tanımlıdır. Özelleştirmek için bu dosyayı düzenleyebilirsiniz.
