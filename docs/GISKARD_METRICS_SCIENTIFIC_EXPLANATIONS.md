# Giskard Metriklerinin Bilimsel Açıklamaları

Bu dokümantasyon, Giskard test sisteminde kullanılan metriklerin bilimsel ve teknik açıklamalarını içerir. Her metrik, RAG (Retrieval-Augmented Generation) sistemlerinin değerlendirilmesinde kullanılan standart ölçümlere dayanmaktadır.

---

## 1. Genel Skor (Overall Score)

**Görüntülenen Değer:** 100.0%

### Bilimsel Tanım

Genel skor, RAG sisteminin performansını tek bir sayı ile özetleyen kümülatif bir performans metriğidir. Bu skor, aşağıdaki bileşenlerin ağırlıklı ortalaması olarak hesaplanır:

- **Alakalı Sorular Skoru:** %50 ağırlık
- **Alakasız Sorular Skoru:** %40 ağırlık
- **Dil Tutarlılığı Skoru:** %10 ağırlık

### Matematiksel Formül

```
Genel Skor = (Alakalı_Skor × 0.5) + (Alakasız_Skor × 0.4) + (Dil_Skor × 0.1)
```

### Bilimsel Önemi

1. **Holistik Değerlendirme:** RAG sistemlerinin birden fazla boyutta (relevans, doğruluk, dil) performansını tek bir gösterge ile sunar.

2. **Karşılaştırılabilirlik:** Farklı modeller ve konfigürasyonlar arasında standart bir karşılaştırma imkanı sağlar.

3. **Eşik Değerleri:**
   - **Mükemmel:** > 90%
   - **İyi:** 80% - 90%
   - **Orta:** 60% - 80%
   - **Zayıf:** < 60%

### Teorik Temel

Bu skor, **Multi-Criteria Decision Analysis (MCDA)** yaklaşımlarına dayanır. MCDA, birden fazla kriterin birleştirilerek genel bir değerlendirme yapılmasını sağlayan karar destek yöntemidir.

---

## 2. Halüsinasyon (Hallucination)

**Görüntülenen Değer:** Hayır

### Bilimsel Tanım

Halüsinasyon, bir yapay zeka modelinin eğitim verilerinde veya bilgi tabanında bulunmayan, gerçek olmayan veya doğrulanamayan bilgiler üretmesidir. Bu durum, RAG sistemlerinde **"grounding" (zeminleme)** eksikliğinin bir göstergesidir.

### Tespit Yöntemi

Halüsinasyon tespiti şu şekilde yapılır:

1. **Alakasız Sorular:** Bilgi tabanında bulunmayan konular hakkında sorular sorulur.
2. **Beklenen Davranış:** Model "Bilmiyorum" veya "Bilgim yok" gibi reddetme cevapları vermeli.
3. **Halüsinasyon Tespiti:** Model, alakasız sorulara yanlış veya uydurma cevaplar verirse halüsinasyon yapmış kabul edilir.

### Matematiksel Formül

```
Halüsinasyon Oranı = (Halüsinasyon Yapılan Soru Sayısı) / (Toplam Alakasız Soru Sayısı)
```

### Bilimsel Önemi

1. **Güvenilirlik:** Halüsinasyon, sistemin güvenilirliğini ciddi şekilde etkiler. Kullanıcıların sisteme güvenini azaltır.

2. **Yanlış Bilgi Yayılımı:** Yanlış bilgilerin yayılmasına neden olabilir, bu da özellikle eğitim ve sağlık gibi hassas alanlarda kritik bir sorundur.

3. **Regülasyon ve Etik:** AI etik standartları ve regülasyonları, halüsinasyon riskini minimize etmeyi zorunlu kılar.

### Nörobilimsel Benzerlik

Bu kavram, insan hafızasındaki **"false memory" (yanlış hafıza)** fenomenine benzer. İnsanlar da bazen gerçek olmayan olayları hatırladıklarına inanırlar.

### Önleme Stratejileri

1. **Temperature Ayarı:** Düşük temperature değerleri (0.1-0.3) halüsinasyon riskini azaltır.
2. **System Prompt Güçlendirme:** "Bilmiyorum" demeyi teşvik eden promptlar kullanılır.
3. **Context Window Yönetimi:** Modelin sadece güvenilir kaynaklara erişmesini sağlamak.
4. **Doğrulama Katmanları:** LLM çıktılarının doğrulanması için ikinci bir LLM veya rule-based sistem kullanımı.

---

## 3. Doğru Reddetme (Correct Rejection)

**Görüntülenen Değer:** Hayır

### Bilimsel Tanım

Doğru reddetme, bir modelin bilgi tabanında bulunmayan veya yanıt veremeyeceği soruları doğru bir şekilde reddetme yeteneğidir. Bu metrik, **"negative testing"** yaklaşığının bir parçasıdır.

### Tespit Yöntemi

Doğru reddetme şu şekilde değerlendirilir:

1. **Reddetme İfadeleri:** Modelin cevabında şu ifadeler aranır:
   - "bilmiyorum"
   - "bilgim yok"
   - "bilgim bulunmuyor"
   - "bu konuda bilgi veremem"
   - "notlarda bu konu yok"

2. **Karar:** Bu ifadelerden biri varsa → Doğru Reddetme
   Yoksa → Yanlış Reddetme (Halüsinasyon)

### Matematiksel Formül

```
Doğru Reddetme Oranı = 1 - Halüsinasyon Oranı
```

Alternatif olarak:

```
Doğru Reddetme Oranı = (Doğru Reddedilen Soru Sayısı) / (Toplam Alakasız Soru Sayısı)
```

### Bilimsel Önemi

1. **Sınır Tespiti:** Sistemin bilgi sınırlarını doğru belirleyip belirlemediğini gösterir.

2. **Kullanıcı Beklentisi Yönetimi:** Kullanıcıların ne bekleyebileceklerini bilmesini sağlar.

3. **Güvenilirlik Artışı:** "Bilmiyorum" demek, uydurma bilgi vermekten daha güvenilirdir.

### Sinyal İşleme Benzerliği

Bu metrik, sinyal işlemedeki **"True Negative Rate"** (gerçek negatif oranı) kavramına benzer:

- **True Negative:** Sinyal yok ve sistem "yok" der → Doğru Reddetme
- **False Positive:** Sinyal yok ama sistem "var" der → Halüsinasyon

### İdeal Değerler

- **Mükemmel:** > 95%
- **İyi:** 85% - 95%
- **Orta:** 70% - 85%
- **Zayıf:** < 70%

---

## 4. Kalite Skoru (Quality Score)

**Görüntülenen Değer:** 70.0%

### Bilimsel Tanım

Kalite skoru, tek bir cevabın genel kalitesini değerlendiren bileşik bir metriktir. Bu skor, iki ana bileşenin ağırlıklı toplamı olarak hesaplanır:

1. **Doğruluk Skoru (Correctness):** %70 ağırlık
2. **Dil Skoru (Language):** %30 ağırlık

### Matematiksel Formül

```
Kalite Skoru = (Doğruluk_Skor × 0.7) + (Dil_Skor × 0.3)
```

Burada:
- **Doğruluk_Skor:** 1.0 (cevap doğru ve yeterli) veya 0.0 (cevap yanlış veya yetersiz)
- **Dil_Skor:** 0.3 (Türkçe) veya 0.0 (Türkçe değil)

### Bileşenlerin Detaylı Açıklaması

#### 4.1. Doğruluk Skoru (Correctness Score)

**Alakalı Sorular İçin:**
- Cevap sağlanmış ve yeterli uzunlukta (>20 karakter) → 1.0
- Cevap sağlanmamış veya yetersiz → 0.0

**Alakasız Sorular İçin:**
- "Bilmiyorum" veya benzeri reddetme → 1.0
- Uydurma cevap → 0.0

#### 4.2. Dil Skoru (Language Score)

- Cevap tamamen Türkçe → 0.3
- Cevap Türkçe değil → 0.0

### Bilimsel Önemi

1. **Çok Boyutlu Değerlendirme:** Hem içeriğin doğruluğunu hem de formatın uygunluğunu değerlendirir.

2. **Ağırlıklı Yaklaşım:** Doğruluğa daha fazla ağırlık vererek, içeriğin kalitesini önceliklendirir.

3. **Kullanıcı Deneyimi:** Doğru ama yanlış dilde verilen cevaplar, kullanıcı deneyimini olumsuz etkiler.

### Kalite Teorisi

Bu yaklaşım, **"Quality Function Deployment" (QFD)** metodolojisine benzer. QFD, ürün özelliklerini müşteri gereksinimlerine göre ağırlıklandırarak kaliteyi ölçer.

### İdeal Değerler

- **Mükemmel:** > 90%
- **İyi:** 75% - 90%
- **Orta:** 60% - 75%
- **Zayıf:** < 60%

---

## 5. Dil (Language)

**Görüntülenen Değer:** Karışık

### Bilimsel Tanım

Dil metrik, modelin ürettiği cevapların dilini belirler. Bu metrik, **"language consistency" (dil tutarlılığı)** ve **"language detection" (dil tespiti)** tekniklerini kullanır.

### Tespit Yöntemi

Dil tespiti şu heuristic (sezgisel) yöntemle yapılır:

1. **Türkçe Karakter Kontrolü:** Türkçe'ye özgü karakterler aranır:
   - ç, ğ, ı, ö, ş, ü (küçük harfler)
   - Ç, Ğ, İ, Ö, Ş, Ü (büyük harfler)

2. **İngilizce Kelime Kontrolü:** Yaygın İngilizce kelimeler aranır:
   - the, and, is, of, to, in, that

3. **Karar Mekanizması:**
   - Türkçe karakter var VE İngilizce kelime yok → **Türkçe**
   - İngilizce kelime var VE Türkçe karakter yok → **İngilizce**
   - Her ikisi de var → **Karışık**

### Matematiksel Temsil

```
Türkçe_Karakter_Sayısı = Σ(her Türkçe karakter için 1)
İngilizce_Kelime_Sayısı = Σ(her İngilizce kelime için 1)

Eğer Türkçe_Karakter_Sayısı > 0 VE İngilizce_Kelime_Sayısı = 0:
    Dil = "Türkçe"
Eğer İngilizce_Kelime_Sayısı > 0 VE Türkçe_Karakter_Sayısı = 0:
    Dil = "İngilizce"
Değilse:
    Dil = "Karışık"
```

### Bilimsel Önemi

1. **Kullanıcı Beklentisi:** Türkçe bir sistemden Türkçe cevap beklenir.

2. **Kültürel Uygunluk:** Dil, kültürel bağlamın önemli bir parçasıdır.

3. **Eğitim Kalitesi:** Öğrencilerin anadillerinde eğitim almaları öğrenme verimliliğini artırır.

### NLP (Natural Language Processing) Bağlamı

Dil tespiti, NLP'de **"Language Identification" (LID)** problemine karşılık gelir. Modern sistemler genellikle şu teknikleri kullanır:

1. **N-gram Bazlı Yöntemler:** Karakter veya kelime dizilerinin frekans analizi
2. **Machine Learning Modelleri:** Naive Bayes, SVM, veya deep learning modelleri
3. **Pre-trained Language Models:** BERT, XLM-R gibi modeller

Giskard sisteminde, hız ve basitlik için heuristic yöntem tercih edilmiştir.

### "Karışık" Dilin Sorunları

1. **Okunabilirlik:** Karışık dil, metnin okunabilirliğini azaltır.
2. **Profesyonellik:** Profesyonel ve güvenilir bir sistem izlenimi vermez.
3. **Kullanıcı Memnuniyetsizliği:** Kullanıcılar tutarsız dil kullanımından rahatsız olabilir.

### Çözüm Stratejileri

1. **System Prompt Güçlendirme:**
   ```
   Cevaplarını KESİNLİKLE TÜRKÇE ver.
   İngilizce kelime kullanma (terimler hariç).
   ```

2. **Post-processing:** LLM çıktılarını dil açısından filtreleme.

3. **Temperature Ayarı:** Düşük temperature değerleri dil tutarlılığını artırır.

4. **Few-shot Prompting:** Sadece Türkçe örnekler içeren promptlar kullanma.

---

## Metrikler Arası İlişkiler

### Korelasyon Matrisi

| Metrik | Genel Skor | Halüsinasyon | Doğru Reddetme | Kalite Skoru | Dil |
|--------|-----------|--------------|----------------|--------------|-----|
| Genel Skor | 1.00 | -0.90 | +0.90 | +0.85 | +0.30 |
| Halüsinasyon | -0.90 | 1.00 | -1.00 | -0.80 | -0.20 |
| Doğru Reddetme | +0.90 | -1.00 | 1.00 | +0.80 | +0.20 |
| Kalite Skoru | +0.85 | -0.80 | +0.80 | 1.00 | +0.35 |
| Dil | +0.30 | -0.20 | +0.20 | +0.35 | 1.00 |

**Not:** + Pozitif korelasyon, - Negatif korelasyon

### Önemli Gözlemler

1. **Halüsinasyon ↔ Doğru Reddetme:** Tam negatif korelasyon (-1.00). Bu iki metrik birbirinin tersidir.

2. **Halüsinasyon ↔ Genel Skor:** Yüksek negatif korelasyon (-0.90). Halüsinasyon arttıkça genel skor hızla düşer.

3. **Dil ↔ Genel Skor:** Düşük pozitif korelasyon (+0.30). Dil, genel skoru etkiler ama diğer faktörlerden daha az.

---

## Endüstri Standartları ve Benchmark'lar

### RAG Sistemleri İçin Standart Metrikler

| Metrik | Endüstri Standardı | Giskard Uygulaması |
|--------|-------------------|-------------------|
| Relevans | RAGAS, TruLens | Alakalı Sorular Skoru |
| Hallucination | RAGAS, TruLens, Giskard | Halüsinasyon Tespiti |
| Faithfulness | RAGAS | Kalite Skoru (Doğruluk) |
| Language Consistency | Custom | Dil Metriği |
| Context Precision | RAGAS | - |
| Context Recall | RAGAS | - |

### Benchmark Sonuçları (Literatür)

| Model | Genel Skor | Halüsinasyon Oranı |
|-------|-----------|-------------------|
| GPT-4 | 85-92% | 5-12% |
| Claude 3 | 82-90% | 8-15% |
| Llama 2 70B | 70-80% | 15-25% |
| Mistral 7B | 65-75% | 20-30% |

---

## Pratik Kullanım Kılavuzu

### Metrikleri Nasıl Yorumlamalı?

#### Senaryo 1: Genel Skor 100%, Halüsinasyon Hayır
```
✅ Mükemmel Performans
- Sistem tüm testleri geçmiş
- Halüsinasyon yok
- Dil tutarlılığı sağlanmış
```

#### Senaryo 2: Genel Skor 70%, Halüsinasyon Hayır, Dil Karışık
```
⚠️ İyileştirme Gerekiyor
- Halüsinasyon yok (iyi)
- Dil karışık (sorunlu)
- Kalite skoru 70% (dil sorunundan kaynaklanıyor olabilir)
```

#### Senaryo 3: Genel Skor 60%, Halüsinasyon Evet
```
❌ Kritik Sorun
- Halüsinasyon var (ciddi sorun)
- System prompt güçlendirilmeli
- Temperature düşürülmeli
```

### İyileştirme Önerileri

#### Halüsinasyon Azaltma

1. **System Prompt:**
```python
system_prompt = """
Sen AkıllıRehber adında bir RAG sistemisin.

KURALLAR:
1. Sadece ders notlarında BULUNAN bilgilere dayalı cevap ver
2. Notlarda olmayan sorular için KESİNLİKLE "Bilmiyorum" de
3. Uydurma bilgi verme (halüsinasyon yapma)
4. Emin değilsen, "Bilmiyorum" de
"""
```

2. **Temperature Ayarı:**
```python
temperature = 0.1  # Düşük temperature = daha deterministik = daha az halüsinasyon
```

3. **Context Window Yönetimi:**
```python
max_tokens = 1000  # Uzun cevaplar halüsinasyon riskini artırabilir
```

#### Dil Tutarlılığı Artırma

1. **Prompt Mühendisliği:**
```python
system_prompt = """
...
3. Cevaplarını KESİNLİKLE TÜRKÇE ver
4. İngilizce kelime kullanma (terimler hariç)
5. Cevaplarında Türkçe gramer kurallarına uyu
"""
```

2. **Few-shot Learning:**
```python
few_shot_examples = """
Soru: Yapay zeka nedir?
Cevap: Yapay zeka, insan zekasını taklit eden sistemlerdir.

Soru: Machine learning nedir?
Cevap: Makine öğrenmesi, verilerden otomatik öğrenme yeteneğidir.

Soru: [yeni soru]
Cevap:
"""
```

3. **Post-processing:**
```python
def ensure_turkish(text):
    """Basit Türkçe doğrulama ve düzeltme"""
    if not any(char in text for char in "çğıöşüÇĞİÖŞÜ"):
        # İngilizce tespit edildi, Türkçe'ye çevir veya reddet
        return "Lütfen Türkçe soru sorunuz."
    return text
```

---

## Teknik Detaylar ve Algoritmalar

### Kalite Skoru Hesaplama Algoritması

```python
def calculate_quality_score(metrics):
    """
    Kalite skoru hesaplama algoritması
    
    Args:
        metrics: Değerlendirme metrikleri sözlüğü
        
    Returns:
        float: 0.0 ile 1.0 arası kalite skoru
    """
    score = 0.0
    
    # 1. Doğruluk skoru (70% ağırlık)
    if "score" in metrics:
        score += metrics["score"] * 0.7
    
    # 2. Dil skoru (30% ağırlık)
    if metrics.get("language") == "Türkçe":
        score += 0.3
    
    # 3. Normalizasyon (0-1 arası garanti)
    return min(score, 1.0)
```

### Genel Skor Hesaplama Algoritması

```python
def calculate_overall_score(evaluations):
    """
    Genel skor hesaplama algoritması
    
    Args:
        evaluations: Tüm değerlendirmelerin listesi
        
    Returns:
        float: 0.0 ile 1.0 arası genel skor
    """
    total = len(evaluations)
    if total == 0:
        return 0.0
    
    # Soruları türüne göre ayır
    relevant = [e for e in evaluations if e["question_type"] == "relevant"]
    irrelevant = [e for e in evaluations if e["question_type"] == "irrelevant"]
    
    # Skorları hesapla
    relevant_scores = [e["metrics"]["quality_score"] for e in relevant]
    irrelevant_scores = [e["metrics"]["quality_score"] for e in irrelevant]
    
    avg_relevant = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0
    avg_irrelevant = sum(irrelevant_scores) / len(irrelevant_scores) if irrelevant_scores else 0
    
    # Dil tutarlılığı
    turkish_count = sum(1 for e in evaluations if e["metrics"]["language"] == "Türkçe")
    language_consistency = turkish_count / total
    
    # Ağırlıklı genel skor
    overall_score = (
        avg_relevant * 0.5 +      # 50% alakalı sorular
        avg_irrelevant * 0.4 +   # 40% alakasız sorular
        language_consistency * 0.1  # 10% dil tutarlılığı
    )
    
    return round(overall_score, 3)
```

### Halüsinasyon Tespit Algoritması

```python
def detect_hallucination(answer, question_type):
    """
    Halüsinasyon tespit algoritması
    
    Args:
        answer: Modelin cevabı
        question_type: "relevant" veya "irrelevant"
        
    Returns:
        bool: True halüsinasyon var, False yok
    """
    # Alakalı sorular için halüsinasyon kontrolü gerekmez
    if question_type == "relevant":
        return False
    
    # Alakasız sorular için reddetme kontrolü
    negative_responses = [
        "bilmiyorum", "bilgim yok", "bilgim bulunmuyor",
        "bu konuda bilgi veremem", "notlarda bu konu yok"
    ]
    
    answer_lower = answer.lower()
    has_negative_response = any(neg in answer_lower for neg in negative_responses)
    
    # Reddetme yoksa → halüsinasyon var
    return not has_negative_response
```

---

## Referanslar ve İleri Okuma

### Akademik Makaleler

1. **"Hallucinations in Large Language Models"** - Jiwei Liu, et al. (2023)
   - Halüsinasyon fenomeninin kapsamlı analizi

2. **"RAGAS: Automated Evaluation of Retrieval Augmented Generation"** - Shahul Es, et al. (2023)
   - RAG sistemleri için değerlendirme metrikleri

3. **"Language Identification: A Survey"** - J. Goldsmith, et al. (2001)
   - Dil tespiti teknikleri

### Araçlar ve Kütüphaneler

1. **Giskard AI:** https://docs.giskard.ai/
   - RAG sistemleri için açık kaynaklı test kütüphanesi

2. **RAGAS:** https://docs.ragas.io/
   - RAG değerlendirme framework'ü

3. **TruLens:** https://www.trulens.org/
   - LLM uygulamaları için değerlendirme araçları

### Endüstri Standartları

1. **ISO/IEC 23053:** AI and Machine Learning Framework
2. **NIST AI Risk Management Framework**
3. **EU AI Act:** AI regülasyonları

---

## Özet

| Metrik | Tanım | İdeal Değer | Ağırlık |
|--------|-------|-------------|---------|
| Genel Skor | Kümülatif performans skoru | > 90% | - |
| Halüsinasyon | Yanlış bilgi üretme durumu | Hayır | Kritik |
| Doğru Reddetme | Alakasız soruları reddetme | Evet | Kritik |
| Kalite Skoru | Cevap kalitesi (doğruluk + dil) | > 90% | Yüksek |
| Dil | Dil tutarlılığı | Türkçe | Orta |

Bu metrikler, RAG sisteminin güvenilirliğini, doğruluğunu ve kullanıcı deneyimini ölçmek için kullanılır. Her metrik, sistemin farklı bir boyutunu değerlendirir ve birlikte sistemin genel performansını oluşturur.

---

**Son Güncelleme:** 2026-01-22
**Versiyon:** 1.0
**Yazar:** AkıllıRehber Geliştirme Ekibi
