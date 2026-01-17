# 3. Metodoloji

## 3.1 Genel Bakış

Bu çalışmada, Türkçe eğitim içeriği için RAG (Retrieval-Augmented Generation) sistemlerinin performansını artırmayı amaçlıyoruz. Odak noktamız, RAG pipeline'ının ürettiği bağlamsal bilginin ilgililik ve kalitesini maksimize etmektir. Bu hedefe ulaşmak için, RAG mimarisinin üç temel bileşenini sistematik olarak optimize ettik:

1. **Hibrit Arama Stratejisi:** BM25 (anahtar kelime tabanlı) ve vektör aramasını (semantik) birleştiren hibrit arama yaklaşımını kullandık ve alpha parametresini optimize ederek optimal dengeyi belirledik.

2. **Chunking Stratejisi Optimizasyonu:** Veri işleme aşamasında doküman bağlamını zenginleştirmek için iki temel chunking yöntemini karşılaştırdık: recursive chunking ve semantic chunking.

3. **Sistem Promptu Optimizasyonu:** LLM'in eğitim bağlamına uygun, doğru ve pedagojik açıdan uygun yanıtlar üretmesini sağlamak için sistem promptunu optimize ettik.

4. **Reranker Entegrasyonu:** İlk aşama sonuçlarına dayanarak, retrieval kalitesini daha da artırmak için reranker bileşenini sisteme entegre ettik ve etkisini değerlendirdik.

**Deneysel Süreç:**

Sistemimizi geliştirirken iteratif bir yaklaşım izledik:

- **Aşama 1:** Hibrit arama alpha değerlerini (0.0, 0.2, 0.5, 0.8, 1.0) sistematik olarak test ettik
- **Aşama 2:** RAGAS framework'ü ile her konfigürasyonu değerlendirdik
- **Aşama 3:** Alpha = 0.8 değerinin en iyi performansı gösterdiğini tespit ettik
- **Aşama 4:** Reranker bileşenini aktif ederek testi tekrarladık
- **Aşama 5:** Sonuçları karşılaştırmalı olarak analiz ettik

Bu metodoloji, Türkçe eğitim içeriği için optimal RAG konfigürasyonunun ampirik olarak belirlenmesini sağlamıştır. Aşağıdaki bölümlerde, her bir bileşen ve deney aşaması detaylı olarak açıklanmaktadır.

---

## 3.2 Veri Seti ve Alan

Deneysel çalışmalarımızı, Milli Eğitim Bakanlığı (MEB) tarafından yayınlanan resmi "Bilişim Teknolojileri Temelleri" müfredat dokümanı kullanarak gerçekleştirdik. Bu doküman, gerçek dünya eğitim alanını temsil eden, alana özgü Türkçe terminoloji ve pedagojik içerik içermektedir.

**Doküman Özellikleri:**
- **Kaynak:** MEB Resmi Müfredat Dokümanı (PDF formatı)
- **Kapsam:** Bilişim teknolojileri temel konuları
- **Dil:** Türkçe
- **Hedef Kitle:** [TODO: Sınıf seviyesi - örn. 9. sınıf öğrencileri]
- **Sayfa Sayısı:** [TODO: Ekleyin]

**Kapsanan Konular:**
- Donanım bileşenleri ve mimarisi
- Yazılım türleri ve işletim sistemleri
- Ağ teknolojileri ve protokoller
- Veri güvenliği ve gizlilik
- Dijital okuryazarlık ve etik
- [TODO: Diğer ilgili konuları ekleyin]

Bu dokümanın seçilmesinin nedeni, standartlaştırılmış eğitim içeriği sunması ve Türkçe eğitim bağlamında RAG sistemlerinin değerlendirilmesi için ideal bir test ortamı oluşturmasıdır.


---

## 3.3 RAG Sistemi Konfigürasyonu

### 3.3.1 Doküman İşleme Bileşenleri

**Chunking Stratejisi Karşılaştırması**

Doküman parçalama (chunking) işlemi için iki farklı stratejiyi karşılaştırdık:

**1. Recursive Chunking (Özyinelemeli Parçalama)**

Recursive chunking, metni hiyerarşik ayırıcılar kullanarak parçalar. Önce paragraf düzeyinde, sonra cümle düzeyinde, en son kelime düzeyinde bölme yapar.

Özellikler:
- Doğal metin yapısını korur (paragraf → cümle → kelime)
- Sabit boyut hedefi ile çalışır
- Overlap (örtüşme) desteği sunar
- Genel amaçlı, çoğu doküman tipine uygun

Avantajları:
- Basit ve hızlı implementasyon
- Tutarlı chunk boyutları
- İyi dokümante edilmiş (LangChain [45])

**2. Semantic Chunking (Anlamsal Parçalama)**

Semantic chunking, metni anlamsal olarak tutarlı birimlere ayırır. Embedding benzerliğine dayalı olarak ilgili cümleleri gruplar.

Özellikler:
- Anlamsal tutarlılığı önceliklendirir
- Değişken chunk boyutları
- Konu değişimlerini algılar
- Bağlamsal bütünlüğü korur

Avantajları:
- Kavramsal ilişkileri chunk içinde korur
- Embedding modellerinin anlamı yakalamak için yeterli bağlam sağlar
- İlgili bilgilerin birden fazla chunk'a bölünmesini önler
- Eğitim içeriğindeki doğal konu sınırlarıyla uyumludur

**Karşılaştırma Sonuçları:**

[TODO: Her iki yöntemle yapılan testlerin sonuçlarını ekleyin]

| Metrik | Recursive Chunking | Semantic Chunking | Kazanan |
|--------|-------------------|-------------------|---------|
| Context Recall | [TODO]% | [TODO]% | [TODO] |
| Context Precision | [TODO]% | [TODO]% | [TODO] |
| Faithfulness | [TODO]% | [TODO]% | [TODO] |
| Answer Relevance | [TODO]% | [TODO]% | [TODO] |

**Seçilen Yöntem:** [TODO: Hangi chunking yöntemi seçildi ve neden?]

Bu yaklaşım, özellikle müfredat dokümanları için etkili olmuştur çünkü bu dokümanlar genellikle belirli öğrenme hedeflerini kapsayan tutarlı bölümler halinde organize edilmiştir.

**Embedding Modeli: OpenAI text-embedding-3-small**

Metin vektörlerini oluşturmak için OpenAI'nin text-embedding-3-small modelini kullandık.

Model özellikleri:
- **Boyut:** 1536 dimensions
- **Dil Desteği:** Çok dilli (Türkçe dahil)
- **Performans:** Yüksek kaliteli semantik temsiller
- **Maliyet:** Uygun fiyatlı API erişimi

Bu modelin seçilme nedenleri:
1. Türkçe için güçlü çok dilli destek
2. Eğitim terminolojisini iyi yakalama kapasitesi
3. Dengeli boyut/performans oranı
4. Geniş kullanım ve doğrulanmış performans

**Vektör Veritabanı: Weaviate**

Chunk'ların vektör temsillerini depolamak ve aramak için Weaviate vektör veritabanını kullandık.

Weaviate'in temel özellikleri:
- **Hibrit Arama:** Vektör ve anahtar kelime aramasını birleştirir
- **Ölçeklenebilirlik:** Büyük doküman koleksiyonlarını destekler
- **Performans:** Hızlı benzerlik araması
- **Esneklik:** Özelleştirilebilir arama parametreleri


### 3.3.2 Hibrit Arama Konfigürasyonu

Weaviate'in hibrit arama özelliği, yoğun vektör araması (semantic search) ile seyrek anahtar kelime aramasını (BM25) birleştirir. Bu kombinasyon, alpha (α) parametresi ile kontrol edilir.

**Alpha Parametresi (α):**

Alpha değeri, vektör benzerliği ile anahtar kelime eşleşmesi arasındaki dengeyi belirler:

- **α = 0.0:** Saf BM25 (100% anahtar kelime tabanlı)
- **α = 0.2:** BM25 ağırlıklı hibrit (%80 anahtar kelime, %20 vektör)
- **α = 0.5:** Dengeli hibrit (%50 anahtar kelime, %50 vektör)
- **α = 0.8:** Vektör ağırlıklı hibrit (%80 vektör, %20 anahtar kelime) ✓ **En iyi performans**
- **α = 1.0:** Saf vektör araması (100% semantik)

**Deneysel Yaklaşım:**

Optimal alpha değerini belirlemek için sistematik bir deney süreci uyguladık:

1. Her bir alpha değeri için ayrı test çalıştırıldı
2. Aynı test seti tüm konfigürasyonlarda kullanıldı
3. Hem retrieval hem de generation metrikleri ölçüldü
4. Sonuçlar karşılaştırmalı olarak analiz edildi

Bu yaklaşım, Türkçe eğitim içeriği için en uygun arama stratejisinin ampirik olarak belirlenmesini sağladı.

### 3.3.3 Cevap Üretimi Bileşenleri

**Büyük Dil Modeli (LLM):**
- **Model:** [TODO: Kullandığınız LLM modelini ekleyin - örn. GPT-4, Gemini-1.5-Pro, Claude-3.5-Sonnet]
- **Temperature:** [TODO: Değeri ekleyin - örn. 0.7]
- **Max Tokens:** [TODO: Değeri ekleyin - örn. 1000]
- **Top-k Retrieval:** [TODO: Kaç chunk getiriliyor - örn. 5]

**Sistem Promptu:**

Sistem promptu, LLM'in eğitim asistanı rolünü üstlenmesi ve müfredata uygun yanıtlar üretmesi için tasarlandı:

```
[TODO: Sistem promptunuzu buraya ekleyin]

Örnek:
"Sen bir eğitim asistanısın. Verilen bağlam bilgilerini kullanarak 
öğrencilerin sorularını yanıtla. Yanıtlarını Türkçe ver.

Kurallar:
1. Sadece verilen bağlamdaki bilgileri kullan
2. Bağlamda olmayan bilgileri uydurma
3. Emin olmadığın konularda bunu belirt
4. Yanıtlarını açık ve anlaşılır tut
5. [Hedef sınıf seviyesi] öğrencilerine uygun dil kullan"
```

**RAG Pipeline:**

1. Kullanıcı sorusu alınır
2. Soru embedding'e dönüştürülür
3. Weaviate'te hibrit arama yapılır (α = 0.8)
4. En ilgili top-k chunk getirilir
5. Chunk'lar ve soru LLM'e gönderilir
6. LLM bağlama dayalı yanıt üretir
7. Yanıt kullanıcıya döndürülür


---

## 3.4 Değerlendirme Metrikleri

Sistemimizin performansını hem retrieval (bilgi getirme) hem de generation (cevap üretme) kalitesi açısından değerlendirmek için iki tamamlayıcı framework kullandık.

### 3.4.1 RAGAS Framework

RAGAS (Retrieval-Augmented Generation Assessment) [42], RAG sistemlerinin çok boyutlu değerlendirilmesi için geliştirilmiş bir framework'tür. Dört temel metrik içerir:

**1. Context Recall (Bağlam Geri Çağırma)**

Ground truth bilgisinin ne kadarının getirilen bağlamlarda yakalandığını ölçer.

```
Context Recall = |Ground Truth ∩ Retrieved Contexts| / |Ground Truth|
```

- **Yüksek değer:** Sistem ilgili tüm bilgiyi getirmiş
- **Düşük değer:** Önemli bilgiler kaçırılmış
- **Aralık:** 0.0 - 1.0

**2. Context Precision (Bağlam Kesinliği)**

Getirilen bağlamların soruyla ne kadar ilgili olduğunu ölçer.

```
Context Precision = |Relevant Contexts| / |Retrieved Contexts|
```

- **Yüksek değer:** Getirilen chunk'lar çok ilgili
- **Düşük değer:** Çok fazla alakasız bilgi getirilmiş
- **Aralık:** 0.0 - 1.0

**3. Faithfulness (Sadakat)**

Üretilen cevabın getirilen bağlamlara ne kadar sadık kaldığını ölçer. Halüsinasyon (uydurma) riskini değerlendirir.

```
Faithfulness = |Claims Supported by Context| / |Total Claims in Answer|
```

- **Yüksek değer:** Cevap tamamen bağlama dayalı
- **Düşük değer:** Cevap uydurma bilgi içeriyor
- **Aralık:** 0.0 - 1.0

**4. Answer Relevance (Cevap İlgisi)**

Üretilen cevabın kullanıcının sorusunu ne kadar iyi yanıtladığını ölçer.

```
Answer Relevance = Semantic Similarity(Question, Answer)
```

- **Yüksek değer:** Cevap soruyu tam olarak yanıtlıyor
- **Düşük değer:** Cevap konudan sapıyor
- **Aralık:** 0.0 - 1.0


### 3.4.2 Metin Benzerlik Metrikleri

RAGAS metriklerine ek olarak, cevap kalitesini ölçmek için yaygın kullanılan metin benzerlik metriklerini de kullandık.

**ROUGE Metrikleri [43]**

ROUGE (Recall-Oriented Understudy for Gisting Evaluation), n-gram örtüşmesini ölçen leksikal benzerlik metrikleridir.

**1. ROUGE-1 (Unigram Overlap)**

Üretilen cevap ile ground truth arasındaki tekli kelime örtüşmesini ölçer.

```
ROUGE-1 = |Unigrams in Both| / |Unigrams in Reference|
```

- Temel kelime düzeyinde benzerliği yakalar
- Kelime sırasını dikkate almaz

**2. ROUGE-2 (Bigram Overlap)**

İki kelimelik dizilerin örtüşmesini ölçer.

```
ROUGE-2 = |Bigrams in Both| / |Bigrams in Reference|
```

- Kelime sırasını kısmen dikkate alır
- Daha katı bir benzerlik ölçüsüdür

**3. ROUGE-L (Longest Common Subsequence)**

En uzun ortak alt dizinin uzunluğunu ölçer.

```
ROUGE-L = LCS(Generated, Reference) / Length(Reference)
```

- Yapısal ve sıralı benzerliği yakalar
- Cümle akışının korunmasını değerlendirir
- Eğitim içeriği için özellikle önemlidir

**BERTScore Metrikleri [44]**

BERTScore, embedding tabanlı semantik benzerlik metrikleridir. Yüzeysel kelime eşleşmesinin ötesinde anlam benzerliğini yakalar.

**1. BERTScore Precision (Kesinlik)**

Üretilen cevaptaki her kelimenin ground truth'ta semantik karşılığının olup olmadığını ölçer.

```
Precision = Σ max_similarity(word_generated, words_reference) / |Generated|
```

- Üretilen cevaptaki gereksiz bilgiyi tespit eder
- Yüksek değer: Tüm bilgi ground truth'ta var

**2. BERTScore Recall (Geri Çağırma)**

Ground truth'taki her kelimenin üretilen cevapte semantik karşılığının olup olmadığını ölçer.

```
Recall = Σ max_similarity(word_reference, words_generated) / |Reference|
```

- Ground truth'taki eksik bilgiyi tespit eder
- Yüksek değer: Tüm önemli bilgi kapsanmış

**3. BERTScore F1 (Harmonik Ortalama)**

Precision ve Recall'un harmonik ortalamasıdır.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- Dengeli bir genel performans göstergesidir
- Hem eksiklik hem de fazlalık cezalandırılır

**BERTScore'un Avantajları:**
- Parafraz ve farklı ifadeleri yakalayabilir
- Semantik eşdeğerliği değerlendirir
- Eğitim bağlamında önemlidir (öğrenciler farklı kelimelerle aynı anlamı ifade edebilir)


### 3.4.3 Metrik Seçim Gerekçesi

**Neden Bu Metrikler?**

1. **Çok Boyutlu Değerlendirme:** RAGAS retrieval kalitesini, metin metrikleri generation kalitesini ölçer
2. **Tamamlayıcılık:** Leksikal (ROUGE) ve semantik (BERTScore) benzerlik birlikte değerlendirilir
3. **Akademik Standart:** Her iki framework de literatürde yaygın kullanılır ve kabul görür
4. **Eğitim Uygunluğu:** Metrikler pedagojik kalite göstergelerine uygun (doğruluk, eksiksizlik, uygunluk)

**Metrik Yorumlama Rehberi:**

| Metrik Değeri | Yorum | Anlamı |
|---------------|-------|--------|
| ≥ 0.80 (80%) | Mükemmel | Çok yüksek kalite, güvenilir |
| 0.60 - 0.79 | İyi | Kabul edilebilir, iyileştirilebilir |
| < 0.60 (60%) | Zayıf | Önemli iyileştirme gerekli |

---

## 3.5 Test Veri Seti

### 3.5.1 Test Seti Oluşturma

Sistemin performansını değerlendirmek için kapsamlı bir test seti oluşturduk.

**Test Seti Özellikleri:**
- **Soru Sayısı:** [TODO: N soru - örn. 50 soru]
- **Kaynak:** MEB Bilişim Teknolojileri Temelleri müfredatı
- **Kapsam:** Müfredatın tüm ana konuları
- **Ground Truth:** Resmi müfredat dokümanından doğrudan alınan cevaplar

**Soru Kategorileri:**

1. **Tanım Soruları** ([TODO: %X])
   - Örnek: "RAM nedir?"
   - Temel kavramların tanımını sorar

2. **Karşılaştırma Soruları** ([TODO: %X])
   - Örnek: "RAM ve ROM arasındaki farklar nelerdir?"
   - İki veya daha fazla kavramı karşılaştırır

3. **Açıklama Soruları** ([TODO: %X])
   - Örnek: "TCP/IP protokol ailesi nasıl çalışır?"
   - Süreç veya mekanizma açıklaması gerektirir

4. **Uygulama Soruları** ([TODO: %X])
   - Örnek: "Güçlü bir parola nasıl oluşturulur?"
   - Pratik uygulama bilgisi gerektirir

**Zorluk Seviyeleri:**

- **Kolay:** Doğrudan tanım ve basit kavramlar ([TODO: %X])
- **Orta:** Karşılaştırma ve açıklama gerektiren ([TODO: %X])
- **Zor:** Çoklu kavram entegrasyonu gerektiren ([TODO: %X])

### 3.5.2 Ground Truth Hazırlama

Her soru için ground truth cevaplar şu şekilde hazırlandı:

1. **Doğrudan Alıntı:** MEB dokümanından ilgili bölüm belirlendi
2. **Doğrulama:** Eğitim uzmanı tarafından kontrol edildi
3. **Standardizasyon:** Tutarlı format ve uzunlukta düzenlendi
4. **Kalite Kontrolü:** Pedagojik uygunluk değerlendirildi

Bu süreç, değerlendirmenin güvenilirliğini ve eğitim standartlarına uygunluğunu garanti eder.


---

## 3.6 Deney Protokolü

### 3.6.1 Hibrit Arama Optimizasyonu Deneyi

**Amaç:** Türkçe eğitim içeriği için optimal alpha değerini belirlemek

**Hipotez:** Vektör ağırlıklı hibrit arama (yüksek alpha), saf vektör veya saf anahtar kelime aramasından daha iyi performans gösterecektir.

**Deney Tasarımı:**

1. **Bağımsız Değişken:** Alpha parametresi (α)
   - Seviyeler: 0.0, 0.2, 0.5, 0.8, 1.0

2. **Bağımlı Değişkenler:**
   - RAGAS metrikleri (4 metrik)
   - ROUGE metrikleri (3 metrik)
   - BERTScore metrikleri (3 metrik)

3. **Kontrol Değişkenleri:**
   - Aynı test seti
   - Aynı embedding modeli
   - Aynı LLM ve parametreler
   - Aynı chunking stratejisi
   - Aynı top-k değeri

**Prosedür:**

```
FOR each alpha_value in [0.0, 0.2, 0.5, 0.8, 1.0]:
    Weaviate'i alpha_value ile yapılandır
    
    FOR each question in test_set:
        1. Soruyu embedding'e dönüştür
        2. Hibrit arama yap (alpha = alpha_value)
        3. Top-k chunk'ları getir
        4. LLM ile cevap üret
        5. Metrikleri hesapla
    
    Aggregate metrikleri kaydet
    
Tüm alpha değerleri için sonuçları karşılaştır
En iyi performansı gösteren alpha'yı belirle
```

### 3.6.2 Veri Toplama

**Otomatik Metrik Hesaplama:**

Her test sorusu için otomatik olarak şunlar kaydedildi:
- Üretilen cevap
- Getirilen chunk'lar (top-k)
- RAGAS skorları
- ROUGE skorları
- BERTScore skorları
- Gecikme süresi (latency)

**Veri Saklama:**

Tüm sonuçlar veritabanında saklandı:
- Test ID
- Soru metni
- Ground truth
- Üretilen cevap
- Alpha değeri
- Tüm metrik skorları
- Timestamp
- Kullanılan model bilgileri

Bu yapı, sonradan detaylı analiz ve karşılaştırma yapılmasını sağladı.

### 3.6.3 İstatistiksel Analiz

**Karşılaştırma Yöntemi:**

1. **Tanımlayıcı İstatistikler:**
   - Her alpha değeri için ortalama, standart sapma
   - Min/max değerler
   - Dağılım grafikleri

2. **Karşılaştırmalı Analiz:**
   - Alpha değerleri arası metrik karşılaştırması
   - En iyi performans gösteren konfigürasyonun belirlenmesi
   - [TODO: İstatistiksel anlamlılık testi yapıldıysa ekleyin - örn. ANOVA, t-test]

3. **Görselleştirme:**
   - Bar chart: Her alpha için metrik skorları
   - Line graph: Alpha değişimine göre metrik trendi
   - Heatmap: Metrik korelasyonları


---

## 3.7 Uygulama Detayları

### 3.7.1 Teknik Altyapı

**Yazılım Bileşenleri:**
- **Backend Framework:** [TODO: FastAPI, Flask, vb.]
- **Frontend:** [TODO: React, Next.js, vb.]
- **Vektör DB:** Weaviate [TODO: versiyon]
- **LLM API:** [TODO: OpenAI, Google, vb.]
- **Embedding API:** OpenAI text-embedding-3-small
- **Değerlendirme:** RAGAS library, rouge-score, bert-score

**Donanım:**
- **Sunucu:** [TODO: Bulut sağlayıcı veya lokal]
- **İşlemci:** [TODO: CPU/GPU bilgisi]
- **Bellek:** [TODO: RAM miktarı]
- **Depolama:** [TODO: Disk kapasitesi]

### 3.7.2 Sistem Akışı

**1. Doküman Yükleme ve İşleme:**

```
PDF Doküman
    ↓
Metin Çıkarma (PDF parsing)
    ↓
Semantik Chunking
    ↓
Embedding Oluşturma (text-embedding-3-small)
    ↓
Weaviate'e Kaydetme
```

**2. Soru-Cevap Akışı:**

```
Kullanıcı Sorusu
    ↓
Soru Embedding'i Oluştur
    ↓
Weaviate Hibrit Arama (α = 0.8)
    ↓
Top-k Chunk Getir
    ↓
LLM'e Gönder (Chunk'lar + Soru + System Prompt)
    ↓
Cevap Üret
    ↓
Kullanıcıya Dön
```

**3. Değerlendirme Akışı:**

```
Test Sorusu + Ground Truth
    ↓
RAG Pipeline ile Cevap Üret
    ↓
RAGAS Metrikleri Hesapla
    ↓
ROUGE Metrikleri Hesapla
    ↓
BERTScore Metrikleri Hesapla
    ↓
Sonuçları Kaydet
```

### 3.7.3 Parametreler ve Ayarlar

**Chunking Parametreleri:**
- **Yöntem:** Semantik chunking
- **Chunk Boyutu:** [TODO: Ortalama kelime/karakter sayısı]
- **Overlap:** [TODO: Varsa overlap miktarı]

**Retrieval Parametreleri:**
- **Top-k:** [TODO: Kaç chunk getiriliyor - örn. 5]
- **Alpha:** 0.8 (optimal değer)
- **Min Relevance Score:** [TODO: Varsa minimum skor eşiği]

**Generation Parametreleri:**
- **Temperature:** [TODO: Değer - örn. 0.7]
- **Max Tokens:** [TODO: Değer - örn. 1000]
- **Top-p:** [TODO: Varsa]
- **Frequency Penalty:** [TODO: Varsa]


---

## 3.8 Kalite Güvencesi ve Doğrulama

### 3.8.1 Veri Kalitesi Kontrolleri

**Doküman İşleme Doğrulaması:**
1. PDF'den metin çıkarma doğruluğu kontrol edildi
2. Chunking sonuçları manuel olarak incelendi
3. Embedding kalitesi örnek sorgularla test edildi
4. Weaviate'e yükleme başarısı doğrulandı

**Test Seti Doğrulaması:**
1. Her soru-cevap çifti eğitim uzmanı tarafından gözden geçirildi
2. Ground truth cevapların müfredata uygunluğu kontrol edildi
3. Soru çeşitliliği ve kapsam analizi yapıldı
4. Zorluk seviyesi dengesi değerlendirildi

### 3.8.2 Sistem Güvenilirliği

**Tekrarlanabilirlik:**
- Tüm deney parametreleri dokümante edildi
- Rastgele seed değerleri sabitlendi (varsa)
- Aynı test seti tüm deneylerde kullanıldı
- Sonuçlar veritabanında saklandı

**Tutarlılık Kontrolleri:**
- Aynı soru birden fazla kez test edildi
- Sonuç varyansı analiz edildi
- Anormal sonuçlar incelendi
- [TODO: Varsa tutarlılık metrikleri ekleyin]

### 3.8.3 Etik Hususlar

**Veri Gizliliği:**
- Kullanılan doküman resmi ve açık kaynaklıdır
- Öğrenci verileri kullanılmamıştır
- Test soruları anonim ve genel niteliktedir

**Akademik Dürüstlük:**
- Tüm kaynaklar uygun şekilde atıf yapılmıştır
- Metodoloji şeffaf bir şekilde açıklanmıştır
- Sonuçlar manipüle edilmemiştir
- Sınırlamalar açıkça belirtilmiştir

---

## 3.9 Metodolojik Sınırlamalar

Bu çalışmanın metodolojik sınırlamaları şunlardır:

### 3.9.1 Kapsam Sınırlamaları

1. **Tek Alan Odağı:**
   - Sadece Bilişim Teknolojileri müfredatı kullanıldı
   - Diğer eğitim alanlarına genelleme sınırlı olabilir

2. **Tek Doküman:**
   - Tek bir resmi müfredat dokümanı kullanıldı
   - Çoklu kaynak entegrasyonu test edilmedi

3. **Dil:**
   - Sadece Türkçe içerik değerlendirildi
   - Çok dilli senaryolar kapsam dışı

### 3.9.2 Teknik Sınırlamalar

1. **Embedding Modeli:**
   - Sadece text-embedding-3-small test edildi
   - Türkçe'ye özel fine-tune edilmiş modeller denenmedi

2. **LLM Seçimi:**
   - [TODO: Tek bir LLM kullanıldıysa belirtin]
   - Farklı LLM'ler farklı sonuçlar verebilir

3. **Chunking Stratejisi:**
   - Sadece semantik chunking kullanıldı
   - Alternatif stratejiler (recursive, fixed-size) karşılaştırılmadı

### 3.9.3 Değerlendirme Sınırlamaları

1. **Otomatik Metrikler:**
   - RAGAS, ROUGE ve BERTScore kullanıldı
   - Pedagojik kalite tam olarak yakalanamayabilir
   - İnsan değerlendirmesi yapılmadı

2. **Test Seti Boyutu:**
   - [TODO: Test seti küçükse belirtin]
   - Daha büyük test setleri daha güvenilir sonuçlar verebilir

3. **Gerçek Kullanım:**
   - Kontrollü test ortamında değerlendirme yapıldı
   - Gerçek sınıf ortamında öğrenci testleri yapılmadı

### 3.9.4 Genelleştirilebilirlik

Bu metodoloji ve bulgular:
- Türkçe eğitim içeriği için geçerlidir
- Benzer yapıdaki müfredat dokümanlarına uygulanabilir
- Farklı diller ve alanlar için adaptasyon gerektirebilir
- Optimal parametreler (α = 0.8) alan-bağımlı olabilir

---

## 3.10 Özet

Bu bölümde, Türkçe eğitim içeriği için optimize edilmiş RAG sistemimizin metodolojisini detaylı olarak açıkladık. Temel yaklaşımımız:

1. **Veri:** MEB resmi müfredat dokümanı
2. **İşleme:** Semantik chunking + text-embedding-3-small
3. **Retrieval:** Weaviate hibrit arama (α optimizasyonu)
4. **Generation:** [LLM] ile bağlam-tabanlı cevap üretimi
5. **Değerlendirme:** RAGAS + ROUGE + BERTScore

Sistematik deneysel yaklaşımımız, **α = 0.8** değerinin Türkçe eğitim içeriği için optimal olduğunu göstermiştir. Bu bulgu, vektör-ağırlıklı hibrit aramanın semantik anlama ile anahtar kelime hassasiyetini dengeli bir şekilde birleştirdiğini ortaya koymaktadır.

Sonraki bölümde, bu metodoloji ile elde edilen deneysel sonuçları ve detaylı analizleri sunacağız.

---

## Kaynaklar

[42] Es, S., James, J., Espinosa-Anke, L., Schockaert, S.: RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv preprint (2023). https://arxiv.org/abs/2309.15217

[43] Lin, C.Y.: ROUGE: a package for automatic evaluation of summaries. In: Proceedings of the Meeting of the Association for Computational Linguistics, Barcelona, Spain, pp. 74–81 (2004)

[44] Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q., Artzi, Y.: BERTScore: Evaluating Text Generation with BERT. arXiv preprint (2019). https://arxiv.org/abs/1904.09675

[TODO: Diğer referansları ekleyin]

---

## Ekler

### Ek A: Sistem Prompt Örnekleri

[TODO: Kullandığınız sistem promptlarını buraya ekleyin]

### Ek B: Test Soruları Örnekleri

[TODO: Örnek test sorularını buraya ekleyin]

### Ek C: Chunking Örnekleri

[TODO: Örnek chunk'ları buraya ekleyin]

### Ek D: Teknik Konfigürasyon Dosyaları

[TODO: Varsa config dosyalarını buraya ekleyin]
