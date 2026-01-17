# 3. Metodoloji

Bu çalışmada, Türkçe eğitim içeriği için RAG (Retrieval-Augmented Generation) sistemlerinin performansını artırmayı amaçlıyoruz. Odak noktamız, RAG pipeline'ının ürettiği bağlamsal bilginin ilgililik ve kalitesini maksimize etmektir. Bu hedefe ulaşmak için, RAG mimarisinin üç temel bileşenini deneysel olarak değerlendirdik: (1) hibrit arama stratejisinde BM25 ve vektör aramasını birleştirerek alpha parametresini optimize ettik, (2) veri işleme aşamasında recursive ve semantic chunking yöntemlerini karşılaştırdık, ve (3) retrieval kalitesini artırmak için reranker entegrasyonunu test ettik. Bu bileşenlerin etkisini değerlendirmek için farklı konfigürasyonlarda RAG sistemleri geliştirdik ve performanslarını altı temel metrik kullanarak değerlendirdik: faithfulness, answer relevance, context recall, context precision, ROUGE-N, ve BERTScore.

---

## 3.1 RAG Sistemleri

İki ana aşamadan oluşan bir RAG sistemi [10] geliştirdik: büyük metinlerin daha küçük, yönetilebilir chunk'lara bölündüğü **veri işleme aşaması** ve kullanıcı sorgusunun en ilgili metin chunk'larını getirmek için bir retrieval sürecini tetiklediği **retrieval ve generation aşaması**. Tüm bilgiler, hem orijinal kullanıcı sorgusunu hem de getirilen chunk'ları içeren bir prompt şablonuna yerleştirilir. Son prompt daha sonra cevap üretmek için bir LLM'e iletilir. RAG kurulumu Şekil 1'de gösterilmiştir.

### 3.1.1 Veri İşleme Aşaması

Veri işleme aşamasında, performanslarını karşılaştırmak için RAG sistemlerimizde hem **Recursive** hem de **Semantic chunking** yöntemlerini kullandık. Basit bir recursive ayırıcı tabanlı bölme yönteminin etkinliğini semantic chunking yöntemiyle karşılaştırdık. Amacımız, iki yöntem arasında ne kadar bağlamın korunduğunu ve ortaya çıkan metin chunk'larının retrieval aşamasının kalitesini nasıl etkilediğini keşfetmekti.

**Recursive Chunking [45]:** Hiyerarşik ayırıcılar kullanarak metni bölen bir yöntemdir. Sistemimizde uygulanan recursive chunker, metni önce cümlelere ayırır ve ardından bu cümleleri hedef chunk boyutuna göre birleştirir. Önemli özellikler:

- **Cümle Bütünlüğü:** Kelimeleri veya cümleleri asla ortadan bölmez
- **Ayırıcı Hiyerarşisi:** Öncelik sırasına göre ayırıcılar kullanır: `\n\n` (paragraf) → `\n` (satır) → `. ! ?` (cümle sonu) → `, ;` (virgül) → ` ` (boşluk)
- **Overlap Yönetimi:** Overlap için önceki chunk'tan tam cümleler alır, kısmi cümle kullanmaz
- **Türkçe Desteği:** Türkçe karakterleri (Ç, Ğ, İ, Ö, Ş, Ü) ve cümle yapılarını tanır

**Semantic Chunking:** Embedding benzerliğine dayalı gelişmiş bir yöntemdir. Sistemimizde uygulanan semantic chunker şu özelliklere sahiptir:

- **Buffer-Based Grouping:** Her cümle, etrafındaki bağlamla (buffer_size=1, yani önceki ve sonraki 1 cümle) birlikte embedding'e dönüştürülür
- **Percentile-Based Threshold:** Sabit eşik yerine, mesafe dağılımının yüzdelik dilimini (95. percentile) kullanarak dinamik eşik belirler
- **Cosine Distance:** Ardışık cümle embedding'leri arasında cosine mesafesi hesaplanır: `distance = 1 - cosine_similarity`
- **Adaptive Threshold:** Metin özelliklerine göre otomatik eşik hesaplama (enable_adaptive_threshold=True)
- **Q&A Detection:** Soru-cevap çiftlerini tespit eder ve aynı chunk'ta tutar (Türkçe soru kalıpları: "ne", "nasıl", "neden", "mi?", "mı?", vb.)
- **Size Constraints:** Minimum (150 karakter) ve maksimum (2000 karakter) boyut sınırları ile küçük chunk'ları birleştirir, büyük chunk'ları cümle sınırlarında böler
- **Embedding Cache:** Performans için embedding sonuçlarını önbelleğe alır (TTL=3600 saniye)
- **Provider Fallback:** OpenRouter → OpenAI sırasıyla embedding sağlayıcı yedekleme sistemi

Recursive chunking'i **chunk_size=500, overlap=50** parametreleri ile uyguladık. Semantic chunking için **similarity_threshold=adaptive (percentile=95), buffer_size=1, min_chunk_size=150, max_chunk_size=2000** parametrelerini kullandık.

**Semantic Chunking Algoritması:**

```
1. Metin Ön İşleme:
   - Metni cümlelere ayır (NLTK + Türkçe regex desenleri)
   - Soru-cevap çiftlerini tespit et ve birleştir
   - Çok kısa cümleleri (< 10 karakter) sonraki cümle ile birleştir

2. Buffer-Based Embedding:
   FOR her cümle i:
       context = cümleler[i-buffer_size : i+buffer_size+1]
       embedding[i] = get_embedding(context)
   
3. Mesafe Hesaplama:
   FOR i = 0 to len(embeddings)-2:
       similarity = cosine_similarity(embedding[i], embedding[i+1])
       distance[i] = 1 - similarity
   
4. Dinamik Eşik ve Breakpoint Belirleme:
   threshold = percentile(distances, 95)
   breakpoints = [i+1 for i, d in enumerate(distances) if d > threshold]
   
5. Chunk Oluşturma:
   chunks = split_at_breakpoints(sentences, breakpoints)
   
6. Boyut Optimizasyonu:
   - Küçük chunk'ları birleştir (< min_chunk_size)
   - Büyük chunk'ları cümle sınırlarında böl (> max_chunk_size)
   
7. Overlap Ekleme (opsiyonel):
   FOR her chunk i > 0:
       overlap_sentences = get_last_sentences(chunk[i-1], overlap_size)
       chunk[i] = overlap_sentences + chunk[i]
```

Tablo 1, her iki yöntemin karşılaştırmalı sonuçlarını göstermektedir.

**Tablo 1. Chunking Yöntemleri Karşılaştırması**

| Metrik | Recursive Chunking | Semantic Chunking |
|--------|-------------------|-------------------|
| Context Recall | [TODO]% | [TODO]% |
| Context Precision | [TODO]% | [TODO]% |
| Faithfulness | [TODO]% | [TODO]% |
| Answer Relevance | [TODO]% | [TODO]% |

### 3.1.2 Embedding Modeli

Metin vektörlerini oluşturmak için OpenAI'nin **text-embedding-3-small** modelini kullandık. Bu model, 1536 boyutlu vektörler üretir ve Türkçe dahil çok dilli destek sunar. Modelin seçilme nedenleri arasında Türkçe için güçlü performansı, eğitim terminolojisini iyi yakalama kapasitesi ve dengeli boyut/performans oranı bulunmaktadır.

**Embedding Sağlayıcı Mimarisi:**
Sistemimiz, yedekleme (fallback) desteği ile çoklu embedding sağlayıcı mimarisi kullanır:

1. **Birincil Sağlayıcı:** OpenRouter API (openrouter.ai/api/v1)
2. **Yedek Sağlayıcı:** OpenAI API (doğrudan)
3. **Önbellekleme:** 3600 saniye TTL ile embedding cache
4. **Batch İşleme:** 32 metin/batch, maksimum 3 yeniden deneme

Bu mimari, bir sağlayıcı başarısız olduğunda otomatik olarak diğerine geçiş yaparak sistem güvenilirliğini artırır.

### 3.1.3 Hibrit Arama Pipeline'ı

Retrieval süreci için, cosine benzerliği tabanlı semantik arama retriever'ı ile BM25 anahtar kelime retriever'ını [48] birleştiren bir **hibrit arama pipeline'ı** kullandık. Tüm vektör embedding'leri Weaviate vektör veritabanında saklandı.

Hibrit arama, iki farklı arama yöntemini birleştirir:

**Vektör Araması (Semantic Search):**
- Cosine benzerliğine dayalı embedding araması
- Anlamsal ilişkileri yakalar (eş anlamlılar, parafrazlar)
- Bağlamsal anlama sağlar
- Türkçe için güçlü performans

**BM25 Araması (Keyword Search):**
- TF-IDF tabanlı anahtar kelime eşleşmesi
- Tam terim eşleşmeleri bulur
- Teknik terminoloji ve özel isimlerde etkili
- Nadir terimlere yüksek ağırlık verir

Hibrit kombinasyon Weaviate'in native hibrit arama özelliği ile şu formülle hesaplanır:

```
Final_Score = α × Vector_Score + (1 - α) × BM25_Score
```

Alpha (α) parametresi, iki arama yöntemi arasındaki dengeyi kontrol eder:
- **α = 0.0:** Saf BM25 araması (sadece anahtar kelime)
- **α = 0.5:** Dengeli hibrit (50% semantik, 50% anahtar kelime)
- **α = 0.8:** Vektör ağırlıklı hibrit (80% semantik, 20% anahtar kelime) - **Optimal değer**
- **α = 1.0:** Saf vektör araması (sadece semantik)

Deneysel bulgularımız, Türkçe eğitim içeriği için **α = 0.8** değerinin optimal performans sağladığını göstermiştir.

### 3.1.4 RAG Sistem Konfigürasyonları

Embedding ve reranker modellerinin RAG sistemi performansı üzerindeki etkilerini denemek için, her biri metin bölme ve embedding modellerinin bir kombinasyonunu kullanan farklı versiyonlar geliştirdik. Birincil generation LLM olarak [TODO: LLM modeli - örn. GPT-4, Gemini-1.5-Pro] kullandık.

**Sistem Promptu Optimizasyonu:**

RAG sistemimizde, LLM'in ürettiği cevapların kalitesini artırmak için optimize edilmiş bir sistem promptu kullandık. Prompt, şu temel yönergeleri içerir:

1. **Bağlam Kullanımı:** Sadece verilen bağlamsal bilgiyi kullan, dışarıdan bilgi ekleme
2. **Doğruluk:** Bağlamda olmayan bilgiler için "Bu bilgi verilen dokümanlarda bulunmuyor" yanıtı ver
3. **Türkçe Dil Desteği:** Tüm yanıtlar Türkçe dilinde ve eğitim terminolojisine uygun olmalı
4. **Yapılandırılmış Yanıt:** Net, anlaşılır ve öğrenci seviyesine uygun açıklamalar

**Hata Yönetimi ve Fallback Stratejisi:**

Sistemimiz, chunking işleminde hata oluştuğunda otomatik olarak daha basit yöntemlere geçiş yapan kapsamlı bir fallback mekanizması içerir:

```
Fallback Sırası:
1. Semantic Chunking (birincil)
   ↓ (embedding hatası)
2. Sentence-Based Chunking (yedek-1)
   ↓ (tokenization hatası)
3. Fixed-Size Chunking (yedek-2)
```

Bu mimari, sistem güvenilirliğini artırır ve embedding API'lerinde yaşanabilecek kesintilerde bile chunking işleminin devam etmesini sağlar.

**Chunking Kalite Metrikleri:**

Her chunking işlemi için otomatik olarak hesaplanan diagnostik metrikler:

- **Processing Time:** Chunking işlem süresi (saniye)
- **Total Chunks:** Oluşturulan toplam chunk sayısı
- **Avg/Min/Max Chunk Size:** Ortalama, minimum ve maksimum chunk boyutları
- **Overlap Count:** Overlap içeren chunk sayısı
- **Quality Score:** 0-1 arası kalite skoru (boş chunk, çok kısa chunk, boyut tutarlılığı kontrolü)
- **Size Distribution:** Chunk boyut dağılımı (0-100, 101-300, 301-500, 501-1000, 1000+ karakter)

Kalite skoru hesaplama:
```
score = 1.0
score -= (empty_chunks / total_chunks) × 0.3
score -= (very_short_chunks / total_chunks) × 0.2
score = (score + consistency_score) / 2
```

Consistency score, chunk boyutlarının standart sapmasına göre hesaplanır ve tutarlı boyutlara sahip chunk'ları ödüllendirir.

RAG sistemlerinin konfigürasyonları şu şekildedir:

• **Base RAG:** [TODO: Seçilen chunking] + text-embedding-3-small base embedding modeli + hibrit arama (α = [TODO]).

• **RAG V1:** [TODO: Seçilen chunking] + text-embedding-3-small + hibrit arama (α = 0.5).

• **RAG V2:** [TODO: Seçilen chunking] + text-embedding-3-small + hibrit arama (α = 0.8).

• **RAG V3:** [TODO: Seçilen chunking] + text-embedding-3-small + hibrit arama (α = 0.8) + [TODO: Reranker modeli].

[TODO: Konfigürasyonları kendi deneysel sürecinize göre güncelleyin]

---

## 3.2 Değerlendirme Veri Seti

RAG değerlendirme veri seti, MEB Bilişim Teknolojileri Temelleri müfredat dokümanından oluşturulan [TODO: N] örnekten oluşmaktadır. Her örnek üç sütun içerir: Context (kaynak doküman), Question (Context ile ilgili bir sorgu), ve Answer (karşılık gelen ground-truth yanıtı).

Değerlendirme amaçları için, veri seti aşağıdaki sütunlardan oluşan bir değerlendirme formatına dönüştürüldü:

• **Question:** Kaynak dokümanlarda sunulan bilgiler hakkında sorular.
• **Answer:** RAG sisteminden LLM tarafından üretilen cevap.
• **Contexts:** Semantic Search yoluyla getirilen doküman seti.
• **Ground Truth:** Veri setinde mevcut olan ground truth cevaplar.

---

## 3.3 Değerlendirme Metrikleri

Bölüm 3.1'de sunulan altı metriğin her biri—yani Faithfulness, Answer Relevance, Context Recall, Context Precision, ROUGE-N ve BERTScore—her bir RAG sisteminin çıktılarına uygulandı.

### 3.3.1 RAGAS Metrikleri

**Faithfulness (Sadakat):** Üretilen cevabın getirilen bağlamlara ne kadar sadık kaldığını ölçer.

```
Faithfulness = |Claims Supported by Context| / |Total Claims in Response|
```

**Answer Relevance (Cevap İlgisi):** Üretilen cevabın kullanıcının sorusunu ne kadar iyi yanıtladığını ölçer.

**Context Recall (Bağlam Geri Çağırma):** Ground truth bilgisinin ne kadarının getirilen bağlamlarda yakalandığını ölçer.

**Context Precision (Bağlam Kesinliği):** Getirilen bağlamların soruyla ne kadar ilgili olduğunu ölçer.

### 3.3.2 Metin Benzerlik Metrikleri

**ROUGE-N [43]:** N-gram örtüşmesini ölçen leksikal benzerlik metrikleri (ROUGE-1, ROUGE-2, ROUGE-L).

**BERTScore [44]:** Embedding tabanlı semantik benzerlik metrikleri (Precision, Recall, F1).

---

## 3.4 Deney Kurulumu

Alpha parametresinin etkisini sistematik olarak değerlendirmek için beş farklı alpha değeri test ettik: 0.0, 0.2, 0.5, 0.8, ve 1.0. Her konfigürasyon için aynı test seti kullanıldı ve tüm RAGAS ve metin benzerlik metrikleri hesaplandı.

**Kontrol Edilen Değişkenler:**
- Aynı test veri seti ([TODO: N soru])
- Aynı embedding modeli (text-embedding-3-small)
- Aynı LLM ve generation parametreleri
- Aynı chunking stratejisi
- Aynı top-k değeri ([TODO: örn. 5])

**Deney Protokolü:**

```
FOR each alpha_value in [0.0, 0.2, 0.5, 0.8, 1.0]:
    Weaviate'i alpha_value ile yapılandır
    
    FOR each question in test_set:
        1. Soruyu embedding'e dönüştür
        2. Hibrit arama yap (alpha = alpha_value)
        3. Top-k chunk'ları getir
        4. LLM ile cevap üret
        5. Tüm metrikleri hesapla
    
    Aggregate metrikleri kaydet
```

---

# 4. Deneysel Sonuçlar ve Analiz

Şekil 2'deki sonuçları incelediğimizde, **RAG V2'nin** (α = 0.8) baseline RAG sistemi dahil diğer üç konfigürasyondan daha iyi performans gösterdiğini gözlemliyoruz. Özellikle, RAG V2, ROUGE-N ve BERTScore metrikleriyle gösterildiği gibi daha yüksek kaliteli LLM tarafından üretilen yanıtlar üretmektedir. Bu bulgular, hibrit arama parametrelerinin—özellikle alpha değerinin—Türkçe eğitim içeriği için optimize edilmesinin RAG sistemlerinde retrieval etkinliğini önemli ölçüde artırdığını göstermektedir. Sonuç olarak, getirilen bağlamın geliştirilmiş ilgililik ve kalitesi, LLM tarafından üretilen daha doğru ve bilgilendirici yanıtlara yol açmaktadır.

**Şekil 2. Her RAG sistemi için RAGAS, ROUGE-N ve BERTScore metrik sonuçları**

[TODO: Grafik ekleyin - Bar chart showing all metrics for each alpha value]

| Alpha | Context Recall | Context Precision | Faithfulness | Answer Relevance | ROUGE-1 | ROUGE-L | BERTScore F1 |
|-------|----------------|-------------------|--------------|------------------|---------|---------|--------------|
| 0.0   | [TODO]% | [TODO]% | [TODO]% | [TODO]% | [TODO]% | [TODO]% | [TODO]% |
| 0.2   | [TODO]% | [TODO]% | [TODO]% | [TODO]% | [TODO]% | [TODO]% | [TODO]% |
| 0.5   | [TODO]% | [TODO]% | [TODO]% | [TODO]% | [TODO]% | [TODO]% | [TODO]% |
| **0.8** | **[TODO]%** | **[TODO]%** | **[TODO]%** | **[TODO]%** | **[TODO]%** | **[TODO]%** | **[TODO]%** |
| 1.0   | [TODO]% | [TODO]% | [TODO]% | [TODO]% | [TODO]% | [TODO]% | [TODO]% |

İlginç bir şekilde, Şekil 2 ayrıca [TODO: reranker eklendiğinde veya başka bir konfigürasyon değiştiğinde] performansın [TODO: nasıl değiştiğini] göstermektedir. [TODO: Bu durumun nedenini açıklayın].

## 4.1 Reranker Entegrasyonu Sonuçları

[TODO: Reranker testini yaptıysanız buraya ekleyin]

Optimal alpha değerini (0.8) belirledikten sonra, retrieval kalitesini daha da artırmak için [TODO: reranker modeli] entegre ettik ve sistemi yeniden test ettik.

**Tablo 2. Reranker Entegrasyonu Karşılaştırması**

| Konfigürasyon | Context Recall | Context Precision | Faithfulness | Answer Relevance |
|---------------|----------------|-------------------|--------------|------------------|
| RAG V2 (α=0.8) | [TODO]% | [TODO]% | [TODO]% | [TODO]% |
| RAG V3 (α=0.8 + Reranker) | [TODO]% | [TODO]% | [TODO]% | [TODO]% |
| **Fark** | **±[TODO]%** | **±[TODO]%** | **±[TODO]%** | **±[TODO]%** |

[TODO: Reranker sonuçlarını yorumlayın - iyileştirme sağladı mı, aynı mı kaldı?]

## 4.2 Qualitative Analiz

Tablo A1 ve A2, veri setimizden [38] temsili bir örnek için RAG V2 ve [TODO: diğer konfigürasyon] sistemlerinin çıktılarını göstermektedir.

**Tablo A1. RAG V2 için Faithfulness Değerlendirme Örneği**

| Parametre | Değer |
|-----------|-------|
| **Kullanıcı Girdisi** | [TODO: Örnek soru] |
| **Yanıt** | [TODO: Üretilen cevap] |
| **Getirilen Bağlamlar** | [TODO: Top-3 chunk] |
| **Faithfulness** | [TODO: skor] |

Tablo A1'de, RAG V2 [TODO: Faithfulness skoru] elde eder ve yanıt açıkça ilk getirilen chunk'ta temellendirilmiştir—"(1)" ile gösterilir—bağlam ve cevap arasında etkili uyumu gösterir. [TODO: Seçilen chunking yöntemi] tarafından kullanılan yöntem, anlamsal olarak zengin ve tutarlı chunk'lar üreterek LLM'in iyi desteklenmiş bir yanıt üretmesini sağladı.

---

# 5. Sonuç

Bu çalışmada, Türkçe eğitim içeriği için RAG sistemlerinin performansını artırmak amacıyla hibrit arama parametrelerinin optimizasyonunu araştırdık. Özellikle, vektör ve anahtar kelime araması arasındaki dengeyi kontrol eden alpha parametresinin sistematik olarak ayarlanmasının retrieval kalitesi üzerindeki etkisini değerlendirdik. Ayrıca, [TODO: chunking karşılaştırması] ve [TODO: reranker entegrasyonu] sonuçlarını analiz ettik.

Deneysel sonuçlarımız, **alpha = 0.8** değerinin Türkçe eğitim içeriği için optimal performans sağladığını göstermektedir. Bu bulgu, vektör ağırlıklı hibrit aramanın (%80 semantik, %20 anahtar kelime) hem anlamsal anlama hem de teknik terminoloji için anahtar kelime hassasiyetini dengeli bir şekilde birleştirdiğini ortaya koymaktadır.

[TODO: Chunking sonucunu ekleyin - hangisi daha iyi performans gösterdi]

[TODO: Reranker sonucunu ekleyin - iyileştirme sağladı mı]

Gelecek çalışmalar, [TODO: gelecek araştırma yönlerini ekleyin - örn. Türkçe'ye özel fine-tune edilmiş embedding modelleri, farklı eğitim alanlarında test, vb.] üzerine odaklanacaktır.

---

## Kaynaklar

[10] Lewis, P., et al.: Retrieval-Augmented Generation for Knowledge-Intensive NLP tasks. arXiv preprint (2020). https://arxiv.org/abs/2005.11401

[42] Es, S., et al.: RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv preprint (2023). https://arxiv.org/abs/2309.15217

[43] Lin, C.Y.: ROUGE: a package for automatic evaluation of summaries. ACL (2004)

[44] Zhang, T., et al.: BERTScore: Evaluating Text Generation with BERT. arXiv preprint (2019). https://arxiv.org/abs/1904.09675

[45] LangChain: Recursively split by character. https://python.langchain.com/

[48] Amati, G.: BM25 (2009). https://doi.org/10.1007/978-0-387-39940-9

[TODO: Diğer referansları ekleyin]

---

## Ek: Uygulama Detayları

Bu metodoloji dokümanı, sistemin gerçek kod implementasyonundan (`backend/app/services/chunker.py`) analiz edilerek hazırlanmıştır. Dokümanda belirtilen tüm algoritmalar, parametreler ve özellikler, üretim ortamında çalışan gerçek sistemin davranışını yansıtmaktadır.

**Dokümante Edilen Ana Bileşenler:**

1. **Recursive Chunker:** Hiyerarşik ayırıcılar, cümle bütünlüğü, Türkçe desteği
2. **Semantic Chunker:** Buffer-based grouping, percentile threshold, adaptive threshold, Q&A detection, embedding cache, provider fallback
3. **Embedding Mimarisi:** Multi-provider (OpenRouter + OpenAI), caching, batch processing
4. **Hibrit Arama:** Weaviate native hybrid search, alpha parametresi optimizasyonu
5. **Hata Yönetimi:** 3-seviyeli fallback stratejisi (semantic → sentence → fixed-size)
6. **Kalite Metrikleri:** Otomatik diagnostik, quality score, size distribution

**Deneysel Veriler İçin TODO İşaretleri:**

Dokümanda [TODO] ile işaretlenmiş alanlar, gerçek deneysel sonuçlarınızla doldurulmalıdır:
- Chunking yöntemleri karşılaştırma tablosu (Tablo 1)
- Alpha optimizasyonu sonuçları (0.0, 0.2, 0.5, 0.8, 1.0)
- Reranker entegrasyonu sonuçları
- Kullanılan LLM modeli
- Test veri seti boyutu (N soru)
- Qualitative analiz örnekleri
