# Literatür Taraması: AkıllıRehber Akademik Makale

## Genel Bakış

Bu doküman, AkıllıRehber akademik makalesi için yapılan literatür taramasını içermektedir. PDF makaleleri Python pdfplumber ile okunarak analiz edilmiştir.

---

## Kategori 1: Doğrudan İlgili - Türkçe RAG Sistemleri

### 1.1 EkoBot: Türkçe Destekli Akıllı Sanal Akademik Danışman
**Kaynak:** Int. J. Adv. Eng. Pure Sci. 2025, 37(2): 196-205
**Yazar:** Ayça Kumluca Topallı (İzmir Ekonomi Üniversitesi)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- İlk Türkçe destekli RAG tabanlı akademik danışman sistemi
- İzmir Ekonomi Üniversitesi "Ön Lisans ve Lisans Eğitim-Öğretim ve Sınav Yönetmeliği" üzerinde çalışıyor
- 100 soru ile test edilmiş, %100 retrieval başarısı (k=5), %82 yanıt-bağlam benzerliği

**Teknik Detaylar:**
- **Embedding:** OpenAI text-embedding-3-large (N=3072 boyut)
- **LLM:** GPT-4o
- **Chunking:** Yönetmelik maddeleri bazında (183 parça, M=183)
- **Benzerlik:** Kosinüs benzerliği
- **Retrieval:** En benzer k=5 madde
- **Değerlendirme:** Kosinüs benzerliği (yanıt vs bağlam)

**Karşılaştırma - Hugging Face BERT vs OpenAI:**
- OpenAI text-embedding-3-large: k=5'te %100 başarı
- HuggingFace bert-base-turkish-cased: k=10'da %100 başarı
- OpenAI Türkçe için daha iyi performans gösteriyor

**Literatür Taraması Bulguları (Tablo 1):**
- 2017-2024 arası 20+ çalışma incelenmiş
- Çoğu İngilizce, RAG öncesi yöntemler (AIML, LSTM, NER)
- RAG kullanan: Oliveira & Matos 2023 (GPT-3), Maryamah 2024 (GPT-3.5), Neupane 2024 (GPT-3.5)
- **Türkçe RAG çalışması yok** - EkoBot bu alanda ilk

**AkıllıRehber Farkları:**
| Özellik | EkoBot | AkıllıRehber |
|---------|--------|--------------|
| Hedef Kitle | Üniversite | Lise |
| Chunking | Madde bazlı (fixed) | Adaptive Semantic |
| Hybrid Search | Yok | Weaviate (BM25+Vector, α=0.7) |
| Türkçe NLP | Temel | Kapsamlı (50+ kısaltma, Q&A detection) |
| Değerlendirme | Kosinüs benzerliği | RAGAS framework |
| Panel Ayrımı | Yok | Öğretmen/Öğrenci |

---

### 1.2 Bridging the Language Gap in RAG: A Case Study on Turkish
**Kaynak:** Ankara Science University, RESEARCHER Vol. 5, No. 1, July 2025
**Yazarlar:** Erdoğan Bıkmaz, Mohammed Briman, Serdar Arslan (Çankaya Üniversitesi)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Türkçe RAG sistemlerinde retrieval ve generation performansını artırmak için fine-tuning
- Embedding model ve reranker model Türkçe verilerle fine-tune edilmiş
- 4 farklı RAG sistemi, 6 metrik ile değerlendirilmiş

**Türkçe RAG Zorlukları (Kritik Bulgular):**
1. **Morfoloji:** Eklemeli yapı (agglutination) performans farklılıkları yaratıyor
2. **Gramer:** Avrupa dilleriyle (İngilizce, Fransızca) paylaşılan yapı az
3. **Dil Difüzyonu:** Arapça, Farsça, İngilizce, Fransızca'dan alıntı kelimeler
4. **Domain-Specific Terminoloji:** Finans, hukuk gibi alanlarda özel terimler

**Teknik Detaylar:**
- **Base Embedding:** multilingual-e5-large (384 boyut)
- **Fine-tuned Embedding:** multilingual-e5-tr-rag
- **Base Reranker:** jina-reranker-v2-base-multilingual
- **Fine-tuned Reranker:** jina-reranker-multilingual-wiki-tr-rag
- **Chunking Yöntemleri:** Recursive vs Prepositional (LLM-based)
- **Retrieval:** Hybrid search (cosine + BM25)
- **Vector DB:** Chroma DB
- **LLM:** gemma-3-27b-it

**RAG Sistem Konfigürasyonları:**
1. Base RAG: Recursive + MiniLM + semantic + ColBERTv2.0
2. RAG V1: Recursive + fine-tuned e5 + hybrid + ColBERTv2.0
3. RAG V2: Recursive + fine-tuned e5 + hybrid + fine-tuned jina (EN İYİ)
4. RAG V3: Prepositional + fine-tuned e5 + hybrid + fine-tuned jina

**Kritik Bulgular:**
- **RAG V2 en iyi performans** - Fine-tuned embedding + reranker
- **Prepositional chunking (RAG V3) kötü performans** - Kısa cümleler bağlam kaybına yol açıyor
- **Faithfulness skoru:** RAG V2 = 1.0, RAG V3 = 0.0 (örnek)
- **Sonuç:** Recursive chunking + fine-tuned retrieval > LLM-based chunking

**Değerlendirme Metrikleri:**
- RAGAS: Faithfulness, Answer Relevance, Context Recall, Context Precision
- NLP: ROUGE-N, BERTScore

**AkıllıRehber İçin Çıkarımlar:**
- Türkçe için fine-tuned embedding modelleri önemli
- Hybrid search (BM25 + Vector) Türkçe için kritik
- Prepositional chunking yerine semantic chunking tercih edilmeli
- Reranker modeli retrieval kalitesini artırıyor

---

## Kategori 2: Eğitimde RAG Uygulamaları

### 2.1 RAG Chatbots for Education: A Survey of Applications
**Kaynak:** Applied Sciences 2025, 15, 4234 (MDPI)
**Yazarlar:** Jakub Swacha, Michał Gracel (University of Szczecin)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- 47 makale analiz edilmiş (2022-2025)
- Eğitimde RAG chatbot uygulamalarının kapsamlı survey'i
- Destek hedefi, tematik kapsam, LLM seçimi, değerlendirme kriterleri

**Destek Hedefleri (Figure 2):**
1. **Öğrenme desteği:** 14 çalışma (en yaygın)
2. **Kaynak bilgiye erişim:** 20 çalışma
3. **Organizasyonel konular:** 7 çalışma
4. **Ders seçimi:** 3 çalışma
5. **Soru üretimi:** 2 çalışma
6. **Öğrenme analitiği:** 2 çalışma

**Tematik Kapsam (Figure 3):**
- Kurumsal bilgi: 11 çalışma
- Sağlık bilimleri: 9 çalışma
- Bilgisayar bilimi: 6 çalışma
- Hukuk: 3 çalışma
- Herhangi bir alan: 10 çalışma

**LLM Kullanımı (Figure 4):**
- **GPT ailesi:** 36 örnek (baskın)
- **LLaMA ailesi:** 15 örnek
- **Diğer:** 8 örnek (Mistral, Claude, Gemma)
- **Belirtilmemiş:** 7 çalışma

**Değerlendirme Kriterleri (Figure 5):**
- Accuracy (bilgi retrieval)
- Relevance (RAGAS)
- Faithfulness (RAGAS)
- Quality of Generated Text (ROUGE, BERTScore)
- User Acceptance (SUS)
- Qualitative Evaluation

**Önerilen Değerlendirme Yaklaşımları (Table 5):**
| Amaç | Önerilen Metrik |
|------|-----------------|
| Yanıt İçeriği | Information retrieval (Precision, Recall, F1) |
| Yanıt Formu | Conversational quality, Generated text quality |
| Kullanıcı Deneyimi | User acceptance (SUS) |
| RAG Kalitesi | RAGAS (Faithfulness, Relevance, Context) |

**Kritik Bulgular:**
- RAGAS sadece 4/47 çalışmada kullanılmış
- Çoğu çalışma hedef-spesifik değerlendirme yapmamış
- Türkçe çalışma survey'de yok

**AkıllıRehber İçin Çıkarımlar:**
- RAGAS kullanımı literatürde nadir - özgün katkı
- Öğretmen/öğrenci panel ayrımı literatürde yok
- Lise odaklı RAG sistemi literatürde yok
- Adaptive semantic chunking literatürde yok

---

### 2.2 UniROBO: AI-Powered GPT-Based University-Specific Chat Assistant
**Kaynak:** Malatya Turgut Ozal University Journal of Engineering and Natural Sciences, Vol 5, Issue 2 (2024)
**Yazarlar:** Serdar Budak, Serpil Aslan

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Malatya Turgut Özal Üniversitesi için geliştirilen chatbot
- Eğitim, yemek menüleri, kampüs etkinlikleri hakkında bilgi
- RAG + Azure AI Search + Fine-tuning

**Teknik Detaylar:**
- **Frontend:** React Native (cross-platform)
- **Backend:** Python + FastAPI
- **Database:** MongoDB
- **AI:** OpenAI API + Fine-tuning
- **Search:** Azure AI Search
- **Mimari:** RAG (Retrieval-Augmented Generation)

**Fine-tuning Talimatları:**
- Sadece Türkçe yanıt ver
- İstenen bilgiye odaklan, ek bilgi verme
- Kullanıcıya dostça yaklaş
- Alakasız sorularda soruyu daha anlamlı sormasını iste
- Sadece üniversite ile ilgili sorulara yanıt ver

**AkıllıRehber Farkları:**
| Özellik | UniROBO | AkıllıRehber |
|---------|---------|--------------|
| Hedef | Üniversite | Lise |
| Chunking | Belirtilmemiş | Adaptive Semantic |
| Search | Azure AI Search | Weaviate Hybrid |
| Fine-tuning | OpenAI | Yok (prompt engineering) |
| Panel | Tek | Öğretmen/Öğrenci ayrı |

---

## Kategori 3: Türkçe NLP ve LLM

### 3.1 TULIP: Adapting Open-Source LLMs for Turkish Financial Tasks
**Kaynak:** ArXiv 2025
**Yazarlar:** İrem Demirtaş, Burak Payzun, Seçil Arslan (Prometeia SPA)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Türkçe finans alanı için LLM adaptasyonu
- Llama 3.1 8B ve Qwen 2.5 7B modelleri
- Continual Pre-training (CPT) + Supervised Fine-tuning (SFT)

**Türkçe LLM Zorlukları:**
1. **Morfoloji:** Eklemeli yapı, küçük modellerde performans düşüklüğü
2. **Gramer:** Avrupa dilleriyle farklı yapı
3. **Dil Difüzyonu:** Arapça, Farsça, İngilizce alıntılar
4. **Domain Terminoloji:** Finans terimleri çeviri zorluğu
5. **Performans Farkı:** Genel domain vs spesifik domain

**Teknik Detaylar:**
- **Base Models:** Llama 3.1 8B, Qwen 2.5 7B
- **CPT Data:** 2.19B token (akademik, merkez bankası, haberler, TRG)
- **SFT Data:** 23K instruction-answer çifti
- **Training:** QLoRA (alpha=128, rank=64)
- **Hardware:** 2x RTX A6000

**SFT Task Dağılımı:**
- Fill-in-the-blank: %19
- Multiple-choice QA: %16
- Summarization: %16
- True-False: %14
- Sentiment Analysis: %14
- Multi-turn QA: %6
- NER: %3

**Benchmark Sonuçları (FINTR-EXAMS):**
| Model | Ortalama Accuracy |
|-------|-------------------|
| Qwen2.5 (base) | 0.600 |
| TULIP-Qwen2.5 (CPT) | 0.668 |
| TULIP-Qwen2.5-IT (CPT+SFT) | 0.659 |
| Llama3.1 (base) | 0.544 |
| TULIP-Llama3.1 (CPT) | 0.583 |

**Kritik Bulgular:**
- CPT tek başına SFT'den daha iyi benchmark sonucu
- Türkçe için Qwen > Llama performansı
- Domain-specific fine-tuning önemli
- Çeviri yaklaşımı yetersiz (domain terimleri)

**AkıllıRehber İçin Çıkarımlar:**
- Türkçe için özel embedding/model seçimi kritik
- Domain-specific terminoloji desteği gerekli
- Eğitim alanı için benzer adaptasyon düşünülebilir

---

## Karşılaştırma Tablosu (Güncellenmiş)

| Çalışma | Yıl | Dil | Hedef | Chunking | Embedding | LLM | Değerlendirme |
|---------|-----|-----|-------|----------|-----------|-----|---------------|
| EkoBot | 2025 | TR | Üniversite | Madde bazlı | text-embedding-3-large | GPT-4o | Kosinüs (%82) |
| Bridging Gap | 2025 | TR | Genel | Recursive | multilingual-e5-tr-rag | gemma-3-27b | RAGAS |
| UniROBO | 2024 | TR | Üniversite | - | Azure AI | GPT + Fine-tune | - |
| TULIP | 2025 | TR | Finans | - | - | Llama/Qwen | FINTR-EXAMS |
| Survey (47) | 2025 | EN | Eğitim | Çeşitli | Çeşitli | GPT (%77) | Çeşitli |
| **AkıllıRehber** | **2025** | **TR** | **Lise** | **Adaptive Semantic** | **Multi-provider** | **Multi-provider** | **RAGAS** |

---

## AkıllıRehber'in Özgün Katkıları (Literatür Destekli)

### 1. Adaptive Semantic Chunking
- **Literatür:** EkoBot fixed-size, Bridging Gap recursive/prepositional
- **AkıllıRehber:** Adaptive threshold ile semantic chunking
- **Fark:** Dinamik eşik değeri, içerik türüne göre adaptasyon

### 2. Kapsamlı Türkçe NLP
- **Literatür:** EkoBot temel Türkçe, TULIP finans odaklı
- **AkıllıRehber:** 50+ kısaltma, tüm soru kalıpları (mi/mı/mu/mü)
- **Fark:** Eğitim alanına özel Türkçe NLP

### 3. Hybrid RAG (Weaviate)
- **Literatür:** Bridging Gap Chroma + hybrid, EkoBot sadece vector
- **AkıllıRehber:** Weaviate (BM25 + Vector, α=0.7)
- **Fark:** Türkçe için optimize edilmiş alpha parametresi

### 4. Öğretmen/Öğrenci Panel Ayrımı
- **Literatür:** Hiçbir çalışmada yok
- **AkıllıRehber:** Ayrı paneller, farklı yetkiler
- **Fark:** Eğitim ortamına özgü tasarım

### 5. RAGAS Değerlendirmesi
- **Literatür:** Survey'de sadece 4/47 çalışma
- **AkıllıRehber:** Sistematik RAGAS kullanımı
- **Fark:** Kapsamlı ve standart değerlendirme

### 6. Lise Odaklı Sistem
- **Literatür:** Tüm çalışmalar üniversite odaklı
- **AkıllıRehber:** Lise öğrencileri ve öğretmenleri
- **Fark:** Farklı hedef kitle, farklı gereksinimler

---

## Makalede Kullanılacak Referanslar

### Türkçe RAG Zorlukları
1. Bıkmaz et al. (2025) - Morfoloji, gramer, terminoloji zorlukları
2. Demirtaş et al. (2025) - Domain-specific adaptasyon gerekliliği
3. Topallı (2025) - Türkçe embedding model karşılaştırması

### Eğitimde RAG
1. Swacha & Gracel (2025) - 47 çalışmalık kapsamlı survey
2. Budak & Aslan (2024) - Türkçe üniversite chatbot

### Değerlendirme Metodolojisi
1. RAGAS framework (Es et al., 2023)
2. Swacha & Gracel (2025) - Değerlendirme kriterleri analizi

### Chunking Stratejileri
1. Bıkmaz et al. (2025) - Recursive vs Prepositional karşılaştırması
2. LangChain documentation - Recursive text splitter

---

## Kategori 4: Ek Türkçe NLP Çalışmaları

### 4.1 Passage Retrieval on Turkish Legal Texts Using BERT
**Kaynak:** METU Master's Thesis, September 2024
**Yazar:** Seda Civelek (Supervisor: Prof. Dr. Nihan Kesim Çiçekli)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Türkçe hukuk metinlerinde passage retrieval
- BM25 + BERT kombinasyonu
- Türkçe hukuk kitaplarından dataset oluşturulmuş

**Kritik Bulgular:**
- BM25 ve BERT kombinasyonu en iyi sonuç
- Türkçe legal domain için contextual embeddings önemli
- Hybrid retrieval (lexical + semantic) Türkçe için etkili

**AkıllıRehber İçin Çıkarımlar:**
- Hybrid search yaklaşımı doğrulanıyor
- Domain-specific dataset oluşturma metodolojisi

---

### 4.2 Enrichment of Turkish Question Answering Systems Using Knowledge Graphs
**Kaynak:** Turkish Journal of Electrical Engineering & Computer Sciences, 2024, 32(4), 516-533
**Yazarlar:** Okan Çiftçi, Fatih Soygazi, Selma Tekir (İzmir Yüksek Teknoloji Enstitüsü)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Türkçe soru-cevap sistemleri için knowledge graph yaklaşımı
- BPMovieKG (Beyazperde Movie Knowledge Graph) oluşturulmuş
- TRMQA (Turkish Movie Question Answering) dataset
- GPT-3.5 Turbo ile karşılaştırma

**Kritik Bulgular:**
- Knowledge graph multihop reasoning için etkili
- GPT-3.5 Turbo bazı 1-hop sorularda hatalı yanıt veriyor
- Graph embedding + question embedding kombinasyonu başarılı
- Türkçe için ilk knowledge graph tabanlı QA sistemi

**Türkçe QA Literatürü (Makaleden):**
- Derici et al. (2007): Closed-domain QA
- Celebi et al. (2008): NER + pattern matching
- HazırCevap (Derici et al.): Summarization-based QA
- THQuAD (Soygazi et al.): Turkish reading comprehension dataset
- Menevşe et al.: İlk Türkçe spoken QA dataset

**AkıllıRehber İçin Çıkarımlar:**
- Türkçe QA sistemleri için knowledge enrichment önemli
- Multihop reasoning desteği düşünülebilir
- Türkçe QA dataset'leri referans olarak kullanılabilir

---

## Kategori 5: Güncel RAG Survey (2025)

### 5.1 Retrieval-Augmented Generation for Educational Application: A Systematic Survey
**Kaynak:** Computers and Education: Artificial Intelligence, 8 (2025) 100417
**Yazarlar:** Zongxi Li, Zijian Wang, Weiming Wang, Kevin Hung, Haoran Xie, Fu Lee Wang (Lingnan University, Hong Kong Metropolitan University)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Eğitimde RAG uygulamalarının sistematik survey'i
- 51 makale analiz edilmiş (2020-2025)
- RAG workflow: Indexing → Retrieval → Generation
- 3 ana uygulama kategorisi

**RAG Workflow Detayları:**

**1. Indexing:**
- Data preprocessing (PDF, PPT, Word → plain text)
- Text chunking (context length limiti nedeniyle)
- Vectorization (BGE, ESE, text-embedding-ada-002)
- Index storage (FAISS, Weaviate, Pinecone)

**2. Retrieval:**
- **Sparse Retrieval:** BM25, KNN
- **Dense Retrieval:** DPR, ColBERTv2, Contriever
- **Hybrid:** Sparse + Dense kombinasyonu

**3. Generation:**
- Concatenation-based generation
- Fusion-in-Decoder (FiD)
- Adaptive fusion (attention mechanisms)
- Prompt engineering, Fine-tuning

**Eğitimde RAG Uygulama Kategorileri:**

**A. Interactive Learning Systems (RQ1):**
- Educational Q&A Systems (Anatbuddy, SCM courses, Math)
- Educational Chatbots
- AI-driven Tutoring Systems
- Adaptive Learning Paths

**B. Educational Content Development & Assessment (RQ2):**
- Automated content generation
- Intelligent assessment mechanisms

**C. Large-Scale Educational Ecosystem (RQ3):**
- Institutional-level implementations
- Technical advancements for scale

**Kritik Zorluklar:**
1. Hallucination problemi
2. Static internal knowledge
3. Explainability eksikliği
4. Personalization yetersizliği
5. Computational costs
6. Multimodal support

**AkıllıRehber İçin Çıkarımlar:**
- Weaviate kullanımı survey'de referans gösterilmiş
- Hybrid retrieval (BM25 + Dense) öneriliyor
- Chunking stratejisi kritik öneme sahip
- Prompt engineering önemli optimizasyon stratejisi

---

## Güncellenmiş Karşılaştırma Tablosu

### Tablo: Türkçe ve Eğitim RAG Sistemleri (2024-2025)

| Çalışma | Yıl | Dil | Domain | Retrieval | Chunking | LLM | Değerlendirme |
|---------|-----|-----|--------|-----------|----------|-----|---------------|
| EkoBot | 2025 | TR | Akademik | Vector | Madde bazlı | GPT-4o | %82 Kosinüs |
| Bridging Gap | 2025 | TR | Genel | Hybrid | Recursive | gemma-3-27b | RAGAS |
| UniROBO | 2024 | TR | Üniversite | Azure AI | - | GPT + FT | - |
| TULIP | 2025 | TR | Finans | - | - | Llama/Qwen | FINTR-EXAMS |
| Legal BERT | 2024 | TR | Hukuk | BM25+BERT | - | - | - |
| KG-QA | 2024 | TR | Film | Graph | - | GPT-3.5 | Accuracy |
| Li et al. Survey | 2025 | EN | Eğitim | Çeşitli | Çeşitli | Çeşitli | Survey |
| Swacha Survey | 2025 | EN | Eğitim | Çeşitli | Çeşitli | GPT (%77) | Survey |
| **AkıllıRehber** | **2025** | **TR** | **Lise Eğitim** | **Hybrid (Weaviate)** | **Adaptive Semantic** | **Multi-provider** | **RAGAS** |

---

## AkıllıRehber'in Literatürdeki Konumu

### Özgün Katkılar (Literatür Destekli)

1. **Adaptive Semantic Chunking**
   - Literatür: Fixed-size (EkoBot), Recursive (Bridging Gap), Prepositional (başarısız)
   - AkıllıRehber: Dinamik threshold ile adaptive chunking
   - Referans: Li et al. (2025) chunking'in kritik önemini vurguluyor

2. **Türkçe Eğitim RAG**
   - Literatür: Türkçe RAG var (EkoBot, UniROBO) ama hepsi üniversite odaklı
   - AkıllıRehber: İlk lise odaklı Türkçe RAG sistemi
   - Referans: Swacha Survey'de lise çalışması yok

3. **Hybrid Search (Weaviate)**
   - Literatür: BM25+BERT (Legal), Chroma+Hybrid (Bridging Gap)
   - AkıllıRehber: Weaviate (BM25+Vector, α=0.7 Türkçe için optimize)
   - Referans: Li et al. (2025) Weaviate'i referans gösteriyor

4. **Kapsamlı Türkçe NLP**
   - Literatür: Temel Türkçe desteği (EkoBot), Domain-specific (TULIP)
   - AkıllıRehber: 50+ kısaltma, tüm soru kalıpları (mi/mı/mu/mü)
   - Referans: Bıkmaz et al. (2025) Türkçe morfoloji zorluklarını belgeliyor

5. **RAGAS Değerlendirmesi**
   - Literatür: Swacha Survey'de 4/47 (%8.5) çalışma RAGAS kullanmış
   - AkıllıRehber: Sistematik RAGAS kullanımı
   - Referans: Es et al. (2023) RAGAS framework

6. **Öğretmen/Öğrenci Panel Ayrımı**
   - Literatür: Hiçbir çalışmada yok
   - AkıllıRehber: Ayrı paneller, farklı yetkiler
   - Referans: Li et al. (2025) personalization eksikliğini zorluk olarak belirtiyor

---

## Sonraki Adımlar

1. [x] EkoBot analizi
2. [x] Bridging the Language Gap analizi
3. [x] RAG Chatbots Survey (Swacha) analizi
4. [x] UniROBO analizi
5. [x] TULIP analizi
6. [x] Passage Retrieval Turkish Legal analizi
7. [x] Turkish QA Knowledge Graph analizi
8. [x] Li et al. RAG Education Survey analizi
9. [ ] Karşılaştırma tablosunun design.md'ye entegrasyonu
10. [ ] Literatür taraması bölümü yazımı


---

## Kategori 6: Yeni Analiz Edilen Makaleler (Ocak 2026)

### 6.1 Turk-LettuceDetect: Hallucination Detection Models for Turkish RAG Applications
**Kaynak:** ArXiv 2025
**Yazarlar:** Selva Taş, Mahmut El Huseyni, Özay Ezerceli, Reyhan Bayraktar, Fatma Betül Terzioğlu (Newmind AI, İstanbul)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Türkçe RAG uygulamaları için ilk hallucination detection modeli
- LettuceDetect framework'ünün Türkçe adaptasyonu
- Token-level classification ile hallucination tespiti
- RAGTruth benchmark dataset'inin Türkçe çevirisi

**Teknik Detaylar:**
- **Modeller:** ModernBERT, TurkEmbed4STS, EuroBERT
- **Dataset:** 17,790 örnek (QA, data-to-text, summarization)
- **Context Length:** 8,192 token
- **F1-Score:** 0.7266 (ModernBERT)
- **Task:** Binary token classification (supported/hallucinated)

**Türkçe RAG Zorlukları (Makaleden):**
- Morfolojik karmaşıklık (agglutinative)
- Zengin çekim sistemi (rich inflectional system)
- Düşük kaynak dili (low-resource language)
- Değerlendirme benchmark'larının yetersizliği

**Kritik Bulgular:**
- ModernBERT en iyi performans
- LLM-as-judge yüksek recall ama düşük precision
- Encoder-based modeller daha verimli
- Türkçe için özel hallucination detection gerekli

**AkıllıRehber İçin Çıkarımlar:**
- Hallucination detection entegrasyonu düşünülebilir
- Türkçe RAG kalite kontrolü için önemli referans
- Token-level classification yaklaşımı

---

### 6.2 LuminaURO: AI-Driven Assistant for Urological Diagnostics
**Kaynak:** Anadolu Kliniği Tıp Bilimleri Dergisi, Mayıs 2025, Cilt 30, Sayı 2
**Yazarlar:** Tuncay Soylu et al. (Sağlık Bilimleri Üniversitesi)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Üroloji alanında RAG tabanlı yapay zeka asistanı
- Çok dilli (Türkçe/İngilizce) doküman işleme
- Novel pooling metodolojisi
- Uzman değerlendirmesi ile validasyon

**Teknik Detaylar:**
- **Yanıt Süresi:** 8-15 saniye
- **Değerlendirme:** OESM, Spacy, T5, BERTScore
- **Türkçe Similarity:** 0.9086 (BERTScore)
- **İngilizce Similarity:** 0.9183 (BERTScore)
- **Uzman Değerlendirmesi:** 0.9408 (Türkçe), 0.9444 (İngilizce)

**Kritik Bulgular:**
- Çok dilli RAG sistemi başarılı
- Uzman validasyonu yüksek skorlar
- Domain-specific RAG sağlık alanında etkili
- Follow-up soru önerisi kullanıcı etkileşimini artırıyor

**AkıllıRehber İçin Çıkarımlar:**
- Çok dilli destek metodolojisi
- Uzman değerlendirmesi yaklaşımı
- Domain-specific RAG tasarımı

---

### 6.3 RAGSmith: Framework for Optimal RAG Composition
**Kaynak:** ArXiv, November 2025
**Yazarlar:** Muhammed Yusuf Kartal, Suha Kagan Kose, Korhan Sevinç, Burak Aktas (TOBB ETÜ, Roketsan)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- RAG pipeline optimizasyonu için modüler framework
- Genetik algoritma ile end-to-end architecture search
- 46,080 olası konfigürasyon
- 6 domain (Math, Law, Finance, Medicine, Defense, CS)

**Teknik Detaylar:**
- **Teknik Aileler:** 9 (retrieval, ranking, augmentation, prompting, generation)
- **Optimizasyon:** Genetic algorithm
- **Metrikler:** recall@k, mAP, nDCG, MRR, LLM-Judge, semantic similarity
- **İyileştirme:** +3.8% ortalama (range +1.2% to +6.9%)
- **Retrieval İyileştirme:** +12.5%
- **Generation İyileştirme:** +7.5%

**Kritik Bulgular:**
- Vector retrieval + post-generation reflection/revision en iyi backbone
- Passage compression hiçbir zaman seçilmemiş
- Domain-dependent choices önemli
- Question type performansı etkiliyor

**AkıllıRehber İçin Çıkarımlar:**
- RAG pipeline optimizasyonu metodolojisi
- Genetik algoritma ile hyperparameter tuning
- Domain-specific konfigürasyon önemi

---

### 6.4 LPITutor: LLM-based Personalized Intelligent Tutoring System
**Kaynak:** PeerJ Computer Science, August 2025
**Yazarlar:** Zhensheng Liu, Prateek Agrawal et al.

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- RAG + Prompt Engineering ile kişiselleştirilmiş öğretim sistemi
- Öğrenci seviyesine göre adaptif içerik
- Accuracy, completeness, clarity, difficulty alignment değerlendirmesi

**Teknik Detaylar:**
- **Yaklaşım:** RAG + Advanced Prompt Engineering
- **Kişiselleştirme:** Learner skill level, question complexity
- **Değerlendirme:** Accuracy, completeness, clarity, coherence, relevance
- **Hedef:** Customized learning content

**Kritik Bulgular:**
- RAG + prompt engineering kombinasyonu etkili
- Difficulty alignment önemli
- Real-time adaptasyon mümkün
- Scalable education technology

**AkıllıRehber İçin Çıkarımlar:**
- Kişiselleştirilmiş öğrenme yaklaşımı
- Prompt engineering stratejileri
- Difficulty alignment metodolojisi

---

### 6.5 Human-in-the-Loop Systems for Adaptive Learning Using Generative AI
**Kaynak:** ArXiv, August 2025
**Yazarlar:** Bhavishya Tarun, Haoze Du, Dinesh Kannan, Edward F. Gehringer (NC State University)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- HITL yaklaşımı ile öğrenci geri bildirimi entegrasyonu
- Feedback tagging sistemi
- RAG ile adaptive content delivery
- STEM eğitiminde uygulama

**Teknik Detaylar:**
- **Yaklaşım:** Human-in-the-Loop + RAG
- **Feedback Tags:** Clarity, correctness, tone
- **Retrieval:** RAG-based content retrieval
- **Adaptasyon:** Real-time response modification

**Kritik Bulgular:**
- Öğrenci geri bildirimi AI yanıtlarını iyileştiriyor
- Tagging sistemi yapılandırılmış feedback sağlıyor
- Iterative learning daha etkili
- Student agency önemli

**AkıllıRehber İçin Çıkarımlar:**
- Öğrenci geri bildirimi entegrasyonu
- Feedback tagging sistemi
- Adaptive content delivery

---

### 6.6 Pedagogical Teacher-Student LLM Agents: Genetic Adaptation + Persona-RAG
**Kaynak:** ArXiv, May 2025
**Yazarlar:** Debdeep Sanyal, Agniva Maiti et al. (KIIT, BITS Pilani, Penn State)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Öğretmen-öğrenci LLM agent simülasyonu
- Genetik algoritma ile öğretmen stratejisi optimizasyonu
- Persona-RAG: Öğrenme stiline göre kişiselleştirilmiş retrieval
- 6 öğrenme stili (VARK + Felder-Silverman)

**Teknik Detaylar:**
- **Öğrenme Stilleri:** Read/Write, Visual, Auditory, Kinesthetic, Intuitive, Sequential
- **Kişilik Özellikleri:** Social, Diligent, Independent, Anxious, Curious
- **Optimizasyon:** Genetic algorithm (500 teacher agents, 50 generations)
- **Persona-RAG:** Learning style-aware retrieval

**Kritik Bulgular:**
- Öğrenme stiline göre RAG kişiselleştirmesi etkili
- Genetik algoritma öğretmen stratejisi optimize ediyor
- Persona-RAG standard RAG'den daha iyi
- Simülasyon gerçek dünya validasyonu ile destekleniyor

**AkıllıRehber İçin Çıkarımlar:**
- Öğrenme stiline göre kişiselleştirme
- Persona-RAG yaklaşımı
- Öğretmen-öğrenci etkileşim modelleme

---

### 6.7 Integrating Generative AI in Higher Education: DS-ASST App
**Kaynak:** Education Journal, May 2025
**Yazarlar:** Zhihua Zhang (Kansai University of International Studies)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Data Science eğitimi için custom GPTs-based AI teaching assistant
- RAG + course-specific materials
- Institutional guidelines for GenAI use

**Teknik Detaylar:**
- **Platform:** Custom GPTs
- **Yaklaşım:** RAG + Course materials
- **Değerlendirme Alanları:** Teaching preparation, active learning, data analysis, advanced learning
- **Zorluklar:** Prompt design optimization, hallucination mitigation

**Kritik Bulgular:**
- RAG ile verified knowledge base hallucination azaltıyor
- Instructor workflow iyileşiyor
- Self-directed learning destekleniyor
- Institutional guidelines önemli

**AkıllıRehber İçin Çıkarımlar:**
- Kurumsal kullanım kılavuzları
- Hallucination mitigation stratejileri
- Teaching assistant tasarımı

---

### 6.8 Turkish Health Consultancy: LLM-Based Virtual Doctor Assistants
**Kaynak:** IDAP'24, September 2024
**Yazarlar:** Muhammed Kayra Bulut, Banu Diri (Yıldız Teknik Üniversitesi)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Türkçe sağlık danışmanlığı için LLM karşılaştırması
- 4 farklı model: Llama2, Llama3, Mistral tabanlı
- 321,179 hasta-doktor soru-cevap çifti
- LoRA ile fine-tuning

**Teknik Detaylar:**
- **Modeller:** Meta-Llama-3-8B, SambaLingo-Turkish-Chat, Trendyol-LLM-7b-chat, Turkish-Llama-8b
- **Dataset:** 321,179 QA çifti
- **Fine-tuning:** LoRA
- **Değerlendirme:** ROUGE, Elo, Winning percentage, Expert evaluation

**Kritik Bulgular:**
- SambaLingo-Turkish-Chat en iyi yanıt doğruluğu
- Trendyol-LLM etik açıdan daha başarılı
- Türkçe için özel fine-tuning gerekli
- Domain-specific veri önemli

**AkıllıRehber İçin Çıkarımlar:**
- Türkçe LLM model karşılaştırması
- Fine-tuning metodolojisi
- Domain-specific dataset oluşturma

---

### 6.9 Chatbots in Education: Role and Future (Turkish Survey)
**Kaynak:** İnönü Üniversitesi Eğitim Fakültesi Dergisi, 2024
**Yazarlar:** Emrah Altun, Süleyman Sadi Seferoğlu (Ondokuz Mayıs Üniversitesi, Hacettepe Üniversitesi)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Eğitimde chatbot kullanımının Türkçe literatür taraması
- WoS Core Collection'dan 52 çalışma analizi
- Pedagojik fonksiyonlar, riskler, tartışmalar

**Kritik Bulgular:**
- Chatbotlar mentor, destek, öğrenme fırsatı sağlayıcı olarak sınıflandırılabilir
- Veri gizliliği önemli endişe
- Türkiye'de pilot çalışmalar öneriliyor
- Öğretmen/yönetici endişeleri ele alınmalı

**AkıllıRehber İçin Çıkarımlar:**
- Türkiye bağlamında chatbot kullanımı
- Veri gizliliği gereksinimleri
- Pedagojik fonksiyon tasarımı

---

### 6.10 Turkish Constitutional Court Decision Prediction with AI
**Kaynak:** Turkish Journal of Nature Science, 2025
**Yazarlar:** Emrah Aydemir, Yusuf Kaçar et al. (Sakarya Üniversitesi)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Türk Anayasa Mahkemesi kararlarının AI ile tahmini
- Kabul edilebilirlik ve hak ihlali tahmini
- NLP + Machine Learning

**Teknik Detaylar:**
- **Başarı Oranı:** %91.56 (kabul edilebilirlik), %97.18 (hak ihlali)
- **Veri:** "Olgular" başlığındaki metinler
- **Yaklaşım:** İki aşamalı tahmin (admissibility + merit)

**Kritik Bulgular:**
- Türkçe hukuk metinlerinde yüksek başarı
- İki aşamalı tahmin etkili
- Veri artırma kullanılmamış

**AkıllıRehber İçin Çıkarımlar:**
- Türkçe metin sınıflandırma metodolojisi
- Hukuk domain'i için NLP yaklaşımları

---

### 6.11 Patient Information Leaflets RAG System
**Kaynak:** Journal of Smart Systems Research, 2024
**Yazarlar:** Serhan Ayberk Kılıç, Kasım Serbest (Sakarya Uygulamalı Bilimler Üniversitesi)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Türkçe ilaç prospektüsleri için RAG sistemi
- OCR + Vector embeddings + Hybrid search
- GPT-3.5 turbo ile yanıt üretimi

**Teknik Detaylar:**
- **OCR:** Azure Computer Vision
- **Embedding:** text-embedding-3-small (1536 boyut)
- **Chunking:** RecursiveCharacterTextSplitter (1000 char, 100 overlap)
- **LLM:** GPT-3.5 turbo
- **Veri:** TITCK'den 9 ilaç prospektüsü

**Kritik Bulgular:**
- Türkçe tıbbi RAG sistemi başarılı
- Hybrid search (semantic + full-text) etkili
- Layperson-friendly yanıtlar önemli

**AkıllıRehber İçin Çıkarımlar:**
- Türkçe domain-specific RAG tasarımı
- OCR entegrasyonu
- Chunking parametreleri

---

### 6.12 LLM and RAG-Based QA for Enterprise Knowledge Management
**Kaynak:** IEEE UBMK-2024
**Yazarlar:** Gürkan Şahin, Karya Varol, Burcu Kuleli Pak (Adesso Türkiye)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Kurumsal bilgi yönetimi için RAG tabanlı QA sistemi
- İK ve bilgi güvenliği içerikleri
- Multiple embedding ve LLM modelleri karşılaştırması

**Teknik Detaylar:**
- **Embedding:** text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
- **Vector DB:** Chroma (LangChain)
- **Değerlendirme:** ROUGE, BLEU, accuracy
- **Domain:** HR, Information Security

**Kritik Bulgular:**
- RAG kurumsal bilgi erişimini hızlandırıyor
- Çoklu embedding model karşılaştırması
- Çalışan verimliliği artıyor

**AkıllıRehber İçin Çıkarımlar:**
- Kurumsal RAG tasarımı
- Embedding model seçimi
- Değerlendirme metrikleri

---

### 6.13 Efficient Learning Content Retrieval with Knowledge Injection
**Kaynak:** ArXiv, November 2024
**Yazarlar:** Batuhan Sarıtürk, Rabia Bayraktar, Merve Elmas Erdem (Huawei Türkiye R&D)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Huawei Talent Platform için domain-specific chatbot
- Phi-2 ve Phi-3 modelleri ile QLoRA fine-tuning
- RAG + Fine-tuning kombinasyonu

**Teknik Detaylar:**
- **Modeller:** Phi-2, Phi-3 Mini Instruct
- **Fine-tuning:** QLoRA (500 QA pairs)
- **RAG:** 420 QA pairs (JSON, PPT, DOC)
- **Değerlendirme:** ROUGE, BERTScore, METEOR, BLEU
- **Precision:** 0.84 (Phi-2 + RAG), F1: 0.82

**Kritik Bulgular:**
- RAG + Fine-tuning kombinasyonu en iyi sonuç
- Small language models (SLM) etkili
- QLoRA memory-efficient fine-tuning
- Domain-specific chatbot başarılı

**AkıllıRehber İçin Çıkarımlar:**
- RAG + Fine-tuning kombinasyonu
- Small language model kullanımı
- QLoRA metodolojisi

---

### 6.14 SMART-SLIC: Domain-Specific RAG with KG and Tensor Factorization
**Kaynak:** ArXiv, October 2024
**Yazarlar:** Ryan C. Barron et al. (Los Alamos National Laboratory)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Domain-specific RAG framework
- Knowledge Graph + Vector Store kombinasyonu
- Nonnegative tensor factorization ile ontology oluşturma
- Chain-of-thought prompting agents

**Teknik Detaylar:**
- **Yaklaşım:** KG + VS + RAG
- **Ontology:** NLP + data mining + tensor factorization
- **Retrieval:** K-Nearest Neighbors with Levenshtein metric
- **Domain:** Malware analysis, anomaly detection

**Kritik Bulgular:**
- KG + VS kombinasyonu hallucination azaltıyor
- LLM kullanmadan KG oluşturma
- Domain-specific ontology önemli
- Chain-of-thought reasoning etkili

**AkıllıRehber İçin Çıkarımlar:**
- Knowledge Graph entegrasyonu
- Domain-specific ontology oluşturma
- Hallucination mitigation stratejileri

---

## Güncellenmiş Kapsamlı Karşılaştırma Tablosu

### Tablo: Tüm Analiz Edilen RAG Sistemleri (2024-2025)

| Çalışma | Yıl | Dil | Domain | Retrieval | Chunking | LLM | Değerlendirme | Özgün Katkı |
|---------|-----|-----|--------|-----------|----------|-----|---------------|-------------|
| EkoBot | 2025 | TR | Akademik | Vector | Madde bazlı | GPT-4o | %82 Kosinüs | İlk Türkçe RAG |
| Bridging Gap | 2025 | TR | Genel | Hybrid | Recursive | gemma-3-27b | RAGAS | Fine-tuned TR embedding |
| UniROBO | 2024 | TR | Üniversite | Azure AI | - | GPT + FT | - | Azure entegrasyonu |
| TULIP | 2025 | TR | Finans | - | - | Llama/Qwen | FINTR-EXAMS | TR finans LLM |
| Turk-LettuceDetect | 2025 | TR | Genel | - | - | ModernBERT | F1: 0.73 | TR hallucination detection |
| LuminaURO | 2025 | TR/EN | Sağlık | RAG | - | - | BERTScore | Çok dilli tıbbi RAG |
| RAGSmith | 2025 | EN | Multi | Genetic | - | Multi | +3.8% | Pipeline optimizasyonu |
| LPITutor | 2025 | EN | Eğitim | RAG | - | LLM | Multi-metric | Personalized tutoring |
| HITL Learning | 2025 | EN | STEM | RAG | - | GPT | - | Feedback tagging |
| Persona-RAG | 2025 | EN | Eğitim | Persona-RAG | - | LLM | - | Learning style RAG |
| DS-ASST | 2025 | EN | Data Science | RAG | - | GPT | - | Institutional guidelines |
| TR Health LLM | 2024 | TR | Sağlık | - | - | Multi | ROUGE, Elo | TR sağlık LLM |
| TR Chatbot Survey | 2024 | TR | Eğitim | - | - | - | Survey | TR literatür taraması |
| TR Court AI | 2025 | TR | Hukuk | - | - | ML | %97 | TR hukuk NLP |
| Patient Leaflets | 2024 | TR | Sağlık | Hybrid | 1000 char | GPT-3.5 | - | TR tıbbi RAG |
| Adesso QA | 2024 | TR | Kurumsal | RAG | - | GPT | ROUGE, BLEU | Kurumsal RAG |
| Huawei Learning | 2024 | EN | Eğitim | RAG+FT | - | Phi-2/3 | F1: 0.82 | QLoRA + RAG |
| SMART-SLIC | 2024 | EN | Güvenlik | KG+VS | - | LLM | - | KG + tensor factorization |
| **AkıllıRehber** | **2025** | **TR** | **Lise Eğitim** | **Hybrid (Weaviate)** | **Adaptive Semantic** | **Multi-provider** | **RAGAS** | **Lise + TR + Adaptive** |

---

## AkıllıRehber'in Güncellenmiş Özgün Katkıları

### Literatür Taraması Sonuçları (30+ Makale)

1. **İlk Lise Odaklı Türkçe RAG Sistemi**
   - Tüm Türkçe RAG çalışmaları üniversite odaklı (EkoBot, UniROBO, LuminaURO)
   - Lise eğitimi için özel tasarım literatürde yok

2. **Adaptive Semantic Chunking**
   - Literatür: Fixed-size, Recursive, Prepositional
   - AkıllıRehber: Dinamik threshold ile adaptive chunking
   - Huawei çalışması 1000 char fixed kullanmış

3. **Weaviate Hybrid Search (α=0.7)**
   - Literatür: Chroma, FAISS, Azure AI Search
   - AkıllıRehber: Türkçe için optimize edilmiş alpha
   - Li et al. Survey Weaviate'i referans gösteriyor

4. **Kapsamlı Türkçe NLP**
   - 50+ kısaltma, tüm soru kalıpları (mi/mı/mu/mü)
   - Turk-LettuceDetect Türkçe morfoloji zorluklarını belgeliyor
   - Bridging Gap fine-tuned embedding önemini vurguluyor

5. **Sistematik RAGAS Değerlendirmesi**
   - Swacha Survey: 4/47 (%8.5) çalışma RAGAS kullanmış
   - AkıllıRehber: Kapsamlı RAGAS entegrasyonu

6. **Öğretmen/Öğrenci Panel Ayrımı**
   - Hiçbir çalışmada yok
   - Persona-RAG öğrenme stiline göre kişiselleştirme öneriyor
   - HITL feedback tagging sistemi benzer yaklaşım

7. **3-Bileşenli Mimari**
   - Çoğu çalışma monolitik veya 2-bileşenli
   - AkıllıRehber: Frontend + Backend + Weaviate

---

## Sonraki Adımlar (Güncellenmiş)

1. [x] EkoBot analizi
2. [x] Bridging the Language Gap analizi
3. [x] RAG Chatbots Survey (Swacha) analizi
4. [x] UniROBO analizi
5. [x] TULIP analizi
6. [x] Passage Retrieval Turkish Legal analizi
7. [x] Turkish QA Knowledge Graph analizi
8. [x] Li et al. RAG Education Survey analizi
9. [x] Turk-LettuceDetect analizi
10. [x] LuminaURO analizi
11. [x] RAGSmith analizi
12. [x] LPITutor analizi
13. [x] HITL Learning analizi
14. [x] Persona-RAG analizi
15. [x] DS-ASST analizi
16. [x] TR Health LLM analizi
17. [x] TR Chatbot Survey analizi
18. [x] TR Court AI analizi
19. [x] Patient Leaflets RAG analizi
20. [x] Adesso QA analizi
21. [x] Huawei Learning analizi
22. [x] SMART-SLIC analizi
23. [ ] Karşılaştırma tablosunun design.md'ye entegrasyonu
24. [ ] Literatür taraması bölümü yazımı

---

## Toplam Analiz Edilen Makale Sayısı: 22+

### Kategorilere Göre Dağılım:
- **Türkçe RAG Sistemleri:** 8 makale
- **Eğitimde RAG:** 6 makale
- **Türkçe NLP/LLM:** 4 makale
- **RAG Survey/Framework:** 4 makale


---

## Kategori 7: Son Eklenen Makaleler (Ocak 2026 - Devam)

### 7.1 Local Generative AI Teaching Assistant System (CE-GAITA)
**Kaynak:** Electronics 2025, 14, 3402 (MDPI)
**Yazarlar:** Jing-Wen Wu, Ming-Hseng Tseng (Chung Shan Medical University, Taiwan)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Yerel olarak deploy edilen (local deployment) generative AI öğretim asistanı
- RAG mimarisi ile PDF tabanlı Q&A platformu
- Veri gizliliği ve güvenliği odaklı tasarım
- %86 ortalama doğruluk oranı

**Teknik Detaylar:**
- **Mimari:** Closed-end generative AI (CE-GAITA)
- **Deployment:** Local (on-premise), cloud-free
- **Değerlendirme:** Standardized test question bank
- **Accuracy:** %86 ortalama
- **Bileşenler:** Open-source LLM, embedding models, vector databases

**Kritik Bulgular:**
- Local deployment veri gizliliğini koruyor
- Cloud maliyetlerinden kaçınma
- Eğitim ortamında ölçeklenebilir çözüm
- Privacy-preserving AI teaching aid

**AkıllıRehber İçin Çıkarımlar:**
- Local deployment seçeneği düşünülebilir
- Veri gizliliği gereksinimleri
- Eğitim ortamı için optimizasyon

---

### 7.2 MDKAG: Multimodal Disciplinary Knowledge-Augmented Generation
**Kaynak:** Applied Sciences 2025, 15, 9095 (MDPI)
**Yazarlar:** Xu Zhao, Guozhong Wang, Yufei Lu (Shanghai University of Engineering Science)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Multimodal disciplinary knowledge graph ile RAG entegrasyonu
- Textbook, slides, classroom videos'dan entity extraction
- ERNIE 3.0 ile named entity recognition
- Answer verification modülü ile hallucination azaltma

**Teknik Detaylar:**
- **Entity Extraction:** ERNIE 3.0 model
- **Knowledge Graph:** Multimodal disciplinary KG (MDKG)
- **Retrieval:** Graph-adjacent passages
- **Verification:** Semantic overlap + entity coverage checks
- **Sonuçlar:** %23 hallucination azalma, %11 accuracy artışı

**Kritik Bulgular:**
- Multimodal RAG text-only RAG'den daha iyi
- Knowledge graph hallucination azaltıyor
- Answer verification kritik
- Incremental graph updates dinamik güncelleme sağlıyor

**AkıllıRehber İçin Çıkarımlar:**
- Knowledge graph entegrasyonu düşünülebilir
- Answer verification mekanizması
- Multimodal içerik desteği

---

### 7.3 RAG Framework for Academic Literature Navigation in Data Science
**Kaynak:** ArXiv, December 2024
**Yazarlar:** Ahmet Yasin Aytar, Kamer Kaya, Kemal Kilic (Sabanci University, İstanbul)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- Data science akademik literatür navigasyonu için RAG
- GROBID ile bibliographic data extraction
- Fine-tuned embedding models
- Semantic chunking + abstract-first retrieval
- RAGAS framework ile değerlendirme

**Teknik Detaylar:**
- **Data Extraction:** GROBID (GeneRation Of BIbliographic Data)
- **Embedding:** Fine-tuned models (academic textbooks)
- **Chunking:** Semantic chunking (not recursive)
- **Retrieval:** Abstract-first method
- **LLM:** GPT-4o
- **Değerlendirme:** RAGAS (50 sample questions)
- **Odak Metrik:** Context Relevance

**5-Stage Enhancement Process:**
1. GROBID ile data cleaning/structuring
2. Academic textbook fine-tuning
3. Semantic chunking (meaningful units)
4. Abstract-first retrieval
5. Advanced prompting techniques

**Kritik Bulgular:**
- Semantic chunking recursive'den daha iyi
- Abstract-first retrieval etkili
- Fine-tuned embedding önemli
- RAGAS Context Relevance ana metrik

**AkıllıRehber İçin Çıkarımlar:**
- Semantic chunking yaklaşımı doğrulanıyor
- Fine-tuned embedding önemi
- RAGAS değerlendirme metodolojisi
- Türk araştırmacıların çalışması (Sabancı Üniversitesi)

---

### 7.4 MufassirQAS: RAG for Religious Question-Answering
**Kaynak:** Turkish Journal of Engineering, 2025, 9(3), 544-559
**Yazarlar:** Ahmet Yusuf Alan, Enis Karaarslan, Omer Aydin (Muğla Sıtkı Koçman Üniversitesi)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- İslami sorular için RAG tabanlı QA sistemi
- Türkçe Kuran çevirisi, hadis koleksiyonları, ilmihal metinleri
- Hallucination önleme ve şeffaflık odaklı
- Kaynak referansları ile güvenilirlik

**Teknik Detaylar:**
- **Veri Kaynakları:** Kuran (Türkçe), Kutub-i Sitte hadisleri, İslami ilmihal
- **Yaklaşım:** Vector database-driven RAG
- **Şeffaflık:** Kaynak sayfa numaraları, referans makaleler
- **Güvenlik:** Zararlı/saldırgan içerik önleme prompt'ları
- **Karşılaştırma:** ChatGPT vs MufassirQAS

**Kritik Bulgular:**
- RAG hallucination'ı azaltıyor
- Kaynak referansları güvenilirliği artırıyor
- Domain-specific RAG genel LLM'den daha iyi
- Hassas konularda özel prompt engineering gerekli

**AkıllıRehber İçin Çıkarımlar:**
- Türkçe domain-specific RAG örneği
- Kaynak referansı gösterimi
- Hassas içerik yönetimi
- Prompt engineering stratejileri

---

### 7.5 RAG and LLM Integration (Vakıf Katılım)
**Kaynak:** IEEE ISAS 2024
**Yazarlar:** Büşra Tural, Zeynep Örpek, Zeynep Destan (Vakıf Katılım, İstanbul)

**Analiz Durumu:** ✅ TAMAMLANDI

**Özet:**
- RAG mimarisinin LLM ile entegrasyonu
- Information Retrieval (IR) sistemleri ile birleştirme
- Dinamik ve güncel bilgi erişimi
- Türkçe kurumsal uygulama

**Teknik Detaylar:**
- **IR Modelleri:** TF-IDF, BM25, Vector Space Model
- **LLM Modelleri:** GPT, BERT
- **Retrieval:** Dense Retrieval Model (DRM)
- **Uygulama Alanları:** Q&A, chatbots, content generation

**RAG Avantajları:**
- Statik dataset sınırlamasını aşma
- Gerçek zamanlı harici kaynak erişimi
- Dinamik ve güncel bilgi üretimi
- Hallucination azaltma

**Kritik Bulgular:**
- RAG LLM'lerin statik dataset sınırlamasını çözüyor
- IR + LLM kombinasyonu etkili
- Chatbot ve Q&A uygulamalarında başarılı
- Türkçe kurumsal ortamda uygulanabilir

**AkıllıRehber İçin Çıkarımlar:**
- Türkçe kurumsal RAG örneği
- IR + LLM entegrasyonu
- Dinamik bilgi güncelleme

---

## Güncellenmiş Toplam Makale Sayısı: 27+

### Kategorilere Göre Dağılım (Güncellenmiş):
- **Türkçe RAG Sistemleri:** 11 makale
- **Eğitimde RAG:** 8 makale
- **Türkçe NLP/LLM:** 4 makale
- **RAG Survey/Framework:** 4 makale

### Yeni Eklenen Türkçe Çalışmalar:
1. **Sabancı Üniversitesi RAG (2024)** - Academic literature navigation, RAGAS evaluation
2. **MufassirQAS (2025)** - İslami QA, Türkçe domain-specific RAG
3. **Vakıf Katılım RAG (2024)** - Kurumsal RAG entegrasyonu

### Yeni Eklenen Eğitim RAG Çalışmaları:
4. **CE-GAITA (2025)** - Local deployment, %86 accuracy
5. **MDKAG (2025)** - Multimodal KG + RAG, %23 hallucination azalma

---

## AkıllıRehber'in Güncellenmiş Özgün Konumu

### Literatür Taraması Sonuçları (27+ Makale)

| Özellik | Literatür Durumu | AkıllıRehber Farkı |
|---------|------------------|-------------------|
| **Hedef Kitle** | Tüm çalışmalar üniversite odaklı | İlk lise odaklı Türkçe RAG |
| **Chunking** | Fixed-size, Recursive, Semantic | Adaptive Semantic (dinamik threshold) |
| **Hybrid Search** | Chroma, FAISS, Azure | Weaviate (α=0.7 Türkçe optimize) |
| **Türkçe NLP** | Temel destek | 50+ kısaltma, tüm soru kalıpları |
| **Değerlendirme** | %8.5 RAGAS kullanımı | Sistematik RAGAS entegrasyonu |
| **Panel Ayrımı** | Hiçbir çalışmada yok | Öğretmen/Öğrenci ayrı paneller |
| **Local Deployment** | CE-GAITA örneği | Docker-based deployment |

### Makalede Vurgulanacak Karşılaştırmalar:

1. **vs EkoBot (2025):** Lise vs Üniversite, Adaptive vs Fixed chunking
2. **vs Bridging Gap (2025):** Weaviate vs Chroma, α=0.7 optimizasyonu
3. **vs Sabancı RAG (2024):** Semantic chunking doğrulaması, RAGAS kullanımı
4. **vs MufassirQAS (2025):** Türkçe domain-specific RAG, kaynak referansları
5. **vs CE-GAITA (2025):** Local deployment, privacy-preserving
6. **vs MDKAG (2025):** Answer verification, hallucination azaltma
