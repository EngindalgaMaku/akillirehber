# Requirements Document: AkıllıRehber Akademik Makale Planlaması

## Introduction

Bu doküman, AkıllıRehber RAG sisteminin Türkiye'de bilimsel bir dergide yayınlanması için makale planlamasını içermektedir. Sistem baştan kurulduğu için içerik önemli ölçüde güncellenecek ve güncel literatürle uyumlu hale getirilecektir.

## Glossary

- **RAG**: Retrieval-Augmented Generation - İçe Aktarımlı Üretim
- **Semantic_Chunking**: Embedding tabanlı anlamsal metin parçalama
- **RAGAS**: RAG Assessment Framework - RAG değerlendirme çerçevesi
- **Adaptive_Threshold**: Metin özelliklerine göre dinamik eşik belirleme
- **Q&A_Detection**: Soru-cevap çiftlerinin otomatik tespiti
- **Embedding_Provider**: Vektör temsili sağlayan servis

## Sistem Özellikleri (Makale Katkıları)

### 1. Gelişmiş Semantic Chunking Mimarisi
- **Embedding tabanlı benzerlik analizi** ile anlamsal parçalama
- **Adaptive threshold** mekanizması (metin çeşitliliğine göre dinamik eşik)
- **Buffer-based sentence grouping** (bağlam korumalı cümle gruplama)
- **Percentile-based breakpoint detection** (yüzdelik dilim tabanlı kesim noktası)

### 2. Kapsamlı Türkçe Dil Desteği
- 50+ Türkçe kısaltma desteği (Dr., Prof., örn., vs., vb., vd., bkz., krş., A.Ş., Ltd., Şti., etc.)
- Tüm Türkçe soru kalıpları (mi/mı/mu/mü + tüm çekimler)
- Türkçe karakter koruması (ç, ğ, ı, ö, ş, ü)
- Ondalık sayı, URL, e-posta koruması

### 3. Çoklu Embedding Provider Desteği
- OpenRouter, OpenAI, HuggingFace, Local Models
- Türkçe-optimize modeller (dbmdz/bert-base-turkish-cased, loodos/bert-turkish-base)
- Otomatik fallback mekanizması
- Embedding cache sistemi

### 4. Gelişmiş Q&A Detection
- Kapsamlı Türkçe soru kalıpları (ne, nasıl, neden, niçin, kim, nerede, hangi, kaç, acaba, merak ediyorum, vb.)
- Semantic similarity ile cevap tespiti
- Q&A çiftlerinin aynı chunk'ta tutulması

### 5. Chunk Kalite Metrikleri
- Semantic coherence score
- Inter-chunk similarity
- Topic distribution analysis
- Quality report generation

### 6. Sistem Promptu Optimizasyonu ve LLM Parametre Yönetimi
- Ders bazlı özelleştirilebilir sistem promptları
- Temperature optimizasyonu (soru tipine göre dinamik ayar)
- LLM parametreleri (max_tokens, top_p, frequency_penalty, presence_penalty)
- Cevap kalitesi iyileştirme stratejileri
- Öğretmen kontrolünde prompt mühendisliği

### 7. PostgreSQL ile Merkezi Veritabanı Mimarisi
- Tüm metadata tek veritabanında yönetim
- İlişkisel veri modeli (Users, Courses, Documents, Chunks, Settings)
- Foreign key ile veri tutarlılığı
- Alembic migration yönetimi
- Kolay yedekleme ve ölçeklenebilirlik

## Makale Yapısı Gereksinimleri

### Requirement 1: Güncellenmiş Özet (Abstract)

**User Story:** Bir okuyucu olarak, makalenin ana katkılarını ve sonuçlarını hızlıca anlamak istiyorum.

#### Acceptance Criteria

1. THE Abstract SHALL highlight the novel semantic chunking approach with adaptive threshold
2. THE Abstract SHALL mention multi-provider embedding support with Turkish optimization
3. THE Abstract SHALL include updated RAGAS evaluation results
4. THE Abstract SHALL emphasize the microservice architecture and teacher-controlled dynamic content
5. THE Abstract SHALL be limited to 250 words in Turkish

### Requirement 2: Güncellenmiş Giriş (Introduction)

**User Story:** Bir araştırmacı olarak, çalışmanın motivasyonunu ve araştırma sorularını net anlamak istiyorum.

#### Acceptance Criteria

1. THE Introduction SHALL present the problem of personalized education at high school level
2. THE Introduction SHALL explain the limitations of existing RAG systems
3. THE Introduction SHALL introduce the novel contributions of AkıllıRehber
4. THE Introduction SHALL present updated research questions reflecting new system capabilities
5. THE Introduction SHALL reference recent 2024-2025 literature

### Requirement 3: Güncellenmiş İlgili Çalışmalar (Related Work)

**User Story:** Bir hakem olarak, çalışmanın literatürdeki konumunu ve özgün katkısını değerlendirmek istiyorum.

#### Acceptance Criteria

1. THE Related Work SHALL include recent 2024-2025 RAG education papers
2. THE Related Work SHALL compare semantic chunking approaches in literature
3. THE Related Work SHALL discuss Turkish NLP and RAG studies
4. THE Related Work SHALL highlight gaps that AkıllıRehber addresses
5. THE Related Work SHALL update the comparison table with new studies

### Requirement 4: Güncellenmiş Sistem Mimarisi (System Architecture)

**User Story:** Bir teknik okuyucu olarak, sistemin nasıl çalıştığını detaylı anlamak istiyorum.

#### Acceptance Criteria

1. THE System Architecture SHALL describe the new semantic chunking pipeline
2. THE System Architecture SHALL explain the adaptive threshold mechanism
3. THE System Architecture SHALL detail the multi-provider embedding system
4. THE System Architecture SHALL present the Q&A detection algorithm
5. THE System Architecture SHALL include updated architecture diagrams

### Requirement 5: Güncellenmiş Deneysel Kurulum (Experimental Setup)

**User Story:** Bir araştırmacı olarak, deneylerin nasıl yapıldığını ve tekrarlanabilirliğini anlamak istiyorum.

#### Acceptance Criteria

1. THE Experimental Setup SHALL describe the new chunking configuration
2. THE Experimental Setup SHALL list all embedding models tested
3. THE Experimental Setup SHALL explain the RAGAS evaluation methodology
4. THE Experimental Setup SHALL describe the test datasets (Turkish and English)
5. THE Experimental Setup SHALL include hardware and software specifications

### Requirement 6: Güncellenmiş Bulgular (Results)

**User Story:** Bir okuyucu olarak, sistemin performansını ve karşılaştırmalı sonuçları görmek istiyorum.

#### Acceptance Criteria

1. THE Results SHALL present RAGAS metrics for the new semantic chunking system
2. THE Results SHALL compare different chunking strategies (semantic vs. fixed-size vs. LLM-based)
3. THE Results SHALL show Turkish vs. English performance comparison
4. THE Results SHALL include chunk quality metrics (coherence, similarity)
5. THE Results SHALL present ablation studies for key components

### Requirement 7: Güncellenmiş Tartışma (Discussion)

**User Story:** Bir araştırmacı olarak, sonuçların yorumunu ve sınırlılıkları anlamak istiyorum.

#### Acceptance Criteria

1. THE Discussion SHALL interpret the RAGAS results in context of literature
2. THE Discussion SHALL discuss the effectiveness of semantic chunking for Turkish
3. THE Discussion SHALL address the adaptive threshold mechanism's impact
4. THE Discussion SHALL acknowledge limitations and future work
5. THE Discussion SHALL answer the research questions posed in introduction

### Requirement 8: Güncellenmiş Sonuç (Conclusion)

**User Story:** Bir okuyucu olarak, çalışmanın ana katkılarını ve gelecek yönelimlerini öğrenmek istiyorum.

#### Acceptance Criteria

1. THE Conclusion SHALL summarize the main contributions
2. THE Conclusion SHALL highlight the novel aspects of the system
3. THE Conclusion SHALL present practical implications for education
4. THE Conclusion SHALL outline future research directions
5. THE Conclusion SHALL be concise (max 300 words)

## Yeni Araştırma Soruları (Güncellenmiş)

### AS1 (Güncellenmiş)
AkıllıRehber'in embedding tabanlı semantic chunking yaklaşımı, sabit boyutlu parçalama ve LLM tabanlı parçalamaya kıyasla RAGAS metrikleri açısından ne düzeyde performans farkı sunmaktadır?

### AS2 (Güncellenmiş)
Adaptive threshold mekanizması, farklı metin türleri ve dil yapıları için chunk kalitesini nasıl etkilemektedir?

### AS3 (Mevcut)
RAG tabanlı soru–cevap sistemlerinin performansı, farklı dil altyapılarına sahip ders materyalleri kullanıldığında nasıl bir değişim göstermektedir?

### AS4 (Yeni)
Çoklu embedding provider desteği ve Türkçe-optimize modeller, Türkçe eğitim içerikleri için retrieval kalitesini nasıl etkilemektedir?

### AS5 (Mevcut - Genişletilmiş)
RAG tabanlı eğitim asistanlarının değerlendirilmesinde, RAGAS metrikleri Türkçe içerikler için güvenilir ve anlamlı sonuçlar üretmekte midir?

## Literatür Güncellemeleri (2024-2025)

### Eklenecek Yeni Çalışmalar

1. **Swacha & Gracel (2025)** - "RAG Chatbots for Education: A Survey of Applications" - MDPI Applied Sciences
2. **Hierarchical Chunking (2025)** - "Evaluating and Enhancing RAG with Hierarchical Chunking" - arXiv
3. **Semantic Chunking Cost Analysis (2024)** - "Is Semantic Chunking Worth the Computational Cost?" - ResearchGate
4. **RAG Evaluation Survey (2025)** - "A Systematic Literature Review of RAG: Techniques, Metrics, and Challenges" - MDPI
5. **Document Chunking Strategies (2025)** - "Best Chunking Strategies for RAG in 2025" - Firecrawl

### Güncellenecek Karşılaştırma Tablosu

| Çalışma | Yıl | Chunking Yöntemi | Dil Desteği | Değerlendirme | Temel Katkı |
|---------|-----|------------------|-------------|---------------|-------------|
| LPITutor | 2025 | Fixed-size | EN | %94 Doğruluk | Çift katmanlı prompt |
| Gaita | 2024 | Recursive | EN | - | Kişiselleştirilmiş CS eğitimi |
| EkoBot | 2025 | Fixed-size | TR | %82 Doğruluk | İlk Türkçe akademik RAG |
| MOOC RAG | 2025 | Semantic | EN | - | Kullanıcı davranış analizi |
| **AkıllıRehber** | **2025** | **Adaptive Semantic** | **TR/EN** | **%88 RAGAS** | **Adaptive threshold, Türkçe Q&A, Multi-provider** |

## Makale Özgün Katkıları (Contributions)

### Katkı 1: Adaptive Semantic Chunking
- Metin özelliklerine göre dinamik threshold belirleme
- Vocabulary diversity ve sentence length analizi
- Percentile-based breakpoint detection

### Katkı 2: Kapsamlı Türkçe NLP Desteği
- 50+ Türkçe kısaltma ve özel durum
- Tüm Türkçe soru kalıpları (mi/mı/mu/mü + çekimler)
- Türkçe-optimize embedding modelleri

### Katkı 3: Çoklu Provider Mimarisi
- OpenRouter, OpenAI, HuggingFace, Local Models
- Otomatik fallback ve cache mekanizması
- Maliyet ve performans optimizasyonu

### Katkı 4: Gelişmiş Q&A Detection
- Semantic similarity ile cevap tespiti
- Q&A çiftlerinin bağlam korumalı tutulması
- Nested question handling

### Katkı 5: Chunk Kalite Metrikleri
- Semantic coherence score
- Inter-chunk similarity
- Quality report generation

### Katkı 6: Sistem Promptu Optimizasyonu
- Ders ve seviye bazlı özelleştirilebilir sistem promptları
- Temperature optimizasyonu (0.0-1.0 aralığında soru tipine göre)
- LLM parametre yönetimi (max_tokens, top_p, penalties)
- Cevap kalitesi iyileştirme stratejileri

### Katkı 7: PostgreSQL ile Merkezi Veritabanı Mimarisi
- Dağınık veri depolamadan merkezi yönetime geçiş
- İlişkisel veri modeli ile veri tutarlılığı
- Alembic migration yönetimi
- Production-ready ölçeklenebilirlik

## Success Metrics

1. **Makale Kabul Oranı**: Hedef dergi için uygun format ve kalite
2. **Literatür Güncelliği**: 2024-2025 çalışmalarının %50+ dahil edilmesi
3. **Teknik Derinlik**: Yeni sistem özelliklerinin detaylı açıklanması
4. **Deneysel Geçerlilik**: RAGAS sonuçlarının güvenilir raporlanması
5. **Özgün Katkı**: En az 3 net özgün katkının vurgulanması
