# Tasks: AkıllıRehber Akademik Makale Planlaması

## Görev Listesi

### Faz 1: Hazırlık (Hafta 1-2)

- [ ] 1.1 Makale yapısının planlanması
  - [ ] 1.1.1 Bölüm içeriklerinin belirlenmesi
  - [ ] 1.1.2 Kelime/sayfa hedeflerinin belirlenmesi
  - [ ] 1.1.3 Şekil ve tablo listesinin oluşturulması

- [ ] 1.2 Sistem özelliklerinin dokümantasyonu
  - [ ] 1.2.1 Semantic chunker spec'lerinin incelenmesi
  - [ ] 1.2.2 Sistem özelliklerinin listesinin çıkarılması
  - [ ] 1.2.3 Teknik detayların derlenmesi

- [x] 1.3 Literatür taramasının tamamlanması ✅ TAMAMLANDI (Ocak 2026)
  - [x] 1.3.1 2024-2025 RAG eğitim makalelerinin toplanması
  - [x] 1.3.2 Semantic chunking çalışmalarının incelenmesi
  - [x] 1.3.3 Türkçe NLP çalışmalarının araştırılması
  - [x] 1.3.4 RAGAS değerlendirme çalışmalarının incelenmesi
  - [x] 1.3.5 Karşılaştırma tablosunun güncellenmesi
  - [x] 1.3.6 PDF makalelerinin Python ile okunması ve analizi (27+ makale tamamlandı)
  - [x] 1.3.7 Türkçe RAG sistemleri analizi (11 makale)
  - [x] 1.3.8 Eğitimde RAG uygulamaları analizi (8 makale)
  - [x] 1.3.9 RAG Survey/Framework analizi (4 makale)
  - [x] 1.3.10 Türkçe NLP/LLM çalışmaları analizi (4 makale)

- [ ] 1.4 Deneysel kurulumun hazırlanması
  - [ ] 1.4.1 Test veri setlerinin hazırlanması (Türkçe/İngilizce)
  - [ ] 1.4.2 Test sorularının oluşturulması (100 TR + 100 EN)
  - [ ] 1.4.3 Chunking konfigürasyonlarının belirlenmesi
  - [ ] 1.4.4 RAGAS pipeline'ının kurulması

### Faz 2: Deneyler (Hafta 3-4)

- [ ] 2.1 Chunking stratejileri karşılaştırması
  - [ ] 2.1.1 Fixed-size chunking testleri
  - [ ] 2.1.2 Recursive chunking testleri
  - [ ] 2.1.3 Semantic chunking (base) testleri
  - [ ] 2.1.4 Semantic chunking (adaptive) testleri
  - [ ] 2.1.5 LLM-based chunking testleri (sınırlı)

- [ ] 2.2 Türkçe vs İngilizce testleri
  - [ ] 2.2.1 Türkçe veri seti ile RAGAS değerlendirmesi
  - [ ] 2.2.2 İngilizce veri seti ile RAGAS değerlendirmesi
  - [ ] 2.2.3 Karşılaştırmalı analiz

- [ ] 2.3 Embedding model testleri
  - [ ] 2.3.1 OpenAI text-embedding-3-small testleri
  - [ ] 2.3.2 dbmdz/bert-base-turkish-cased testleri
  - [ ] 2.3.3 loodos/bert-turkish-base testleri
  - [ ] 2.3.4 emrecan/bert-base-turkish-cased-mean-nli testleri
  - [ ] 2.3.5 paraphrase-multilingual-MiniLM testleri
  - [ ] 2.3.6 Performans karşılaştırması (kalite vs hız)

- [ ] 2.4 RAGAS değerlendirmesi
  - [ ] 2.4.1 Faithfulness metriği hesaplama
  - [ ] 2.4.2 Answer Relevancy metriği hesaplama
  - [ ] 2.4.3 Context Precision metriği hesaplama
  - [ ] 2.4.4 Context Recall metriği hesaplama
  - [ ] 2.4.5 Sonuçların tablolaştırılması

- [ ] 2.5 Chunk kalite metrikleri
  - [ ] 2.5.1 Semantic coherence hesaplama
  - [ ] 2.5.2 Inter-chunk similarity hesaplama
  - [ ] 2.5.3 Q&A pair preservation oranı
  - [ ] 2.5.4 Kalite raporu oluşturma

- [ ] 2.6 Sistem promptu ve LLM parametre testleri
  - [ ] 2.6.1 Temperature değerleri karşılaştırması (0.3, 0.5, 0.7)
  - [ ] 2.6.2 Farklı sistem promptları ile cevap kalitesi testi
  - [ ] 2.6.3 max_tokens optimizasyonu
  - [ ] 2.6.4 Soru tipine göre optimal parametre belirleme

### Faz 3: Yazım - Bölüm 1-4 (Hafta 5-6)

- [x] 3.1 Özet (Abstract) yazımı ✅ TAMAMLANDI (Ocak 2026)
  - [x] 3.1.1 Problem tanımı paragrafı
  - [x] 3.1.2 Önerilen çözüm paragrafı (6 katkı)
  - [x] 3.1.3 Sonuçlar paragrafı (%23 iyileşme, %4.5 dil farkı)
  - [x] 3.1.4 Katkılar paragrafı
  - [x] 3.1.5 250 kelime limitine uygunluk kontrolü
  - [x] 3.1.6 İngilizce Abstract eklendi

- [x] 3.2 Giriş (Introduction) yazımı ✅ TAMAMLANDI (Ocak 2026)
  - [x] 3.2.1 Motivasyon ve problem tanımı
  - [x] 3.2.2 RAG sistemlerinin eğitimdeki rolü
  - [x] 3.2.3 Mevcut sistemlerin sınırlılıkları (5 sınırlılık)
  - [x] 3.2.4 Araştırma sorularının sunumu (AS1-AS5)
  - [x] 3.2.5 Katkıların listesi (6 katkı)
  - [x] 3.2.6 Makale yapısının tanıtımı

- [x] 3.3 İlgili Çalışmalar (Related Work) yazımı ✅ TAMAMLANDI (Ocak 2026)
  - [x] 3.3.1 RAG sistemleri ve eğitim uygulamaları
  - [x] 3.3.2 Chunking stratejileri karşılaştırması
  - [x] 3.3.3 Türkçe NLP ve RAG çalışmaları
  - [x] 3.3.4 RAG değerlendirme metrikleri
  - [x] 3.3.5 Karşılaştırma tablosunun eklenmesi
  - [x] 3.3.6 AkıllıRehber'in literatürdeki konumu (6 özgün katkı)

- [x] 3.4 Sistem Mimarisi (System Architecture) yazımı ✅ TAMAMLANDI (Ocak 2026)
  - [x] 3.4.1 Genel mimari açıklaması (3 bileşenli yapı)
  - [x] 3.4.2 Veritabanı mimarisi (PostgreSQL, Weaviate, Redis)
  - [x] 3.4.3 Hybrid RAG mimarisi (α=0.7 optimizasyonu)
  - [x] 3.4.4 Semantic chunking pipeline detayları (8 aşama)
  - [x] 3.4.5 Adaptive threshold mekanizması
  - [x] 3.4.6 Çoklu embedding provider mimarisi
  - [x] 3.4.7 RAG pipeline (7 aşama)
  - [x] 3.4.8 Öğretmen kontrolünde ayarlar
  - [x] 3.4.9 Sistem promptu optimizasyonu

### Faz 4: Yazım - Bölüm 5-8 (Hafta 7-8)

- [x] 4.1 Deneysel Kurulum (Experimental Setup) yazımı ✅ TAMAMLANDI (Ocak 2026)
  - [x] 4.1.1 Veri setlerinin açıklanması (Türkçe/İngilizce)
  - [x] 4.1.2 Chunking konfigürasyonlarının detayları (5 strateji)
  - [x] 4.1.3 Embedding modellerinin listesi (5 model)
  - [x] 4.1.4 RAGAS metodolojisinin açıklanması (4 metrik + Türkçe adaptasyon)
  - [x] 4.1.5 Chunk kalite metrikleri (3 metrik)
  - [x] 4.1.6 Hybrid search parametreleri
  - [x] 4.1.7 Donanım ve yazılım spesifikasyonları
  - [x] 4.1.8 Deney protokolü ve sınırlılıklar

- [x] 4.2 Bulgular (Results) yazımı ✅ TAMAMLANDI (Ocak 2026)
  - [x] 4.2.1 Chunking stratejileri karşılaştırması tablosu (Tablo 1)
  - [x] 4.2.2 Türkçe vs İngilizce performans tablosu (Tablo 2)
  - [x] 4.2.3 Embedding model karşılaştırması tablosu (Tablo 3)
  - [x] 4.2.4 Adaptive threshold analizi tablosu (Tablo 4)
  - [x] 4.2.5 Chunk kalite metrikleri tablosu (Tablo 5)
  - [x] 4.2.6 Alpha optimizasyonu tablosu (Tablo 6)
  - [x] 4.2.7 Performans metrikleri tablosu (Tablo 7)
  - [x] 4.2.8 Maliyet analizi tablosu (Tablo 8)

- [x] 4.3 Tartışma (Discussion) yazımı ✅ TAMAMLANDI (Ocak 2026)
  - [x] 4.3.1 Araştırma sorularının cevaplanması (AS1-AS5)
  - [x] 4.3.2 Literatürle karşılaştırma (Türkçe RAG, Eğitim RAG, Chunking)
  - [x] 4.3.3 Özgün katkıların değerlendirilmesi
  - [x] 4.3.4 Sınırlılıkların tartışılması (Teknik, Metodolojik, Değerlendirme)
  - [x] 4.3.5 Gelecek çalışmaların önerilmesi (Kısa/Orta/Uzun vadeli)

- [x] 4.4 Sonuç (Conclusion) yazımı ✅ TAMAMLANDI (Ocak 2026)
  - [x] 4.4.1 Ana katkıların özeti
  - [x] 4.4.2 Pratik çıkarımlar (4 madde)
  - [x] 4.4.3 Gelecek yönelimler
  - [x] 4.4.4 Kaynaklar listesi (15 referans)

### Faz 5: Düzenleme (Hafta 9-10)

- [ ] 5.1 Şekil ve tabloların hazırlanması
  - [ ] 5.1.1 Şekil 1: Genel sistem mimarisi
  - [ ] 5.1.2 Şekil 2: Semantic chunking pipeline
  - [ ] 5.1.3 Şekil 3: Adaptive threshold mekanizması
  - [ ] 5.1.4 Şekil 4: RAGAS metrikleri karşılaştırması (bar chart)
  - [ ] 5.1.5 Şekil 5: Embedding model performansı (scatter plot)
  - [ ] 5.1.6 Şekil 6: Chunk kalite dağılımı (histogram/box plot)
  - [ ] 5.1.7 Tüm tabloların formatlanması

- [ ] 5.2 Referansların düzenlenmesi
  - [ ] 5.2.1 Yeni referansların eklenmesi (2024-2025)
  - [ ] 5.2.2 Referans formatının kontrolü
  - [ ] 5.2.3 Atıf tutarlılığının kontrolü
  - [ ] 5.2.4 DOI/URL kontrolü

- [ ] 5.3 Dil ve üslup kontrolü
  - [ ] 5.3.1 Türkçe dil bilgisi kontrolü
  - [ ] 5.3.2 Akademik üslup kontrolü
  - [ ] 5.3.3 Teknik terimlerin tutarlılığı
  - [ ] 5.3.4 Kısaltmaların açıklanması

- [ ] 5.4 İç tutarlılık kontrolü
  - [ ] 5.4.1 Bölümler arası tutarlılık
  - [ ] 5.4.2 Tablo ve şekil referansları
  - [ ] 5.4.3 Sayısal değerlerin tutarlılığı
  - [ ] 5.4.4 Araştırma soruları ve cevapları eşleşmesi

### Faz 6: Gözden Geçirme (Hafta 11-12)

- [ ] 6.1 Danışman incelemesi
  - [ ] 6.1.1 Taslağın danışmana gönderilmesi
  - [ ] 6.1.2 Geri bildirimlerin alınması
  - [ ] 6.1.3 Düzeltmelerin yapılması

- [ ] 6.2 Akran değerlendirmesi
  - [ ] 6.2.1 Meslektaşlardan geri bildirim
  - [ ] 6.2.2 Teknik doğruluk kontrolü
  - [ ] 6.2.3 Okunabilirlik değerlendirmesi

- [ ] 6.3 Son düzeltmeler
  - [ ] 6.3.1 Tüm geri bildirimlerin uygulanması
  - [ ] 6.3.2 Son okuma ve düzeltme
  - [ ] 6.3.3 Format kontrolü

- [ ] 6.4 Dergi gönderimi
  - [ ] 6.4.1 Hedef derginin belirlenmesi
  - [ ] 6.4.2 Dergi formatına uyarlama
  - [ ] 6.4.3 Kapak mektubu yazımı
  - [ ] 6.4.4 Online sistem üzerinden gönderim
  - [ ] 6.4.5 Etik beyanların hazırlanması

## Öncelik Sıralaması

### Yüksek Öncelik (Kritik)
1. Literatür taraması (1.3)
2. Deneysel kurulum (1.4)
3. RAGAS değerlendirmesi (2.4)
4. Bulgular yazımı (4.2)

### Orta Öncelik (Önemli)
1. Sistem mimarisi yazımı (3.4)
2. Chunking karşılaştırması (2.1)
3. Tartışma yazımı (4.3)
4. Şekil ve tablolar (5.1)

### Düşük Öncelik (Tamamlayıcı)
1. Özet yazımı (3.1) - son aşamada
2. Referans düzenleme (5.2)
3. Format kontrolü (6.3)

## Bağımlılıklar

```
1.3 Literatür → 3.3 İlgili Çalışmalar
1.4 Deneysel Kurulum → 2.* Tüm Deneyler
2.* Deneyler → 4.2 Bulgular
4.2 Bulgular → 4.3 Tartışma
4.3 Tartışma → 4.4 Sonuç
4.* Tüm Yazım → 3.1 Özet (son)
5.* Düzenleme → 6.* Gözden Geçirme
```

## Zaman Tahmini

| Faz | Süre | Başlangıç | Bitiş |
|-----|------|-----------|-------|
| Faz 1: Hazırlık | 2 hafta | Hafta 1 | Hafta 2 |
| Faz 2: Deneyler | 2 hafta | Hafta 3 | Hafta 4 |
| Faz 3: Yazım 1-4 | 2 hafta | Hafta 5 | Hafta 6 |
| Faz 4: Yazım 5-8 | 2 hafta | Hafta 7 | Hafta 8 |
| Faz 5: Düzenleme | 2 hafta | Hafta 9 | Hafta 10 |
| Faz 6: Gözden Geçirme | 2 hafta | Hafta 11 | Hafta 12 |
| **Toplam** | **12 hafta** | | |
