# RAGAS Evaluation System - Requirements

## Overview
RAG sisteminin kalitesini değerlendirmek için RAGAS (Retrieval Augmented Generation Assessment) framework'ünü entegre etme.

## Functional Requirements

### FR-1: Test Seti Yönetimi
- Kullanıcı ders bazında soru-cevap test setleri oluşturabilmeli
- Her test seti birden fazla soru-beklenen cevap çifti içerebilmeli
- Test setleri JSON formatında import/export edilebilmeli
- Ground truth (beklenen cevap) ve context (beklenen kaynak) tanımlanabilmeli

### FR-2: Çoklu Test Çalıştırma
- Bir test seti üzerinde toplu değerlendirme yapılabilmeli
- Farklı ayarlarla (chunk size, overlap, embedding model) karşılaştırmalı testler
- Test sonuçları geçmişi tutulmalı
- Paralel test çalıştırma desteği

### FR-3: RAGAS Metrikleri
- **Faithfulness**: Cevabın kaynaklara sadakati (0-1)
- **Answer Relevancy**: Cevabın soruyla ilgisi (0-1)
- **Context Precision**: Getirilen context'in doğruluğu (0-1)
- **Context Recall**: Gerekli context'in ne kadarının getirildiği (0-1)
- **Answer Correctness**: Cevabın doğruluğu (ground truth ile karşılaştırma)

### FR-4: Sonuç Görselleştirme
- Metrik skorları tablo ve grafik olarak gösterilmeli
- Test geçmişi karşılaştırması
- Ders bazında performans trendi
- Detaylı soru bazında analiz

### FR-5: Raporlama
- PDF/Excel formatında rapor export
- Özet ve detaylı rapor seçenekleri

## Non-Functional Requirements

### NFR-1: Performance
- 100 soruluk test seti 5 dakika içinde tamamlanmalı
- Paralel işleme ile hız optimizasyonu

### NFR-2: Scalability
- Docker container olarak çalışmalı
- Backend ile API üzerinden iletişim

### NFR-3: Reliability
- Test kesintiye uğrarsa kaldığı yerden devam edebilmeli
- Hata durumunda detaylı log

## User Stories

### US-1: Test Seti Oluşturma
**As a** öğretmen
**I want to** dersim için soru-cevap test seti oluşturmak
**So that** RAG sistemimin kalitesini ölçebilirim

### US-2: Toplu Değerlendirme
**As a** öğretmen
**I want to** test setimi çalıştırıp sonuçları görmek
**So that** sistemin hangi sorularda başarısız olduğunu anlayabilirim

### US-3: Karşılaştırmalı Test
**As a** öğretmen
**I want to** farklı ayarlarla testleri karşılaştırmak
**So that** en iyi konfigürasyonu bulabilirim

### US-4: Performans Takibi
**As a** öğretmen
**I want to** zaman içindeki performans değişimini görmek
**So that** iyileştirmelerin etkisini ölçebilirim
