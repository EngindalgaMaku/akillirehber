# Bilimsel Makale için RAG Sistemi İyileştirmeleri - Gereksinimler

## Giriş

Bu doküman, Türkiye'de lise düzeyinde RAG tabanlı chatbot kullanımı üzerine hazırlanacak bilimsel makale için mevcut sistemin analizini ve bilimsel değer katacak iyileştirme önerilerini içermektedir.

## Mevcut Sistemin Güçlü Yönleri

### 1. Kapsamlı RAGAS Entegrasyonu
- **Faithfulness**: Cevabın kaynak metinlere sadakati
- **Answer Relevancy**: Cevabın soruyla ilgisi
- **Context Precision**: Alınan bağlamın hassasiyeti
- **Context Recall**: Bağlam hatırlama oranı
- **Answer Correctness**: Cevap doğruluğu

### 2. Gelişmiş Chunking Stratejileri
- **Recursive Chunking**: Hiyerarşik metin bölümleme
- **Semantic Chunking**: Anlam tabanlı bölümleme (embedding similarity)
- **Agentic Chunking**: LLM destekli akıllı bölümleme
- **Sentence-based**: Cümle bazlı bölümleme

### 3. Hibrit Arama Sistemi
- Vector search (semantic)
- Keyword search (BM25)
- Hybrid search (alpha parametresi ile ayarlanabilir)

### 4. Çoklu LLM Desteği
- OpenRouter (GPT-4, Claude, Llama)
- Claude.gg
- API Claude.gg (GPT-5, O3, Grok, DeepSeek, Gemini)
- Groq
- OpenAI

### 5. Hızlı Test ve Değerlendirme
- Quick Test özelliği
- Batch evaluation
- Test set yönetimi
- Sonuç karşılaştırma

## Bilimsel Makale için Önerilen İyileştirmeler

### Gereksinim 1: Türkçe Dil Desteği Optimizasyonu

**Kullanıcı Hikayesi:** Araştırmacı olarak, Türkçe dilin özelliklerini dikkate alan bir RAG sistemi geliştirmek istiyorum, böylece lise öğrencileri için daha etkili sonuçlar elde edebilirim.

#### Kabul Kriterleri

1.1. WHEN Türkçe metin işlendiğinde THEN sistem Türkçe karakterleri (ç, ğ, ı, ö, ş, ü) doğru şekilde işlemeli

1.2. WHEN chunking yapıldığında THEN Türkçe cümle yapısı ve noktalama kuralları dikkate alınmalı

1.3. WHEN embedding oluşturulduğunda THEN Türkçe için optimize edilmiş modeller kullanılmalı

1.4. WHEN stop words filtrelenmesi yapıldığında THEN Türkçe stop words listesi kullanılmalı

1.5. WHEN semantic similarity hesaplanırken THEN Türkçe morfolojik yapı dikkate alınmalı

1.6. WHEN öğrenci soru sorduğunda THEN sistem kişisel bilgileri (TC kimlik, telefon, e-posta, isim) tespit etmeli ve maskelemeli

1.7. WHEN kişisel bilgi tespit edildiğinde THEN sistem öğrenciyi uyarmalı ve maskelenmiş metni LLM'e göndermeli

1.8. WHEN uygunsuz içerik tespit edildiğinde THEN sistem içeriği filtrelemeli veya reddetmeli

1.9. WHEN PII tespiti yapıldığında THEN sistem güvenlik logu oluşturmalı (KVKK uyumluluğu için)

1.10. WHEN API'ye veri gönderilmeden önce THEN sistem gizlilik kontrolü yapmalı ve öğrenci verilerini korumalı

### Gereksinim 2: Eğitim Seviyesi Adaptasyonu

**Kullanıcı Hikayesi:** Öğretmen olarak, öğrenci seviyesine göre cevap karmaşıklığını ayarlayabilmek istiyorum, böylece her öğrenci kendi seviyesinde öğrenebilir.

#### Kabul Kriterleri

2.1. WHEN öğrenci profili oluşturulduğunda THEN sistem sınıf seviyesi (9, 10, 11, 12) bilgisini kaydetmeli

2.2. WHEN cevap üretildiğinde THEN sistem öğrenci seviyesine uygun kelime dağarcığı kullanmalı

2.3. WHEN karmaşık kavramlar açıklanırken THEN sistem seviyeye uygun örnekler vermeli

2.4. WHEN cevap uzunluğu belirlenirken THEN öğrenci seviyesi dikkate alınmalı

2.5. WHEN teknik terimler kullanıldığında THEN lise seviyesine uygun açıklamalar eklenmeli

### Gereksinim 3: Öğrenme Analitikleri ve İzleme

**Kullanıcı Hikayesi:** Araştırmacı olarak, öğrenci-sistem etkileşimlerini analiz edebilmek istiyorum, böylece RAG sisteminin eğitsel etkisini ölçebilirim.

#### Kabul Kriterleri

3.1. WHEN öğrenci soru sorduğunda THEN sistem soru kategorisini (kavramsal, prosedürel, faktüel) otomatik sınıflandırmalı

3.2. WHEN cevap verildiğinde THEN sistem öğrenci memnuniyetini (thumbs up/down) kaydetmeli

3.3. WHEN öğrenci oturumu tamamlandığında THEN sistem öğrenme metriklerini (soru sayısı, konu dağılımı, ortalama etkileşim süresi) hesaplamalı

3.4. WHEN öğretmen dashboard'a eriştiğinde THEN sistem sınıf bazında öğrenme analitiği raporları sunmalı

3.5. WHEN zaman içinde veri biriktiğinde THEN sistem öğrenme trendlerini görselleştirmeli

### Gereksinim 4: Çoklu Kaynak Referanslama

**Kullanıcı Hikayesi:** Öğrenci olarak, verilen cevabın hangi kaynaklardan geldiğini görmek istiyorum, böylece bilginin güvenilirliğini değerlendirebilirim.

#### Kabul Kriterleri

4.1. WHEN cevap üretildiğinde THEN sistem kullanılan her kaynak chunk'ını referans numarası ile işaretlemeli

4.2. WHEN kaynak gösterildiğinde THEN sistem kaynak doküman adı, sayfa numarası ve ilgili bölümü göstermeli

4.3. WHEN birden fazla kaynak kullanıldığında THEN sistem kaynakları relevance score'a göre sıralamalı

4.4. WHEN öğrenci kaynağa tıkladığında THEN sistem orijinal dokümanın ilgili bölümünü vurgulamalı

4.5. WHEN çelişkili bilgiler varsa THEN sistem farklı kaynakları karşılaştırmalı ve belirtmeli

### Gereksinim 5: Sokratik Öğretim Modu

**Kullanıcı Hikayesi:** Öğretmen olarak, sistemin doğrudan cevap vermek yerine öğrenciyi düşünmeye yönlendirmesini istiyorum, böylece derin öğrenme gerçekleşebilir.

#### Kabul Kriterleri

5.1. WHEN Sokratik mod aktif edildiğinde THEN sistem doğrudan cevap vermek yerine yönlendirici sorular sorma

5.2. WHEN öğrenci yanlış cevap verdiğinde THEN sistem ipucu vererek doğru yöne yönlendirmeli

5.3. WHEN öğrenci doğru cevaba yaklaştığında THEN sistem olumlu pekiştirme yapmalı

5.4. WHEN öğrenci takıldığında THEN sistem kademeli ipuçları (scaffolding) sunmalı

5.5. WHEN öğrenci doğru cevaba ulaştığında THEN sistem öğrenme sürecini özetlemeli

### Gereksinim 6: Çoklu Modalite Desteği

**Kullanıcı Hikayesi:** Öğrenci olarak, sadece metin değil görsel ve diyagram içeren materyallerle de çalışabilmek istiyorum, böylece daha iyi anlayabilirim.

#### Kabul Kriterleri

6.1. WHEN PDF'de görsel/diyagram tespit edildiğinde THEN sistem görseli OCR ile işlemeli

6.2. WHEN görsel içerik sorgulandığında THEN sistem vision model kullanarak görseli analiz etmeli

6.3. WHEN matematiksel formül varsa THEN sistem LaTeX formatında formülü çıkarmalı

6.4. WHEN tablo içeriği sorgulandığında THEN sistem tablo yapısını koruyarak bilgi vermeli

6.5. WHEN grafik/şema açıklanırken THEN sistem görsel-metin ilişkisini kurmalı

### Gereksinim 7: Adaptif Zorluk Seviyesi

**Kullanıcı Hikayesi:** Sistem olarak, öğrencinin performansına göre soru ve cevap zorluğunu otomatik ayarlamak istiyorum, böylece optimal öğrenme gerçekleşebilir.

#### Kabul Kriterleri

7.1. WHEN öğrenci başarılı cevaplar verdiğinde THEN sistem daha karmaşık sorular önerebilmeli

7.2. WHEN öğrenci zorlandığında THEN sistem daha basit açıklamalar sunmalı

7.3. WHEN öğrenci performansı izlendiğinde THEN sistem zorluk seviyesini dinamik ayarlamalı

7.4. WHEN öğrenci "zone of proximal development" içinde olduğunda THEN sistem optimal zorlukta içerik sunmalı

7.5. WHEN öğrenci ilerlemesi kaydedildiğinde THEN sistem kişiselleştirilmiş öğrenme yolu oluşturmalı

### Gereksinim 8: Bilimsel Değerlendirme Metrikleri

**Kullanıcı Hikayesi:** Araştırmacı olarak, sistemin eğitsel etkisini bilimsel olarak ölçebilmek istiyorum, böylece makale için kanıt sunabilirim.

#### Kabul Kriterleri

8.1. WHEN sistem kullanıldığında THEN Bloom Taksonomisi seviyelerinde soru dağılımı ölçülmeli

8.2. WHEN öğrenci etkileşimi gerçekleştiğinde THEN bilişsel yük (cognitive load) göstergeleri toplanmalı

8.3. WHEN öğrenme oturumu tamamlandığında THEN öğrenme kazanımları (learning gains) hesaplanmalı

8.4. WHEN sistem performansı değerlendirildiğinde THEN eğitsel metrikler (pedagogical metrics) raporlanmalı

8.5. WHEN karşılaştırma yapıldığında THEN kontrol grubu ile deneysel grup metrikleri karşılaştırılabilmeli

### Gereksinim 9: Yanlış Kavrama Tespiti

**Kullanıcı Hikayesi:** Öğretmen olarak, öğrencilerin yaygın yanlış kavramalarını tespit edebilmek istiyorum, böylece müdahale edebilirim.

#### Kabul Kriterleri

9.1. WHEN öğrenci yanlış cevap verdiğinde THEN sistem yanlış kavrama türünü sınıflandırmalı

9.2. WHEN yaygın yanlış kavramalar tespit edildiğinde THEN sistem öğretmene bildirim göndermeli

9.3. WHEN yanlış kavrama düzeltilirken THEN sistem kavramsal değişim stratejileri kullanmalı

9.4. WHEN öğrenci grubu analiz edildiğinde THEN sistem sınıf genelinde yanlış kavrama haritası oluşturmalı

9.5. WHEN müdahale planlandığında THEN sistem hedefli açıklama önerileri sunmalı

### Gereksinim 10: Çevrimdışı Mod ve Düşük Kaynak Desteği

**Kullanıcı Hikayesi:** Öğrenci olarak, internet bağlantısı olmadan veya düşük kaynaklı cihazlarda da sistemi kullanabilmek istiyorum, böylece her yerden erişebilirim.

#### Kabul Kriterleri

10.1. WHEN internet bağlantısı kesildiğinde THEN sistem temel özellikleri çevrimdışı sunabilmeli

10.2. WHEN düşük kaynaklı cihaz kullanıldığında THEN sistem hafif modeller kullanmalı

10.3. WHEN çevrimdışı modda çalışırken THEN sistem önceden indirilen materyallere erişim sağlamalı

10.4. WHEN bağlantı geri geldiğinde THEN sistem çevrimdışı aktiviteleri senkronize etmeli

10.5. WHEN mobil cihazda kullanıldığında THEN sistem responsive ve touch-friendly olmalı

## Bilimsel Katkı Alanları

### 1. Türkçe RAG Sistemleri
- Türkçe için optimize edilmiş chunking stratejileri
- Türkçe embedding modellerinin karşılaştırmalı analizi
- Türkçe stop words ve morfolojik analiz etkisi

### 2. Eğitsel RAG Uygulamaları
- Lise seviyesi için RAG sistem tasarımı
- Öğrenci-RAG etkileşim paternleri
- Eğitsel metriklerle RAG performans değerlendirmesi

### 3. Adaptif Öğrenme Sistemleri
- RAG tabanlı adaptif zorluk ayarlama
- Öğrenci modellemesi ve kişiselleştirme
- Sokratik öğretim yöntemlerinin RAG'e entegrasyonu

### 4. Çoklu Modalite ve RAG
- Görsel-metin entegrasyonu
- Matematiksel içerik işleme
- Tablo ve grafik anlama

### 5. Yanlış Kavrama Tespiti
- RAG sistemlerinde otomatik yanlış kavrama tespiti
- Kavramsal değişim stratejileri
- Öğretmen müdahale sistemleri

## Öncelik Sıralaması

### Yüksek Öncelik (Makale için Kritik)
1. Türkçe Dil Desteği Optimizasyonu (Gereksinim 1)
2. Öğrenme Analitikleri ve İzleme (Gereksinim 3)
3. Bilimsel Değerlendirme Metrikleri (Gereksinim 8)

### Orta Öncelik (Bilimsel Değer Katacak)
4. Eğitim Seviyesi Adaptasyonu (Gereksinim 2)
5. Çoklu Kaynak Referanslama (Gereksinim 4)
6. Yanlış Kavrama Tespiti (Gereksinim 9)

### Düşük Öncelik (İyileştirme)
7. Sokratik Öğretim Modu (Gereksinim 5)
8. Adaptif Zorluk Seviyesi (Gereksinim 7)
9. Çoklu Modalite Desteği (Gereksinim 6)
10. Çevrimdışı Mod (Gereksinim 10)

## Mevcut Sistemin Bilimsel Makale için Kullanılabilir Özellikleri

### Güçlü Yönler
1. ✅ Kapsamlı RAGAS metrikleri (5 farklı metrik)
2. ✅ Gelişmiş chunking stratejileri (4 farklı yöntem)
3. ✅ Hibrit arama (vector + keyword)
4. ✅ Çoklu LLM desteği
5. ✅ Test set yönetimi ve batch evaluation
6. ✅ Quick test özelliği
7. ✅ Sonuç karşılaştırma ve analiz

### Eksik Yönler (Makale için Eklenebilir)
1. ❌ Türkçe'ye özel optimizasyonlar
2. ❌ Eğitsel metrikler ve öğrenme analitikleri
3. ❌ Öğrenci seviyesi adaptasyonu
4. ❌ Kaynak referanslama sistemi
5. ❌ Yanlış kavrama tespiti
6. ❌ Bloom Taksonomisi entegrasyonu
7. ❌ Öğrenci performans izleme
8. ❌ Sokratik öğretim modu

## Sonuç

Mevcut sistem, güçlü bir RAG altyapısına sahip ancak bilimsel makale için **eğitsel boyut** ve **Türkçe optimizasyonları** eklenmesi gerekiyor. Önerilen iyileştirmeler, sistemi sadece teknik bir RAG uygulamasından, **bilimsel araştırma değeri olan eğitsel bir platforma** dönüştürecektir.
