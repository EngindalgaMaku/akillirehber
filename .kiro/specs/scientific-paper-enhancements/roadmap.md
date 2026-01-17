# Yüksek Öncelikli Özellikler - Uygulama Yol Haritası

## Genel Bakış

Bu doküman, bilimsel makale için kritik öneme sahip 3 yüksek öncelikli özelliğin detaylı uygulama yol haritasını içermektedir.

## 🎯 Hedef Özellikler

1. **Türkçe Dil Desteği Optimizasyonu**
2. **Öğrenme Analitikleri ve İzleme**
3. **Bilimsel Değerlendirme Metrikleri**

---

## 1️⃣ TÜRKÇE DİL DESTEĞİ OPTİMİZASYONU

### 📊 Mevcut Durum Analizi

**Güçlü Yönler:**
- ✅ Semantic chunker zaten Türkçe karakterleri destekliyor (ç, ğ, ı, ö, ş, ü)
- ✅ RecursiveChunker'da Türkçe regex pattern var: `[A-ZÇĞİÖŞÜa-zçğıöşü0-9]`
- ✅ SemanticChunker'da Türkçe soru kalıpları tanımlı

**Eksiklikler:**
- ❌ Türkçe stop words listesi yok
- ❌ Türkçe morfolojik analiz yok
- ❌ Türkçe için optimize edilmiş embedding modelleri test edilmemiş
- ❌ Türkçe cümle yapısı için özel chunking stratejisi yok

### 🛠️ Uygulama Adımları

#### Faz 1: Türkçe Stop Words Entegrasyonu (1-2 gün)

**Yapılacaklar:**
- [ ] Türkçe stop words listesi oluştur (TDK + NLP kaynakları)
- [ ] `backend/app/services/turkish_nlp.py` modülü oluştur
- [ ] Stop words filtreleme fonksiyonu ekle
- [ ] Chunking servislerine entegre et
- [ ] A/B testi için metrik topla (stop words ile/siz karşılaştırma)

**Bilimsel Katkı:**
- Türkçe stop words'ün RAG performansına etkisi (RAGAS metrikleri ile ölçülebilir)
- Retrieval precision/recall üzerindeki etki

**Teknik Detaylar:**
```python
# Örnek implementasyon
TURKISH_STOP_WORDS = {
    'bir', 've', 'veya', 'ama', 'fakat', 'çünkü', 'için',
    'ile', 'bu', 'şu', 'o', 'ne', 'nasıl', 'neden', 'gibi',
    'kadar', 'daha', 'en', 'çok', 'az', 'var', 'yok'
    # ... toplam ~200-300 kelime
}
```

#### Faz 2: Türkçe Embedding Model Karşılaştırması (2-3 gün)

**Yapılacaklar:**
- [ ] Test edilecek modeller:
  - `intfloat/multilingual-e5-large` (çok dilli, Türkçe destekli)
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  - `openai/text-embedding-3-small` (baseline)
  - `openai/text-embedding-3-large`
- [ ] Her model için embedding kalitesi testi
- [ ] RAGAS metrikleri ile karşılaştırma
- [ ] Latency ve maliyet analizi
- [ ] Sonuçları raporla

**Bilimsel Katkı:**
- Türkçe için en iyi embedding modelinin belirlenmesi
- Model seçiminin RAG performansına etkisi
- Makale için karşılaştırmalı tablo

**Test Metodolojisi:**
1. Aynı test set'i kullan (Türkçe sorular)
2. Her model için evaluation run oluştur
3. RAGAS metriklerini karşılaştır
4. İstatistiksel anlamlılık testi yap

#### Faz 3: Türkçe Morfolojik Analiz (3-4 gün)

**Yapılacaklar:**
- [ ] Zemberek-NLP veya TurkishNLP kütüphanesi entegrasyonu
- [ ] Kök bulma (stemming) fonksiyonu
- [ ] Kelime türü analizi (isim, fiil, sıfat)
- [ ] Chunking'de morfolojik bilgi kullanımı
- [ ] Semantic similarity hesaplamada morfoloji etkisi

**Bilimsel Katkı:**
- Türkçe morfolojisinin RAG sistemlerinde kullanımı
- Agglutinative dil özelliklerinin chunking'e etkisi

**Teknik Yaklaşım:**
```python
# Zemberek-NLP Python wrapper kullanımı
from zemberek import TurkishMorphology

morphology = TurkishMorphology.create_with_defaults()
analysis = morphology.analyze("öğrencilerin")
# Kök: öğrenci, Ekler: +ler+in
```

#### Faz 4: Türkçe-Optimized Chunking Stratejisi (2-3 gün)

**Yapılacaklar:**
- [ ] `TurkishRecursiveChunker` sınıfı oluştur
- [ ] Türkçe cümle yapısına özel separators
- [ ] Türkçe soru-cevap çiftlerini birlikte tut
- [ ] Ek yapısını dikkate alan split logic
- [ ] Performans karşılaştırması

**Türkçe'ye Özel Separators:**
```python
TURKISH_SEPARATORS = [
    "\n\n",  # Paragraf
    "\n",    # Satır
    ". ",    # Cümle sonu
    "! ",    # Ünlem
    "? ",    # Soru
    "; ",    # Noktalı virgül
    ", ",    # Virgül
    " "      # Boşluk
]
```

### 📈 Başarı Metrikleri

- **Teknik:** RAGAS metriklerinde %5-10 iyileşme
- **Bilimsel:** 3-4 karşılaştırmalı tablo/grafik
- **Makale:** "Turkish Language Optimization in RAG Systems" bölümü

---

## 2️⃣ ÖĞRENME ANALİTİKLERİ VE İZLEME

### 📊 Mevcut Durum Analizi

**Güçlü Yönler:**
- ✅ User ve Course modelleri mevcut
- ✅ RAGAS evaluation sistemi var
- ✅ Test question tracking var

**Eksiklikler:**
- ❌ Öğrenci etkileşim kaydı yok
- ❌ Soru kategorilendirme sistemi yok
- ❌ Öğrenme metrikleri hesaplanmıyor
- ❌ Dashboard/raporlama yok

### 🛠️ Uygulama Adımları

#### Faz 1: Veritabanı Şeması Genişletme (1 gün)

**Yeni Tablolar:**

1. **StudentInteraction** (Öğrenci Etkileşimi)
```python
class StudentInteraction(Base):
    id: int
    student_id: int  # FK to User
    course_id: int   # FK to Course
    question_text: str
    question_category: str  # 'conceptual', 'procedural', 'factual'
    generated_answer: str
    retrieved_contexts: JSON
    satisfaction_rating: int  # 1-5 or thumbs up/down
    interaction_duration_ms: int
    bloom_level: str  # 'remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'
    created_at: datetime
```

2. **LearningSession** (Öğrenme Oturumu)
```python
class LearningSession(Base):
    id: int
    student_id: int
    course_id: int
    started_at: datetime
    ended_at: datetime
    total_questions: int
    avg_interaction_duration_ms: int
    topics_covered: JSON  # List of topics
    session_summary: JSON  # Metrics
```

3. **QuestionCategory** (Soru Kategorisi)
```python
class QuestionCategory(Base):
    id: int
    name: str  # 'conceptual', 'procedural', 'factual'
    description: str
    bloom_levels: JSON  # Associated Bloom levels
```

#### Faz 2: Soru Kategorilendirme Sistemi (2-3 gün)

**Yapılacaklar:**
- [ ] LLM-based soru kategorilendirme
- [ ] Bloom Taksonomisi seviye belirleme
- [ ] Konu/kavram çıkarımı (topic extraction)
- [ ] Kategori güven skoru

**LLM Prompt Örneği:**
```python
CATEGORIZATION_PROMPT = """
Aşağıdaki soruyu analiz et ve kategorize et:

Soru: {question}

Lütfen şunları belirle:
1. Soru Kategorisi: kavramsal/prosedürel/faktüel
2. Bloom Seviyesi: hatırlama/anlama/uygulama/analiz/değerlendirme/yaratma
3. Ana Konu: (örn: "fotosentez", "hücre bölünmesi")
4. Zorluk Seviyesi: 1-5

JSON formatında yanıt ver.
"""
```

**Bilimsel Katkı:**
- Otomatik soru kategorilendirme doğruluğu
- Bloom Taksonomisi dağılımı analizi
- Öğrenci soru paternleri

#### Faz 3: Etkileşim Tracking API (2 gün)

**Yapılacaklar:**
- [ ] POST `/api/interactions` endpoint
- [ ] Gerçek zamanlı etkileşim kaydı
- [ ] Session yönetimi
- [ ] Batch analytics hesaplama

**API Endpoint:**
```python
@router.post("/interactions")
async def create_interaction(
    interaction: InteractionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 1. Soruyu kategorize et (LLM)
    category = await categorize_question(interaction.question)
    
    # 2. Etkileşimi kaydet
    db_interaction = StudentInteraction(
        student_id=current_user.id,
        question_text=interaction.question,
        question_category=category.type,
        bloom_level=category.bloom_level,
        # ...
    )
    db.add(db_interaction)
    
    # 3. Session güncelle
    update_learning_session(current_user.id, db)
    
    return db_interaction
```

#### Faz 4: Analytics Dashboard (3-4 gün)

**Yapılacaklar:**
- [ ] Öğretmen dashboard sayfası
- [ ] Sınıf bazında istatistikler
- [ ] Öğrenci bazında detaylar
- [ ] Zaman serisi grafikleri
- [ ] Bloom dağılımı görselleştirme
- [ ] Konu haritası (topic map)

**Dashboard Metrikleri:**
1. **Genel İstatistikler:**
   - Toplam soru sayısı
   - Ortalama etkileşim süresi
   - Memnuniyet oranı
   - Aktif öğrenci sayısı

2. **Soru Dağılımı:**
   - Kategori bazında (pasta grafik)
   - Bloom seviyesi (bar chart)
   - Zaman içinde trend (line chart)

3. **Öğrenci Performansı:**
   - Bireysel ilerleme
   - Konu bazında başarı
   - Zorluk seviyesi adaptasyonu

### 📈 Başarı Metrikleri

- **Teknik:** 1000+ etkileşim kaydı
- **Bilimsel:** Öğrenci davranış paternleri analizi
- **Makale:** "Learning Analytics in RAG-based Educational Systems" bölümü

---

## 3️⃣ BİLİMSEL DEĞERLENDİRME METRİKLERİ

### 📊 Mevcut Durum Analizi

**Güçlü Yönler:**
- ✅ 5 RAGAS metriği mevcut
- ✅ Evaluation framework hazır
- ✅ Test set yönetimi var

**Eksiklikler:**
- ❌ Eğitsel metrikler yok
- ❌ Bloom Taksonomisi entegrasyonu yok
- ❌ Bilişsel yük ölçümü yok
- ❌ Öğrenme kazanımı hesaplaması yok

### 🛠️ Uygulama Adımları

#### Faz 1: Bloom Taksonomisi Entegrasyonu (2 gün)

**Yapılacaklar:**
- [ ] Bloom seviyesi tanımları
- [ ] Soru-Bloom mapping
- [ ] Test set'lere Bloom metadata ekleme
- [ ] Bloom dağılımı raporlama

**Bloom Seviyeleri:**
```python
class BloomLevel(str, enum.Enum):
    REMEMBER = "remember"      # Hatırlama
    UNDERSTAND = "understand"  # Anlama
    APPLY = "apply"           # Uygulama
    ANALYZE = "analyze"       # Analiz
    EVALUATE = "evaluate"     # Değerlendirme
    CREATE = "create"         # Yaratma

BLOOM_KEYWORDS = {
    "remember": ["tanımla", "listele", "adlandır", "hatırla"],
    "understand": ["açıkla", "özetle", "karşılaştır", "sınıflandır"],
    "apply": ["uygula", "göster", "çöz", "kullan"],
    "analyze": ["analiz et", "ayır", "incele", "karşılaştır"],
    "evaluate": "değerlendir", "eleştir", "savun", "karar ver"],
    "create": ["tasarla", "oluştur", "yarat", "planla"]
}
```

**Database Değişikliği:**
```python
# TestQuestion tablosuna ekle
bloom_level = Column(String(50), nullable=True)
cognitive_complexity = Column(Integer, nullable=True)  # 1-5
```

#### Faz 2: Bilişsel Yük Göstergeleri (2-3 gün)

**Yapılacaklar:**
- [ ] Etkileşim süresi analizi
- [ ] Soru karmaşıklığı skoru
- [ ] Cevap uzunluğu vs anlaşılırlık
- [ ] Tekrar soru sorma oranı
- [ ] Bilişsel yük indeksi hesaplama

**Bilişsel Yük Formülü:**
```python
def calculate_cognitive_load(interaction):
    """
    Bilişsel yük göstergesi (0-1 arası)
    """
    # Faktörler:
    # 1. Etkileşim süresi (normalize)
    time_factor = min(interaction.duration_ms / 60000, 1.0)  # 1 dakika max
    
    # 2. Soru karmaşıklığı
    complexity = calculate_question_complexity(interaction.question)
    
    # 3. Cevap uzunluğu
    answer_length_factor = len(interaction.answer) / 1000  # normalize
    
    # 4. Tekrar soru oranı
    repeat_factor = get_repeat_question_rate(interaction.student_id)
    
    # Ağırlıklı ortalama
    cognitive_load = (
        0.3 * time_factor +
        0.3 * complexity +
        0.2 * answer_length_factor +
        0.2 * repeat_factor
    )
    
    return min(cognitive_load, 1.0)
```

**Bilimsel Katkı:**
- RAG sistemlerinde bilişsel yük ölçümü
- Öğrenci zorlanma noktalarının tespiti

#### Faz 3: Öğrenme Kazanımları (Learning Gains) (3 gün)

**Yapılacaklar:**
- [ ] Pre-test / Post-test sistemi
- [ ] Konu bazında ilerleme takibi
- [ ] Normalized learning gain hesaplama
- [ ] Kontrol grubu karşılaştırması

**Learning Gain Formülü:**
```python
def calculate_normalized_gain(pre_score, post_score, max_score=100):
    """
    Hake's normalized gain
    g = (post - pre) / (max - pre)
    """
    if max_score - pre_score == 0:
        return 0.0
    
    gain = (post_score - pre_score) / (max_score - pre_score)
    return gain

# Yorumlama:
# g < 0.3  : Düşük kazanım
# 0.3 ≤ g < 0.7 : Orta kazanım
# g ≥ 0.7  : Yüksek kazanım
```

**Database Şeması:**
```python
class LearningAssessment(Base):
    id: int
    student_id: int
    course_id: int
    topic: str
    assessment_type: str  # 'pre-test', 'post-test', 'formative'
    score: float
    max_score: float
    bloom_distribution: JSON  # Her seviyeden kaç soru
    completed_at: datetime

class LearningGain(Base):
    id: int
    student_id: int
    course_id: int
    topic: str
    pre_test_id: int
    post_test_id: int
    normalized_gain: float
    gain_category: str  # 'low', 'medium', 'high'
    calculated_at: datetime
```

#### Faz 4: Eğitsel Metrikler Dashboard (2-3 gün)

**Yapılacaklar:**
- [ ] Pedagogical metrics sayfası
- [ ] Bloom dağılımı görselleştirme
- [ ] Bilişsel yük heatmap
- [ ] Learning gains timeline
- [ ] Karşılaştırmalı analiz (kontrol vs deney grubu)

**Metrikler:**
1. **Bloom Taksonomisi Dağılımı:**
   - Soru seviyesi dağılımı
   - Öğrenci performansı seviye bazında
   - Hedef vs gerçekleşen dağılım

2. **Bilişsel Yük Analizi:**
   - Ortalama bilişsel yük
   - Yüksek yük anları
   - Öğrenci bazında karşılaştırma

3. **Öğrenme Kazanımları:**
   - Normalized gain ortalaması
   - Konu bazında kazanımlar
   - Zaman içinde ilerleme

4. **Karşılaştırmalı Analiz:**
   - RAG kullanan vs kullanmayan
   - Farklı chunking stratejileri
   - Farklı LLM modelleri

### 📈 Başarı Metrikleri

- **Teknik:** 10+ yeni eğitsel metrik
- **Bilimsel:** İstatistiksel anlamlılık testleri
- **Makale:** "Pedagogical Evaluation of RAG Systems" bölümü

---

## 🗓️ GENEL UYGULAMA TAKVIMI

### Hafta 1-2: Türkçe Optimizasyonu
- Gün 1-2: Stop words + entegrasyon
- Gün 3-5: Embedding model karşılaştırması
- Gün 6-9: Morfolojik analiz
- Gün 10-12: Turkish chunking stratejisi

### Hafta 3-4: Öğrenme Analitikleri
- Gün 13: Database şeması
- Gün 14-16: Soru kategorilendirme
- Gün 17-18: Tracking API
- Gün 19-22: Analytics dashboard

### Hafta 5-6: Bilimsel Metrikler
- Gün 23-24: Bloom entegrasyonu
- Gün 25-27: Bilişsel yük
- Gün 28-30: Learning gains
- Gün 31-33: Metrikler dashboard

### Hafta 7: Test ve Dokümantasyon
- Gün 34-36: Entegrasyon testleri
- Gün 37-38: Performans optimizasyonu
- Gün 39-40: Dokümantasyon ve makale yazımı

**Toplam Süre:** ~8 hafta (2 ay)

---

## 📊 BİLİMSEL MAKALE İÇİN ÇIKTILAR

### Karşılaştırmalı Tablolar
1. Türkçe embedding modelleri karşılaştırması
2. Stop words etkisi (RAGAS metrikleri)
3. Chunking stratejileri performansı
4. Bloom seviyesi dağılımı
5. Learning gains istatistikleri

### Grafikler
1. RAGAS metrikleri zaman serisi
2. Bilişsel yük heatmap
3. Öğrenci etkileşim paternleri
4. Konu bazında öğrenme kazanımları
5. Kontrol vs deney grubu karşılaştırması

### İstatistiksel Analizler
1. T-test (kontrol vs deney)
2. ANOVA (farklı stratejiler)
3. Korelasyon analizi (metrikler arası)
4. Regresyon (öğrenme kazanımı tahmin)

---

## 🎯 ÖNCELİKLENDİRME ÖNERİSİ

Eğer zaman kısıtlı ise, şu sırayla ilerleyin:

### Minimum Viable Product (MVP) - 4 hafta
1. ✅ Türkçe stop words (1 gün)
2. ✅ Embedding model karşılaştırması (3 gün)
3. ✅ Temel etkileşim tracking (2 gün)
4. ✅ Soru kategorilendirme (2 gün)
5. ✅ Bloom entegrasyonu (2 gün)
6. ✅ Basit analytics dashboard (3 gün)
7. ✅ Learning gains hesaplama (2 gün)
8. ✅ Test ve dokümantasyon (5 gün)

### Full Implementation - 8 hafta
Yukarıdaki tam takvimi takip edin.

---

## 💡 EK ÖNERİLER

### Hızlı Kazanımlar (Quick Wins)
1. **Türkçe stop words:** Hemen uygulanabilir, etkisi ölçülebilir
2. **Bloom kategorilendirme:** LLM ile kolay, bilimsel değer yüksek
3. **Etkileşim logging:** Basit ama veri toplamaya hemen başlar

### Uzun Vadeli Değer
1. **Morfolojik analiz:** Türkçe için unique katkı
2. **Learning gains:** Eğitsel etki kanıtı
3. **Bilişsel yük:** Yeni araştırma alanı

### Makale Stratejisi
- Her özellik için ayrı bölüm
- Karşılaştırmalı sonuçlar vurgula
- Türkçe'ye özgü zorlukları ve çözümleri detaylandır
- İstatistiksel anlamlılık testleri ekle

---

## 📚 KAYNAKLAR VE ARAÇLAR

### Türkçe NLP
- Zemberek-NLP: https://github.com/ahmetaa/zemberek-nlp
- TurkishNLP: https://github.com/turkish-nlp
- Turkish Stop Words: TDK + literatür

### Embedding Models
- Hugging Face Model Hub
- OpenAI Embeddings API
- Sentence Transformers

### Eğitsel Metrikler
- Bloom's Taxonomy literatürü
- Cognitive Load Theory
- Hake's Normalized Gain

### Veri Analizi
- Pandas, NumPy
- Matplotlib, Plotly
- SciPy (istatistiksel testler)

---

## ✅ SONRAKI ADIMLAR

1. **Bu roadmap'i inceleyin ve onaylayın**
2. **Başlangıç noktasını seçin** (MVP veya Full)
3. **İlk özelliği seçin** (öneri: Türkçe stop words)
4. **Design document oluşturalım** (teknik detaylar için)
5. **Implementation'a başlayalım**

**Hangi özellikle başlamak istersiniz?**
