# Türkçe Stop Words - Detaylı Plan

## Gün 1: Stop Words Listesi Oluşturma

### 1. Kaynak Toplama (2 saat)
- TDK (Türk Dil Kurumu) yaygın kelimeler
- NLTK Turkish stop words
- Zemberek-NLP stop words
- Akademik makalelerden frekans analizi

### 2. Liste Oluşturma (3 saat)
```python
TURKISH_STOP_WORDS = {
    # Bağlaçlar
    've', 'veya', 'ya da', 'ama', 'fakat', 'ancak', 'lakin',
    
    # Edatlar  
    'ile', 'için', 'gibi', 'kadar', 'daha', 'en',
    
    # Zamirler
    'ben', 'sen', 'o', 'biz', 'siz', 'onlar',
    'bu', 'şu', 'o', 'bunlar', 'şunlar', 'onlar',
    
    # Soru kelimeleri (dikkatli!)
    'ne', 'nasıl', 'neden', 'niçin', 'nerede', 'kim',
    
    # Yardımcı fiiller
    'olmak', 'etmek', 'yapmak', 'var', 'yok',
    
    # Toplam ~200-300 kelime
}
```

### 3. Kategorilendirme (2 saat)
- Kesinlikle çıkarılacaklar (bağlaçlar, edatlar)
- Dikkatli çıkarılacaklar (soru kelimeleri - eğitim bağlamında önemli!)
- Bağlama göre çıkarılacaklar

## Gün 2: Entegrasyon ve Test

### 1. Service Oluşturma (2 saat)
```python
# backend/app/services/turkish_nlp.py

class TurkishNLPService:
    def __init__(self):
        self.stop_words = self._load_stop_words()
    
    def remove_stop_words(self, text: str, 
                         preserve_questions: bool = True) -> str:
        """
        Stop words'leri çıkar
        
        Args:
            preserve_questions: Soru kelimelerini koru (eğitim için önemli)
        """
        words = text.split()
        
        if preserve_questions:
            # Soru kelimelerini koruyarak filtrele
            filtered = [w for w in words 
                       if w.lower() not in self.stop_words 
                       or w.lower() in self.QUESTION_WORDS]
        else:
            filtered = [w for w in words 
                       if w.lower() not in self.stop_words]
        
        return ' '.join(filtered)
```

### 2. Chunking Entegrasyonu (3 saat)
- RecursiveChunker'a ekle
- SemanticChunker'a ekle
- Opsiyonel parametre olarak ekle

### 3. A/B Test Hazırlığı (2 saat)
- Stop words ile/siz karşılaştırma
- RAGAS metrikleri toplama
- Test set hazırlama

---

# Embedding Model Karşılaştırması - Detaylı Plan

## Gün 1-2: Model Seçimi ve Test Ortamı

### Test Edilecek Modeller

1. **OpenAI text-embedding-3-small** (Baseline)
   - Boyut: 1536
   - Maliyet: Düşük
   - Çok dilli destek: ✅

2. **OpenAI text-embedding-3-large**
   - Boyut: 3072
   - Maliyet: Orta
   - Performans: Yüksek

3. **intfloat/multilingual-e5-large**
   - Boyut: 1024
   - Maliyet: Ücretsiz (self-hosted)
   - Türkçe: ✅✅

4. **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**
   - Boyut: 768
   - Maliyet: Ücretsiz
   - Türkçe: ✅

## Gün 3: Test Execution

### Test Metodolojisi
```python
# Test script
async def compare_embedding_models():
    models = [
        'openai/text-embedding-3-small',
        'openai/text-embedding-3-large',
        'intfloat/multilingual-e5-large',
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    ]
    
    test_set = load_turkish_test_set()  # 100 Türkçe soru
    
    results = {}
    for model in models:
        # Her model için evaluation run
        run = create_evaluation_run(
            test_set_id=test_set.id,
            embedding_model=model
        )
        
        # RAGAS metrikleri topla
        metrics = await run_evaluation(run.id)
        
        results[model] = {
            'faithfulness': metrics.avg_faithfulness,
            'answer_relevancy': metrics.avg_answer_relevancy,
            'context_precision': metrics.avg_context_precision,
            'context_recall': metrics.avg_context_recall,
            'latency_ms': metrics.avg_latency_ms,
            'cost_per_1k': calculate_cost(model)
        }
    
    return results
```

### Karşılaştırma Kriterleri
1. RAGAS metrikleri (5 metrik)
2. Latency (ms)
3. Maliyet ($/1000 token)
4. Türkçe performansı (özel test set)

---

# Morfolojik Analiz - Detaylı Plan

## Gün 1-2: Zemberek-NLP Entegrasyonu

### Kurulum
```bash
pip install zemberek-python
```

### Temel Kullanım
```python
from zemberek import TurkishMorphology

morphology = TurkishMorphology.create_with_defaults()

# Kelime analizi
analysis = morphology.analyze("öğrencilerin")
# Sonuç: öğrenci+Noun+A3pl+P3sg+Gen

# Kök bulma
stems = morphology.stem_and_lemmatize("öğrencilerin")
# Sonuç: ['öğrenci']
```

## Gün 3-4: RAG Entegrasyonu

### Use Case 1: Chunking'de Morfoloji
```python
def turkish_aware_chunking(text: str):
    """
    Türkçe morfolojisini dikkate alan chunking
    """
    # Kelimeleri köklerine ayır
    words = text.split()
    stemmed = [morphology.stem(word)[0] for word in words]
    
    # Aynı kökten gelen kelimeleri birlikte tut
    # Örnek: "öğrenci", "öğrenciler", "öğrencinin" -> aynı chunk
```

### Use Case 2: Semantic Search'te Morfoloji
```python
def morphology_aware_search(query: str):
    """
    Morfolojik varyasyonları da ara
    """
    # Query'yi köklerine ayır
    query_stems = get_stems(query)
    
    # Hem orijinal hem köklerle ara
    results = hybrid_search(
        query=query,
        query_variants=query_stems
    )
```

---

# Türkçe-Optimized Chunking - Detaylı Plan

## Gün 1-2: TurkishRecursiveChunker

### Özellikler
1. Türkçe cümle yapısına özel separators
2. Soru-cevap çiftlerini birlikte tut
3. Ek yapısını dikkate al
4. Türkçe noktalama kuralları

### Implementasyon
```python
class TurkishRecursiveChunker(RecursiveChunker):
    """Türkçe için optimize edilmiş chunker"""
    
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
    
    QUESTION_PATTERNS = [
        r'\?$',  # Soru işareti ile biten
        r'^(Ne|Nasıl|Neden|Niçin|Kim|Nerede|Hangi|Kaç)\b',
        r'\b(mi|mı|mu|mü)\?$',  # Soru eki
    ]
    
    def chunk(self, text: str, **kwargs):
        # Soru-cevap çiftlerini tespit et
        qa_pairs = self._detect_qa_pairs(text)
        
        # QA çiftlerini birlikte tut
        chunks = self._chunk_with_qa_awareness(text, qa_pairs)
        
        return chunks
    
    def _detect_qa_pairs(self, text: str) -> List[Tuple[int, int]]:
        """Soru-cevap çiftlerini tespit et"""
        sentences = self._split_sentences(text)
        qa_pairs = []
        
        for i, sent in enumerate(sentences):
            if self._is_question(sent):
                # Sonraki cümle cevap olabilir
                if i + 1 < len(sentences):
                    qa_pairs.append((i, i + 1))
        
        return qa_pairs
```

## Gün 3: Test ve Karşılaştırma

### Karşılaştırma
- Standard RecursiveChunker
- TurkishRecursiveChunker
- SemanticChunker

### Metrikler
- Chunk kalitesi
- Soru-cevap bütünlüğü
- RAGAS performansı

---

# Toplam Süre ve Öncelik

## Revize Edilmiş Süre (Gizlilik Dahil)

1. **Gizlilik Koruma:** 3-4 gün (ÖNCELİK!)
2. **Stop Words:** 2 gün
3. **Embedding Karşılaştırma:** 3 gün
4. **Morfolojik Analiz:** 4 gün
5. **Turkish Chunking:** 3 gün

**TOPLAM:** 15-16 gün (~3 hafta)

## Önerilen Sıralama

1. ✅ **Gizlilik Sistemi** (Hemen başla - etik onay için gerekli!)
2. ✅ **Stop Words** (Hızlı kazanım)
3. ✅ **Embedding Karşılaştırma** (Bilimsel değer)
4. ✅ **Morfolojik Analiz** (Türkçe'ye özgü katkı)
5. ✅ **Turkish Chunking** (İleri seviye)
