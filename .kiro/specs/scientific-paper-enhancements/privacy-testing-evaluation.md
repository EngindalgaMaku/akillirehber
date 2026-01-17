# Gizlilik Sistemi - Test ve Bilimsel Değerlendirme

## 🎯 Hedef

Gizlilik sisteminin **bilimsel olarak** ne kadar iyi çalıştığını ölçmek için:
1. Kapsamlı test dataset'i
2. Evaluation metrikleri
3. Test sayfası (frontend)
4. Automated testing
5. Bilimsel raporlama

---

## 📊 BİLİMSEL DEĞERLENDİRME METRİKLERİ

### 1. PII Detection Metrikleri

#### Precision (Kesinlik)
```
Precision = True Positives / (True Positives + False Positives)
```
- Tespit edilen PII'ların kaçı gerçekten PII?
- Yüksek precision = Az false positive

#### Recall (Duyarlılık)
```
Recall = True Positives / (True Positives + False Negatives)
```
- Gerçek PII'ların kaçı tespit edildi?
- Yüksek recall = Az false negative

#### F1 Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- Precision ve Recall'un harmonik ortalaması
- Genel performans göstergesi

#### Accuracy (Doğruluk)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Genel doğruluk oranı

### 2. PII Türüne Göre Metrikler

Her PII türü için ayrı:
- TC Kimlik: Precision, Recall, F1
- Telefon: Precision, Recall, F1
- E-posta: Precision, Recall, F1
- İsim: Precision, Recall, F1
- IBAN: Precision, Recall, F1

### 3. Performans Metrikleri

- **Latency:** Tespit süresi (ms)
- **Throughput:** Saniyede işlenen metin sayısı
- **Memory Usage:** Bellek kullanımı

### 4. Kullanıcı Deneyimi Metrikleri

- **False Positive Rate:** Yanlış alarm oranı
- **User Satisfaction:** Kullanıcı memnuniyeti
- **Masking Quality:** Maskeleme kalitesi

---

## 🗂️ TEST DATASET OLUŞTURMA

### Dataset Yapısı

```python
@dataclass
class TestCase:
    """Tek bir test case"""
    id: int
    text: str  # Test metni
    expected_pii: List[Dict]  # Beklenen PII'lar
    category: str  # 'positive', 'negative', 'edge_case'
    difficulty: str  # 'easy', 'medium', 'hard'
    description: str


# Örnek test case
test_case = TestCase(
    id=1,
    text="Benim adım Ahmet Yılmaz ve TC kimlik numaram 12345678901",
    expected_pii=[
        {
            'type': 'isim',
            'text': 'Ahmet Yılmaz',
            'start': 12,
            'end': 24
        },
        {
            'type': 'tc_kimlik',
            'text': '12345678901',
            'start': 50,
            'end': 61
        }
    ],
    category='positive',
    difficulty='easy',
    description='Basit isim ve TC kimlik tespiti'
)
```

### Test Kategorileri

#### 1. Positive Cases (PII var)
```python
POSITIVE_CASES = [
    # TC Kimlik
    {
        'text': 'TC kimlik numaram 12345678901',
        'expected': ['tc_kimlik']
    },
    
    # Telefon
    {
        'text': 'Telefon: 0532 123 45 67',
        'expected': ['telefon']
    },
    
    # E-posta
    {
        'text': 'E-postam ahmet@example.com',
        'expected': ['email']
    },
    
    # İsim
    {
        'text': 'Ben Ayşe Demir',
        'expected': ['isim']
    },
    
    # Çoklu PII
    {
        'text': 'Ben Mehmet, TC: 12345678901, Tel: 0532 123 45 67',
        'expected': ['isim', 'tc_kimlik', 'telefon']
    },
]
```

#### 2. Negative Cases (PII yok)
```python
NEGATIVE_CASES = [
    'Fotosentez nedir?',
    'Osmanlı İmparatorluğu ne zaman kuruldu?',
    'Matematik sorusu: 2+2=?',
    'Hücre bölünmesi nasıl olur?',
    'Python programlama dili nedir?',
]
```

#### 3. Edge Cases (Zor durumlar)
```python
EDGE_CASES = [
    # Benzer ama PII değil
    {
        'text': 'Telefon numarası 123 456 78 90',  # Geçersiz format
        'expected': []
    },
    
    # Kısmi PII
    {
        'text': 'TC kimlik numaram 123456789',  # 11 hane değil
        'expected': []
    },
    
    # Bağlam önemli
    {
        'text': 'Ahmet Bey dedi ki...',  # "Ahmet" isim mi yoksa hitap mı?
        'expected': ['isim']  # veya []
    },
    
    # Çoklu format
    {
        'text': 'Tel: 05321234567 veya 0532-123-45-67',
        'expected': ['telefon', 'telefon']
    },
]
```

### Dataset Dosyası

**Dosya:** `backend/tests/data/pii_test_dataset.json`

```json
{
  "version": "1.0",
  "created_at": "2024-01-15",
  "total_cases": 100,
  "categories": {
    "positive": 50,
    "negative": 30,
    "edge_case": 20
  },
  "test_cases": [
    {
      "id": 1,
      "text": "Benim adım Ahmet Yılmaz ve TC kimlik numaram 12345678901",
      "expected_pii": [
        {
          "type": "isim",
          "text": "Ahmet Yılmaz",
          "start": 12,
          "end": 24,
          "confidence": 0.9
        },
        {
          "type": "tc_kimlik",
          "text": "12345678901",
          "start": 50,
          "end": 61,
          "confidence": 1.0
        }
      ],
      "category": "positive",
      "difficulty": "easy",
      "description": "Basit isim ve TC kimlik tespiti"
    },
    // ... 99 more cases
  ]
}
```

---

## 🧪 EVALUATION SCRIPT

**Dosya:** `backend/tests/evaluate_pii_detection.py`

```python
"""
PII Detection Evaluation Script
Gizlilik sisteminin bilimsel değerlendirmesi
"""

import json
import time
from typing import List, Dict
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from app.services.pii_detection import TurkishPIIDetector, PIIType


@dataclass
class EvaluationResult:
    """Değerlendirme sonucu"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    avg_latency_ms: float
    per_type_metrics: Dict[str, Dict]


class PIIDetectionEvaluator:
    """PII Detection değerlendirici"""
    
    def __init__(self, test_dataset_path: str):
        self.detector = TurkishPIIDetector()
        self.test_cases = self._load_test_dataset(test_dataset_path)
    
    def _load_test_dataset(self, path: str) -> List[Dict]:
        """Test dataset'ini yükle"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['test_cases']
    
    def evaluate(self) -> EvaluationResult:
        """Tam değerlendirme yap"""
        print("🔍 PII Detection Evaluation başlıyor...")
        print(f"📊 Test case sayısı: {len(self.test_cases)}")
        
        results = []
        latencies = []
        
        for i, test_case in enumerate(self.test_cases):
            if (i + 1) % 10 == 0:
                print(f"  İşlenen: {i + 1}/{len(self.test_cases)}")
            
            # Tespit yap
            start_time = time.time()
            detection_result = self.detector.detect(test_case['text'])
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
            # Sonucu değerlendir
            evaluation = self._evaluate_single_case(
                test_case,
                detection_result
            )
            results.append(evaluation)
        
        # Metrikleri hesapla
        metrics = self._calculate_metrics(results)
        metrics['avg_latency_ms'] = sum(latencies) / len(latencies)
        
        print("\n✅ Değerlendirme tamamlandı!")
        return metrics
    
    def _evaluate_single_case(self, test_case: Dict, detection_result) -> Dict:
        """Tek bir case'i değerlendir"""
        expected_types = {pii['type'] for pii in test_case['expected_pii']}
        detected_types = {match.pii_type.value for match in detection_result.matches}
        
        # True Positive: Hem expected hem detected
        tp = len(expected_types & detected_types)
        
        # False Positive: Detected ama expected değil
        fp = len(detected_types - expected_types)
        
        # False Negative: Expected ama detected değil
        fn = len(expected_types - detected_types)
        
        # True Negative: Ne expected ne detected
        tn = 1 if len(expected_types) == 0 and len(detected_types) == 0 else 0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'expected_types': list(expected_types),
            'detected_types': list(detected_types),
            'test_case_id': test_case['id']
        }
    
    def _calculate_metrics(self, results: List[Dict]) -> EvaluationResult:
        """Metrikleri hesapla"""
        tp = sum(r['tp'] for r in results)
        fp = sum(r['fp'] for r in results)
        fn = sum(r['fn'] for r in results)
        tn = sum(r['tn'] for r in results)
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # PII türüne göre metrikler
        per_type_metrics = self._calculate_per_type_metrics(results)
        
        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            avg_latency_ms=0.0,  # Daha sonra set edilecek
            per_type_metrics=per_type_metrics
        )
    
    def _calculate_per_type_metrics(self, results: List[Dict]) -> Dict:
        """PII türüne göre metrikleri hesapla"""
        pii_types = [
            'tc_kimlik', 'telefon', 'email', 'isim', 
            'iban', 'kredi_karti', 'dogum_tarihi'
        ]
        
        per_type = {}
        
        for pii_type in pii_types:
            tp = sum(1 for r in results if pii_type in r['expected_types'] and pii_type in r['detected_types'])
            fp = sum(1 for r in results if pii_type not in r['expected_types'] and pii_type in r['detected_types'])
            fn = sum(1 for r in results if pii_type in r['expected_types'] and pii_type not in r['detected_types'])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_type[pii_type] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        return per_type
    
    def generate_report(self, result: EvaluationResult, output_path: str):
        """Bilimsel rapor oluştur"""
        print("\n📄 Rapor oluşturuluyor...")
        
        # 1. Genel metrikler
        print("\n=== GENEL METRİKLER ===")
        print(f"Precision: {result.precision:.4f}")
        print(f"Recall: {result.recall:.4f}")
        print(f"F1 Score: {result.f1_score:.4f}")
        print(f"Accuracy: {result.accuracy:.4f}")
        print(f"Avg Latency: {result.avg_latency_ms:.2f} ms")
        
        # 2. Confusion Matrix
        print("\n=== CONFUSION MATRIX ===")
        print(f"True Positives: {result.true_positives}")
        print(f"False Positives: {result.false_positives}")
        print(f"True Negatives: {result.true_negatives}")
        print(f"False Negatives: {result.false_negatives}")
        
        # 3. PII türüne göre
        print("\n=== PII TÜRÜNE GÖRE METRİKLER ===")
        for pii_type, metrics in result.per_type_metrics.items():
            print(f"\n{pii_type.upper()}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        # 4. Grafikler oluştur
        self._create_visualizations(result, output_path)
        
        # 5. JSON rapor
        self._save_json_report(result, output_path)
        
        print(f"\n✅ Rapor kaydedildi: {output_path}")
    
    def _create_visualizations(self, result: EvaluationResult, output_path: str):
        """Görselleştirmeler oluştur"""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # 1. PII türüne göre F1 Score
        pii_types = list(result.per_type_metrics.keys())
        f1_scores = [result.per_type_metrics[t]['f1_score'] for t in pii_types]
        
        plt.figure(figsize=(10, 6))
        plt.bar(pii_types, f1_scores)
        plt.xlabel('PII Türü')
        plt.ylabel('F1 Score')
        plt.title('PII Türüne Göre F1 Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_path}/f1_scores_by_type.png')
        plt.close()
        
        # 2. Precision vs Recall
        precisions = [result.per_type_metrics[t]['precision'] for t in pii_types]
        recalls = [result.per_type_metrics[t]['recall'] for t in pii_types]
        
        plt.figure(figsize=(10, 6))
        x = range(len(pii_types))
        width = 0.35
        plt.bar([i - width/2 for i in x], precisions, width, label='Precision')
        plt.bar([i + width/2 for i in x], recalls, width, label='Recall')
        plt.xlabel('PII Türü')
        plt.ylabel('Score')
        plt.title('Precision vs Recall')
        plt.xticks(x, pii_types, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_path}/precision_recall.png')
        plt.close()
        
        # 3. Confusion Matrix Heatmap
        cm = [[result.true_positives, result.false_positives],
              [result.false_negatives, result.true_negatives]]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.xticks([0, 1], ['Predicted Positive', 'Predicted Negative'])
        plt.yticks([0, 1], ['Actual Positive', 'Actual Negative'])
        plt.title('Confusion Matrix')
        
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i][j], ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{output_path}/confusion_matrix.png')
        plt.close()
    
    def _save_json_report(self, result: EvaluationResult, output_path: str):
        """JSON rapor kaydet"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_metrics': {
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'accuracy': result.accuracy,
                'avg_latency_ms': result.avg_latency_ms
            },
            'confusion_matrix': {
                'true_positives': result.true_positives,
                'false_positives': result.false_positives,
                'true_negatives': result.true_negatives,
                'false_negatives': result.false_negatives
            },
            'per_type_metrics': result.per_type_metrics
        }
        
        with open(f'{output_path}/evaluation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


# Kullanım
if __name__ == '__main__':
    evaluator = PIIDetectionEvaluator('tests/data/pii_test_dataset.json')
    result = evaluator.evaluate()
    evaluator.generate_report(result, 'evaluation_results')
```

**Çalıştırma:**
```bash
cd backend
python tests/evaluate_pii_detection.py
```

**Çıktı:**
```
evaluation_results/
├── evaluation_report.json
├── f1_scores_by_type.png
├── precision_recall.png
└── confusion_matrix.png
```

Devam edelim mi? Sonraki adım frontend test sayfası!
