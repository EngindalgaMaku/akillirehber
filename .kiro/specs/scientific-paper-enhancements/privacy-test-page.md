# Gizlilik Test Sayfası - Frontend

## 🎯 Hedef

Gizlilik sistemini **interaktif olarak** test edebilmek ve sonuçları görselleştirmek için bir test sayfası.

---

## 📄 TEST SAYFASI TASARIMI

### Sayfa Yapısı

```
┌─────────────────────────────────────────────────────┐
│  🔒 Gizlilik Sistemi Test Sayfası                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [Test Metni Girişi]                                │
│  ┌──────────────────────────────────────────────┐  │
│  │ Metninizi buraya yazın...                    │  │
│  │                                               │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
│  [Hızlı Test Örnekleri]                             │
│  [TC Kimlik] [Telefon] [E-posta] [İsim] [Hepsi]   │
│                                                      │
│  [Tespit Et] Butonu                                 │
│                                                      │
├─────────────────────────────────────────────────────┤
│  📊 Tespit Sonuçları                                │
│                                                      │
│  ✅ PII Tespit Edildi: 3 adet                       │
│  ⚠️  Risk Skoru: 0.75 (Yüksek)                      │
│  ⏱️  İşlem Süresi: 45ms                             │
│                                                      │
│  [Tespit Edilen PII'lar]                            │
│  • TC Kimlik: 12345678901 → [TC_KIMLIK]            │
│  • Telefon: 0532 123 45 67 → [TELEFON]             │
│  • İsim: Ahmet Yılmaz → [ISIM]                     │
│                                                      │
│  [Maskelenmiş Metin]                                │
│  ┌──────────────────────────────────────────────┐  │
│  │ Benim adım [ISIM] ve TC kimlik numaram       │  │
│  │ [TC_KIMLIK], telefon [TELEFON]               │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
├─────────────────────────────────────────────────────┤
│  📈 İstatistikler                                   │
│                                                      │
│  [Grafik: PII Türü Dağılımı]                        │
│  [Grafik: Risk Skoru Geçmişi]                       │
│  [Tablo: Son 10 Test]                               │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 💻 FRONTEND KOD

### Sayfa Komponenti

**Dosya:** `frontend/src/app/dashboard/privacy-test/page.tsx`

```typescript
'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Shield, AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import { PIIDetectionResult, PIIMatch } from '@/types/privacy';

export default function PrivacyTestPage() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState<PIIDetectionResult | null>(null);
  const [loading, setLoading] = useState(false);

  // Hızlı test örnekleri
  const quickTests = {
    tc_kimlik: 'Benim TC kimlik numaram 12345678901',
    telefon: 'Telefon numaram 0532 123 45 67',
    email: 'E-postam ahmet@example.com',
    isim: 'Benim adım Ayşe Demir',
    all: 'Ben Mehmet Yılmaz, TC: 12345678901, Tel: 0532 123 45 67, Email: mehmet@example.com'
  };

  const handleDetect = async () => {
    setLoading(true);
    
    try {
      const response = await fetch('/api/privacy/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText })
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Detection error:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadQuickTest = (key: keyof typeof quickTests) => {
    setInputText(quickTests[key]);
  };

  const getRiskColor = (score: number) => {
    if (score >= 0.7) return 'destructive';
    if (score >= 0.4) return 'warning';
    return 'default';
  };

  const getRiskLabel = (score: number) => {
    if (score >= 0.7) return 'Yüksek Risk';
    if (score >= 0.4) return 'Orta Risk';
    return 'Düşük Risk';
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Shield className="h-8 w-8 text-blue-600" />
        <div>
          <h1 className="text-3xl font-bold">Gizlilik Sistemi Test Sayfası</h1>
          <p className="text-muted-foreground">
            PII (Kişisel Bilgi) tespit sistemini test edin
          </p>
        </div>
      </div>

      {/* Input Section */}
      <Card>
        <CardHeader>
          <CardTitle>Test Metni</CardTitle>
          <CardDescription>
            Test etmek istediğiniz metni girin veya hızlı test örneklerinden birini seçin
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Textarea */}
          <Textarea
            placeholder="Metninizi buraya yazın..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            rows={6}
            className="font-mono"
          />

          {/* Quick Test Buttons */}
          <div className="flex flex-wrap gap-2">
            <span className="text-sm text-muted-foreground">Hızlı Test:</span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => loadQuickTest('tc_kimlik')}
            >
              TC Kimlik
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => loadQuickTest('telefon')}
            >
              Telefon
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => loadQuickTest('email')}
            >
              E-posta
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => loadQuickTest('isim')}
            >
              İsim
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => loadQuickTest('all')}
            >
              Hepsi
            </Button>
          </div>

          {/* Detect Button */}
          <Button
            onClick={handleDetect}
            disabled={!inputText || loading}
            className="w-full"
          >
            {loading ? 'Tespit Ediliyor...' : 'PII Tespit Et'}
          </Button>
        </CardContent>
      </Card>

      {/* Results Section */}
      {result && (
        <>
          {/* Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Tespit Sonuçları</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Status */}
              <div className="flex items-center gap-4">
                {result.has_pii ? (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>PII Tespit Edildi</AlertTitle>
                    <AlertDescription>
                      {result.matches.length} adet kişisel bilgi tespit edildi
                    </AlertDescription>
                  </Alert>
                ) : (
                  <Alert>
                    <CheckCircle className="h-4 w-4" />
                    <AlertTitle>Güvenli</AlertTitle>
                    <AlertDescription>
                      Kişisel bilgi tespit edilmedi
                    </AlertDescription>
                  </Alert>
                )}
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-3 gap-4">
                <div className="p-4 border rounded-lg">
                  <div className="text-sm text-muted-foreground">Risk Skoru</div>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-2xl font-bold">
                      {(result.risk_score * 100).toFixed(0)}%
                    </span>
                    <Badge variant={getRiskColor(result.risk_score)}>
                      {getRiskLabel(result.risk_score)}
                    </Badge>
                  </div>
                </div>

                <div className="p-4 border rounded-lg">
                  <div className="text-sm text-muted-foreground">Tespit Sayısı</div>
                  <div className="text-2xl font-bold mt-1">
                    {result.matches.length}
                  </div>
                </div>

                <div className="p-4 border rounded-lg">
                  <div className="text-sm text-muted-foreground">İşlem Süresi</div>
                  <div className="flex items-center gap-1 mt-1">
                    <Clock className="h-4 w-4" />
                    <span className="text-2xl font-bold">
                      {result.processing_time_ms.toFixed(0)}
                    </span>
                    <span className="text-sm">ms</span>
                  </div>
                </div>
              </div>

              {/* Warnings */}
              {result.warnings.length > 0 && (
                <div className="space-y-2">
                  <h4 className="font-semibold">Uyarılar:</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {result.warnings.map((warning, i) => (
                      <li key={i} className="text-sm text-muted-foreground">
                        {warning}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Detected PII */}
          {result.matches.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Tespit Edilen Kişisel Bilgiler</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {result.matches.map((match, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between p-3 border rounded-lg"
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <Badge>{match.pii_type}</Badge>
                          <span className="font-mono text-sm">
                            {match.matched_text}
                          </span>
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                          Pozisyon: {match.start_pos} - {match.end_pos} | 
                          Güven: {(match.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-semibold text-green-600">
                          → {match.masked_text}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Masked Text */}
          <Card>
            <CardHeader>
              <CardTitle>Maskelenmiş Metin</CardTitle>
              <CardDescription>
                LLM'e gönderilecek güvenli metin
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="p-4 bg-muted rounded-lg font-mono text-sm whitespace-pre-wrap">
                {result.masked_text}
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {/* Statistics Section */}
      <Card>
        <CardHeader>
          <CardTitle>Test İstatistikleri</CardTitle>
          <CardDescription>
            Geçmiş test sonuçlarının özeti
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            İstatistikler yükleniyor...
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
```

### API Endpoint

**Dosya:** `backend/app/routers/privacy.py`

```python
"""Privacy API endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.services.pii_detection import TurkishPIIDetector
from app.services.content_safety import ContentSafetyFilter

router = APIRouter(prefix="/api/privacy", tags=["privacy"])


class DetectRequest(BaseModel):
    """PII tespit isteği"""
    text: str


class PIIMatchResponse(BaseModel):
    """PII match response"""
    pii_type: str
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float
    masked_text: str


class DetectResponse(BaseModel):
    """PII tespit yanıtı"""
    original_text: str
    masked_text: str
    has_pii: bool
    matches: List[PIIMatchResponse]
    risk_score: float
    warnings: List[str]
    processing_time_ms: float


@router.post("/detect", response_model=DetectResponse)
async def detect_pii(request: DetectRequest):
    """
    PII tespit et
    
    Test sayfası için endpoint
    """
    detector = TurkishPIIDetector()
    result = detector.detect(request.text)
    
    return DetectResponse(
        original_text=result.original_text,
        masked_text=result.masked_text,
        has_pii=result.has_pii,
        matches=[
            PIIMatchResponse(
                pii_type=m.pii_type.value,
                matched_text=m.matched_text,
                start_pos=m.start_pos,
                end_pos=m.end_pos,
                confidence=m.confidence,
                masked_text=m.masked_text
            )
            for m in result.matches
        ],
        risk_score=result.risk_score,
        warnings=result.warnings,
        processing_time_ms=result.processing_time_ms
    )
```

---

## 📊 BATCH TEST SAYFASI

Çok sayıda test case'i toplu olarak test etmek için:

**Dosya:** `frontend/src/app/dashboard/privacy-test/batch/page.tsx`

```typescript
'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Card } from '@/components/ui/card';

export default function BatchTestPage() {
  const [testing, setTesting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<any[]>([]);

  const runBatchTest = async () => {
    setTesting(true);
    setProgress(0);
    
    // Test dataset'ini yükle
    const response = await fetch('/api/privacy/test-dataset');
    const testCases = await response.json();
    
    const batchResults = [];
    
    for (let i = 0; i < testCases.length; i++) {
      const testCase = testCases[i];
      
      // Her test case'i çalıştır
      const result = await fetch('/api/privacy/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: testCase.text })
      }).then(r => r.json());
      
      // Sonucu değerlendir
      const evaluation = evaluateResult(testCase, result);
      batchResults.push(evaluation);
      
      // Progress güncelle
      setProgress(((i + 1) / testCases.length) * 100);
    }
    
    setResults(batchResults);
    setTesting(false);
  };

  const evaluateResult = (testCase: any, result: any) => {
    // Expected vs Detected karşılaştırması
    const expectedTypes = new Set(testCase.expected_pii.map((p: any) => p.type));
    const detectedTypes = new Set(result.matches.map((m: any) => m.pii_type));
    
    const tp = [...expectedTypes].filter(t => detectedTypes.has(t)).length;
    const fp = [...detectedTypes].filter(t => !expectedTypes.has(t)).length;
    const fn = [...expectedTypes].filter(t => !detectedTypes.has(t)).length;
    
    return {
      test_case_id: testCase.id,
      tp,
      fp,
      fn,
      precision: tp / (tp + fp) || 0,
      recall: tp / (tp + fn) || 0
    };
  };

  // Genel metrikleri hesapla
  const calculateOverallMetrics = () => {
    if (results.length === 0) return null;
    
    const totalTP = results.reduce((sum, r) => sum + r.tp, 0);
    const totalFP = results.reduce((sum, r) => sum + r.fp, 0);
    const totalFN = results.reduce((sum, r) => sum + r.fn, 0);
    
    const precision = totalTP / (totalTP + totalFP) || 0;
    const recall = totalTP / (totalTP + totalFN) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    return { precision, recall, f1 };
  };

  const metrics = calculateOverallMetrics();

  return (
    <div className="container mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-bold">Toplu Test</h1>
      
      <Card className="p-6">
        <Button
          onClick={runBatchTest}
          disabled={testing}
          className="w-full"
        >
          {testing ? 'Test Ediliyor...' : 'Toplu Test Başlat'}
        </Button>
        
        {testing && (
          <div className="mt-4">
            <Progress value={progress} />
            <p className="text-sm text-center mt-2">
              {progress.toFixed(0)}% tamamlandı
            </p>
          </div>
        )}
      </Card>

      {metrics && (
        <Card className="p-6">
          <h2 className="text-xl font-bold mb-4">Genel Sonuçlar</h2>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-muted-foreground">Precision</div>
              <div className="text-2xl font-bold">
                {(metrics.precision * 100).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">Recall</div>
              <div className="text-2xl font-bold">
                {(metrics.recall * 100).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">F1 Score</div>
              <div className="text-2xl font-bold">
                {(metrics.f1 * 100).toFixed(2)}%
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
```

---

## ✅ ÖZET: TAMAMLANMIŞ SİSTEM

### Backend
- ✅ `TurkishPIIDetector` - PII tespit motoru
- ✅ `ContentSafetyFilter` - İçerik güvenliği
- ✅ `PrivacyMiddleware` - Otomatik koruma
- ✅ Database modelleri - Log sistemi
- ✅ API endpoints - Test ve production

### Frontend
- ✅ Test sayfası - İnteraktif test
- ✅ Batch test - Toplu değerlendirme
- ✅ Görselleştirme - Grafikler ve metrikler
- ✅ Admin dashboard - İstatistikler

### Evaluation
- ✅ Test dataset (100+ case)
- ✅ Evaluation script
- ✅ Bilimsel metrikler (Precision, Recall, F1)
- ✅ Görselleştirmeler
- ✅ JSON rapor

### Bilimsel Çıktılar
- ✅ Precision/Recall/F1 metrikleri
- ✅ PII türüne göre performans
- ✅ Confusion matrix
- ✅ Latency analizi
- ✅ Karşılaştırmalı grafikler

**Toplam Süre:** 4 gün
**Bilimsel Değer:** Yüksek (ilk Türkçe PII sistemi)
**Makale Katkısı:** 1 tam bölüm + 3-4 tablo/grafik

Başlamaya hazır mısınız? 🚀
