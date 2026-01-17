# Türkçe Dil Optimizasyonu - Detaylı Uygulama Planı

## 🎯 Genel Bakış

Bu doküman, Türkçe dil desteği optimizasyonunun detaylı teknik planını ve **öğrenci gizliliği koruma** sistemini içermektedir.

---

## 🔒 YENİ: GİZLİLİK VE GÜVENLİK KATMANI (ÖNCELİK!)

### Neden Önemli?

- ✅ API tabanlı LLM kullanımı (OpenAI, Claude, vb.)
- ✅ Öğrenci verileri 3. parti servislere gidiyor
- ✅ KVKK/GDPR uyumluluğu gerekli
- ✅ Lise öğrencileri (reşit olmayan kullanıcılar)
- ✅ Bilimsel makale için etik onay gerekli

### 📊 Risk Analizi

**Potansiyel Riskler:**
1. 🔴 Öğrenci adı, soyadı, TC kimlik no
2. 🔴 Telefon numarası, e-posta adresi
3. 🔴 Ev adresi, okul bilgileri
4. 🟡 Hassas kişisel bilgiler (sağlık, din, vb.)
5. 🟡 Diğer öğrencilerin isimleri
6. 🟢 Uygunsuz içerik (küfür, şiddet)

### 🛡️ Uygulama: PII (Personally Identifiable Information) Filtreleme

#### Faz 0: Gizlilik Koruma Sistemi (3-4 gün) - ÖNCELİKLİ!

**Yapılacaklar:**

**1. PII Detection Service (2 gün)**


```python
# backend/app/services/pii_filter.py

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class PIIDetection:
    """PII tespit sonucu"""
    has_pii: bool
    pii_types: List[str]
    masked_text: str
    confidence: float
    warnings: List[str]

class TurkishPIIFilter:
    """Türkçe için PII filtreleme servisi"""
    
    # Türkçe isim listesi (en yaygın 1000 isim)
    TURKISH_NAMES = {
        'ahmet', 'mehmet', 'ali', 'ayşe', 'fatma', 'zeynep',
        'mustafa', 'hüseyin', 'emine', 'hatice', 'ibrahim',
        # ... toplam ~1000 isim
    }
    
    # Türkçe soyisim kalıpları
    SURNAME_PATTERNS = [
        r'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+oğlu\b',  # -oğlu
        r'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+can\b',   # -can
        r'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+er\b',    # -er
        r'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+ay\b',    # -ay
    ]
    
    # Regex patterns
    PATTERNS = {
        'tc_kimlik': r'\b[1-9]\d{10}\b',  # 11 haneli TC
        'telefon': r'\b0?5\d{2}\s?\d{3}\s?\d{2}\s?\d{2}\b',  # Türk telefon
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'iban': r'\bTR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b',
        'adres': r'\b(sokak|cadde|mahalle|apartman|daire|no:)\b',
    }
    
    def detect_and_mask(self, text: str) -> PIIDetection:
        """
        Metinde PII tespit et ve maskele
        
        Returns:
            PIIDetection: Tespit sonucu ve maskelenmiş metin
        """
        pii_types = []
        masked_text = text
        warnings = []
        
        # 1. TC Kimlik No kontrolü
        if self._detect_tc_kimlik(text):
            pii_types.append('tc_kimlik')
            masked_text = self._mask_tc_kimlik(masked_text)
            warnings.append('TC Kimlik numarası tespit edildi ve maskelendi')
        
        # 2. Telefon numarası
        if self._detect_phone(text):
            pii_types.append('telefon')
            masked_text = self._mask_phone(masked_text)
            warnings.append('Telefon numarası tespit edildi ve maskelendi')
        
        # 3. E-posta adresi
        if self._detect_email(text):
            pii_types.append('email')
            masked_text = self._mask_email(masked_text)
            warnings.append('E-posta adresi tespit edildi ve maskelendi')
        
        # 4. İsim-soyisim tespiti (NER benzeri)
        names = self._detect_names(text)
        if names:
            pii_types.append('isim')
            masked_text = self._mask_names(masked_text, names)
            warnings.append(f'{len(names)} isim tespit edildi ve maskelendi')
        
        # 5. Adres bilgisi
        if self._detect_address(text):
            pii_types.append('adres')
            warnings.append('Adres bilgisi tespit edildi')
        
        # 6. IBAN
        if self._detect_iban(text):
            pii_types.append('iban')
            masked_text = self._mask_iban(masked_text)
            warnings.append('IBAN tespit edildi ve maskelendi')
        
        has_pii = len(pii_types) > 0
        confidence = self._calculate_confidence(pii_types)
        
        return PIIDetection(
            has_pii=has_pii,
            pii_types=pii_types,
            masked_text=masked_text,
            confidence=confidence,
            warnings=warnings
        )
    
    def _detect_tc_kimlik(self, text: str) -> bool:
        """TC Kimlik numarası tespit et"""
        matches = re.findall(self.PATTERNS['tc_kimlik'], text)
        # TC kimlik algoritması ile doğrula
        for match in matches:
            if self._validate_tc_kimlik(match):
                return True
        return False
    
    def _validate_tc_kimlik(self, tc: str) -> bool:
        """TC Kimlik numarası algoritması ile doğrula"""
        if len(tc) != 11:
            return False
        
        digits = [int(d) for d in tc]
        
        # İlk hane 0 olamaz
        if digits[0] == 0:
            return False
        
        # 10. hane kontrolü
        sum_odd = sum(digits[0:9:2])
        sum_even = sum(digits[1:8:2])
        if (sum_odd * 7 - sum_even) % 10 != digits[9]:
            return False
        
        # 11. hane kontrolü
        if sum(digits[0:10]) % 10 != digits[10]:
            return False
        
        return True
    
    def _detect_names(self, text: str) -> List[str]:
        """İsim-soyisim tespit et"""
        names = []
        words = text.split()
        
        for i, word in enumerate(words):
            # Büyük harfle başlayan kelimeler
            if word[0].isupper() and len(word) > 2:
                word_lower = word.lower()
                
                # Yaygın isim listesinde var mı?
                if word_lower in self.TURKISH_NAMES:
                    names.append(word)
                    
                    # Sonraki kelime soyisim olabilir
                    if i + 1 < len(words):
                        next_word = words[i + 1]
                        if next_word[0].isupper():
                            names.append(next_word)
        
        return names
    
    def _mask_tc_kimlik(self, text: str) -> str:
        """TC Kimlik numarasını maskele"""
        return re.sub(
            self.PATTERNS['tc_kimlik'],
            '[TC_KIMLIK_MASKELENDI]',
            text
        )
    
    def _mask_phone(self, text: str) -> str:
        """Telefon numarasını maskele"""
        return re.sub(
            self.PATTERNS['telefon'],
            '[TELEFON_MASKELENDI]',
            text
        )
    
    def _mask_email(self, text: str) -> str:
        """E-posta adresini maskele"""
        return re.sub(
            self.PATTERNS['email'],
            '[EMAIL_MASKELENDI]',
            text
        )
    
    def _mask_names(self, text: str, names: List[str]) -> str:
        """İsimleri maskele"""
        for name in names:
            text = text.replace(name, '[ISIM_MASKELENDI]')
        return text
    
    def _mask_iban(self, text: str) -> str:
        """IBAN'ı maskele"""
        return re.sub(
            self.PATTERNS['iban'],
            '[IBAN_MASKELENDI]',
            text
        )
    
    def _calculate_confidence(self, pii_types: List[str]) -> float:
        """Tespit güven skoru hesapla"""
        if not pii_types:
            return 1.0  # PII yok, güvenli
        
        # Her PII tipi için risk skoru
        risk_scores = {
            'tc_kimlik': 1.0,  # En yüksek risk
            'telefon': 0.9,
            'email': 0.8,
            'iban': 0.9,
            'isim': 0.6,
            'adres': 0.7,
        }
        
        max_risk = max(risk_scores.get(pii, 0.5) for pii in pii_types)
        return max_risk
```

**2. Content Safety Filter (1 gün)**

```python
# backend/app/services/content_safety.py

class ContentSafetyFilter:
    """İçerik güvenliği filtresi"""
    
    # Türkçe uygunsuz kelimeler (küfür, hakaret)
    INAPPROPRIATE_WORDS = {
        # Küfür listesi (gerçek implementasyonda daha kapsamlı)
        'küfür1', 'küfür2', # ...
    }
    
    # Hassas konular
    SENSITIVE_TOPICS = {
        'şiddet': ['öldür', 'vur', 'döv', 'kan'],
        'cinsellik': ['cinsel', 'seks', 'tecavüz'],
        'uyuşturucu': ['uyuşturucu', 'esrar', 'kokain'],
        'intihar': ['intihar', 'kendini öldür', 'canına kıy'],
    }
    
    def check_content(self, text: str) -> Dict:
        """İçerik güvenliği kontrolü"""
        issues = []
        
        # 1. Küfür kontrolü
        if self._contains_profanity(text):
            issues.append({
                'type': 'profanity',
                'severity': 'high',
                'message': 'Uygunsuz kelime tespit edildi'
            })
        
        # 2. Hassas konu kontrolü
        sensitive = self._detect_sensitive_topics(text)
        if sensitive:
            issues.append({
                'type': 'sensitive_topic',
                'topics': sensitive,
                'severity': 'medium',
                'message': f'Hassas konu tespit edildi: {", ".join(sensitive)}'
            })
        
        # 3. Spam/tekrar kontrolü
        if self._is_spam(text):
            issues.append({
                'type': 'spam',
                'severity': 'low',
                'message': 'Spam benzeri içerik'
            })
        
        return {
            'is_safe': len(issues) == 0,
            'issues': issues,
            'filtered_text': self._filter_inappropriate(text)
        }
    
    def _contains_profanity(self, text: str) -> bool:
        """Küfür içeriyor mu?"""
        text_lower = text.lower()
        return any(word in text_lower for word in self.INAPPROPRIATE_WORDS)
    
    def _detect_sensitive_topics(self, text: str) -> List[str]:
        """Hassas konuları tespit et"""
        text_lower = text.lower()
        detected = []
        
        for topic, keywords in self.SENSITIVE_TOPICS.items():
            if any(keyword in text_lower for keyword in keywords):
                detected.append(topic)
        
        return detected
    
    def _filter_inappropriate(self, text: str) -> str:
        """Uygunsuz içeriği filtrele"""
        filtered = text
        for word in self.INAPPROPRIATE_WORDS:
            filtered = re.sub(
                r'\b' + word + r'\b',
                '[FİLTRELENDİ]',
                filtered,
                flags=re.IGNORECASE
            )
        return filtered
```

**3. Privacy Middleware (1 gün)**

```python
# backend/app/middleware/privacy_middleware.py

from fastapi import Request, HTTPException
from app.services.pii_filter import TurkishPIIFilter
from app.services.content_safety import ContentSafetyFilter

class PrivacyMiddleware:
    """Gizlilik koruma middleware"""
    
    def __init__(self):
        self.pii_filter = TurkishPIIFilter()
        self.content_filter = ContentSafetyFilter()
    
    async def __call__(self, request: Request, call_next):
        """Her request'te PII kontrolü yap"""
        
        # Sadece chat/question endpoint'lerinde kontrol et
        if request.url.path in ['/api/chat', '/api/questions', '/api/interactions']:
            body = await request.json()
            
            # Soru metnini kontrol et
            if 'question' in body:
                question = body['question']
                
                # 1. PII kontrolü
                pii_result = self.pii_filter.detect_and_mask(question)
                
                if pii_result.has_pii:
                    # PII tespit edildi - logla ve maskele
                    await self._log_pii_detection(
                        user_id=request.state.user.id,
                        pii_types=pii_result.pii_types,
                        original_text=question[:50]  # İlk 50 karakter
                    )
                    
                    # Maskelenmiş metni kullan
                    body['question'] = pii_result.masked_text
                    body['pii_detected'] = True
                    body['pii_warnings'] = pii_result.warnings
                
                # 2. İçerik güvenliği kontrolü
                safety_result = self.content_filter.check_content(question)
                
                if not safety_result['is_safe']:
                    # Uygunsuz içerik - reddet veya uyar
                    if any(issue['severity'] == 'high' for issue in safety_result['issues']):
                        raise HTTPException(
                            status_code=400,
                            detail={
                                'error': 'inappropriate_content',
                                'message': 'Uygunsuz içerik tespit edildi',
                                'issues': safety_result['issues']
                            }
                        )
                    
                    # Orta/düşük seviye - uyar ama devam et
                    body['content_warnings'] = safety_result['issues']
        
        response = await call_next(request)
        return response
    
    async def _log_pii_detection(self, user_id: int, pii_types: List[str], original_text: str):
        """PII tespitini logla (güvenlik için)"""
        # Database'e kaydet
        log_entry = PIIDetectionLog(
            user_id=user_id,
            pii_types=pii_types,
            text_preview=original_text,  # Sadece önizleme
            detected_at=datetime.utcnow()
        )
        # ... kaydet
```

**4. Database Şeması (Gizlilik Logları)**

```python
# backend/app/models/db_models.py

class PIIDetectionLog(Base):
    """PII tespit logları"""
    __tablename__ = "pii_detection_logs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    pii_types = Column(JSON)  # ['tc_kimlik', 'telefon']
    text_preview = Column(String(100))  # İlk 50 karakter (güvenlik için)
    action_taken = Column(String(50))  # 'masked', 'blocked', 'warned'
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", backref="pii_detections")

class ContentSafetyLog(Base):
    """İçerik güvenliği logları"""
    __tablename__ = "content_safety_logs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    issue_type = Column(String(50))  # 'profanity', 'sensitive_topic'
    severity = Column(String(20))  # 'low', 'medium', 'high'
    action_taken = Column(String(50))
    detected_at = Column(DateTime, default=datetime.utcnow)
```

**5. Frontend Uyarı Sistemi**

```typescript
// frontend/src/components/PrivacyWarning.tsx

export function PrivacyWarning({ warnings }: { warnings: string[] }) {
  if (!warnings || warnings.length === 0) return null;
  
  return (
    <Alert variant="warning">
      <AlertTitle>Gizlilik Uyarısı</AlertTitle>
      <AlertDescription>
        <ul>
          {warnings.map((warning, i) => (
            <li key={i}>{warning}</li>
          ))}
        </ul>
        <p className="mt-2 text-sm">
          Kişisel bilgileriniz korunmak için maskelenmiştir.
        </p>
      </AlertDescription>
    </Alert>
  );
}
```

### 📊 Bilimsel Katkı (Gizlilik)

1. **Türkçe PII Tespiti:** İlk Türkçe-spesifik PII filtreleme sistemi
2. **Eğitim Bağlamında Gizlilik:** Lise öğrencileri için özel koruma
3. **Etik AI Kullanımı:** KVKK/GDPR uyumlu RAG sistemi
4. **Metrikler:**
   - PII tespit doğruluğu (precision/recall)
   - False positive oranı
   - Kullanıcı deneyimi etkisi

### 📈 Test Senaryoları

```python
# Test cases
test_cases = [
    {
        'input': 'Benim adım Ahmet Yılmaz ve TC kimlik numaram 12345678901',
        'expected_pii': ['isim', 'tc_kimlik'],
        'expected_output': 'Benim adım [ISIM_MASKELENDI] ve TC kimlik numaram [TC_KIMLIK_MASKELENDI]'
    },
    {
        'input': 'Telefon numaram 0532 123 45 67',
        'expected_pii': ['telefon'],
        'expected_output': 'Telefon numaram [TELEFON_MASKELENDI]'
    },
    {
        'input': 'Fotosentez nedir?',
        'expected_pii': [],
        'expected_output': 'Fotosentez nedir?'  # Değişmez
    }
]
```

---

## 📚 FAZ 1: TÜRKÇE STOP WORDS (2 gün)

### Hedef
Türkçe'ye özgü stop words listesi oluşturup RAG sistemine entegre etmek.

### Detaylı Adımlar

**Gün 1: Stop Words Listesi Oluşturma**

1. **Kaynak Toplama (2 saat)**
