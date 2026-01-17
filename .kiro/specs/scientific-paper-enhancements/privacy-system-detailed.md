# Gizlilik Koruma Sistemi - Detaylı Uygulama ve Test Planı

## 🎯 Hedef

Türkçe öğrenci verilerini koruyacak, bilimsel olarak test edilebilir, tam çalışır bir PII (Personally Identifiable Information) filtreleme sistemi.

---

## 📋 GENEL MİMARİ

```
Öğrenci Sorusu
    ↓
[1. PII Detection Layer]
    ↓
[2. Content Safety Layer]
    ↓
[3. Masking & Logging]
    ↓
[4. LLM API Call]
    ↓
[5. Response Processing]
    ↓
Öğrenciye Cevap + Uyarılar
```

---

## 🔧 DETAYLI UYGULAMA ADIMLARI

### GÜN 1: PII Detection Service (Backend)

#### Adım 1.1: Temel Yapı Oluşturma (2 saat)

**Dosya:** `backend/app/services/pii_detection.py`

```python
"""
PII (Personally Identifiable Information) Detection Service
Türkçe öğrenci verilerini korumak için tasarlanmıştır.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """PII türleri"""
    TC_KIMLIK = "tc_kimlik"
    TELEFON = "telefon"
    EMAIL = "email"
    ISIM = "isim"
    ADRES = "adres"
    IBAN = "iban"
    KREDI_KARTI = "kredi_karti"
    DOGUM_TARIHI = "dogum_tarihi"


@dataclass
class PIIMatch:
    """Tespit edilen PII"""
    pii_type: PIIType
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float  # 0.0 - 1.0
    masked_text: str


@dataclass
class PIIDetectionResult:
    """PII tespit sonucu"""
    original_text: str
    masked_text: str
    has_pii: bool
    matches: List[PIIMatch]
    risk_score: float  # 0.0 - 1.0
    warnings: List[str]
    processing_time_ms: float


class TurkishPIIDetector:
    """Türkçe için PII tespit motoru"""
    
    def __init__(self):
        self._load_resources()
    
    def _load_resources(self):
        """Kaynakları yükle"""
        # Türkçe isim listesi (en yaygın 1000 isim)
        self.TURKISH_FIRST_NAMES = self._load_turkish_names()
        
        # Türkçe soyisim kalıpları
        self.SURNAME_SUFFIXES = [
            'oğlu', 'can', 'er', 'ay', 'han', 'gül', 'kaya', 'yıldız'
        ]
        
        # Regex patterns
        self.PATTERNS = {
            PIIType.TC_KIMLIK: r'\b[1-9]\d{10}\b',
            PIIType.TELEFON: r'\b0?5\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b',
            PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PIIType.IBAN: r'\bTR\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{2}\b',
            PIIType.KREDI_KARTI: r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
            PIIType.DOGUM_TARIHI: r'\b\d{2}[./]\d{2}[./]\d{4}\b',
        }
    
    def _load_turkish_names(self) -> set:
        """Türkçe isim listesi yükle"""
        # Gerçek implementasyonda dosyadan yüklenecek
        return {
            'ahmet', 'mehmet', 'ali', 'ayşe', 'fatma', 'zeynep',
            'mustafa', 'hüseyin', 'emine', 'hatice', 'ibrahim',
            'can', 'ece', 'elif', 'yusuf', 'ömer', 'meryem',
            # ... toplam ~1000 isim
        }
    
    def detect(self, text: str) -> PIIDetectionResult:
        """
        Metinde PII tespit et
        
        Args:
            text: Kontrol edilecek metin
            
        Returns:
            PIIDetectionResult: Tespit sonucu
        """
        import time
        start_time = time.time()
        
        matches: List[PIIMatch] = []
        warnings: List[str] = []
        
        # 1. TC Kimlik No
        tc_matches = self._detect_tc_kimlik(text)
        matches.extend(tc_matches)
        if tc_matches:
            warnings.append(f'{len(tc_matches)} TC Kimlik numarası tespit edildi')
        
        # 2. Telefon
        phone_matches = self._detect_phone(text)
        matches.extend(phone_matches)
        if phone_matches:
            warnings.append(f'{len(phone_matches)} telefon numarası tespit edildi')
        
        # 3. E-posta
        email_matches = self._detect_email(text)
        matches.extend(email_matches)
        if email_matches:
            warnings.append(f'{len(email_matches)} e-posta adresi tespit edildi')
        
        # 4. İsim-Soyisim
        name_matches = self._detect_names(text)
        matches.extend(name_matches)
        if name_matches:
            warnings.append(f'{len(name_matches)} isim tespit edildi')
        
        # 5. IBAN
        iban_matches = self._detect_iban(text)
        matches.extend(iban_matches)
        if iban_matches:
            warnings.append(f'{len(iban_matches)} IBAN tespit edildi')
        
        # 6. Kredi Kartı
        cc_matches = self._detect_credit_card(text)
        matches.extend(cc_matches)
        if cc_matches:
            warnings.append(f'{len(cc_matches)} kredi kartı numarası tespit edildi')
        
        # 7. Doğum Tarihi
        dob_matches = self._detect_date_of_birth(text)
        matches.extend(dob_matches)
        if dob_matches:
            warnings.append(f'{len(dob_matches)} doğum tarihi tespit edildi')
        
        # Maskelenmiş metin oluştur
        masked_text = self._apply_masking(text, matches)
        
        # Risk skoru hesapla
        risk_score = self._calculate_risk_score(matches)
        
        processing_time = (time.time() - start_time) * 1000
        
        return PIIDetectionResult(
            original_text=text,
            masked_text=masked_text,
            has_pii=len(matches) > 0,
            matches=matches,
            risk_score=risk_score,
            warnings=warnings,
            processing_time_ms=processing_time
        )
    
    def _detect_tc_kimlik(self, text: str) -> List[PIIMatch]:
        """TC Kimlik numarası tespit et ve doğrula"""
        matches = []
        pattern = self.PATTERNS[PIIType.TC_KIMLIK]
        
        for match in re.finditer(pattern, text):
            tc_no = match.group()
            
            # TC Kimlik algoritması ile doğrula
            if self._validate_tc_kimlik(tc_no):
                matches.append(PIIMatch(
                    pii_type=PIIType.TC_KIMLIK,
                    matched_text=tc_no,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=1.0,  # Algoritma doğrulaması yapıldı
                    masked_text='[TC_KIMLIK]'
                ))
        
        return matches
    
    def _validate_tc_kimlik(self, tc: str) -> bool:
        """
        TC Kimlik numarası algoritması ile doğrula
        
        Kurallar:
        1. 11 haneli olmalı
        2. İlk hane 0 olamaz
        3. 10. hane = (1+3+5+7+9. hanelerin toplamı * 7 - 2+4+6+8. hanelerin toplamı) mod 10
        4. 11. hane = (1-10. hanelerin toplamı) mod 10
        """
        if len(tc) != 11:
            return False
        
        try:
            digits = [int(d) for d in tc]
        except ValueError:
            return False
        
        # İlk hane 0 olamaz
        if digits[0] == 0:
            return False
        
        # 10. hane kontrolü
        sum_odd = sum(digits[0:9:2])  # 1, 3, 5, 7, 9. haneler
        sum_even = sum(digits[1:8:2])  # 2, 4, 6, 8. haneler
        
        tenth_digit = (sum_odd * 7 - sum_even) % 10
        if tenth_digit != digits[9]:
            return False
        
        # 11. hane kontrolü
        eleventh_digit = sum(digits[0:10]) % 10
        if eleventh_digit != digits[10]:
            return False
        
        return True
    
    def _detect_phone(self, text: str) -> List[PIIMatch]:
        """Türk telefon numarası tespit et"""
        matches = []
        pattern = self.PATTERNS[PIIType.TELEFON]
        
        for match in re.finditer(pattern, text):
            phone = match.group()
            
            # Türk telefon formatı kontrolü
            digits = re.sub(r'[\s\-]', '', phone)
            
            # 0 ile başlıyorsa çıkar
            if digits.startswith('0'):
                digits = digits[1:]
            
            # 5XX ile başlamalı (cep telefonu)
            if len(digits) == 10 and digits.startswith('5'):
                matches.append(PIIMatch(
                    pii_type=PIIType.TELEFON,
                    matched_text=phone,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.95,
                    masked_text='[TELEFON]'
                ))
        
        return matches
    
    def _detect_email(self, text: str) -> List[PIIMatch]:
        """E-posta adresi tespit et"""
        matches = []
        pattern = self.PATTERNS[PIIType.EMAIL]
        
        for match in re.finditer(pattern, text):
            email = match.group()
            
            matches.append(PIIMatch(
                pii_type=PIIType.EMAIL,
                matched_text=email,
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.98,
                masked_text='[EMAIL]'
            ))
        
        return matches
    
    def _detect_names(self, text: str) -> List[PIIMatch]:
        """
        İsim-soyisim tespit et
        
        Strateji:
        1. Büyük harfle başlayan kelimeleri bul
        2. Türkçe isim listesinde kontrol et
        3. Ardışık büyük harfli kelimeler = isim + soyisim
        """
        matches = []
        words = text.split()
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Büyük harfle başlıyor mu?
            if len(word) > 2 and word[0].isupper():
                word_clean = re.sub(r'[^\w]', '', word)
                word_lower = word_clean.lower()
                
                # Türkçe isim listesinde var mı?
                if word_lower in self.TURKISH_FIRST_NAMES:
                    # İsim bulundu
                    full_name = word_clean
                    confidence = 0.8
                    
                    # Sonraki kelime soyisim olabilir
                    if i + 1 < len(words):
                        next_word = words[i + 1]
                        if len(next_word) > 2 and next_word[0].isupper():
                            next_clean = re.sub(r'[^\w]', '', next_word)
                            full_name = f"{word_clean} {next_clean}"
                            confidence = 0.9
                            i += 1  # Sonraki kelimeyi atla
                    
                    # Pozisyon bul
                    start_pos = text.find(full_name)
                    if start_pos != -1:
                        matches.append(PIIMatch(
                            pii_type=PIIType.ISIM,
                            matched_text=full_name,
                            start_pos=start_pos,
                            end_pos=start_pos + len(full_name),
                            confidence=confidence,
                            masked_text='[ISIM]'
                        ))
            
            i += 1
        
        return matches
    
    def _detect_iban(self, text: str) -> List[PIIMatch]:
        """IBAN tespit et"""
        matches = []
        pattern = self.PATTERNS[PIIType.IBAN]
        
        for match in re.finditer(pattern, text):
            iban = match.group()
            
            matches.append(PIIMatch(
                pii_type=PIIType.IBAN,
                matched_text=iban,
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.99,
                masked_text='[IBAN]'
            ))
        
        return matches
    
    def _detect_credit_card(self, text: str) -> List[PIIMatch]:
        """Kredi kartı numarası tespit et"""
        matches = []
        pattern = self.PATTERNS[PIIType.KREDI_KARTI]
        
        for match in re.finditer(pattern, text):
            cc = match.group()
            digits = re.sub(r'[\s\-]', '', cc)
            
            # Luhn algoritması ile doğrula
            if self._validate_luhn(digits):
                matches.append(PIIMatch(
                    pii_type=PIIType.KREDI_KARTI,
                    matched_text=cc,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=1.0,
                    masked_text='[KREDI_KARTI]'
                ))
        
        return matches
    
    def _validate_luhn(self, card_number: str) -> bool:
        """Luhn algoritması ile kredi kartı doğrula"""
        def digits_of(n):
            return [int(d) for d in str(n)]
        
        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        
        return checksum % 10 == 0
    
    def _detect_date_of_birth(self, text: str) -> List[PIIMatch]:
        """Doğum tarihi tespit et"""
        matches = []
        pattern = self.PATTERNS[PIIType.DOGUM_TARIHI]
        
        for match in re.finditer(pattern, text):
            date = match.group()
            
            # Tarih formatı kontrolü (basit)
            parts = re.split(r'[./]', date)
            if len(parts) == 3:
                day, month, year = parts
                
                # Geçerli tarih mi?
                if (1 <= int(day) <= 31 and 
                    1 <= int(month) <= 12 and 
                    1900 <= int(year) <= 2020):  # Lise öğrencisi için makul aralık
                    
                    matches.append(PIIMatch(
                        pii_type=PIIType.DOGUM_TARIHI,
                        matched_text=date,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.85,
                        masked_text='[DOGUM_TARIHI]'
                    ))
        
        return matches
    
    def _apply_masking(self, text: str, matches: List[PIIMatch]) -> str:
        """
        PII'ları maskele
        
        Strateji: Sondan başa doğru maskele (pozisyon kayması olmasın)
        """
        # Pozisyona göre sırala (sondan başa)
        sorted_matches = sorted(matches, key=lambda m: m.start_pos, reverse=True)
        
        masked = text
        for match in sorted_matches:
            masked = (
                masked[:match.start_pos] + 
                match.masked_text + 
                masked[match.end_pos:]
            )
        
        return masked
    
    def _calculate_risk_score(self, matches: List[PIIMatch]) -> float:
        """
        Risk skoru hesapla (0.0 - 1.0)
        
        Yüksek risk: TC Kimlik, Kredi Kartı, IBAN
        Orta risk: Telefon, E-posta
        Düşük risk: İsim, Doğum Tarihi
        """
        if not matches:
            return 0.0
        
        risk_weights = {
            PIIType.TC_KIMLIK: 1.0,
            PIIType.KREDI_KARTI: 1.0,
            PIIType.IBAN: 0.9,
            PIIType.TELEFON: 0.7,
            PIIType.EMAIL: 0.6,
            PIIType.DOGUM_TARIHI: 0.5,
            PIIType.ISIM: 0.4,
        }
        
        total_risk = sum(
            risk_weights.get(match.pii_type, 0.5) * match.confidence 
            for match in matches
        )
        
        # Normalize (max 1.0)
        max_possible_risk = len(matches) * 1.0
        normalized_risk = min(total_risk / max_possible_risk, 1.0)
        
        return normalized_risk
```

Bu ilk adımı tamamladık. Devam edelim mi?
