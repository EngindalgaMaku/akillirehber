# Privacy Protection System - Design Document

## Overview

The Privacy Protection System is a multi-layered security framework designed to protect Turkish high school students' personally identifiable information (PII) when using API-based Large Language Models for educational purposes. The system operates as a transparent middleware layer that intercepts, analyzes, and sanitizes all student-generated content before it reaches external LLM APIs.

**Key Design Principles:**
1. **Defense in Depth**: Multiple layers of protection (PII detection, content safety, risk assessment)
2. **Fail-Safe**: System blocks requests when in doubt rather than allowing potential privacy breaches
3. **Transparency**: Students are informed when their content is modified or blocked
4. **Performance**: Minimal latency impact (<100ms) to maintain good user experience
5. **Auditability**: Comprehensive logging for KVKK/GDPR compliance
6. **Scientific Rigor**: Measurable, testable, and reproducible results for academic publication

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Student Client                           │
│                    (Frontend React/Next.js)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST /api/chat
                             │ { question: "..." }
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Privacy Middleware                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  1. PII Detection Layer                             │  │  │
│  │  │     - TurkishPIIDetector                            │  │  │
│  │  │     - Regex patterns + validation algorithms        │  │  │
│  │  │     - Turkish name database                         │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  2. Content Safety Layer                            │  │  │
│  │  │     - ContentSafetyFilter                           │  │  │
│  │  │     - Profanity detection                           │  │  │
│  │  │     - Sensitive topic detection                     │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  3. Risk Assessment & Decision                      │  │  │
│  │  │     - Calculate risk score (0.0-1.0)                │  │  │
│  │  │     - Block if risk > 0.8                           │  │  │
│  │  │     - Mask if 0.3 < risk ≤ 0.8                      │  │  │
│  │  │     - Allow if risk ≤ 0.3                           │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  4. Logging & Audit                                 │  │  │
│  │  │     - PIIDetectionLog (database)                    │  │  │
│  │  │     - ContentSafetyLog (database)                   │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                             │                                    │
│                             │ Masked/Sanitized Question          │
│                             ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              RAG Pipeline                                 │  │
│  │  - Vector search                                          │  │
│  │  - Context retrieval                                      │  │
│  │  - LLM API call (OpenAI/Claude)                          │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Student Input**: Student submits question via chat interface
2. **Middleware Interception**: Privacy middleware intercepts the request
3. **PII Detection**: TurkishPIIDetector scans for 7 PII types
4. **Content Safety Check**: ContentSafetyFilter checks for inappropriate content
5. **Risk Calculation**: System calculates overall risk score
6. **Decision Making**:
   - **High Risk (>0.8)**: Block request, return error to student
   - **Medium Risk (0.3-0.8)**: Mask PII, log warning, forward to LLM
   - **Low Risk (<0.3)**: Allow with minimal logging
7. **Logging**: Record detection events in database
8. **LLM Processing**: Sanitized question sent to RAG pipeline
9. **Response**: Answer returned to student with privacy warnings if applicable

## Components and Interfaces

### 1. TurkishPIIDetector

**Purpose**: Detect and classify Turkish PII in text

**Interface**:
```python
class TurkishPIIDetector:
    def __init__(self):
        """Initialize detector with Turkish name database and regex patterns"""
        
    def detect(self, text: str) -> PIIDetectionResult:
        """
        Detect all PII in the given text
        
        Args:
            text: Input text to analyze
            
        Returns:
            PIIDetectionResult containing:
                - original_text: Original input
                - masked_text: Text with PII replaced by tokens
                - has_pii: Boolean indicating if PII was found
                - matches: List of PIIMatch objects
                - risk_score: Float 0.0-1.0
                - warnings: List of warning messages
                - processing_time_ms: Detection latency
        """
```

**Detection Methods**:
- `_detect_tc_kimlik()`: Validates using 10th/11th digit algorithm
- `_detect_phone()`: Regex + format validation for Turkish mobile numbers
- `_detect_email()`: RFC-compliant email regex
- `_detect_names()`: Turkish name database lookup + capitalization rules
- `_detect_iban()`: TR-format IBAN detection
- `_detect_credit_card()`: Luhn algorithm validation
- `_detect_date_of_birth()`: Date format + range validation (1900-2020)

**Key Algorithms**:

*TC Kimlik Validation*:
```
1. Must be 11 digits
2. First digit cannot be 0
3. 10th digit = (sum of odd positions * 7 - sum of even positions) mod 10
4. 11th digit = (sum of first 10 digits) mod 10
```

*Luhn Algorithm (Credit Card)*:
```
1. Double every second digit from right
2. If doubled digit > 9, subtract 9
3. Sum all digits
4. Valid if sum mod 10 = 0
```

### 2. ContentSafetyFilter

**Purpose**: Detect inappropriate content and sensitive topics

**Interface**:
```python
class ContentSafetyFilter:
    def __init__(self):
        """Initialize with profanity list and sensitive topic keywords"""
        
    def check(self, text: str) -> Dict:
        """
        Check text for content safety issues
        
        Returns:
            {
                'is_safe': bool,
                'issues': List[Dict],  # Each issue has type, severity, message
                'filtered_text': str,  # Text with profanity masked
                'risk_level': str  # 'low', 'medium', 'high'
            }
        """
```

**Detection Categories**:
- **Profanity**: Turkish profanity word list
- **Violence**: Keywords like öldür, vur, döv, kan, silah
- **Self-Harm**: Keywords like intihar, kendimi öldür, canıma kıy
- **Drugs**: Keywords like uyuşturucu, esrar, kokain
- **Sexual Content**: Inappropriate sexual keywords
- **Spam**: >70% uppercase OR <30% unique words

### 3. PrivacyMiddleware

**Purpose**: FastAPI middleware to automatically protect all endpoints

**Interface**:
```python
class PrivacyMiddleware:
    def __init__(self):
        """Initialize with PII detector and content filter"""
        
    async def __call__(self, request: Request, call_next):
        """
        Intercept requests to protected endpoints
        
        Protected endpoints:
            - /api/chat
            - /api/questions
            - /api/interactions
            - /api/ragas/quick-test
        """
```

**Processing Logic**:
1. Check if endpoint is protected
2. Extract text field from request body
3. Run PII detection
4. Run content safety check
5. Calculate risk score
6. Make decision (block/mask/allow)
7. Log to database
8. Modify request body if needed
9. Forward to next handler

### 4. Database Models

**PIIDetectionLog**:
```python
class PIIDetectionLog(Base):
    __tablename__ = "pii_detection_logs"
    
    id: int  # Primary key
    user_id: Optional[int]  # Foreign key to users
    pii_types: List[str]  # JSON array of detected PII types
    text_preview: str  # First 100 characters only
    risk_score: float  # 0.0-1.0
    action_taken: str  # 'blocked', 'masked', 'allowed'
    detected_at: datetime
```

**ContentSafetyLog**:
```python
class ContentSafetyLog(Base):
    __tablename__ = "content_safety_logs"
    
    id: int  # Primary key
    user_id: Optional[int]  # Foreign key to users
    issue_type: str  # 'profanity', 'violence', 'self_harm', etc.
    severity: str  # 'low', 'medium', 'high'
    action_taken: str  # 'blocked', 'warned', 'allowed'
    detected_at: datetime
```

### 5. API Endpoints

**POST /api/privacy/detect**:
```python
Request:
{
    "text": "Benim TC kimlik numaram 12345678901"
}

Response:
{
    "has_pii": true,
    "matches": [
        {
            "pii_type": "tc_kimlik",
            "matched_text": "12345678901",
            "start_pos": 26,
            "end_pos": 37,
            "confidence": 1.0,
            "masked_text": "[TC_KIMLIK]"
        }
    ],
    "masked_text": "Benim TC kimlik numaram [TC_KIMLIK]",
    "risk_score": 0.85,
    "warnings": ["1 TC Kimlik numarası tespit edildi"],
    "processing_time_ms": 45.2
}
```

**GET /api/privacy/stats**:
```python
Response:
{
    "total_detections": 1234,
    "pii_type_breakdown": {
        "tc_kimlik": 45,
        "telefon": 123,
        "email": 234,
        "isim": 567,
        "iban": 12,
        "kredi_karti": 3,
        "dogum_tarihi": 250
    },
    "risk_distribution": {
        "high": 45,
        "medium": 234,
        "low": 955
    },
    "actions_taken": {
        "blocked": 45,
        "masked": 234,
        "allowed": 955
    }
}
```

## Data Models

### PIIMatch
```python
@dataclass
class PIIMatch:
    pii_type: PIIType  # Enum: TC_KIMLIK, TELEFON, EMAIL, etc.
    matched_text: str  # The actual PII found
    start_pos: int  # Character position in original text
    end_pos: int  # Character position in original text
    confidence: float  # 0.0-1.0
    masked_text: str  # Replacement token like [TC_KIMLIK]
```

### PIIDetectionResult
```python
@dataclass
class PIIDetectionResult:
    original_text: str
    masked_text: str
    has_pii: bool
    matches: List[PIIMatch]
    risk_score: float  # 0.0-1.0
    warnings: List[str]
    processing_time_ms: float
```

### ContentSafetyResult
```python
@dataclass
class ContentSafetyResult:
    is_safe: bool
    issues: List[ContentIssue]
    filtered_text: str
    risk_level: str  # 'low', 'medium', 'high'
```

### ContentIssue
```python
@dataclass
class ContentIssue:
    type: ContentIssueType  # Enum: PROFANITY, VIOLENCE, SELF_HARM, etc.
    severity: str  # 'low', 'medium', 'high'
    message: str
    keywords: Optional[List[str]]  # Keywords that triggered detection
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: TC Kimlik Detection Completeness
*For any* valid TC Kimlik number embedded in text, the system should detect it with 100% confidence.

**Validates: Requirements 1.1**

### Property 2: Phone Number Detection Accuracy
*For any* Turkish mobile phone number in standard formats (05XX XXX XX XX, 5XX XXX XX XX, 05XXXXXXXXX), the system should detect it with at least 95% confidence.

**Validates: Requirements 1.2**

### Property 3: Email Detection Completeness
*For any* RFC-compliant email address, the system should detect it with at least 98% confidence.

**Validates: Requirements 1.3**

### Property 4: Turkish Name Detection
*For any* Turkish first name from the database followed by a capitalized word, the system should detect it as a full name with at least 80% confidence.

**Validates: Requirements 1.4**

### Property 5: Multiple PII Independence
*For any* text containing multiple PII types, the system should detect each type independently without one detection interfering with another.

**Validates: Requirements 1.8**

### Property 6: Masking Preserves Structure
*For any* text with detected PII, the masked version should have the same number of words and preserve punctuation positions.

**Validates: Requirements 2.3**

### Property 7: Masking Order Independence
*For any* text with multiple PII instances, masking them in any order should produce the same final masked text.

**Validates: Requirements 2.2**

### Property 8: Original and Masked Text Storage
*For any* PII detection result, both original_text and masked_text fields should be non-empty and different when PII is detected.

**Validates: Requirements 2.5**

### Property 9: Risk Score Blocking Threshold
*For any* text with risk score > 0.8, the system should block the request and return an error response.

**Validates: Requirements 3.6**

### Property 10: Risk Score Masking Threshold
*For any* text with risk score between 0.3 and 0.8, the system should mask PII and allow the request with warnings.

**Validates: Requirements 3.7**

### Property 11: Profanity Detection
*For any* text containing words from the profanity list, the system should flag it as high severity.

**Validates: Requirements 4.1**

### Property 12: Self-Harm Content Blocking
*For any* text containing self-harm keywords, the system should flag it as high severity and block the request.

**Validates: Requirements 4.3**

### Property 13: Spam Pattern Detection
*For any* text where >70% of characters are uppercase OR <30% of words are unique, the system should flag it as spam.

**Validates: Requirements 4.6**

### Property 14: High Severity Content Blocking
*For any* content flagged as high severity, the system should block the request and return an error.

**Validates: Requirements 4.7**

### Property 15: PII Detection Logging
*For any* PII detection event, a log entry should be created with pii_types, risk_score, action_taken, and timestamp fields populated.

**Validates: Requirements 5.1**

### Property 16: Log Text Preview Truncation
*For any* logged PII detection, the text_preview field should contain at most 100 characters.

**Validates: Requirements 5.2**

### Property 17: Detection Latency
*For any* text between 50-200 words, PII detection should complete in less than 100ms on average.

**Validates: Requirements 6.1**

### Property 18: Evaluation Precision Calculation
*For any* test dataset, Precision should be calculated as TP / (TP + FP) for each PII type.

**Validates: Requirements 8.1**

### Property 19: Evaluation Recall Calculation
*For any* test dataset, Recall should be calculated as TP / (TP + FN) for each PII type.

**Validates: Requirements 8.2**

### Property 20: Evaluation F1 Score Calculation
*For any* test dataset, F1 Score should be calculated as 2 * (Precision * Recall) / (Precision + Recall) for each PII type.

**Validates: Requirements 8.3**

### Property 21: Turkish Character Handling
*For any* text containing Turkish characters (ç, ğ, ı, ö, ş, ü), the system should process them correctly without encoding errors.

**Validates: Requirements 10.1, 10.2**

### Property 22: Turkish Name Database Coverage
*For any* name from the Turkish name database, when it appears capitalized in text, it should be detected with at least 80% confidence.

**Validates: Requirements 10.3**

## Error Handling

### Error Categories

1. **Validation Errors** (HTTP 400):
   - Invalid request format
   - Missing required fields
   - Malformed text input

2. **Privacy Violations** (HTTP 400):
   - High-risk PII detected (risk > 0.8)
   - High-severity content detected (self-harm, extreme violence)

3. **System Errors** (HTTP 500):
   - Database connection failures
   - Turkish name database loading errors
   - Regex compilation errors

4. **Performance Errors** (HTTP 503):
   - Detection timeout (>1000ms)
   - System overload (>100 concurrent requests)

### Error Response Format

```python
{
    "error": "high_risk_pii_detected",
    "message": "Yüksek riskli kişisel bilgi tespit edildi",
    "details": {
        "risk_score": 0.92,
        "pii_types": ["tc_kimlik", "kredi_karti"],
        "warnings": [
            "1 TC Kimlik numarası tespit edildi",
            "1 kredi kartı numarası tespit edildi"
        ]
    },
    "masked_text": "Benim TC kimlik numaram [TC_KIMLIK] ve kartım [KREDI_KARTI]",
    "timestamp": "2024-01-15T10:30:45Z"
}
```

### Fail-Safe Behavior

- **PII Detection Failure**: If detector crashes, block request (fail-closed)
- **Content Safety Failure**: If filter crashes, allow request but log error (fail-open for educational content)
- **Database Logging Failure**: Continue processing but log error to file system
- **Turkish Name Database Missing**: Use regex-only detection with reduced confidence

## Testing Strategy

### Dual Testing Approach

The system uses both **unit tests** and **property-based tests** to ensure comprehensive coverage:

- **Unit Tests**: Verify specific examples, edge cases, and error conditions
- **Property Tests**: Verify universal properties across all inputs using randomized testing

Both approaches are complementary and necessary for comprehensive correctness validation.

### Unit Testing

**Focus Areas**:
- Specific PII examples (known valid TC Kimlik numbers, phone formats)
- Edge cases (empty input, very long text, special characters)
- Error conditions (invalid formats, malformed data)
- Integration points (middleware, database, API endpoints)

**Example Unit Tests**:
```python
def test_tc_kimlik_specific_valid():
    """Test with known valid TC Kimlik"""
    detector = TurkishPIIDetector()
    result = detector.detect("TC: 12345678901")
    assert result.has_pii
    assert len(result.matches) == 1
    assert result.matches[0].pii_type == PIIType.TC_KIMLIK

def test_empty_input():
    """Test with empty string"""
    detector = TurkishPIIDetector()
    result = detector.detect("")
    assert not result.has_pii
    assert len(result.matches) == 0
```

### Property-Based Testing

**Configuration**:
- Minimum 100 iterations per property test
- Use Hypothesis library for Python
- Each property test references its design document property number

**Property Test Structure**:
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=11, max_size=11, alphabet='0123456789'))
def test_property_1_tc_kimlik_detection(tc_number):
    """
    Property 1: TC Kimlik Detection Completeness
    Feature: privacy-protection-system, Property 1
    """
    # Generate valid TC Kimlik using algorithm
    if not is_valid_tc_kimlik(tc_number):
        tc_number = generate_valid_tc_kimlik()
    
    detector = TurkishPIIDetector()
    result = detector.detect(f"TC: {tc_number}")
    
    # Property: All valid TC Kimlik numbers should be detected
    assert result.has_pii
    assert any(m.pii_type == PIIType.TC_KIMLIK for m in result.matches)
    assert any(m.confidence == 1.0 for m in result.matches)
```

**Property Test Tags**:
Each property test must include a comment with:
- Feature name: `privacy-protection-system`
- Property number: `Property 1`, `Property 2`, etc.
- Property description from design document

### Test Dataset

**Structure**:
- **50 Positive Cases**: Text with various PII types
- **30 Negative Cases**: Educational content with no PII
- **20 Edge Cases**: Boundary conditions, ambiguous cases

**Example Test Cases**:
```json
{
    "id": 1,
    "text": "Benim adım Ahmet Yılmaz ve TC kimlik numaram 12345678901",
    "expected_pii": [
        {"type": "isim", "text": "Ahmet Yılmaz"},
        {"type": "tc_kimlik", "text": "12345678901"}
    ],
    "category": "positive",
    "difficulty": "easy"
}
```

### Evaluation Metrics

**Scientific Metrics** (for academic paper):
- **Precision**: TP / (TP + FP) - How many detected PII are actually PII?
- **Recall**: TP / (TP + FN) - How many actual PII were detected?
- **F1 Score**: Harmonic mean of Precision and Recall
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Per-Type Metrics**: Separate metrics for each PII type
- **Latency**: Average detection time in milliseconds
- **Throughput**: Requests processed per second

**Success Criteria**:
- Precision ≥ 0.85
- Recall ≥ 0.80
- F1 Score ≥ 0.82
- Latency < 100ms (average)
- False Positive Rate < 0.15

### Test Automation

**Evaluation Script**: `backend/tests/evaluate_pii_detection.py`
- Loads test dataset
- Runs detection on all test cases
- Calculates all metrics
- Generates visualizations (bar charts, confusion matrices)
- Exports JSON report

**Continuous Testing**:
- Run unit tests on every commit
- Run property tests nightly (100 iterations each)
- Run full evaluation weekly
- Track metrics over time for regression detection

## Performance Considerations

### Optimization Strategies

1. **Caching**:
   - Turkish name database loaded once at startup
   - Regex patterns pre-compiled
   - Risk weight dictionary cached

2. **Efficient Algorithms**:
   - Back-to-front masking to avoid position recalculation
   - Early exit on high-risk detection
   - Parallel processing for independent checks

3. **Resource Management**:
   - Connection pooling for database
   - Async I/O for logging
   - Memory-efficient text processing

### Performance Targets

- **Latency**: <100ms for 95th percentile
- **Throughput**: ≥10 requests/second
- **Memory**: <500MB for detector instance
- **CPU**: <50% utilization under normal load

## Security Considerations

### Data Protection

1. **PII Storage**: Only store first 100 characters of original text in logs
2. **Database Encryption**: Encrypt logs at rest
3. **Access Control**: Restrict log access to administrators only
4. **Data Retention**: Automatically delete logs after 6 months

### Attack Vectors

1. **Evasion Attempts**: Students trying to bypass detection
   - Mitigation: Multiple detection methods, fuzzy matching
2. **Performance Attacks**: Sending very long texts to cause timeouts
   - Mitigation: Input length limits, timeout protection
3. **False Positive Exploitation**: Triggering false positives to disrupt service
   - Mitigation: Confidence thresholds, human review for edge cases

## Deployment Considerations

### Environment Variables

```bash
# Turkish name database path
TURKISH_NAMES_DB_PATH=/app/data/turkish_names.txt

# Risk thresholds
PII_RISK_BLOCK_THRESHOLD=0.8
PII_RISK_MASK_THRESHOLD=0.3

# Performance settings
PII_DETECTION_TIMEOUT_MS=1000
MAX_CONCURRENT_DETECTIONS=100

# Logging
PII_LOG_RETENTION_DAYS=180
ENABLE_DETAILED_LOGGING=true
```

### Monitoring

**Key Metrics to Monitor**:
- Detection latency (p50, p95, p99)
- Detection rate (PII found per 100 requests)
- Block rate (requests blocked per 100 requests)
- False positive rate (estimated from user feedback)
- System resource usage (CPU, memory, database connections)

### Rollout Strategy

1. **Phase 1**: Deploy to test environment with synthetic data
2. **Phase 2**: Deploy to staging with real data, shadow mode (log only, don't block)
3. **Phase 3**: Deploy to production with low risk threshold (0.9)
4. **Phase 4**: Gradually lower threshold to target (0.8) while monitoring false positives
5. **Phase 5**: Enable full content safety filtering

## Future Enhancements

### Potential Improvements (Out of Scope for V1)

1. **Machine Learning**: Train ML model on Turkish PII for better name detection
2. **Context-Aware Detection**: Use NLP to understand context (e.g., "Ahmet Bey" vs "Ahmet" as a name)
3. **Multi-Language Support**: Extend to other languages beyond Turkish
4. **Real-Time Dashboard**: Live monitoring of detections and blocks
5. **User Feedback Loop**: Allow students to report false positives
6. **Adaptive Thresholds**: Automatically adjust risk thresholds based on false positive rate
7. **Image PII Detection**: Detect PII in uploaded images using OCR
8. **Historical Data Anonymization**: Batch process and anonymize existing data

## Conclusion

This design provides a comprehensive, scientifically rigorous approach to protecting Turkish students' privacy while maintaining system performance and usability. The multi-layered architecture ensures defense in depth, while the property-based testing approach provides strong correctness guarantees suitable for academic publication.

The system balances security (fail-safe blocking), performance (<100ms latency), and compliance (KVKK/GDPR logging) to create a production-ready privacy protection solution for educational AI applications.
