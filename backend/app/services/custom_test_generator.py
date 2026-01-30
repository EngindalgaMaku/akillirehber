"""
Custom Test Generation Service

Uses course chunks from Weaviate + configured LLM to generate
Bloom taxonomy aligned test questions.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
import random

from app.models.db_models import Course
from app.services.weaviate_service import WeaviateService

logger = logging.getLogger(__name__)


class CustomTestGenerator:
    """Generate test questions using course chunks and LLM"""
    
    # RAGAS uyumlu Bloom prompts
    BLOOM_PROMPTS = {
        "remembering": """
Aşağıdaki ders içeriğinden Bloom Taksonomisi HATIRLAMAYA (Remembering - Alt Basamak) seviyesinde bir soru oluştur.

İÇERİK:
{context}

ÖNEMLİ KURALLAR:
1. Soru DOĞRUDAN içerikten bilgi çekmeyi gerektirmeli (tanım, liste, terim)
2. Cevap MUTLaka yukarıdaki içerikte GEÇMELİ (alıntı veya parafraz)
3. Cevap DETAYLI olmalı (en az 2-3 cümle, tek kelime DEĞİL)
4. "Nedir?", "Nelerdir?", "Tanımlayınız" gibi sorular kullan
5. Soru self-contained olmalı ("metne göre" KULLANMA)

RAGAS EVOLUTION: simple (doğrudan bilgi çekme)

ÖRNEK:
SORU: RAM çeşitleri nelerdir ve her birinin temel özellikleri nedir?
CEVAP: RAM (Random Access Memory) iki ana çeşide ayrılır: SRAM ve DRAM. SRAM (Static RAM), transistörler kullanarak veri saklar ve çok hızlıdır ancak pahalıdır, genellikle önbellek olarak kullanılır. DRAM (Dynamic RAM), kapasitörler kullanarak veri saklar, daha yavaş ama ucuzdur ve ana bellek olarak kullanılır. DRAM düzenli olarak yenilenmesi (refresh) gerekir.

ŞİMDİ SEN OLUŞTUR (cevap MUTLAKA yukarıdaki içerikte geçmeli):

SORU: [Tanım/liste sorusu]
CEVAP: [İçerikten alınan DETAYLI cevap - 2-3 cümle]
""",
        "understanding_applying": """
Aşağıdaki ders içeriğinden Bloom Taksonomisi ANLAMA/UYGULAMA (Understanding & Applying - Orta Basamak) seviyesinde bir soru oluştur.

İÇERİK:
{context}

ÖNEMLİ KURALLAR:
1. Soru bir SENARYO veya DURUM içermeli
2. Öğrenci bilgiyi YORUMLAMALI veya UYGULAMALI
3. "Nasıl", "Neden", "Hangi durumda" gibi sorular kullan
4. Cevap içerikteki bilgileri KULLANARAK çözüm sunmalı
5. Cevap DETAYLI ve AKIL YÜRÜTME içermeli (3-4 cümle)

RAGAS EVOLUTION: reasoning veya conditional (yorumlama ve uygulama)

ÖRNEK:
SORU: Bir bilgisayar sürekli reset atıyorsa, dökümandaki bilgiler ışığında hangi donanım arızasından şüphelenilmelidir ve neden?
CEVAP: Sürekli reset atma sorunu genellikle güç kaynağı (PSU) yetersizliği veya arızasından kaynaklanır. Güç kaynağı, sistemin tüm bileşenlerine kararlı voltaj sağlamalıdır. Eğer PSU yeterli güç üretemiyorsa veya voltaj dalgalanmaları varsa, sistem kendini korumak için reset atar. Ayrıca aşırı ısınma da bu soruna yol açabilir çünkü işlemci veya GPU kritik sıcaklığa ulaştığında sistem otomatik olarak kapanır veya reset atar.

ŞİMDİ SEN OLUŞTUR (cevap içeriğe DAYALI olmalı):

SORU: [Senaryo/durum sorusu]
CEVAP: [İçeriğe dayalı, akıl yürütme içeren DETAYLI cevap - 3-4 cümle]
""",
        "analyzing_evaluating": """
Aşağıdaki ders içeriğinden Bloom Taksonomisi ANALİZ/DEĞERLENDİRME (Analyzing & Evaluating - Üst Basamak) seviyesinde bir soru oluştur.

İÇERİK:
{context}

ÖNEMLİ KURALLAR:
1. Soru KARŞILAŞTIRMA, ANALİZ veya DEĞERLENDİRME gerektirmeli
2. Öğrenci bilgiler arasında BAĞ KURMALI
3. "Karşılaştırınız", "Analiz ediniz", "Değerlendiriniz" gibi sorular kullan
4. Cevap içerikteki bilgilerle SENTEZ yapmalı
5. Cevap KAPSAMLI ve ÇOK BOYUTLU olmalı (4-5 cümle)

RAGAS EVOLUTION: multi_context veya comparative (karşılaştırma ve analiz)

ÖRNEK:
SORU: SATA ve NVMe SSD'ler arasındaki hız farkının oyun performansına etkisini dökümandaki verilerle analiz ediniz.
CEVAP: SATA SSD'ler maksimum 600 MB/s hıza ulaşırken, NVMe SSD'ler PCIe bağlantısı sayesinde 3500-7000 MB/s hıza ulaşabilir. Ancak oyun performansında bu fark beklenenin altındadır çünkü oyunlar genellikle küçük dosyalar yerine büyük texture ve model dosyaları yükler. Oyun yükleme süreleri NVMe'de %20-40 daha hızlı olsa da, oyun içi FPS performansı neredeyse aynıdır. Bunun nedeni, oyun performansının daha çok GPU ve CPU hızına bağlı olması, depolama hızının sadece yükleme anlarında etkili olmasıdır. Sonuç olarak, NVMe oyun deneyimini iyileştirir ama oyun performansını dramatik şekilde değiştirmez.

ŞİMDİ SEN OLUŞTUR (cevap içeriğe dayalı sentez olmalı):

SORU: [Karşılaştırma/analiz sorusu]
CEVAP: [İçeriğe dayalı, kapsamlı ANALİZ - 4-5 cümle]
"""
    }
    
    # Bloom dağılımı
    BLOOM_DISTRIBUTION = {
        "remembering": 0.30,
        "understanding_applying": 0.40,
        "analyzing_evaluating": 0.30
    }
    
    DEFAULT_TEST_SYSTEM_PROMPT = (
        "Sen bir eğitim uzmanısın. "
        "RAGAS evaluation için uygun test soruları oluşturuyorsun. "
        "CEVAP MUTLAKA verilen içerikte geçmeli veya içerikten türetilmeli. "
        "Cevaplar detaylı olmalı (2-5 cümle)."
    )
    
    def __init__(self):
        """Initialize the generator"""
        self.weaviate_service = WeaviateService()
    
    async def generate_from_course(
        self,
        course: Course,
        total_questions: int = 50,
        bloom_distribution: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Generate test questions from course chunks
        
        Args:
            course: Course object with LLM settings
            total_questions: Total questions to generate
            bloom_distribution: Custom Bloom distribution (optional)
            
        Returns:
            Generated questions and metadata
        """
        # Bloom dağılımını ayarla
        distribution = bloom_distribution or self.BLOOM_DISTRIBUTION
        
        # Soru sayılarını hesapla
        question_counts = self._calculate_question_counts(
            total_questions, distribution
        )
        
        logger.info(
            f"Generating {total_questions} questions for "
            f"course {course.id}"
        )
        logger.info(f"Bloom distribution: {question_counts}")
        
        # Course chunks'ları al
        chunks = await self._get_course_chunks(course.id)
        
        if not chunks:
            raise ValueError(f"No chunks found for course {course.id}")
        
        logger.info(f"Found {len(chunks)} chunks for course")
        
        # Get course settings for LLM configuration
        if not course.settings:
            raise ValueError(f"Course {course.id} has no settings configured")
        
        # LLM service oluştur
        from app.services.llm_service import LLMService
        
        llm_service = LLMService(
            provider=course.settings.llm_provider,
            model=course.settings.llm_model,
            temperature=course.settings.llm_temperature or 0.7,
            max_tokens=2500
        )
        
        all_questions = []
        
        # Her Bloom seviyesi için soru üret
        for bloom_level, count in question_counts.items():
            if count == 0:
                continue
            
            logger.info(
                f"Generating {count} questions for {bloom_level}"
            )
            
            # Her soru için farklı chunk kullan
            for i in range(count):
                try:
                    # Chunk seç (döngüsel)
                    chunk = chunks[i % len(chunks)]
                    context = chunk.get('content', '')
                    
                    # Context çok uzunsa kısalt
                    if len(context) > 2000:
                        context = context[:2000] + "..."
                    
                    # Prompt oluştur
                    prompt = self.BLOOM_PROMPTS[bloom_level].format(
                        context=context
                    )
                    
                    # Select system prompt by Bloom level (DB-configurable)
                    settings = getattr(course, "settings", None)
                    system_prompt = None
                    if settings is not None:
                        if bloom_level == "remembering":
                            system_prompt = getattr(settings, "system_prompt_remembering", None)
                        elif bloom_level == "understanding_applying":
                            system_prompt = getattr(settings, "system_prompt_understanding_applying", None)
                        elif bloom_level == "analyzing_evaluating":
                            system_prompt = getattr(settings, "system_prompt_analyzing_evaluating", None)

                        if not system_prompt:
                            system_prompt = getattr(settings, "system_prompt", None)

                    if not system_prompt:
                        system_prompt = self.DEFAULT_TEST_SYSTEM_PROMPT
                    
                    # LLM'den cevap al
                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    
                    response = llm_service.generate_response(messages)
                    
                    # Parse et
                    parsed = self._parse_llm_response(response)
                    
                    if parsed:
                        question_dict = {
                            "question": parsed["question"],
                            "ground_truth": parsed["answer"],
                            "alternative_ground_truths": [],
                            "expected_contexts": [context],
                            "question_metadata": {
                                "bloom_level": bloom_level,
                                "generated_by": "custom_llm",
                                "generated_at": (
                                    datetime.now(timezone.utc).isoformat()
                                ),
                                "chunk_id": chunk.get('chunk_id'),
                                "llm_provider": course.settings.llm_provider,
                                "llm_model": course.settings.llm_model
                            }
                        }
                        all_questions.append(question_dict)
                        
                        logger.info(
                            f"Generated question "
                            f"{len(all_questions)}/{total_questions}"
                        )
                    
                except Exception as e:
                    logger.error(
                        f"Error generating question {i+1} for "
                        f"{bloom_level}: {e}"
                    )
                    continue
        
        # İstatistikler
        bloom_stats = {}
        for q in all_questions:
            level = q["question_metadata"]["bloom_level"]
            bloom_stats[level] = bloom_stats.get(level, 0) + 1
        
        result = {
            "questions": all_questions,
            "statistics": {
                "total_generated": len(all_questions),
                "requested": total_questions,
                "bloom_distribution": {
                    level: {
                        "count": count,
                        "percentage": (
                            round((count / len(all_questions)) * 100, 2)
                            if all_questions else 0
                        )
                    }
                    for level, count in bloom_stats.items()
                },
                "chunks_used": len(chunks),
                "llm_provider": course.settings.llm_provider,
                "llm_model": course.settings.llm_model
            }
        }
        
        logger.info(
            f"Successfully generated {len(all_questions)} questions"
        )
        
        return result
    
    async def _get_course_chunks(self, course_id: int) -> List[Dict]:
        """Get all chunks for a course from Weaviate"""
        try:
            # Weaviate client al
            client = self.weaviate_service._get_client()
            collection_name = f"Course_{course_id}"
            
            # Collection var mı kontrol et
            if not client.collections.exists(collection_name):
                logger.warning(
                    f"Collection {collection_name} does not exist"
                )
                return []
            
            # Collection'ı al
            collection = client.collections.get(collection_name)
            
            # Tüm chunk'ları çek (limit 1000)
            response = collection.query.fetch_objects(limit=1000)
            
            chunks = []
            for obj in response.objects:
                chunks.append({
                    'chunk_id': obj.properties.get('chunk_id'),
                    'document_id': obj.properties.get('document_id'),
                    'content': obj.properties.get('content', ''),
                    'chunk_index': obj.properties.get('chunk_index', 0)
                })
            # Shuffle chunks to ensure random selection for question generation
            random.shuffle(chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error fetching chunks: {e}")
            return []
    
    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """Parse LLM response - sadece SORU ve CEVAP"""
        try:
            lines = response.strip().split('\n')
            parsed = {}
            current_field = None
            field_content = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('SORU:'):
                    if current_field and field_content:
                        parsed[current_field] = ' '.join(field_content)
                    current_field = 'question'
                    field_content = [line.replace('SORU:', '').strip()]
                elif line.startswith('CEVAP:'):
                    if current_field and field_content:
                        parsed[current_field] = ' '.join(field_content)
                    current_field = 'answer'
                    field_content = [line.replace('CEVAP:', '').strip()]
                elif line and current_field:
                    # Multi-line content
                    field_content.append(line)
            
            # Add last field
            if current_field and field_content:
                parsed[current_field] = ' '.join(field_content)
            
            # Validate
            if 'question' in parsed and 'answer' in parsed:
                # Cevap yeterince uzun mu? (en az 10 kelime)
                if len(parsed['answer'].split()) < 10:
                    logger.warning(
                        f"Answer too short: {parsed['answer']}"
                    )
                    return None
                return parsed
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None
    
    def _calculate_question_counts(
        self,
        total_questions: int,
        distribution: Dict[str, float]
    ) -> Dict[str, int]:
        """Calculate question counts per Bloom level"""
        counts = {}
        remaining = total_questions
        
        for level, ratio in distribution.items():
            count = int(total_questions * ratio)
            counts[level] = count
            remaining -= count
        
        # Kalan soruları ilk kategoriye ekle
        if remaining > 0:
            first_level = list(distribution.keys())[0]
            counts[first_level] += remaining
        
        return counts