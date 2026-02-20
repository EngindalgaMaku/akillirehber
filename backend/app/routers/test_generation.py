"""Test Generation API endpoints - Custom LLM-based generation"""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import asyncio
import json
import random

from app.database import get_db
from app.models.db_models import User, TestSet, TestQuestion, Course
from app.services.auth_service import get_current_teacher
from app.services.course_service import verify_course_access
from app.services.custom_test_generator import CustomTestGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/test-generation", tags=["test-generation"])


@router.get("/status")
async def get_generation_status(
    current_user: User = Depends(get_current_teacher),
):
    """Check test generation service status"""
    return {
        "available": True,
        "message": "Custom LLM-based test generation is ready",
        "custom_generation_available": True,
    }


@router.post("/generate-from-course")
async def generate_from_course(
    test_set_id: int = Form(...),
    total_questions: int = Form(50),
    remembering_ratio: float = Form(0.30),
    understanding_applying_ratio: float = Form(0.40),
    analyzing_evaluating_ratio: float = Form(0.30),
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Ders içeriğinden (Weaviate chunks) LLM kullanarak test soruları üret
    
    AVANTAJLAR:
    - Dersin kendi LLM ayarlarını kullanır
    - Weaviate'teki mevcut chunks'lardan üretir
    - Gerçek Bloom taxonomy kontrolü
    - Ekstra dependency yok (chromadb gibi)
    
    Args:
        test_set_id: Soruların ekleneceği test set ID
        total_questions: Üretilecek toplam soru sayısı
        remembering_ratio: Hatırlama seviyesi oranı
        understanding_applying_ratio: Anlama/Uygulama seviyesi oranı
        analyzing_evaluating_ratio: Analiz/Değerlendirme seviyesi oranı
    """
    # Verify test set exists and user has access
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    # Get course with settings
    course = db.query(Course).filter(Course.id == test_set.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Ensure settings are loaded
    if not course.settings:
        raise HTTPException(
            status_code=400,
            detail="Course settings not configured. Please configure LLM settings first."
        )
    
    # Validate ratios sum to 1.0
    total_ratio = remembering_ratio + understanding_applying_ratio + analyzing_evaluating_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Bloom ratios must sum to 1.0 (current sum: {total_ratio})"
        )
    
    # Custom bloom distribution
    bloom_distribution = {
        "remembering": remembering_ratio,
        "understanding_applying": understanding_applying_ratio,
        "analyzing_evaluating": analyzing_evaluating_ratio
    }
    
    logger.info(
        f"Generating {total_questions} questions from course {course.id} "
        f"for test set {test_set_id}"
    )
    
    try:
        # Initialize custom generator
        generator = CustomTestGenerator()
        
        # Generate questions
        result = await generator.generate_from_course(
            course=course,
            total_questions=total_questions,
            bloom_distribution=bloom_distribution
        )
        
        # Save questions to database
        saved_count = 0
        for q_data in result["questions"]:
            question = TestQuestion(
                test_set_id=test_set_id,
                question=q_data["question"],
                ground_truth=q_data["ground_truth"],
                alternative_ground_truths=q_data.get("alternative_ground_truths"),
                expected_contexts=q_data.get("expected_contexts"),
                question_metadata=q_data.get("question_metadata")
            )
            db.add(question)
            saved_count += 1
        
        db.commit()
        
        logger.info(f"Saved {saved_count} questions to test set {test_set_id}")
        
        return {
            "success": True,
            "test_set_id": test_set_id,
            "generated_count": result["statistics"]["total_generated"],
            "saved_count": saved_count,
            "statistics": result["statistics"],
            "message": f"Successfully generated {saved_count} questions from course content"
        }
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-from-course-stream")
async def generate_from_course_stream(
    test_set_id: int = Form(...),
    total_questions: int = Form(50),
    remembering_ratio: float = Form(0.30),
    understanding_applying_ratio: float = Form(0.40),
    analyzing_evaluating_ratio: float = Form(0.30),
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Ders içeriğinden test soruları üret - streaming response ile
    """
    # Verify test set exists and user has access
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    # Get course with settings
    course = db.query(Course).filter(Course.id == test_set.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Ensure settings are loaded
    if not course.settings:
        raise HTTPException(
            status_code=400,
            detail="Course settings not configured. Please configure LLM settings first."
        )
    
    # Validate ratios
    total_ratio = remembering_ratio + understanding_applying_ratio + analyzing_evaluating_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Bloom ratios must sum to 1.0 (current sum: {total_ratio})"
        )
    
    bloom_distribution = {
        "remembering": remembering_ratio,
        "understanding_applying": understanding_applying_ratio,
        "analyzing_evaluating": analyzing_evaluating_ratio
    }
    
    async def generate():
        """SSE generator for streaming progress"""
        try:
            # Send start event
            yield f"data: {json.dumps({'event': 'start', 'message': 'Loading course chunks...', 'progress': 0})}\n\n"
            await asyncio.sleep(0)
            
            # Initialize generator
            generator = CustomTestGenerator()
            
            # Generate questions
            yield f"data: {json.dumps({'event': 'progress', 'message': 'Generating questions with LLM...', 'progress': 20})}\n\n"
            await asyncio.sleep(0)
            
            result = await generator.generate_from_course(
                course=course,
                total_questions=total_questions,
                bloom_distribution=bloom_distribution
            )
            
            yield f"data: {json.dumps({'event': 'progress', 'message': 'Saving questions...', 'progress': 80})}\n\n"
            await asyncio.sleep(0)
            
            # Save to database
            saved_count = 0
            for q_data in result["questions"]:
                question = TestQuestion(
                    test_set_id=test_set_id,
                    question=q_data["question"],
                    ground_truth=q_data["ground_truth"],
                    alternative_ground_truths=q_data.get("alternative_ground_truths"),
                    expected_contexts=q_data.get("expected_contexts"),
                    question_metadata=q_data.get("question_metadata")
                )
                db.add(question)
                saved_count += 1
            
            db.commit()
            
            # Send completion event
            completion_data = {
                "event": "complete",
                "test_set_id": test_set_id,
                "generated_count": result["statistics"]["total_generated"],
                "saved_count": saved_count,
                "statistics": result["statistics"],
                "progress": 100
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            await asyncio.sleep(0)
            
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            error_data = {
                "event": "error",
                "error": str(e),
                "progress": -1
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            await asyncio.sleep(0)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/generate-with-quality-filter")
async def generate_with_quality_filter(
    test_set_id: int = Form(...),
    target_questions: int = Form(10),
    min_rouge1_score: float = Form(0.60),
    remembering_ratio: float = Form(0.30),
    understanding_applying_ratio: float = Form(0.40),
    analyzing_evaluating_ratio: float = Form(0.30),
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Kalite filtreli soru üretimi - Her soru test edilir, kalitesiz olanlar elenir
    
    Akış:
    1. Soru üret
    2. Semantic similarity ile test et
    3. ROUGE-1 >= min_rouge1_score ise kabul et
    4. Değilse reddet ve yeni soru üret
    5. Real-time progress gönder
    6. Target sayıya ulaşınca dur
    """
    # Verify test set exists and user has access
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    # Get course with settings
    course = db.query(Course).filter(Course.id == test_set.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if not course.settings:
        raise HTTPException(
            status_code=400,
            detail="Course settings not configured"
        )
    
    # Validate ratios
    total_ratio = remembering_ratio + understanding_applying_ratio + analyzing_evaluating_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Bloom ratios must sum to 1.0 (current sum: {total_ratio})"
        )
    
    bloom_distribution = {
        "remembering": remembering_ratio,
        "understanding_applying": understanding_applying_ratio,
        "analyzing_evaluating": analyzing_evaluating_ratio
    }
    
    async def generate_with_filter():
        """SSE generator with quality filtering"""
        from app.services.semantic_similarity_service import SemanticSimilarityService
        
        try:
            # Send start event
            yield f"data: {json.dumps({'event': 'start', 'message': 'Başlatılıyor...', 'accepted': 0, 'rejected': 0, 'target': target_questions})}\n\n"
            await asyncio.sleep(0)
            
            # Initialize services
            generator = CustomTestGenerator()
            similarity_service = SemanticSimilarityService(db)
            
            # Get chunks
            yield f"data: {json.dumps({'event': 'progress', 'message': 'Ders içeriği yükleniyor...'})}\n\n"
            await asyncio.sleep(0)
            
            chunks = await generator._get_course_chunks(course.id)
            if not chunks:
                raise ValueError("No chunks found for course")
            
            accepted_questions = []
            rejected_count = 0
            attempts = 0
            max_attempts = target_questions * 5  # Maximum 5x attempts
            
            # Calculate target per bloom level
            bloom_targets = {
                level: int(target_questions * ratio)
                for level, ratio in bloom_distribution.items()
            }
            # Adjust for rounding
            remaining = target_questions - sum(bloom_targets.values())
            if remaining > 0:
                bloom_targets["remembering"] += remaining
            
            bloom_accepted = {level: 0 for level in bloom_distribution.keys()}
            
            while len(accepted_questions) < target_questions and attempts < max_attempts:
                attempts += 1
                
                # Select bloom level based on what's needed
                needed_levels = [
                    level for level, target in bloom_targets.items()
                    if bloom_accepted[level] < target
                ]
                if not needed_levels:
                    break
                
                import random
                bloom_level = random.choice(needed_levels)
                
                try:
                    # Filter chunks for bloom level
                    if bloom_level == "remembering":
                        available_chunks = chunks
                    elif bloom_level == "understanding_applying":
                        available_chunks = generator._filter_chunks_for_understanding_applying(chunks)
                        if not available_chunks:
                            available_chunks = chunks
                    elif bloom_level == "analyzing_evaluating":
                        available_chunks = generator._filter_chunks_for_analyzing_evaluating(chunks)
                        if not available_chunks:
                            available_chunks = chunks
                    else:
                        available_chunks = chunks
                    
                    # Select random chunk
                    chunk = random.choice(available_chunks)
                    context = chunk.get('content', '')
                    if len(context) > 2000:
                        context = context[:2000] + "..."
                    
                    # Generate question
                    from app.services.llm_service import LLMService
                    
                    prompt = generator.BLOOM_PROMPTS[bloom_level].format(context=context)
                    
                    # Get system prompt
                    settings = course.settings
                    if bloom_level == "remembering":
                        system_prompt = getattr(settings, "system_prompt_remembering", None)
                    elif bloom_level == "understanding_applying":
                        system_prompt = getattr(settings, "system_prompt_understanding_applying", None)
                    elif bloom_level == "analyzing_evaluating":
                        system_prompt = getattr(settings, "system_prompt_analyzing_evaluating", None)
                    else:
                        system_prompt = None
                    
                    if not system_prompt:
                        system_prompt = getattr(settings, "system_prompt", None)
                    if not system_prompt:
                        system_prompt = generator.DEFAULT_TEST_SYSTEM_PROMPT
                    
                    llm_service = LLMService(
                        provider=course.settings.llm_provider,
                        model=course.settings.llm_model,
                        temperature=course.settings.llm_temperature or 0.7,
                        max_tokens=2500
                    )
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = llm_service.generate_response(messages)
                    parsed = generator._parse_llm_response(response)
                    
                    if not parsed:
                        rejected_count += 1
                        yield f"data: {json.dumps({'event': 'rejected', 'reason': 'Parse failed', 'accepted': len(accepted_questions), 'rejected': rejected_count, 'target': target_questions})}\n\n"
                        await asyncio.sleep(0)
                        continue
                    
                    # RAG sistemine soruyu sor ve cevabını al
                    try:
                        rag_answer, rag_contexts, rag_model = similarity_service.generate_answer(
                            course_id=course.id,
                            question=parsed["question"],
                            llm_provider=course.settings.llm_provider,
                            llm_model=course.settings.llm_model,
                            embedding_model=course.settings.default_embedding_model
                        )
                    except Exception as rag_error:
                        # RAG sistemi cevap üretemedi - kötü soru, red et
                        rejected_count += 1
                        yield f"data: {json.dumps({'event': 'rejected', 'question': parsed['question'][:100], 'bloom_level': bloom_level, 'rouge1': 0, 'reason': f'RAG failed: {str(rag_error)[:50]}', 'accepted': len(accepted_questions), 'rejected': rejected_count, 'target': target_questions})}\n\n"
                        await asyncio.sleep(0)
                        continue
                    
                    # RAG cevabını ground truth ile karşılaştır
                    metrics = similarity_service.compute_all_metrics(
                        rag_answer,              # RAG sisteminin ürettiği cevap
                        [parsed["answer"]],      # Ground truth (LLM'in beklenen cevabı)
                        course.settings.default_embedding_model,
                        retrieved_contexts=rag_contexts,
                        lang="tr"
                    )
                    
                    rouge1_score = metrics.get('rouge1', 0)
                    
                    # Check quality
                    if rouge1_score >= min_rouge1_score:
                        # Accept question
                        topic = generator._extract_topic_from_chunk(context, llm_service)
                        
                        document_info = {}
                        if chunk.get('document_id'):
                            document_info['document_id'] = chunk.get('document_id')
                        if chunk.get('document_name'):
                            document_info['document_name'] = chunk.get('document_name')
                        
                        question_data = {
                            "question": parsed["question"],
                            "ground_truth": parsed["answer"],
                            "alternative_ground_truths": [],
                            "expected_contexts": [context],
                            "question_metadata": {
                                "bloom_level": bloom_level,
                                "topic": topic,
                                "chunk_id": chunk.get('chunk_id'),
                                **document_info,
                                "generated_at": datetime.now(timezone.utc).isoformat(),
                                "rouge1_score": rouge1_score,
                            }
                        }
                        
                        accepted_questions.append(question_data)
                        bloom_accepted[bloom_level] += 1
                        
                        yield f"data: {json.dumps({'event': 'accepted', 'question': parsed['question'][:100], 'bloom_level': bloom_level, 'rouge1': round(rouge1_score * 100, 1), 'accepted': len(accepted_questions), 'rejected': rejected_count, 'target': target_questions})}\n\n"
                        await asyncio.sleep(0)
                    else:
                        # Reject question
                        rejected_count += 1
                        yield f"data: {json.dumps({'event': 'rejected', 'question': parsed['question'][:100], 'bloom_level': bloom_level, 'rouge1': round(rouge1_score * 100, 1), 'reason': f'ROUGE-1 too low ({rouge1_score:.2f})', 'accepted': len(accepted_questions), 'rejected': rejected_count, 'target': target_questions})}\n\n"
                        await asyncio.sleep(0)
                
                except Exception as e:
                    logger.error(f"Error generating question: {e}")
                    rejected_count += 1
                    yield f"data: {json.dumps({'event': 'rejected', 'reason': str(e), 'accepted': len(accepted_questions), 'rejected': rejected_count, 'target': target_questions})}\n\n"
                    await asyncio.sleep(0)
                    continue
            
            # Save accepted questions to database
            yield f"data: {json.dumps({'event': 'progress', 'message': 'Sorular kaydediliyor...'})}\n\n"
            await asyncio.sleep(0)
            
            for q_data in accepted_questions:
                question = TestQuestion(
                    test_set_id=test_set_id,
                    question=q_data["question"],
                    ground_truth=q_data["ground_truth"],
                    alternative_ground_truths=q_data.get("alternative_ground_truths"),
                    expected_contexts=q_data.get("expected_contexts"),
                    question_metadata=q_data.get("question_metadata")
                )
                db.add(question)
            
            db.commit()
            
            # Send completion
            completion_data = {
                "event": "complete",
                "test_set_id": test_set_id,
                "accepted": len(accepted_questions),
                "rejected": rejected_count,
                "total_attempts": attempts,
                "message": f"{len(accepted_questions)} kaliteli soru oluşturuldu!"
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            await asyncio.sleep(0)
            
        except Exception as e:
            logger.error(f"Error in quality-filtered generation: {e}")
            error_data = {
                "event": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            await asyncio.sleep(0)
    
    return StreamingResponse(
        generate_with_filter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/generate-from-pdf")
async def generate_test_from_pdf(
    current_user: User = Depends(get_current_teacher),
):
    """
    PDF dosyasından soru üretimi - Şu anda desteklenmiyor
    
    Not: PDF üretimi için DeepEval kaldırıldı. 
    Lütfen ders içeriğinden üretim kullanın.
    """
    raise HTTPException(
        status_code=501,
        detail="PDF generation is not supported. Please use course content generation instead."
    )


@router.post("/generate-from-pdf-stream")
async def generate_test_from_pdf_stream(
    current_user: User = Depends(get_current_teacher),
):
    """
    PDF dosyasından soru üretimi - Şu anda desteklenmiyor
    
    Not: PDF üretimi için DeepEval kaldırıldı. 
    Lütfen ders içeriğinden üretim kullanın.
    """
    raise HTTPException(
        status_code=501,
        detail="PDF generation is not supported. Please use course content generation instead."
    )


@router.get("/bloom-levels")
async def get_bloom_levels(
    current_user: User = Depends(get_current_teacher),
):
    """Get Bloom taxonomy level information"""
    return {
        "levels": [
            {
                "id": "remembering",
                "name": "Hatırlama",
                "description": "Temel tanım ve bilgi soruları (simple evolution)",
                "examples": ["X nedir?", "Y'nin özellikleri nelerdir?"],
                "default_ratio": 0.30
            },
            {
                "id": "understanding_applying",
                "name": "Anlama/Uygulama",
                "description": "Senaryo ve problem çözme soruları (reasoning/conditional evolution)",
                "examples": ["Bu durumda hangi çözüm uygulanmalıdır?", "Nasıl çalışır?"],
                "default_ratio": 0.40
            },
            {
                "id": "analyzing_evaluating",
                "name": "Analiz/Değerlendirme",
                "description": "Karşılaştırma ve değerlendirme soruları (multi_context/comparative evolution)",
                "examples": ["X ve Y'yi karşılaştırın", "Avantaj ve dezavantajlarını analiz edin"],
                "default_ratio": 0.30
            }
        ],
        "total_ratio": 1.0
    }