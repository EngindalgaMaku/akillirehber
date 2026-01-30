"""Test Generation API endpoints - Custom LLM-based generation"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import asyncio
import json

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