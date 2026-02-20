"""Course management service."""

from typing import List, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.db_models import (
    Course,
    Document,
    User,
    UserRole,
    CourseSettings,
    Chunk,
    DiagnosticReport,
    ChunkQualityMetrics,
    ProcessingStatus,
)
from app.models.schemas import CourseCreate, CourseUpdate
from app.services.weaviate_service import get_weaviate_service

# Default system prompt for educational assistant (optimized for RAGAS/ROUGE)
DEFAULT_SYSTEM_PROMPT = (
    "Sen sağlanan ders dokümanlarındaki bilgileri yapılandırılmış şekilde sunan "
    "bir bilgi çıkarma sistemisin.\n\n"
    "CEVAPLAMA KURALLARI:\n"
    "1. BAĞLAM SADAKATI: Yalnızca sağlanan bağlam bilgilerini kullan. "
    "Bağlamda olmayan bilgiyi kesinlikle ekleme. Bağlamda yeterli bilgi yoksa "
    "'Bu konu hakkında ders dokümanlarında yeterli bilgi bulunmuyor.' de.\n"
    "2. DOĞRUDAN CEVAP: 'Dökümana göre...', 'Anladığım kadarıyla...' gibi "
    "giriş cümleleri kullanma. Doğrudan cevaba başla.\n"
    "3. TERMİNOLOJİ SADAKATI: Cevabında bağlamdaki anahtar terimleri, teknik "
    "ifadeleri ve tanımları aynen kullan. Eş anlamlılarla değiştirme.\n"
    "4. YAPILANDIRMA: Tanımları yaparken özneyle başla ve tam cümle kur. "
    "Sıralama veya liste varsa madde işaretleri kullan.\n"
    "5. KAPSAM: Soruyla doğrudan ilgili tüm bilgiyi bağlamdan çıkar. "
    "Eksik bırakma ama bağlam dışına da çıkma.\n"
    "6. META VERİ YASAĞI: Cevap içinde döküman ID, kaynak adı veya teknik "
    "meta veri gösterme.\n"
    "7. UZUNLUK: Kısa tanım soruları için 2-3 cümle, açıklama soruları için "
    "gerektiği kadar detay ver. Gereksiz tekrar yapma."
)


DEFAULT_TEST_SYSTEM_PROMPT_REMEMBERING = (
    "Sen bir eğitim uzmanısın ve Bloom Taksonomisi 'Hatırlama' seviyesinde "
    "soru üretiyorsun. "
    "Görev: İçerikten DOĞRUDAN bilgi çekmeyi gerektiren, "
    "tanım/liste/terim odaklı bir soru üret. "
    "KURALLAR: "
    "(1) Soru self-contained olmalı (metne göre gibi ifadeler "
    "kullanma). "
    "(2) Cevap mutlaka verilen içerikte geçmeli veya içerikten "
    "parafraz olmalı. "
    "(3) Cevap detaylı olmalı (en az 2-3 cümle). "
    "(4) Çıktı formatı: 'SORU:' ve 'CEVAP:' satırları."
)


DEFAULT_TEST_SYSTEM_PROMPT_UNDERSTANDING_APPLYING = (
    "Sen bir eğitim uzmanısın ve Bloom Taksonomisi 'Anlama/Uygulama' "
    "seviyesinde soru üretiyorsun. "
    "Görev: İçerikteki bilgiyi YORUMLAMA veya UYGULAMA gerektiren "
    "senaryolu bir soru üret. "
    "KURALLAR: "
    "(1) Soru self-contained olmalı. "
    "(2) Cevap içerikteki kavramları kullanarak akıl yürütmeli ve uygulanabilir "
    "açıklama vermeli (3-4 cümle). "
    "(3) İçeriğe dayalı kal; içerikte olmayan iddialar ekleme. "
    "(4) Çıktı formatı: 'SORU:' ve 'CEVAP:' satırları."
)


DEFAULT_TEST_SYSTEM_PROMPT_ANALYZING_EVALUATING = (
    "Sen bir eğitim uzmanısın ve Bloom Taksonomisi 'Analiz/Değerlendirme' seviyesinde "
    "soru üretiyorsun. "
    "Görev: Karşılaştırma, analiz veya değerlendirme gerektiren bir soru üret. "
    "KURALLAR: "
    "(1) Soru self-contained olmalı. "
    "(2) Cevap içerikteki bilgilerle sentez yapmalı, çok boyutlu olmalı "
    "(4-5 cümle). "
    "(3) İçeriğe dayalı kal; içerikte olmayan iddialar ekleme. "
    "(4) Çıktı formatı: 'SORU:' ve 'CEVAP:' satırları."
)


def get_course_by_id(db: Session, course_id: int) -> Optional[Course]:
    """Get a course by ID."""
    return db.query(Course).filter(Course.id == course_id).first()


def get_courses_by_teacher(db: Session, teacher_id: int) -> List[Course]:
    """Get all courses created by a teacher (including inactive)."""
    return (
        db.query(Course)
        .filter(Course.teacher_id == teacher_id)
        .order_by(Course.created_at.desc())
        .all()
    )


def get_all_active_courses(db: Session) -> List[Course]:
    """Get all active courses (for students)."""
    return (
        db.query(Course)
        .filter(Course.is_active.is_(True))
        .order_by(Course.created_at.desc())
        .all()
    )


def get_course_with_document_count(db: Session, course: Course) -> dict:
    """Get course data with document count."""
    doc_count = (
        db.query(func.count(Document.id))
        .filter(Document.course_id == course.id)
        .scalar()
    )
    return {
        "id": course.id,
        "name": course.name,
        "description": course.description,
        "teacher_id": course.teacher_id,
        "is_active": course.is_active,
        "created_at": course.created_at,
        "updated_at": course.updated_at,
        "document_count": doc_count or 0,
    }


def create_course(
    db: Session, course_data: CourseCreate, teacher: User
) -> Course:
    """Create a new course."""
    if teacher.role not in (UserRole.TEACHER, UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers and admins can create courses",
        )

    db_course = Course(
        name=course_data.name,
        description=course_data.description,
        teacher_id=teacher.id,
    )
    db.add(db_course)
    db.commit()
    db.refresh(db_course)
    return db_course


def update_course(
    db: Session, course_id: int, course_data: CourseUpdate, teacher: User
) -> Course:
    """Update an existing course."""
    course = get_course_by_id(db, course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found",
        )

    if course.teacher_id != teacher.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own courses",
        )

    # Update fields if provided
    if course_data.name is not None:
        course.name = course_data.name
    if course_data.description is not None:
        course.description = course_data.description
    if course_data.is_active is not None:
        course.is_active = course_data.is_active

    db.commit()
    db.refresh(course)
    return course


def delete_course(db: Session, course_id: int, teacher: User) -> bool:
    """Delete a course and all related data permanently.

    This performs a hard delete including:
    - Weaviate vectors (entire course collection)
    - Diagnostic reports
    - Chunk quality metrics
    - Processing statuses
    - Chunks
    - Documents
    - Course settings
    - Course itself
    """
    course = get_course_by_id(db, course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found",
        )

    if course.teacher_id != teacher.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own courses",
        )

    # 1. Delete Weaviate vectors for the entire course
    try:
        weaviate_service = get_weaviate_service()
        weaviate_service.delete_by_course(course_id)
    except Exception:
        # Continue even if Weaviate deletion fails
        pass

    # 2. Get all document IDs for this course
    document_ids = [
        doc.id
        for doc in db.query(Document.id)
        .filter(Document.course_id == course_id)
        .all()
    ]

    if document_ids:
        # 3. Delete diagnostic reports for all documents
        db.query(DiagnosticReport).filter(
            DiagnosticReport.document_id.in_(document_ids)
        ).delete(synchronize_session=False)

        # 4. Delete chunk quality metrics for all documents
        db.query(ChunkQualityMetrics).filter(
            ChunkQualityMetrics.document_id.in_(document_ids)
        ).delete(synchronize_session=False)

        # 5. Delete processing statuses for all documents
        db.query(ProcessingStatus).filter(
            ProcessingStatus.document_id.in_(document_ids)
        ).delete(synchronize_session=False)

        # 6. Delete all chunks for all documents
        db.query(Chunk).filter(
            Chunk.document_id.in_(document_ids)
        ).delete(synchronize_session=False)

        # 7. Delete all documents
        db.query(Document).filter(
            Document.course_id == course_id
        ).delete(synchronize_session=False)

    # 8. Delete course settings
    db.query(CourseSettings).filter(
        CourseSettings.course_id == course_id
    ).delete(synchronize_session=False)

    # 9. Delete the course itself
    db.delete(course)
    db.commit()

    return True


def verify_course_access(db: Session, course_id: int, user: User) -> Course:
    """Verify user has access to a course and return it."""
    course = get_course_by_id(db, course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found",
        )

    # Admin can access any course
    if user.role == UserRole.ADMIN:
        return course

    # Teachers can access their own courses (active or inactive)
    if user.role == UserRole.TEACHER and course.teacher_id == user.id:
        return course

    # Students can only access active courses
    if not course.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found",
        )

    return course


def verify_course_ownership(
    db: Session, course_id: int, teacher: User
) -> Course:
    """Verify teacher owns the course."""
    course = get_course_by_id(db, course_id)
    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found",
        )

    # Admin can manage any course
    if teacher.role == UserRole.ADMIN:
        return course

    if course.teacher_id != teacher.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own courses",
        )

    return course


def get_or_create_settings(db: Session, course_id: int) -> CourseSettings:
    """Get or create course settings."""
    settings = db.query(CourseSettings).filter(
        CourseSettings.course_id == course_id
    ).first()

    if not settings:
        settings = CourseSettings(
            course_id=course_id,
            default_chunk_strategy="recursive",
            default_chunk_size=500,
            default_overlap=50,
            default_embedding_model="openai/text-embedding-3-small",
            search_alpha=0.5,
            search_top_k=5,
            llm_provider="openrouter",
            llm_model="openai/gpt-4o-mini",
            llm_temperature=0.7,
            llm_max_tokens=1000,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            system_prompt_remembering=DEFAULT_TEST_SYSTEM_PROMPT_REMEMBERING,
            system_prompt_understanding_applying=(
                DEFAULT_TEST_SYSTEM_PROMPT_UNDERSTANDING_APPLYING
            ),
            system_prompt_analyzing_evaluating=DEFAULT_TEST_SYSTEM_PROMPT_ANALYZING_EVALUATING,
        )
        db.add(settings)
        db.commit()
        db.refresh(settings)

    return settings
