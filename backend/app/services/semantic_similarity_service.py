"""Semantic Similarity Service for measuring answer quality.

This service provides functionality to compute semantic similarity between
generated answers and ground truth answers using embedding vectors and
cosine similarity, as well as ROUGE and BERTScore metrics.
"""

from typing import List, Tuple, Dict, Any, Optional
from functools import lru_cache
import os
import re
import threading
import numpy as np
from sqlalchemy.orm import Session
import logging

from app.services.embedding_service import get_embedding_service
from app.services.llm_service import get_llm_service
from app.services.weaviate_service import get_weaviate_service, SearchResult
from app.services.rerank_service import get_rerank_service
from app.services.embedding_cache import EmbeddingCache
from app.models.db_models import Course

logger = logging.getLogger(__name__)

# Initialize embedding cache
_embedding_cache = EmbeddingCache(ttl=3600, max_entries=10000)

# Thread lock for BERTScore computation (model loading is not thread-safe)
_bertscore_lock = threading.Lock()

# Module-level cache for original BERTScore (shared across threads)
@lru_cache(maxsize=256)
def _cached_bertscore(
    cands: Tuple[str, ...],
    refs: Tuple[str, ...],
    model: str,
    language: str,
    use_rescale: bool,
) -> Tuple[float, float, float]:
    from bert_score import score as bert_score
    logger.info(f"Computing BERTScore with lang={language}, model={model}")
    P, R, F1 = bert_score(
        list(cands),
        list(refs),
        lang=language,
        model_type=model,
        verbose=False,
        rescale_with_baseline=use_rescale,
    )
    logger.info(
        f"BERTScore computed: P={P.mean().item():.4f}, "
        f"R={R.mean().item():.4f}, F1={F1.mean().item():.4f}"
    )
    return (
        float(P.mean().item()),
        float(R.mean().item()),
        float(F1.mean().item()),
    )


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 to 1, where 1 means identical direction.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score between 0.0 and 1.0
        Returns 0.0 if either vector has zero length
    """
    if not vec1 or not vec2:
        return 0.0

    v1 = np.array(vec1)
    v2 = np.array(vec2)

    # Compute norms
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Handle zero-length vectors
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    # Compute dot product
    dot_product = np.dot(v1, v2)

    # Compute cosine similarity
    similarity = float(dot_product / (norm_v1 * norm_v2))

    # Clamp to [0, 1] range (cosine can be negative, but we want 0-1)
    # For text embeddings, negative similarity is rare and treated as 0
    return max(0.0, min(1.0, similarity))


class SemanticSimilarityService:
    """Service for computing semantic similarity between texts.

    This service uses embedding models to convert texts to vectors and
    computes cosine similarity to measure semantic closeness. It also
    supports ROUGE and BERTScore metrics using lightweight implementations.
    """

    def __init__(self, db: Session):
        """Initialize the semantic similarity service.

        Args:
            db: Database session for accessing course settings
        """
        self.db = db
        self.embedding_service = get_embedding_service()

    def _is_no_info_answer(self, text: Optional[str]) -> bool:
        if not text:
            return True
        t = text.strip().lower()
        if not t:
            return True
        patterns = [
            r"\bbilgi\s+bulunamad[ıi]\b",
            r"\bbulunamad[ıi]\b",
            r"\bbulunamam[ıi]şt[ıi]r\b",
            r"\bders\s+materyallerinde\b.*\bbilgi\s+bulunamad[ıi]\b",
            r"\bders\s+dok[üu]manlar[ıi]nda\b.*\bbilgi\s+bulunamad[ıi]\b",
        ]
        return any(re.search(p, t) is not None for p in patterns)

    def compute_similarity(
        self,
        text1: str,
        text2: str,
        embedding_model: str
    ) -> float:
        """Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            embedding_model: Embedding model to use

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Check cache for text1
        cached_emb1 = _embedding_cache.get(text1, embedding_model)
        if cached_emb1 is None:
            embedding1 = self.embedding_service.get_embedding(
                text1, model=embedding_model
            )
            _embedding_cache.set(text1, embedding_model, embedding1)
        else:
            embedding1 = cached_emb1

        # Check cache for text2
        cached_emb2 = _embedding_cache.get(text2, embedding_model)
        if cached_emb2 is None:
            embedding2 = self.embedding_service.get_embedding(
                text2, model=embedding_model
            )
            _embedding_cache.set(text2, embedding_model, embedding2)
        else:
            embedding2 = cached_emb2

        # Compute cosine similarity
        score = cosine_similarity(embedding1, embedding2)

        return score

    def find_best_match(
        self,
        generated_answer: str,
        ground_truths: List[str],
        embedding_model: str
    ) -> Tuple[float, str, List[Dict[str, Any]]]:
        """Find the best matching ground truth for a generated answer.

        Computes similarity against all ground truths and returns the
        maximum.

        Args:
            generated_answer: The generated answer to evaluate
            ground_truths: List of ground truth answers
            embedding_model: Embedding model to use

        Returns:
            Tuple of (max_score, best_match_text, all_scores)
            where all_scores is a list of dicts with 'ground_truth'
            and 'score'
        """
        if not ground_truths:
            return 0.0, "", []

        # Batch compute embeddings for all ground truths
        # First check cache for all texts
        all_texts = [generated_answer] + ground_truths
        found, missing = _embedding_cache.get_batch(all_texts, embedding_model)

        # Compute embeddings for missing texts in batch
        if missing:
            missing_texts = [all_texts[i] for i in missing]
            new_embeddings = self.embedding_service.get_embeddings(
                missing_texts, model=embedding_model, input_type="document"
            )
            # Cache new embeddings
            _embedding_cache.set_batch(
                missing_texts, embedding_model, new_embeddings
            )
            # Update found dict
            for i, emb in zip(missing, new_embeddings):
                found[i] = emb

        # Get generated answer embedding
        gen_emb = found[0]

        # Compute similarities in parallel
        all_scores = []
        max_score = 0.0
        best_match = ground_truths[0]

        for i, ground_truth in enumerate(ground_truths):
            gt_emb = found[i + 1]  # +1 because index 0 is generated_answer
            score = cosine_similarity(gen_emb, gt_emb)

            all_scores.append({
                "ground_truth": ground_truth,
                "score": score
            })

            if score > max_score:
                max_score = score
                best_match = ground_truth

        return max_score, best_match, all_scores

    def compute_rouge_scores(
        self,
        generated_answer: str,
        ground_truth: str
    ) -> Optional[Dict[str, float]]:
        """Compute ROUGE scores using official rouge-score library.

        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
        n-gram overlap between texts. Uses the standard rouge-score library
        for accurate, academically accepted results.

        Args:
            generated_answer: The generated answer to evaluate
            ground_truth: The reference ground truth answer

        Returns:
            Dict with rouge1, rouge2, rougeL F1 scores
        """
        try:
            from rouge_score import rouge_scorer

            # Initialize scorer without stemmer for Turkish
            scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=False
            )

            # Compute scores (reference first, then generated)
            scores = scorer.score(ground_truth, generated_answer)

            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougel': scores['rougeL'].fmeasure,
            }
        except Exception as e:
            logger.error(f"Error computing ROUGE scores: {e}")
            return None

    def compute_original_bertscore(
        self,
        generated_answer: str,
        ground_truth: str,
        lang: str = "tr",
    ) -> Dict[str, float]:
        try:
            logger.info("Starting original BERTScore computation...")
            from transformers.utils import logging as hf_logging
            import bert_score as bert_score_pkg

            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()

            model_type = os.getenv(
                "ORIGINAL_BERTSCORE_MODEL",
                "bert-base-multilingual-cased",
            )
            logger.info(f"Using BERTScore model: {model_type}")

            # bert-score ships rescale baseline files per (lang, model).
            # If missing, enabling rescale emits warnings and has no effect.
            baseline_path = os.path.join(
                os.path.dirname(bert_score_pkg.__file__),
                "rescale_baseline",
                lang,
                f"{model_type}.tsv",
            )
            use_rescale = os.path.exists(baseline_path)
            logger.info(f"Baseline path: {baseline_path}, exists: {use_rescale}")

            # Use module-level lock to prevent concurrent model loading
            with _bertscore_lock:
                p, r, f1 = _cached_bertscore(
                    (generated_answer,),
                    (ground_truth,),
                    model_type,
                    lang,
                    use_rescale,
                )

            logger.info(f"Original BERTScore result: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
            return {
                "precision": p,
                "recall": r,
                "f1": f1,
            }
        except Exception as e:
            logger.error(f"Error computing original BERTScore: {e}", exc_info=True)
            raise

    def compute_bertscore(
        self,
        generated_answer: str,
        ground_truth: str,
        embedding_model: str = "openai/text-embedding-3-small"
    ) -> Optional[Dict[str, float]]:
        """Compute BERTScore using embeddings.

        Uses embedding service to compute semantic similarity as BERTScore.
        This is a lightweight approximation that doesn't require HF API.

        Args:
            generated_answer: The generated answer to evaluate
            ground_truth: The reference ground truth answer
            embedding_model: Embedding model to use (defaults to OpenAI)

        Returns:
            Dict with precision, recall, f1 scores, or None if unavailable
        """
        try:
            if self._is_no_info_answer(generated_answer):
                return {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                }
            # Use our own embedding service (OpenAI or local)
            # This is faster and doesn't require HF API
            emb1 = self.embedding_service.get_embedding(
                generated_answer,
                model=embedding_model
            )
            emb2 = self.embedding_service.get_embedding(
                ground_truth,
                model=embedding_model
            )

            if not emb1 or not emb2:
                return None

            # Compute cosine similarity
            similarity = cosine_similarity(emb1, emb2)

            # Use similarity as approximation for all three metrics
            return {
                'precision': similarity,
                'recall': similarity,
                'f1': similarity,
            }
        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}")
            return None

    def compute_all_metrics(
        self,
        generated_answer: str,
        ground_truths: List[str],
        embedding_model: str,
        retrieved_contexts: List[str] = None,
        lang: str = "tr"
    ) -> Dict[str, Any]:
        """Compute all available metrics for answer evaluation.

        Computes cosine similarity, ROUGE, and BERTScore metrics.

        Args:
            generated_answer: The generated answer to evaluate
            ground_truths: List of ground truth answers
            embedding_model: Embedding model to use for cosine similarity
            retrieved_contexts: List of retrieved chunks from RAG (not used)
            lang: Language code (not used in lightweight implementation)

        Returns:
            Dict containing all computed metrics
        """
        # Compute cosine similarity (find best match)
        max_score, best_match, all_scores = self.find_best_match(
            generated_answer,
            ground_truths,
            embedding_model
        )

        result = {
            'similarity_score': max_score,
            'best_match_ground_truth': best_match,
            'all_scores': all_scores,
            'hit_at_1': None,  # Not used
            'mrr': None,  # Not used
            'rouge1': None,
            'rouge2': None,
            'rougel': None,
            'bertscore_precision': None,
            'bertscore_recall': None,
            'bertscore_f1': None,
            'original_bertscore_precision': None,
            'original_bertscore_recall': None,
            'original_bertscore_f1': None,
        }

        if self._is_no_info_answer(generated_answer):
            logger.info(
                "SemanticSimilarity: no-info answer detected; "
                "short-circuiting metrics. gen_len=%d gt_count=%d",
                len(generated_answer or ""),
                len(ground_truths or []),
            )
            result['rouge1'] = 0.0
            result['rouge2'] = 0.0
            result['rougel'] = 0.0
            result['bertscore_precision'] = 0.0
            result['bertscore_recall'] = 0.0
            result['bertscore_f1'] = 0.0
            result['original_bertscore_precision'] = 0.0
            result['original_bertscore_recall'] = 0.0
            result['original_bertscore_f1'] = 0.0
            return result

        # Compute ROUGE scores against best match
        if best_match:
            rouge_scores = self.compute_rouge_scores(
                generated_answer,
                best_match
            )
            if rouge_scores:
                result.update(rouge_scores)

            # Compute BERTScore against best match via HF API
            bertscore = self.compute_bertscore(
                generated_answer,
                best_match,
                embedding_model
            )
            if bertscore:
                result['bertscore_precision'] = bertscore['precision']
                result['bertscore_recall'] = bertscore['recall']
                result['bertscore_f1'] = bertscore['f1']
            else:
                logger.warning("compute_bertscore returned None for question: %s...", generated_answer[:50])

            # Compute original BERTScore (bert-score library)
            try:
                original_bertscore = self.compute_original_bertscore(
                    generated_answer,
                    best_match,
                    lang=lang,
                )
                logger.info(
                    "Original BERTScore computed: P=%.4f R=%.4f F1=%.4f",
                    original_bertscore['precision'],
                    original_bertscore['recall'],
                    original_bertscore['f1']
                )
                result['original_bertscore_precision'] = (
                    original_bertscore['precision']
                )
                result['original_bertscore_recall'] = original_bertscore['recall']
                result['original_bertscore_f1'] = original_bertscore['f1']
            except Exception as e:
                logger.error("Failed to compute original BERTScore, skipping: %s", e, exc_info=True)
                # Leave as None - already initialized in result dict

        return result

    def generate_answer(
        self,
        course_id: int,
        question: str,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        use_direct_llm: bool = False,
    ) -> Tuple[str, List[str], str]:
        """Generate an answer using the RAG pipeline.

        Uses the course's configured LLM and retrieves relevant context
        from the vector database.

        Args:
            course_id: Course ID
            question: Question to answer
            llm_provider: Optional LLM provider override
            llm_model: Optional LLM model override
            use_direct_llm: If True, bypass RAG and call LLM directly

        Returns:
            Tuple of (generated_answer, retrieved_contexts, llm_model_used)
        """
        # Get course and settings
        course = self.db.query(Course).filter(
            Course.id == course_id
        ).first()
        if not course:
            raise ValueError(f"Course {course_id} not found")

        settings = course.settings
        if not settings:
            raise ValueError(
                f"Course {course_id} has no settings configured"
            )

        # Use override or course settings for LLM
        actual_provider = (
            llm_provider or settings.llm_provider
        )
        actual_model = llm_model or settings.llm_model
        llm_model_used = f"{actual_provider}/{actual_model}"

        # ==================== DIRECT LLM MODE ====================
        if use_direct_llm:
            logger.info(
                "Direct LLM mode: bypassing RAG for course %d", course_id
            )
            print(f"[DIRECT LLM] *** DIRECT LLM MODE ACTIVE *** Course: {course_id}, Question: {question[:80]}...")
            messages = [
                {"role": "system", "content": "Sen yardımcı bir asistansın. Soruları kendi bilginle yanıtla."},
                {"role": "user", "content": question},
            ]
            llm_service = get_llm_service(
                provider=actual_provider,
                model=actual_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            generated_answer = llm_service.generate_response(messages)
            print(f"[DIRECT LLM] Answer (first 200 chars): {generated_answer[:200]}...")
            print(f"[DIRECT LLM] Contexts returned: [] (empty - no RAG)")
            return generated_answer, [], llm_model_used
        else:
            print(f"[DIRECT LLM] RAG mode active for course {course_id}")
        # ==================== END DIRECT LLM MODE ====================

        # Get query embedding
        query_embedding_model = (
            embedding_model or settings.default_embedding_model
        )
        weaviate_service = get_weaviate_service()

        def _search(q_vector: Optional[List[float]]):
            if q_vector:
                return weaviate_service.hybrid_search(
                    course_id=course_id,
                    query=question,
                    query_vector=q_vector,
                    alpha=settings.search_alpha,
                    limit=settings.search_top_k,
                )
            return weaviate_service.keyword_search(
                course_id=course_id,
                query=question,
                limit=settings.search_top_k,
            )

        query_vector = self.embedding_service.get_embedding(
            question,
            model=query_embedding_model,
        )
        search_results = _search(query_vector)

        if (
            settings.enable_reranker
            and settings.reranker_provider
            and search_results
        ):
            try:
                documents_for_reranking = [
                    {
                        "id": str(r.chunk_id),
                        "content": r.content,
                        "score": r.score,
                        "document_id": r.document_id,
                        "chunk_index": r.chunk_index,
                    }
                    for r in search_results
                ]
                rerank_service = get_rerank_service()
                reranked_docs = rerank_service.rerank(
                    query=question,
                    documents=documents_for_reranking,
                    provider=settings.reranker_provider,
                    model=settings.reranker_model,
                    top_k=min(
                        settings.reranker_top_k or 10,
                        len(documents_for_reranking),
                    ),
                )
                search_results = [
                    SearchResult(
                        chunk_id=int(doc["id"]),
                        document_id=doc.get("document_id", 0),
                        content=doc.get("content", ""),
                        chunk_index=doc.get("chunk_index", 0),
                        score=doc.get("relevance_score", doc.get("score", 0)),
                    )
                    for doc in reranked_docs
                ]
            except Exception as e:
                logger.warning("Reranker failed in semantic-similarity: %s", e)

        # Filter by minimum relevance score if configured
        min_score = settings.min_relevance_score or 0.0
        if min_score > 0 and search_results:
            search_results = [
                r for r in search_results if r.score >= min_score
            ]

        if (
            not search_results
            and query_embedding_model != settings.default_embedding_model
        ):
            fallback_vector = self.embedding_service.get_embedding(
                question,
                model=settings.default_embedding_model,
            )
            search_results = _search(fallback_vector)

            if (
                settings.enable_reranker
                and settings.reranker_provider
                and search_results
            ):
                try:
                    documents_for_reranking = [
                        {
                            "id": str(r.chunk_id),
                            "content": r.content,
                            "score": r.score,
                            "document_id": r.document_id,
                            "chunk_index": r.chunk_index,
                        }
                        for r in search_results
                    ]
                    rerank_service = get_rerank_service()
                    reranked_docs = rerank_service.rerank(
                        query=question,
                        documents=documents_for_reranking,
                        provider=settings.reranker_provider,
                        model=settings.reranker_model,
                        top_k=min(
                            settings.reranker_top_k or 10,
                            len(documents_for_reranking),
                        ),
                    )
                    search_results = [
                        SearchResult(
                            chunk_id=int(doc["id"]),
                            document_id=doc.get("document_id", 0),
                            content=doc.get("content", ""),
                            chunk_index=doc.get("chunk_index", 0),
                            score=doc.get(
                                "relevance_score",
                                doc.get("score", 0),
                            ),
                        )
                        for doc in reranked_docs
                    ]
                except Exception as e:
                    logger.warning(
                        "Reranker failed in semantic-similarity: %s",
                        e,
                    )

            if min_score > 0 and search_results:
                search_results = [
                    r for r in search_results if r.score >= min_score
                ]

        # Build context from search results
        contexts = [result.content for result in search_results]
        context_text = "\n\n".join(contexts)

        # Handle no results
        if not search_results:
            return (
                "Bu konuyla ilgili ders materyallerinde bilgi "
                "bulunamadı.",
                [],
                f"{actual_provider}/{actual_model}"
            )

        # Build messages for LLM
        default_system_prompt = (
            "Sen sağlanan ders dokümanlarındaki bilgileri yapılandırılmış "
            "şekilde sunan bir bilgi çıkarma sistemisin.\n\n"
            "CEVAPLAMA KURALLARI:\n"
            "1. Yalnızca sağlanan bağlam bilgilerini kullan. Bağlamda "
            "olmayan bilgiyi kesinlikle ekleme.\n"
            "2. Doğrudan cevaba başla, giriş cümleleri kullanma.\n"
            "3. Bağlamdaki anahtar terimleri ve teknik ifadeleri aynen "
            "kullan. Eş anlamlılarla değiştirme.\n"
            "4. Soruyla ilgili tüm bilgiyi bağlamdan çıkar, eksik "
            "bırakma ama bağlam dışına çıkma."
        )

        system_prompt = (
            settings.active_prompt_template.content
            if (
                getattr(settings, "active_prompt_template", None) is not None
                and getattr(settings.active_prompt_template, "content", None)
            )
            else (
                settings.system_prompt
                if settings.system_prompt
                else default_system_prompt
            )
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Aşağıda ders dokümanlarından alınan bağlam bilgileri verilmiştir.

Bağlam:
{context_text}

Soru: {question}

Yukarıdaki bağlam bilgilerini kullanarak soruyu yanıtla. Cevabında bağlamdaki teknik terimleri ve ifadeleri aynen kullan. Bağlamda olmayan bilgi ekleme."""}
        ]

        # Generate answer using LLM (with optional override)
        llm_service = get_llm_service(
            provider=actual_provider,
            model=actual_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )

        logger.debug("Using LLM: %s/%s", actual_provider, actual_model)

        generated_answer = llm_service.generate_response(messages)
        llm_model_used = f"{actual_provider}/{actual_model}"

        return generated_answer, contexts, llm_model_used


def get_semantic_similarity_service(
    db: Session
) -> SemanticSimilarityService:
    """Factory function to create a SemanticSimilarityService instance.

    Args:
        db: Database session

    Returns:
        SemanticSimilarityService instance
    """
    return SemanticSimilarityService(db)
