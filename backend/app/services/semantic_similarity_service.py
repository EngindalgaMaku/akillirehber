"""Semantic Similarity Service for measuring answer quality.

This service provides functionality to compute semantic similarity between
generated answers and ground truth answers using embedding vectors and
cosine similarity, as well as ROUGE and BERTScore metrics.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
import logging

from app.services.embedding_service import get_embedding_service
from app.services.llm_service import get_llm_service
from app.services.weaviate_service import get_weaviate_service
from app.models.db_models import Course

logger = logging.getLogger(__name__)


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
        # Get embeddings for both texts
        embedding1 = self.embedding_service.get_embedding(
            text1, model=embedding_model
        )
        embedding2 = self.embedding_service.get_embedding(
            text2, model=embedding_model
        )

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

        all_scores = []
        max_score = 0.0
        best_match = ground_truths[0]

        for ground_truth in ground_truths:
            score = self.compute_similarity(
                generated_answer,
                ground_truth,
                embedding_model
            )

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

    def compute_bertscore(
        self,
        generated_answer: str,
        ground_truth: str
    ) -> Optional[Dict[str, float]]:
        """Compute BERTScore using embeddings.

        Uses embedding service to compute semantic similarity as BERTScore.
        This is a lightweight approximation that doesn't require HF API.

        Args:
            generated_answer: The generated answer to evaluate
            ground_truth: The reference ground truth answer

        Returns:
            Dict with precision, recall, f1 scores, or None if unavailable
        """
        try:
            # Use our own embedding service (OpenAI or local)
            # This is faster and doesn't require HF API
            emb1 = self.embedding_service.get_embedding(
                generated_answer,
                model="openai/text-embedding-3-small"
            )
            emb2 = self.embedding_service.get_embedding(
                ground_truth,
                model="openai/text-embedding-3-small"
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

        Computes cosine similarity, ROUGE, BERTScore, Hit@1, and MRR metrics.

        Args:
            generated_answer: The generated answer to evaluate
            ground_truths: List of ground truth answers
            embedding_model: Embedding model to use for cosine similarity
            retrieved_contexts: List of retrieved chunks from RAG (for Hit@1/MRR)
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

        # Compute Hit@1 and MRR based on retrieved contexts
        # These measure if the ground truth is present in retrieved chunks
        hit_at_1 = 0.0
        mrr = 0.0
        
        if retrieved_contexts and len(retrieved_contexts) > 0 and ground_truths:
            # For each ground truth, check if it appears in retrieved contexts
            # Find the highest-ranked context that contains ground truth info
            best_rank = float('inf')
            
            # Lower threshold for more lenient matching (0.5 = 50% similarity)
            RELEVANCE_THRESHOLD = 0.5
            
            for ground_truth in ground_truths:
                for rank, context in enumerate(retrieved_contexts, start=1):
                    # Check if this context is relevant to ground truth
                    # Use embedding similarity as relevance measure
                    similarity = self.compute_similarity(
                        context,
                        ground_truth,
                        embedding_model
                    )
                    
                    # Log for debugging
                    logger.debug(
                        f"Chunk rank {rank} vs ground_truth similarity: {similarity:.3f}"
                    )
                    
                    # If similarity is above threshold, consider this context relevant
                    if similarity >= RELEVANCE_THRESHOLD:
                        if rank < best_rank:
                            best_rank = rank
                            logger.info(
                                f"Found relevant chunk at rank {rank} "
                                f"(similarity: {similarity:.3f})"
                            )
                        break  # Found relevant context for this ground truth
            
            # Calculate Hit@1 and MRR based on best rank
            if best_rank != float('inf'):
                hit_at_1 = 1.0 if best_rank == 1 else 0.0
                mrr = 1.0 / best_rank
                logger.info(
                    f"Retrieval metrics - Hit@1: {hit_at_1}, "
                    f"MRR: {mrr:.3f} (best_rank: {best_rank})"
                )
            else:
                # Ground truth not found in any retrieved context
                hit_at_1 = 0.0
                mrr = 0.0
                logger.warning(
                    "No relevant chunk found for ground truth "
                    f"(threshold: {RELEVANCE_THRESHOLD})"
                )

        result = {
            'similarity_score': max_score,
            'best_match_ground_truth': best_match,
            'all_scores': all_scores,
            'hit_at_1': hit_at_1,
            'mrr': mrr,
            'rouge1': None,
            'rouge2': None,
            'rougel': None,
            'bertscore_precision': None,
            'bertscore_recall': None,
            'bertscore_f1': None,
        }

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
                best_match
            )
            if bertscore:
                result['bertscore_precision'] = bertscore['precision']
                result['bertscore_recall'] = bertscore['recall']
                result['bertscore_f1'] = bertscore['f1']

        return result

    def generate_answer(
        self,
        course_id: int,
        question: str,
        llm_provider: str = None,
        llm_model: str = None
    ) -> Tuple[str, List[str], str]:
        """Generate an answer using the RAG pipeline.

        Uses the course's configured LLM and retrieves relevant context
        from the vector database.

        Args:
            course_id: Course ID
            question: Question to answer
            llm_provider: Optional LLM provider override
            llm_model: Optional LLM model override

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
        actual_provider = llm_provider or settings.llm_provider
        actual_model = llm_model or settings.llm_model

        # Get query embedding
        query_vector = self.embedding_service.get_embedding(
            question,
            model=settings.default_embedding_model
        )

        # Search for relevant chunks using hybrid search
        weaviate_service = get_weaviate_service()
        search_results = weaviate_service.hybrid_search(
            course_id=course_id,
            query=question,
            query_vector=query_vector,
            alpha=settings.search_alpha,
            limit=settings.search_top_k
        )

        # Filter by minimum relevance score if configured
        min_score = settings.min_relevance_score or 0.0
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
            "Sen bir eğitim asistanısın. Verilen bağlam bilgilerini "
            "kullanarak\nöğrencilerin sorularını yanıtla. Yanıtlarını "
            "Türkçe ver.\n\nKurallar:\n1. Sadece verilen bağlamdaki "
            "bilgileri kullan\n2. Bağlamda olmayan bilgileri uydurma\n"
            "3. Emin olmadığın konularda bunu belirt\n"
            "4. Yanıtlarını açık ve anlaşılır tut"
        )

        system_prompt = (
            settings.system_prompt
            if settings.system_prompt
            else default_system_prompt
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Bağlam:
{context_text}

Soru: {question}"""}
        ]

        # Generate answer using LLM (with optional override)
        llm_service = get_llm_service(
            provider=actual_provider,
            model=actual_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )

        # Log system prompt for debugging
        print(f"\n{'='*80}")
        print(f"🔍 SEMANTIC SIMILARITY - GENERATING ANSWER")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"LLM: {actual_provider}/{actual_model}")
        print(f"System Prompt (first 200 chars):\n{system_prompt[:200]}...")
        print(f"{'='*80}\n")
        
        logger.info(
            "Generating answer with system prompt (first 100 chars): %s...",
            system_prompt[:100]
        )
        logger.info("Using LLM: %s/%s", actual_provider, actual_model)

        generated_answer = llm_service.generate_response(messages)
        llm_model_used = f"{actual_provider}/{actual_model}"
        
        print(f"✅ Generated answer (first 100 chars): {generated_answer[:100]}...\n")

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
