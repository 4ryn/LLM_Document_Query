from typing import List, Dict, Any, Tuple
import logging
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.utils.helpers import optimize_memory, log_memory_usage
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        
        # Initialize TF-IDF for keyword search
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        self.tfidf_matrix = None
        self.stored_chunks = []
        
    def _preprocess_query(self, query: str) -> str:
        """Enhanced query preprocessing"""
        # Remove special characters but keep question words
        query = re.sub(r'[^\w\s\?]', ' ', query)
        # Normalize whitespace
        query = ' '.join(query.split())
        return query.lower()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction based on TF-IDF scores
        if self.tfidf_matrix is not None:
            try:
                tfidf_vec = self.tfidf_vectorizer.transform([text])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                scores = tfidf_vec.toarray()[0]
                
                # Get top keywords
                top_indices = np.argsort(scores)[-10:]
                keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
                return keywords
            except:
                pass
        
        # Fallback: simple keyword extraction
        words = text.lower().split()
        # Filter out common stop words and short words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        return keywords[:10]
    
    def _hybrid_search(self, query: str, semantic_results: List[Dict], top_k: int = 10) -> List[Dict]:
        """Combine semantic and keyword search results"""
        if not self.stored_chunks or self.tfidf_matrix is None:
            return semantic_results[:top_k]
        
        try:
            # Keyword search using TF-IDF
            query_tfidf = self.tfidf_vectorizer.transform([query])
            keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            
            # Create keyword results
            keyword_results = []
            for i, score in enumerate(keyword_scores):
                if i < len(self.stored_chunks) and score > 0.01:  # Minimum threshold
                    keyword_results.append({
                        'chunk_index': i,
                        'keyword_score': float(score),
                        'text': self.stored_chunks[i]['text']
                    })
            
            # Sort by keyword score
            keyword_results.sort(key=lambda x: x['keyword_score'], reverse=True)
            
            # Combine and rerank results
            combined_results = {}
            
            # Add semantic results
            for i, result in enumerate(semantic_results[:top_k]):
                chunk_id = result.get('id', f"semantic_{i}")
                combined_results[chunk_id] = {
                    **result,
                    'semantic_score': result.get('score', 0.0),
                    'keyword_score': 0.0,
                    'rank_semantic': i + 1
                }
            
            # Add keyword scores to matching chunks
            for kw_result in keyword_results[:top_k]:
                # Try to match with semantic results by text similarity
                for chunk_id, combined_result in combined_results.items():
                    if self._texts_similar(combined_result.get('text', ''), kw_result['text']):
                        combined_result['keyword_score'] = kw_result['keyword_score']
                        break
                else:
                    # Add as new result if not found in semantic
                    chunk_id = f"keyword_{kw_result['chunk_index']}"
                    if chunk_id not in combined_results:
                        combined_results[chunk_id] = {
                            'id': chunk_id,
                            'text': kw_result['text'],
                            'semantic_score': 0.0,
                            'keyword_score': kw_result['keyword_score'],
                            'score': kw_result['keyword_score'],  # Fallback score
                            'rank_semantic': 999
                        }
            
            # Calculate hybrid scores and rerank
            for result in combined_results.values():
                # Weighted combination: 70% semantic, 30% keyword
                semantic_weight = 0.7
                keyword_weight = 0.3
                
                result['hybrid_score'] = (
                    semantic_weight * result['semantic_score'] + 
                    keyword_weight * result['keyword_score']
                )
                result['score'] = result['hybrid_score']  # Update main score
            
            # Sort by hybrid score
            final_results = list(combined_results.values())
            final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to semantic: {e}")
            return semantic_results[:top_k]
    
    def _texts_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar (simple overlap-based)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
            
        overlap = len(words1.intersection(words2))
        similarity = overlap / min(len(words1), len(words2))
        return similarity >= threshold
    
    def _rerank_results(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """Advanced reranking using multiple signals"""
        if len(results) <= top_k:
            return results
        
        query_lower = query.lower()
        query_keywords = self._extract_keywords(query)
        
        for result in results:
            text = result.get('text', '').lower()
            
            # Calculate reranking features
            features = {
                'exact_match': 1.0 if any(keyword in text for keyword in query_keywords) else 0.0,
                'question_type_match': self._calculate_question_type_score(query_lower, text),
                'length_score': min(len(text) / 500, 1.0),  # Prefer moderate length
                'position_score': 1.0 / (results.index(result) + 1),  # Original ranking
                'keyword_density': self._calculate_keyword_density(query_keywords, text)
            }
            
            # Weighted reranking score
            rerank_score = (
                0.3 * result.get('hybrid_score', result.get('score', 0.0)) +  # Original score
                0.25 * features['exact_match'] +
                0.2 * features['question_type_match'] +
                0.1 * features['length_score'] +
                0.1 * features['position_score'] +
                0.05 * features['keyword_density']
            )
            
            result['rerank_score'] = rerank_score
            result['rerank_features'] = features
        
        # Sort by rerank score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return results[:top_k]
    
    def _calculate_question_type_score(self, query: str, text: str) -> float:
        """Score based on question type matching"""
        question_patterns = {
            'what': ['definition', 'description', 'explanation', 'meaning'],
            'how': ['process', 'method', 'steps', 'procedure', 'way'],
            'when': ['time', 'date', 'period', 'duration', 'schedule'],
            'where': ['location', 'place', 'address', 'site'],
            'why': ['reason', 'cause', 'purpose', 'benefit'],
            'who': ['person', 'people', 'individual', 'organization'],
            'which': ['option', 'choice', 'alternative', 'selection']
        }
        
        score = 0.0
        for q_word, keywords in question_patterns.items():
            if q_word in query:
                for keyword in keywords:
                    if keyword in text:
                        score += 0.2
                        break
        
        return min(score, 1.0)
    
    def _calculate_keyword_density(self, keywords: List[str], text: str) -> float:
        """Calculate keyword density in text"""
        if not keywords:
            return 0.0
        
        text_words = text.lower().split()
        if not text_words:
            return 0.0
        
        keyword_count = sum(1 for word in text_words if word in keywords)
        return keyword_count / len(text_words)
    
    async def process_document_and_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Enhanced document processing and query answering"""
        try:
            log_memory_usage("Start of enhanced query processing")
            
            # Step 1: Process document with enhanced chunking
            logger.info(f"Processing document: {document_url}")
            processed_doc = await self.document_processor.process_document(document_url)
            
            # Store chunks for keyword search
            self.stored_chunks = processed_doc["chunks"]
            
            # Build TF-IDF matrix for keyword search
            if self.stored_chunks:
                chunk_texts = [chunk.get('text', '') for chunk in self.stored_chunks]
                try:
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)
                    logger.info("TF-IDF matrix built successfully")
                except Exception as e:
                    logger.warning(f"TF-IDF matrix build failed: {e}")
                    self.tfidf_matrix = None
            
            # Step 2: Store chunks in vector database
            logger.info("Storing document chunks...")
            await self.embedding_service.store_document_chunks(
                document_url, 
                processed_doc["chunks"]
            )
            
            # Step 3: Answer questions with enhanced pipeline
            answers = []
            for i, question in enumerate(questions):
                try:
                    logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
                    
                    # Preprocess query
                    processed_query = self._preprocess_query(question)
                    
                    # Step 3a: Semantic search
                    semantic_chunks = await self.embedding_service.search_similar_chunks(
                        query=processed_query,
                        limit=10  # Get more candidates for reranking
                    )
                    
                    if not semantic_chunks:
                        answers.append("I couldn't find relevant information to answer this question in the provided document.")
                        continue
                    
                    # Step 3b: Hybrid search (combine semantic + keyword)
                    hybrid_chunks = self._hybrid_search(processed_query, semantic_chunks, top_k=10)
                    
                    # Step 3c: Rerank results
                    final_chunks = self._rerank_results(processed_query, hybrid_chunks, top_k=5)
                    
                    # Step 3d: Generate answer using LLM with enhanced context
                    result = await self.llm_service.generate_answer(
                        query=question,  # Use original question for LLM
                        context_chunks=final_chunks
                    )
                    
                    answers.append(result["answer"])
                    
                    # Log performance metrics
                    logger.info(f"Question {i+1} - Found {len(final_chunks)} relevant chunks")
                    if final_chunks:
                        avg_score = sum(chunk.get('rerank_score', 0) for chunk in final_chunks) / len(final_chunks)
                        logger.info(f"Average rerank score: {avg_score:.3f}")
                    
                    # Clean up after each question
                    optimize_memory()
                    
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {e}")
                    answers.append("An error occurred while processing this question. Please try again.")
            
            log_memory_usage("End of enhanced query processing")
            return answers
            
        except Exception as e:
            logger.error(f"Error in enhanced query processing: {e}")
            raise