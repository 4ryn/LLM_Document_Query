import os
import logging
import uuid
from typing import List, Dict, Any, Optional
import hashlib
import json
from datetime import datetime, timedelta

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from app.config import settings
from app.utils.helpers import optimize_memory, log_memory_usage

# Configure logging for the embedding service
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Enhanced embedding service with better models, caching, and retrieval strategies
    """
    def __init__(self):
        # Use a more powerful embedding model
        # BAAI/bge-small-en-v1.5 is currently one of the best performing small models
        self.model_name = 'BAAI/bge-small-en-v1.5'
        try:
            self.model = SentenceTransformer(self.model_name)
            self.vector_size = 384  # Dimension for bge-small-en-v1.5
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}, falling back to all-MiniLM-L6-v2: {e}")
            self.model_name = 'all-MiniLM-L6-v2'
            self.model = SentenceTransformer(self.model_name)
            self.vector_size = 384
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        
        self.collection_name = settings.qdrant_collection_name
        
        # In-memory cache for embeddings
        self.embedding_cache = {}
        self.cache_max_size = 1000
        
        # Ensure the Qdrant collection exists on startup
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """
        Enhanced collection setup with proper indexing
        """
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                
                # Create index for faster filtering
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_url",
                    field_schema="keyword"
                )
                
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="timestamp",
                    field_schema="datetime"
                )
                
                logger.info(f"Created collection with indexes: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _manage_cache(self):
        """Manage cache size"""
        if len(self.embedding_cache) > self.cache_max_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.embedding_cache.keys())[:len(self.embedding_cache) - self.cache_max_size + 100]
            for key in keys_to_remove:
                del self.embedding_cache[key]
    
    def create_embeddings(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        Enhanced embedding creation with caching and query optimization
        """
        try:
            log_memory_usage("Before embedding creation")
            
            all_embeddings = []
            texts_to_embed = []
            cache_indices = []
            
            # Check cache first
            if use_cache:
                for i, text in enumerate(texts):
                    cache_key = self._get_cache_key(text)
                    if cache_key in self.embedding_cache:
                        all_embeddings.append(self.embedding_cache[cache_key])
                        cache_indices.append(i)
                    else:
                        texts_to_embed.append((i, text))
            else:
                texts_to_embed = list(enumerate(texts))
            
            # Embed uncached texts
            if texts_to_embed:
                batch_size = 8  # Optimal batch size for memory/speed balance
                
                for batch_start in range(0, len(texts_to_embed), batch_size):
                    batch = texts_to_embed[batch_start:batch_start + batch_size]
                    batch_texts = [item[1] for item in batch]
                    
                    # Add instruction prefix for better retrieval (for bge models)
                    if 'bge' in self.model_name.lower():
                        prefixed_texts = [f"Represent this document for search: {text}" for text in batch_texts]
                    else:
                        prefixed_texts = batch_texts
                    
                    batch_embeddings = self.model.encode(
                        prefixed_texts, 
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    )
                    
                    # Store in cache and results
                    for (original_idx, original_text), embedding in zip(batch, batch_embeddings):
                        embedding_list = embedding.tolist()
                        
                        if use_cache:
                            cache_key = self._get_cache_key(original_text)
                            self.embedding_cache[cache_key] = embedding_list
                        
                        # Insert at correct position
                        while len(all_embeddings) <= original_idx:
                            all_embeddings.append(None)
                        all_embeddings[original_idx] = embedding_list
                    
                    optimize_memory()
            
            # Manage cache size
            if use_cache:
                self._manage_cache()
            
            log_memory_usage("After embedding creation")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def create_query_embedding(self, query: str) -> List[float]:
        """Create optimized embedding for search queries"""
        # Add query prefix for better retrieval (for bge models)
        if 'bge' in self.model_name.lower():
            prefixed_query = f"Represent this query for searching relevant documents: {query}"
        else:
            prefixed_query = query
        
        embedding = self.model.encode([prefixed_query], normalize_embeddings=True)[0]
        return embedding.tolist()
    
    async def store_document_chunks(self, document_url: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Enhanced chunk storage with metadata and deduplication
        """
        try:
            log_memory_usage("Before storing chunks")
            
            # Clean up old entries for this document first
            await self._cleanup_old_document(document_url)
            
            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunks]
            
            # Create embeddings
            embeddings = self.create_embeddings(texts)
            
            # Prepare points for Qdrant with enhanced metadata
            points = []
            point_ids = []
            timestamp = datetime.now().isoformat()
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                # Enhanced payload with more metadata
                payload = {
                    "document_url": document_url,
                    "chunk_index": chunk["index"],
                    "text": chunk["text"][:2000],  # Increased text storage
                    "word_count": chunk["word_count"],
                    "timestamp": timestamp,
                    "section_title": chunk.get("section_title", ""),
                    "key_terms": chunk.get("key_terms", []),
                    "unique_word_ratio": chunk.get("unique_word_ratio", 0.0),
                    "readability_score": chunk.get("readability_score", 0.0),
                    "contains_numbers": chunk.get("contains_numbers", False),
                    "contains_questions": chunk.get("contains_questions", False),
                    "model_version": self.model_name
                }
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                )
            
            # Store in Qdrant in batches
            batch_size = 25  # Smaller batches for stability
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                logger.debug(f"Stored batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
            logger.info(f"Stored {len(points)} enhanced chunks in Qdrant")
            
            # Clean up
            del embeddings, points
            optimize_memory()
            
            log_memory_usage("After storing chunks")
            return point_ids
            
        except Exception as e:
            logger.error(f"Error storing chunks in Qdrant: {e}")
            raise
    
    async def _cleanup_old_document(self, document_url: str):
        """Remove old chunks for the same document"""
        try:
            # Delete existing points for this document
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_url",
                            match=MatchValue(value=document_url)
                        )
                    ]
                )
            )
            logger.info(f"Cleaned up old chunks for document: {document_url}")
        except Exception as e:
            logger.warning(f"Error cleaning up old document chunks: {e}")
    
    async def search_similar_chunks(self, query: str, limit: int = 10, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Enhanced similarity search with filtering and metadata
        """
        try:
            log_memory_usage("Before similarity search")
            
            # Create optimized query embedding
            query_embedding = self.create_query_embedding(query)
            
            # Search with higher limit for potential reranking
            search_limit = min(limit * 2, 50)  # Get more candidates
            
            # Search in Qdrant with score threshold
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=search_limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Format and enhance results
            results = []
            for result in search_results:
                if result.payload:
                    # Calculate additional relevance signals
                    text = result.payload.get("text", "")
                    relevance_boost = self._calculate_relevance_boost(query, result.payload)
                    
                    enhanced_result = {
                        "id": result.id,
                        "score": float(result.score),
                        "adjusted_score": float(result.score) * relevance_boost,
                        "text": text,
                        "document_url": result.payload.get("document_url", ""),
                        "chunk_index": result.payload.get("chunk_index", -1),
                        "section_title": result.payload.get("section_title", ""),
                        "word_count": result.payload.get("word_count", 0),
                        "key_terms": result.payload.get("key_terms", []),
                        "readability_score": result.payload.get("readability_score", 0.0),
                        "relevance_boost": relevance_boost,
                        "metadata": {
                            "unique_word_ratio": result.payload.get("unique_word_ratio", 0.0),
                            "contains_numbers": result.payload.get("contains_numbers", False),
                            "contains_questions": result.payload.get("contains_questions", False),
                            "timestamp": result.payload.get("timestamp", "")
                        }
                    }
                    results.append(enhanced_result)
            
            # Sort by adjusted score and limit results
            results.sort(key=lambda x: x["adjusted_score"], reverse=True)
            final_results = results[:limit]
            
            optimize_memory()
            log_memory_usage("After similarity search")
            
            logger.info(f"Found {len(final_results)} relevant chunks (score threshold: {score_threshold})")
            if final_results:
                avg_score = sum(r["adjusted_score"] for r in final_results) / len(final_results)
                logger.info(f"Average adjusted score: {avg_score:.3f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            raise
    
    def _calculate_relevance_boost(self, query: str, payload: Dict[str, Any]) -> float:
        """Calculate relevance boost based on chunk metadata"""
        boost = 1.0
        query_lower = query.lower()
        
        # Boost for section title relevance
        section_title = payload.get("section_title", "").lower()
        if section_title and any(word in section_title for word in query_lower.split()):
            boost *= 1.2
        
        # Boost for key terms overlap
        key_terms = payload.get("key_terms", [])
        if key_terms:
            query_words = set(query_lower.split())
            term_overlap = len(query_words.intersection(set(key_terms)))
            if term_overlap > 0:
                boost *= (1.0 + 0.1 * term_overlap)
        
        # Boost for readability (more readable chunks might be better)
        readability = payload.get("readability_score", 0.0)
        if readability > 0.7:
            boost *= 1.1
        
        # Boost for moderate length chunks
        word_count = payload.get("word_count", 0)
        if 100 <= word_count <= 500:  # Sweet spot for chunk length
            boost *= 1.15
        
        # Boost for questions if query is a question
        if "?" in query and payload.get("contains_questions", False):
            boost *= 1.1
        
        # Boost for numbers if query contains numbers
        if any(c.isdigit() for c in query) and payload.get("contains_numbers", False):
            boost *= 1.1
        
        return min(boost, 2.0)  # Cap boost at 2x
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "total_points": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "status": collection_info.status.name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}