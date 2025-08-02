import os
import logging
import uuid
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import settings
from app.utils.helpers import optimize_memory, log_memory_usage

# Configure logging for the embedding service
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Handles all operations related to text embedding and vector database interaction
    using SentenceTransformers and Qdrant.
    """
    def __init__(self):
        # Use a lightweight model for memory efficiency
        # This model generates a 384-dimensional vector
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = 384  # Dimension for all-MiniLM-L6-v2
        
        # Ensure the Qdrant collection exists on startup
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """
        Checks if the Qdrant collection exists and creates it if not.
        This prevents runtime errors when trying to interact with a non-existent collection.
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
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Creates embeddings for a list of texts with memory optimization.
        Processes texts in batches to reduce memory pressure.
        """
        try:
            log_memory_usage("Before embedding creation")
            
            # Process in smaller batches to save memory
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings = self.model.encode(batch, normalize_embeddings=True)
                all_embeddings.extend(embeddings.tolist())
                
                # Clean up after each batch
                optimize_memory()
            
            log_memory_usage("After embedding creation")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    async def store_document_chunks(self, document_url: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Stores document chunks and their embeddings in Qdrant.
        """
        try:
            log_memory_usage("Before storing chunks")
            
            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunks]
            
            # Create embeddings
            embeddings = self.create_embeddings(texts)
            
            # Prepare points for Qdrant
            points = []
            point_ids = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "document_url": document_url,
                            "chunk_index": chunk["index"],
                            "text": chunk["text"][:1000],  # Limit payload size
                            "word_count": chunk["word_count"]
                        }
                    )
                )
            
            # Store in Qdrant in batches
            batch_size = 50
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
            
            logger.info(f"Stored {len(points)} chunks in Qdrant")
            
            # Clean up
            del embeddings, points
            optimize_memory()
            
            log_memory_usage("After storing chunks")
            return point_ids
            
        except Exception as e:
            logger.error(f"Error storing chunks in Qdrant: {e}")
            raise
    
    async def search_similar_chunks(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for similar chunks in Qdrant based on a query.
        Returns the most relevant chunks.
        """
        try:
            log_memory_usage("Before similarity search")
            
            # Create query embedding
            query_embedding = self.create_embeddings([query])[0]
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=min(limit, settings.max_chunks_per_query)
            )
            
            # Format results, safely accessing the payload's text key
            results = []
            for result in search_results:
                if result.payload:
                    results.append({
                        "id": result.id,
                        "score": result.score,
                        "text": result.payload.get("text", "Text not found."),  # Use .get() for safety
                        "document_url": result.payload.get("document_url", "URL not found."),
                        "chunk_index": result.payload.get("chunk_index", -1)
                    })
            
            optimize_memory()
            log_memory_usage("After similarity search")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            # Reraise the exception after logging to allow the query_service to handle it
            raise