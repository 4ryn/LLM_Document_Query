from typing import List, Dict, Any
import logging
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.utils.helpers import optimize_memory, log_memory_usage

logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
    
    async def process_document_and_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Process document and answer questions"""
        try:
            log_memory_usage("Start of query processing")
            
            # Step 1: Process document
            logger.info(f"Processing document: {document_url}")
            processed_doc = await self.document_processor.process_document(document_url)
            
            # Step 2: Store chunks in vector database
            logger.info("Storing document chunks...")
            await self.embedding_service.store_document_chunks(
                document_url, 
                processed_doc["chunks"]
            )
            
            # Step 3: Answer questions
            answers = []
            for i, question in enumerate(questions):
                try:
                    logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
                    
                    # Search for relevant chunks
                    similar_chunks = await self.embedding_service.search_similar_chunks(
                        query=question,
                        limit=5
                    )
                    
                    if not similar_chunks:
                        answers.append("I couldn't find relevant information to answer this question in the provided document.")
                        continue
                    
                    # Generate answer using LLM
                    result = await self.llm_service.generate_answer(
                        query=question,
                        context_chunks=similar_chunks
                    )
                    
                    answers.append(result["answer"])
                    
                    # Clean up after each question
                    optimize_memory()
                    
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {e}")
                    answers.append("An error occurred while processing this question. Please try again.")
            
            log_memory_usage("End of query processing")
            return answers
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            raise