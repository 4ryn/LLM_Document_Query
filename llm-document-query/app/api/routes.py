from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Annotated
import logging
from app.models.schemas import DocumentRequest, DocumentResponse, HealthResponse
from app.services.query_service import QueryService
from app.config import settings
from app.utils.helpers import get_memory_usage, optimize_memory

logger = logging.getLogger(__name__)
router = APIRouter()

async def verify_bearer_token(authorization: Annotated[str, Header()]):
    """Verify Bearer token"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    token = authorization.split(" ")[1]
    if token != settings.api_bearer_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

@router.post("/hackrx/run", response_model=DocumentResponse)
async def process_document_query(
    request: DocumentRequest,
    token: str = Depends(verify_bearer_token)
):
    """Main endpoint for processing documents and answering questions"""
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Memory check before processing
        memory_usage = get_memory_usage()
        if memory_usage > 400:  # 400MB threshold
            optimize_memory()
            logger.warning(f"High memory usage detected: {memory_usage:.2f} MB")
        
        query_service = QueryService()
        
        # Process document and answer questions
        answers = await query_service.process_document_and_query(
            document_url=request.documents,
            questions=request.questions
        )
        
        # Final memory cleanup
        optimize_memory()
        
        return DocumentResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in process_document_query: {e}")
        optimize_memory()  # Clean up on error
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    memory_usage = get_memory_usage()
    
    return HealthResponse(
        status="healthy",
        message="LLM Document Query System is running",
        memory_usage_mb=memory_usage
    )

@router.get("/")
async def root():
    """Root endpoint"""
    return {"message": "LLM Document Query System", "status": "running"}