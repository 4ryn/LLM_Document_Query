from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DocumentRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]

class DocumentResponse(BaseModel):
    answers: List[str]

class HealthResponse(BaseModel):
    status: str
    message: str
    memory_usage_mb: Optional[float] = None

class ChunkData(BaseModel):
    text: str
    index: int
    metadata: Dict[str, Any]

class QueryResult(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]