from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Supabase Configuration
    supabase_url: str
    supabase_key: str
    
    # Qdrant Configuration
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "hackrx_documents"
    
    # OpenRouter Configuration
    openrouter_api_key: str
    
    # API Configuration
    api_bearer_token: str
    
    # App Configuration
    app_name: str = "LLM Document Query System"
    debug: bool = False
    
    # Memory Optimization
    max_chunk_size: int = 800
    chunk_overlap: int = 100
    max_chunks_per_query: int = 5
    app_url: str = "http://localhost:8000"
    app_name: str = "Document QA App"
    
    class Config:
        env_file = ".env"

settings = Settings()