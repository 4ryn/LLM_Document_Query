from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
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
    app_url: str = "http://localhost:8000"
    debug: bool = False
    
    # ENHANCED CHUNKING CONFIGURATION
    max_chunk_size: int = 600  # Reduced for better semantic coherence
    chunk_overlap: int = 50    # Reduced overlap for efficiency
    max_chunks_per_query: int = 7  # Increased for better context
    
    # EMBEDDING CONFIGURATION
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # Better performing model
    embedding_cache_size: int = 1000
    embedding_batch_size: int = 8
    
    # SEARCH CONFIGURATION
    search_score_threshold: float = 0.3  # Minimum relevance score
    hybrid_search_enabled: bool = Field(
        default=True, 
        alias="enable_hybrid_search"  # Accept both names
    )
    reranking_enabled: bool = Field(
        default=True,
        alias="enable_reranking"  # Accept both names
    )
    max_search_results: int = 20  # Get more candidates for reranking
    
    # LLM CONFIGURATION
    primary_llm_model: str = "anthropic/claude-3-haiku"
    fallback_llm_models: List[str] = [
        "google/gemini-pro",
        "openai/gpt-3.5-turbo",
        "microsoft/wizardlm-2-8x22b"
    ]
    llm_timeout: int = 45
    llm_max_tokens: int = 500
    llm_temperature: float = 0.1
    
    # PERFORMANCE OPTIMIZATION
    memory_optimization_enabled: bool = True
    concurrent_requests_limit: int = 3
    cache_embeddings: bool = True
    
    # QUALITY SETTINGS
    min_chunk_word_count: int = 20
    max_chunk_word_count: int = 800
    quality_threshold: float = 0.4  # Minimum quality score for chunks
    
    # DOCUMENT PROCESSING
    max_document_size: int = 15 * 1024 * 1024  # 15MB limit
    max_text_length: int = 100000  # 100k characters
    max_chunks_per_document: int = 100
    
    # TEXT PROCESSING
    enable_semantic_chunking: bool = True
    enable_section_detection: bool = True
    clean_text_aggressively: bool = True
    preserve_document_structure: bool = True
    
    # MONITORING AND LOGGING
    log_performance_metrics: bool = True
    log_chunk_quality: bool = True
    log_search_results: bool = True
    
    # EXPERIMENTAL FEATURES
    enable_query_expansion: bool = False  # Expand queries with synonyms
    enable_answer_validation: bool = False  # Validate answer quality
    enable_context_compression: bool = True  # Compress context for efficiency
    
    # RETRY CONFIGURATION
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # RATE LIMITING
    requests_per_minute: int = 60
    burst_requests: int = 10
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
        populate_by_name = True  # Allow using both field names and aliases
        
        # Allow for environment variable overrides
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            if field_name in ['fallback_llm_models']:
                # Parse comma-separated list
                return [model.strip() for model in raw_val.split(',') if model.strip()]
            return raw_val

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Validate configuration
        self._validate_config()
        
        # Set derived configurations
        self._set_derived_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        if self.max_chunk_size < 100:
            raise ValueError("max_chunk_size must be at least 100")
        
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError("chunk_overlap must be less than max_chunk_size")
        
        if self.search_score_threshold < 0 or self.search_score_threshold > 1:
            raise ValueError("search_score_threshold must be between 0 and 1")
        
        if self.max_chunks_per_query < 1:
            raise ValueError("max_chunks_per_query must be at least 1")
    
    def _set_derived_config(self):
        """Set derived configuration values"""
        # Adjust batch size based on model
        if 'bge' in self.embedding_model.lower():
            self.embedding_batch_size = min(self.embedding_batch_size, 6)  # BGE models prefer smaller batches
        
        # Adjust search parameters based on performance settings
        if not self.debug:
            self.log_chunk_quality = False
            self.log_search_results = False
    
    def get_chunking_config(self) -> dict:
        """Get chunking-specific configuration"""
        return {
            'max_chunk_size': self.max_chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'min_word_count': self.min_chunk_word_count,
            'max_word_count': self.max_chunk_word_count,
            'semantic_chunking': self.enable_semantic_chunking,
            'section_detection': self.enable_section_detection
        }
    
    def get_search_config(self) -> dict:
        """Get search-specific configuration"""
        return {
            'score_threshold': self.search_score_threshold,
            'max_results': self.max_search_results,
            'hybrid_search': self.hybrid_search_enabled,
            'reranking': self.reranking_enabled,
            'max_chunks_per_query': self.max_chunks_per_query
        }
    
    def get_llm_config(self) -> dict:
        """Get LLM-specific configuration"""
        return {
            'primary_model': self.primary_llm_model,
            'fallback_models': self.fallback_llm_models,
            'timeout': self.llm_timeout,
            'max_tokens': self.llm_max_tokens,
            'temperature': self.llm_temperature
        }
    
    def get_performance_config(self) -> dict:
        """Get performance-specific configuration"""
        return {
            'memory_optimization': self.memory_optimization_enabled,
            'concurrent_limit': self.concurrent_requests_limit,
            'cache_embeddings': self.cache_embeddings,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay
        }

# Create global settings instance
settings = Settings()

# Utility functions for configuration
def is_production() -> bool:
    """Check if running in production environment"""
    return not settings.debug and os.getenv('ENVIRONMENT', '').lower() == 'production'

def get_model_config(model_name: str) -> dict:
    """Get model-specific configuration"""
    model_configs = {
        'BAAI/bge-small-en-v1.5': {
            'batch_size': 6,
            'max_length': 512,
            'query_prefix': 'Represent this query for searching relevant documents: '
        },
        'all-MiniLM-L6-v2': {
            'batch_size': 10,
            'max_length': 384,
            'query_prefix': ''
        }
    }
    
    return model_configs.get(model_name, {
        'batch_size': 8,
        'max_length': 512,
        'query_prefix': ''
    })

def get_environment_info() -> dict:
    """Get environment information for debugging"""
    return {
        'debug_mode': settings.debug,
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'app_name': settings.app_name,
        'embedding_model': settings.embedding_model,
        'primary_llm': settings.primary_llm_model,
        'optimizations_enabled': {
            'semantic_chunking': settings.enable_semantic_chunking,
            'hybrid_search': settings.hybrid_search_enabled,
            'reranking': settings.reranking_enabled,
            'memory_optimization': settings.memory_optimization_enabled
        }
    }