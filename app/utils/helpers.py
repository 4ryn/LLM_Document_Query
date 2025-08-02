import psutil
import gc
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def optimize_memory():
    """Force garbage collection to free memory"""
    gc.collect()

def log_memory_usage(operation: str):
    """Log memory usage for debugging"""
    memory_mb = get_memory_usage()
    logger.info(f"{operation} - Memory usage: {memory_mb:.2f} MB")

def extract_filename_from_url(url: str) -> str:
    """Extract filename from URL"""
    return url.split('/')[-1].split('?')[0]

def chunk_text_optimized(text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict[str, Any]]:
    """Memory-optimized text chunking"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "index": len(chunks),
            "word_count": len(chunk_words)
        })
    
    # Clear the words list to free memory
    del words
    optimize_memory()
    
    return chunks