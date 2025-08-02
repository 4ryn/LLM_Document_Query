import sys
import logging
from typing import List, Dict, Any

# Debug: Check if openai is properly imported
try:
    from openai import AsyncOpenAI
    import openai
    print(f"OpenAI version: {openai.__version__}")
    print(f"OpenAI module path: {openai.__file__}")
except ImportError as e:
    print(f"OpenAI import error: {e}")
    raise

from app.config import settings
from app.utils.helpers import optimize_memory, log_memory_usage

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        try:
            # Configure OpenRouter client with AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            
            # Use a cost-effective model
            self.model = "microsoft/wizardlm-2-8x22b"
            logger.info(f"LLMService initialized with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Error initializing LLMService: {e}")
            raise
    
    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using LLM with context"""
        try:
            log_memory_usage("Before LLM processing")
            
            # Build context from chunks
            context_parts = []
            max_chunks = getattr(settings, 'max_chunks_per_query', 5)
            
            for i, chunk in enumerate(context_chunks[:max_chunks]):
                # Handle different chunk formats
                text_content = chunk.get('text', '') or chunk.get('content', '')
                if text_content:
                    context_parts.append(f"[Source {i+1}]: {text_content[:500]}")
            
            context = "\n\n".join(context_parts)
            
            # Create optimized prompt
            system_message = "You are an expert document analyst. Based on the provided context, answer the user's question accurately and concisely. If the information is not available in the context, clearly state this."
            
            user_message = f"""CONTEXT:
{context}

QUESTION: {query}

Please provide a direct, accurate answer based only on the information in the context."""
            
            # Make async API call with proper OpenAI 1.0+ syntax
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=300,
                temperature=0.1,
                timeout=30,
                # Optional OpenRouter headers
                extra_headers={
                    "HTTP-Referer": getattr(settings, 'app_url', 'http://localhost:8000'),
                    "X-Title": getattr(settings, 'app_name', 'Document QA App')
                }
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on similarity scores
            confidence = 0.0
            if context_chunks:
                scores = [chunk.get('score', 0.0) for chunk in context_chunks]
                confidence = sum(scores) / len(scores) if scores else 0.0
            
            result = {
                "answer": answer,
                "confidence": confidence,
                "sources": [
                    {
                        "chunk_index": chunk.get("chunk_index", i),
                        "score": chunk.get("score", 0.0),
                        "text_preview": (chunk.get('text', '') or chunk.get('content', ''))[:100] + "..."
                    }
                    for i, chunk in enumerate(context_chunks)
                ]
            }
            
            optimize_memory()
            log_memory_usage("After LLM processing")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error args: {e.args}")
            
            # Return fallback response
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "confidence": 0.0,
                "sources": [],
                "error": str(e)
            }
    
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the provided text"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user", 
                        "content": f"Please provide a concise summary of the following text in no more than {max_length} words:\n\n{text}\n\nSummary:"
                    }
                ],
                max_tokens=300,
                temperature=0.1,
                timeout=30
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary"
    
    async def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """Extract key points from the provided text"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Extract the {num_points} most important key points from the following text. Format as a numbered list:\n\n{text}\n\nKey Points:"
                    }
                ],
                max_tokens=400,
                temperature=0.1,
                timeout=30
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the response into a list
            points = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets and add to list
                    clean_point = line.lstrip('0123456789.-• ').strip()
                    if clean_point:
                        points.append(clean_point)
            
            return points[:num_points]  # Ensure we don't exceed requested number
            
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return ["Error extracting key points"]


# Debug function to check for old OpenAI usage
def check_openai_usage():
    """Debug function to check if old OpenAI syntax is being used anywhere"""
    import importlib
    import pkgutil
    
    # Check if any loaded modules are using old openai syntax
    for name, module in sys.modules.items():
        if hasattr(module, '__file__') and module.__file__:
            try:
                with open(module.__file__, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'openai.ChatCompletion' in content:
                        print(f"WARNING: Old OpenAI syntax found in {module.__file__}")
            except:
                pass