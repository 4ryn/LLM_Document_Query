import sys
import logging
from typing import List, Dict, Any, Optional
import asyncio
import json
import re

try:
    from openai import AsyncOpenAI
    import openai
    logger = logging.getLogger(__name__)
    logger.info(f"OpenAI version: {openai.__version__}")
except ImportError as e:
    logger.error(f"OpenAI import error: {e}")
    raise

from app.config import settings
from app.utils.helpers import optimize_memory, log_memory_usage

class LLMService:
    def __init__(self):
        try:
            # Configure OpenRouter client with AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            
            # Use a high-performance model optimized for reasoning
            # Consider these alternatives based on performance needs:
            # - "anthropic/claude-3-haiku" (fastest, good quality)
            # - "google/gemini-pro" (balanced)
            # - "openai/gpt-4-turbo-preview" (highest quality)
            self.model = "anthropic/claude-3-haiku"  # Fast and reliable
            
            # Fallback models in order of preference
            self.fallback_models = [
                "google/gemini-pro",
                "openai/gpt-3.5-turbo",
                "microsoft/wizardlm-2-8x22b"
            ]
            
            logger.info(f"LLMService initialized with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Error initializing LLMService: {e}")
            raise
    
    def _build_enhanced_system_prompt(self) -> str:
        """Create an optimized system prompt for document QA"""
        return """You are an expert document analyst and question-answering assistant. Your task is to provide accurate, helpful answers based on the provided document context.

GUIDELINES:
1. ACCURACY: Base your answers strictly on the provided context. If information isn't available, clearly state this.
2. COMPLETENESS: Provide comprehensive answers that fully address the question.
3. CLARITY: Use clear, concise language that's easy to understand.
4. STRUCTURE: Organize your response logically with relevant details.
5. CONTEXT AWARENESS: Consider the document structure and section context when answering.

RESPONSE FORMAT:
- Provide direct answers to the question
- Include relevant details and specifics when available
- If multiple aspects are relevant, address them systematically
- For numerical data, include exact figures when provided
- For processes or procedures, outline steps clearly

IMPORTANT: If the context doesn't contain sufficient information to answer the question accurately, clearly state "Based on the provided document, I cannot find sufficient information to answer this question" rather than making assumptions."""
    
    def _extract_question_type(self, query: str) -> str:
        """Identify the type of question for targeted prompting"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'what are', 'what does', 'define']):
            return 'definition'
        elif any(word in query_lower for word in ['how', 'how to', 'steps', 'process']):
            return 'procedure'
        elif any(word in query_lower for word in ['when', 'time', 'date', 'period']):
            return 'temporal'
        elif any(word in query_lower for word in ['where', 'location', 'place']):
            return 'location'
        elif any(word in query_lower for word in ['why', 'reason', 'because', 'purpose']):
            return 'causal'
        elif any(word in query_lower for word in ['who', 'person', 'people']):
            return 'entity'
        elif any(word in query_lower for word in ['which', 'what', 'option', 'choice']):
            return 'selection'
        elif any(word in query_lower for word in ['how much', 'how many', 'cost', 'price', 'amount']):
            return 'quantitative'
        elif '?' in query:
            return 'general_question'
        else:
            return 'general'
    
    def _build_context_string(self, context_chunks: List[Dict[str, Any]], max_chunks: int = 5) -> str:
        """Build optimized context string with metadata"""
        context_parts = []
        
        for i, chunk in enumerate(context_chunks[:max_chunks]):
            text_content = chunk.get('text', '') or chunk.get('content', '')
            section_title = chunk.get('section_title', '')
            score = chunk.get('adjusted_score', chunk.get('score', 0))
            
            if text_content:
                # Add section context if available
                section_info = f" (Section: {section_title})" if section_title else ""
                header = f"[Context {i+1}{section_info} | Relevance: {score:.2f}]"
                
                # Clean and format text
                clean_text = text_content.strip()
                if len(clean_text) > 800:  # Limit context length
                    clean_text = clean_text[:800] + "..."
                
                context_parts.append(f"{header}\n{clean_text}")
        
        return "\n\n".join(context_parts)
    
    def _create_targeted_prompt(self, query: str, context: str, question_type: str) -> str:
        """Create question-type specific prompts for better accuracy"""
        
        base_prompt = f"""CONTEXT:
{context}

QUESTION: {query}

"""
        
        type_specific_instructions = {
            'definition': "Provide a clear, comprehensive definition based on the context. Include any relevant details, characteristics, or examples mentioned in the document.",
            
            'procedure': "Outline the process or procedure step-by-step. Include any prerequisites, warnings, or important notes mentioned in the context.",
            
            'temporal': "Provide specific time-related information including dates, periods, durations, or deadlines as mentioned in the context.",
            
            'quantitative': "Provide exact numbers, amounts, costs, or quantities as specified in the context. Include any conditions or qualifiers.",
            
            'causal': "Explain the reasons, causes, or purposes as described in the context. Include any contributing factors or consequences mentioned.",
            
            'location': "Provide specific location information including addresses, places, or geographical references mentioned in the context.",
            
            'entity': "Identify and describe the relevant people, organizations, or entities mentioned in the context.",
            
            'selection': "Compare and present the available options or choices described in the context. Highlight key differences if mentioned.",
            
            'general_question': "Provide a comprehensive answer addressing all aspects of the question based on the available context.",
            
            'general': "Analyze the context and provide relevant information that addresses the query comprehensively."
        }
        
        instruction = type_specific_instructions.get(question_type, type_specific_instructions['general'])
        
        return base_prompt + f"INSTRUCTION: {instruction}\n\nProvide your answer based strictly on the context provided above:"
    
    async def _make_llm_request(self, messages: List[Dict[str, str]], model: str = None, timeout: int = 45) -> str:
        """Make LLM request with error handling and retries"""
        current_model = model or self.model
        
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    max_tokens=500,  # Increased for more comprehensive answers
                    temperature=0.1,  # Low temperature for factual accuracy
                    top_p=0.9,
                    frequency_penalty=0.1,
                    presence_penalty=0.1,
                    extra_headers={
                        "HTTP-Referer": getattr(settings, 'app_url', 'http://localhost:8000'),
                        "X-Title": getattr(settings, 'app_name', 'Document QA App')
                    }
                ),
                timeout=timeout
            )
            
            return response.choices[0].message.content.strip()
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout with model {current_model}")
            raise
        except Exception as e:
            logger.error(f"Error with model {current_model}: {e}")
            raise
    
    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced answer generation with advanced prompting strategies"""
        try:
            log_memory_usage("Before LLM processing")
            
            if not context_chunks:
                return {
                    "answer": "I couldn't find relevant information in the document to answer this question.",
                    "confidence": 0.0,
                    "sources": [],
                    "question_type": "unknown"
                }
            
            # Analyze question type
            question_type = self._extract_question_type(query)
            logger.debug(f"Detected question type: {question_type}")
            
            # Build optimized context
            max_chunks = getattr(settings, 'max_chunks_per_query', 5)
            context = self._build_context_string(context_chunks, max_chunks)
            
            # Create targeted prompt
            system_prompt = self._build_enhanced_system_prompt()
            user_prompt = self._create_targeted_prompt(query, context, question_type)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Try primary model first, then fallbacks
            answer = None
            model_used = None
            
            for attempt, model in enumerate([self.model] + self.fallback_models):
                try:
                    logger.debug(f"Attempting with model: {model}")
                    answer = await self._make_llm_request(messages, model)
                    model_used = model
                    break
                except Exception as e:
                    logger.warning(f"Model {model} failed: {e}")
                    if attempt == len(self.fallback_models):  # Last attempt
                        raise
                    continue
            
            if not answer:
                raise Exception("All models failed")
            
            # Calculate enhanced confidence score
            confidence = self._calculate_confidence(context_chunks, answer, question_type)
            
            # Prepare enhanced response
            result = {
                "answer": answer,
                "confidence": confidence,
                "question_type": question_type,
                "model_used": model_used,
                "sources": [
                    {
                        "chunk_index": chunk.get("chunk_index", i),
                        "score": chunk.get("adjusted_score", chunk.get("score", 0.0)),
                        "section_title": chunk.get("section_title", ""),
                        "text_preview": (chunk.get('text', '') or chunk.get('content', ''))[:150] + "...",
                        "relevance_boost": chunk.get("relevance_boost", 1.0)
                    }
                    for i, chunk in enumerate(context_chunks[:max_chunks])
                ],
                "metadata": {
                    "context_chunks_used": len(context_chunks[:max_chunks]),
                    "total_context_length": len(context),
                    "avg_chunk_score": sum(c.get("adjusted_score", c.get("score", 0)) for c in context_chunks[:max_chunks]) / min(max_chunks, len(context_chunks)),
                    "processing_time": "optimized"
                }
            }
            
            optimize_memory()
            log_memory_usage("After LLM processing")
            
            logger.info(f"Generated answer with confidence: {confidence:.3f}, model: {model_used}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            logger.error(f"Error type: {type(e)}")
            
            # Return enhanced fallback response
            return {
                "answer": "I apologize, but I encountered an error while processing your question. This might be due to temporary service issues. Please try rephrasing your question or try again in a moment.",
                "confidence": 0.0,
                "question_type": self._extract_question_type(query) if query else "unknown",
                "sources": [],
                "error": str(e),
                "model_used": "none"
            }
    
    def _calculate_confidence(self, context_chunks: List[Dict[str, Any]], answer: str, question_type: str) -> float:
        """Calculate enhanced confidence score"""
        if not context_chunks or not answer:
            return 0.0
        
        # Base confidence from chunk scores
        scores = [chunk.get('adjusted_score', chunk.get('score', 0.0)) for chunk in context_chunks]
        base_confidence = sum(scores) / len(scores) if scores else 0.0
        
        # Boost factors
        confidence_boost = 1.0
        
        # Boost for answer length (not too short, not too long)
        answer_length = len(answer.split())
        if 10 <= answer_length <= 100:
            confidence_boost *= 1.1
        elif answer_length < 5:
            confidence_boost *= 0.8
        
        # Boost for specific answer patterns
        if any(phrase in answer.lower() for phrase in [
            'based on the document', 'according to the context', 
            'the document states', 'specifically mentions'
        ]):
            confidence_boost *= 1.15
        
        # Penalty for uncertainty phrases
        if any(phrase in answer.lower() for phrase in [
            'i cannot find', 'not mentioned', 'unclear', 
            'insufficient information', 'not specified'
        ]):
            confidence_boost *= 0.7
        
        # Question type specific adjustments
        type_adjustments = {
            'definition': 1.1,
            'quantitative': 1.2,  # Numbers are usually precise
            'temporal': 1.15,
            'procedure': 1.05,
            'general': 0.95
        }
        
        confidence_boost *= type_adjustments.get(question_type, 1.0)
        
        # Final confidence calculation
        final_confidence = min(base_confidence * confidence_boost, 1.0)
        return max(final_confidence, 0.0)
    
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Enhanced summary generation"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at creating concise, informative summaries. Focus on the most important information and key points."
                },
                {
                    "role": "user",
                    "content": f"Please provide a comprehensive yet concise summary of the following text in no more than {max_length} words. Focus on key facts, main points, and important details:\n\n{text}\n\nSummary:"
                }
            ]
            
            return await self._make_llm_request(messages, timeout=30)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary"
    
    async def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """Enhanced key point extraction"""
        try:
            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert at identifying and extracting the most important key points from documents. Focus on actionable information, key facts, and critical details."
                },
                {
                    "role": "user",
                    "content": f"Extract the {num_points} most important and actionable key points from the following text. Format as a clear, numbered list with each point being concise but informative:\n\n{text}\n\nKey Points:"
                }
            ]
            
            response_text = await self._make_llm_request(messages, timeout=30)
            
            # Parse the response into a list
            points = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    # Remove numbering/bullets and add to list
                    clean_point = re.sub(r'^[\d\-•*.\s]+', '', line).strip()
                    if clean_point and len(clean_point) > 10:  # Filter out too short points
                        points.append(clean_point)
            
            return points[:num_points]  # Ensure we don't exceed requested number
            
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return ["Error extracting key points"]
    
    async def analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure and content"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a document analysis expert. Analyze the structure, main topics, and organization of documents."
                },
                {
                    "role": "user", 
                    "content": f"Analyze the following document and provide:\n1. Main topics/sections\n2. Document type (policy, manual, guide, etc.)\n3. Key themes\n4. Structure assessment\n\nDocument:\n{text[:2000]}...\n\nAnalysis:"
                }
            ]
            
            analysis = await self._make_llm_request(messages, timeout=30)
            
            # Parse analysis into structured format
            return {
                "analysis": analysis,
                "word_count": len(text.split()),
                "estimated_reading_time": len(text.split()) // 200,  # ~200 WPM
                "complexity": "analyzed"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document structure: {e}")
            return {"analysis": "Error analyzing document", "word_count": 0}
    
    async def validate_answer_quality(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """Validate the quality and accuracy of generated answers"""
        try:
            validation_prompt = f"""Evaluate the following question-answer pair based on the provided context:

QUESTION: {question}
ANSWER: {answer}
CONTEXT: {context[:1000]}...

Assess:
1. Accuracy: Is the answer factually correct based on context?
2. Completeness: Does it fully address the question?
3. Relevance: Is it directly relevant to the question?
4. Clarity: Is it clear and well-structured?

Provide a brief evaluation and score (1-10) for each criterion."""

            messages = [
                {
                    "role": "system",
                    "content": "You are a quality assurance expert for AI-generated answers. Provide objective, critical evaluation."
                },
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ]
            
            evaluation = await self._make_llm_request(messages, timeout=25)
            
            # Extract scores (simple regex approach)
            scores = {}
            for criterion in ['accuracy', 'completeness', 'relevance', 'clarity']:
                pattern = rf'{criterion}.*?(\d+)'
                match = re.search(pattern, evaluation.lower())
                if match:
                    scores[criterion] = int(match.group(1))
                else:
                    scores[criterion] = 5  # Default neutral score
            
            overall_score = sum(scores.values()) / len(scores)
            
            return {
                "evaluation": evaluation,
                "scores": scores,
                "overall_score": overall_score,
                "quality_rating": "high" if overall_score >= 8 else "medium" if overall_score >= 6 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error validating answer quality: {e}")
            return {"evaluation": "Error in validation", "overall_score": 5.0}
    
    def _optimize_for_speed(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Optimize messages for faster processing"""
        # Truncate very long contexts while preserving key information
        for message in messages:
            if len(message["content"]) > 3000:
                content = message["content"]
                if "CONTEXT:" in content:
                    # Keep question and truncate context smartly
                    parts = content.split("QUESTION:")
                    if len(parts) == 2:
                        context_part = parts[0]
                        question_part = "QUESTION:" + parts[1]
                        
                        # Truncate context but keep first and last parts
                        if len(context_part) > 2000:
                            context_lines = context_part.split('\n')
                            keep_first = context_lines[:10]  # First 10 lines
                            keep_last = context_lines[-10:]  # Last 10 lines
                            truncated_context = '\n'.join(keep_first + ['...[content truncated]...'] + keep_last)
                            message["content"] = truncated_context + "\n\n" + question_part
        
        return messages
    
    async def batch_generate_answers(self, queries_and_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate answers for multiple questions efficiently"""
        try:
            tasks = []
            for item in queries_and_contexts:
                task = self.generate_answer(
                    query=item["query"],
                    context_chunks=item["context_chunks"]
                )
                tasks.append(task)
            
            # Process in parallel with concurrency limit
            semaphore = asyncio.Semaphore(3)  # Limit concurrent requests
            
            async def limited_generate(task):
                async with semaphore:
                    return await task
            
            results = await asyncio.gather(*[limited_generate(task) for task in tasks])
            return results
            
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            return [{"answer": "Error in batch processing", "confidence": 0.0} for _ in queries_and_contexts]


# Utility functions for debugging and monitoring
def check_model_availability():
    """Check which models are available through OpenRouter"""
    try:
        import httpx
        response = httpx.get("https://openrouter.ai/api/v1/models")
        if response.status_code == 200:
            models = response.json()
            available_models = [model["id"] for model in models.get("data", [])]
            logger.info(f"Available models: {len(available_models)}")
            return available_models
        else:
            logger.warning("Could not fetch available models")
            return []
    except Exception as e:
        logger.error(f"Error checking model availability: {e}")
        return []

def estimate_token_count(text: str) -> int:
    """Rough estimation of token count"""
    # Approximation: 1 token ≈ 0.75 words for English
    words = len(text.split())
    return int(words / 0.75)

def optimize_prompt_length(prompt: str, max_tokens: int = 3000) -> str:
    """Optimize prompt length to stay within token limits"""
    estimated_tokens = estimate_token_count(prompt)
    
    if estimated_tokens <= max_tokens:
        return prompt
    
    # Calculate reduction ratio
    reduction_ratio = max_tokens / estimated_tokens
    target_length = int(len(prompt) * reduction_ratio * 0.9)  # 10% buffer
    
    # Smart truncation preserving structure
    if "CONTEXT:" in prompt and "QUESTION:" in prompt:
        parts = prompt.split("QUESTION:")
        context_part = parts[0]
        question_part = "QUESTION:" + parts[1] if len(parts) > 1 else ""
        
        # Preserve question, truncate context
        available_for_context = target_length - len(question_part)
        if available_for_context > 0:
            truncated_context = context_part[:available_for_context] + "...[truncated]"
            return truncated_context + "\n\n" + question_part
    
    # Fallback: simple truncation
    return prompt[:target_length] + "...[truncated]"