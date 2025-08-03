import requests
import PyPDF2
from docx import Document as DocxDocument
import io
from typing import List, Dict, Any
import logging
from urllib.parse import urlparse, unquote
from app.config import settings
from app.utils.helpers import optimize_memory, log_memory_usage
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class DocumentProcessor:
    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024  # 10MB limit
        self.stop_words = set(stopwords.words('english'))
    
    def get_file_type_from_url(self, url: str) -> str:
        """Extract file type from URL, handling query parameters and URL encoding"""
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path
            decoded_path = unquote(path)
            
            if decoded_path.lower().endswith('.pdf'):
                return 'pdf'
            elif decoded_path.lower().endswith('.docx'):
                return 'docx'
            elif decoded_path.lower().endswith('.doc'):
                return 'doc'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Error determining file type from URL: {e}")
            return 'unknown'
    
    def get_file_type_from_content(self, content: bytes) -> str:
        """Determine file type from content headers"""
        try:
            if content.startswith(b'%PDF'):
                return 'pdf'
            if content.startswith(b'PK\x03\x04'):
                return 'docx'
            if content.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
                return 'doc'
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error determining file type from content: {e}")
            return 'unknown'
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\bPage \d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove isolated single characters (often OCR artifacts)
        text = re.sub(r'\b\w\b', '', text)
        
        # Fix common OCR issues
        text = text.replace('fi', 'fi').replace('fl', 'fl')
        
        # Normalize punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'!{2,}', '!', text)
        
        return text.strip()
    
    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect document sections based on headings and structure"""
        sections = []
        lines = text.split('\n')
        current_section = ""
        section_title = "Introduction"
        
        heading_patterns = [
            r'^[A-Z][A-Z\s]{10,}$',  # ALL CAPS headings
            r'^\d+\.\s+[A-Z]',       # Numbered headings
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*:$',  # Title Case with colon
            r'^[IV]+\.\s+',          # Roman numerals
            r'^Chapter\s+\d+',       # Chapter headings
            r'^Section\s+\d+',       # Section headings
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            is_heading = False
            for pattern in heading_patterns:
                if re.match(pattern, line):
                    # Save previous section
                    if current_section.strip():
                        sections.append({
                            'title': section_title,
                            'content': current_section.strip(),
                            'type': 'section'
                        })
                    
                    # Start new section
                    section_title = line
                    current_section = ""
                    is_heading = True
                    break
            
            if not is_heading:
                current_section += line + "\n"
        
        # Add final section
        if current_section.strip():
            sections.append({
                'title': section_title,
                'content': current_section.strip(),
                'type': 'section'
            })
        
        return sections if sections else [{'title': 'Document', 'content': text, 'type': 'document'}]
    
    def _semantic_chunking(self, text: str, max_chunk_size: int = 800, overlap: int = 100) -> List[Dict[str, Any]]:
        """Enhanced semantic chunking strategy"""
        chunks = []
        
        # First, detect sections
        sections = self._detect_sections(text)
        
        for section_idx, section in enumerate(sections):
            section_content = section['content']
            section_title = section['title']
            
            # Split into sentences
            sentences = sent_tokenize(section_content)
            
            if not sentences:
                continue
            
            current_chunk = ""
            current_sentences = []
            chunk_word_count = 0
            
            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_words = len(word_tokenize(sentence))
                
                # Check if adding this sentence would exceed chunk size
                if chunk_word_count + sentence_words > max_chunk_size and current_chunk:
                    # Create chunk
                    chunk_data = self._create_chunk(
                        text=current_chunk.strip(),
                        sentences=current_sentences,
                        section_title=section_title,
                        chunk_index=len(chunks)
                    )
                    chunks.append(chunk_data)
                    
                    # Start new chunk with overlap
                    if overlap > 0 and len(current_sentences) > 1:
                        # Take last few sentences for overlap
                        overlap_sentences = current_sentences[-2:]  # Last 2 sentences
                        current_chunk = " ".join(overlap_sentences)
                        current_sentences = overlap_sentences
                        chunk_word_count = sum(len(word_tokenize(s)) for s in overlap_sentences)
                    else:
                        current_chunk = ""
                        current_sentences = []
                        chunk_word_count = 0
                
                # Add sentence to current chunk
                current_chunk += (" " + sentence if current_chunk else sentence)
                current_sentences.append(sentence)
                chunk_word_count += sentence_words
            
            # Add final chunk if not empty
            if current_chunk.strip():
                chunk_data = self._create_chunk(
                    text=current_chunk.strip(),
                    sentences=current_sentences,
                    section_title=section_title,
                    chunk_index=len(chunks)
                )
                chunks.append(chunk_data)
        
        return chunks
    
    def _create_chunk(self, text: str, sentences: List[str], section_title: str, chunk_index: int) -> Dict[str, Any]:
        """Create enhanced chunk with metadata"""
        words = word_tokenize(text.lower())
        
        # Calculate chunk quality metrics
        unique_words = set(word for word in words if word not in self.stop_words and word not in string.punctuation)
        
        # Extract key terms (simple frequency-based)
        word_freq = {}
        for word in unique_words:
            if len(word) > 3:  # Filter out short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        key_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'index': chunk_index,
            'text': text,
            'word_count': len(words),
            'sentence_count': len(sentences),
            'section_title': section_title,
            'key_terms': [term[0] for term in key_terms],
            'unique_word_ratio': len(unique_words) / len(words) if words else 0,
            'avg_sentence_length': sum(len(word_tokenize(s)) for s in sentences) / len(sentences) if sentences else 0,
            'contains_numbers': bool(re.search(r'\d+', text)),
            'contains_questions': bool(re.search(r'\?', text)),
            'readability_score': self._calculate_readability(text)
        }
    
    def _calculate_readability(self, text: str) -> float:
        """Simple readability score (Flesch Reading Ease approximation)"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Count syllables (rough approximation)
        syllables = 0
        for word in words:
            word = word.lower()
            if word.endswith('e'):
                word = word[:-1]
            syllable_count = max(1, len([char for char in word if char in 'aeiouy']))
            syllables += syllable_count
        
        avg_syllables_per_word = syllables / len(words) if words else 1
        
        # Simplified Flesch Reading Ease
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score)) / 100  # Normalize to 0-1
    
    async def download_document(self, url: str) -> bytes:
        """Download document with memory optimization"""
        try:
            log_memory_usage("Before document download")
            
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_file_size:
                raise ValueError(f"File too large: {content_length} bytes")
            
            content = response.content
            
            log_memory_usage("After document download")
            return content
            
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            raise
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF with enhanced cleaning"""
        try:
            log_memory_usage("Before PDF processing")
            
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Clean each page separately
                    cleaned_page = self._clean_text(page_text)
                    if cleaned_page:  # Only add non-empty pages
                        text += cleaned_page + "\n\n"
            
            pdf_file.close()
            del content
            optimize_memory()
            
            log_memory_usage("After PDF processing")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX with enhanced structure preservation"""
        try:
            log_memory_usage("Before DOCX processing")
            
            doc_file = io.BytesIO(content)
            doc = DocxDocument(doc_file)
            
            text_parts = []
            for paragraph in doc.paragraphs:
                para_text = paragraph.text.strip()
                if para_text:
                    # Preserve heading structure
                    if paragraph.style.name.startswith('Heading'):
                        text_parts.append(f"\n{para_text}\n")
                    else:
                        text_parts.append(para_text)
            
            text = "\n".join(text_parts)
            cleaned_text = self._clean_text(text)
            
            doc_file.close()
            del content
            optimize_memory()
            
            log_memory_usage("After DOCX processing")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise
    
    async def process_document(self, url: str) -> Dict[str, Any]:
        """Enhanced document processing with semantic chunking"""
        logger.info(f"Processing document: {url}")
        
        try:
            # Download document
            content = await self.download_document(url)
            
            # Determine file type
            file_type = self.get_file_type_from_url(url)
            if file_type == 'unknown':
                file_type = self.get_file_type_from_content(content)
            
            logger.info(f"Detected file type: {file_type}")
            
            # Extract text
            if file_type == 'pdf':
                text = self.extract_text_from_pdf(content)
            elif file_type in ['docx', 'doc']:
                text = self.extract_text_from_docx(content)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Quality check
            if len(text.strip()) < 100:
                raise ValueError("Extracted text is too short (less than 100 characters)")
            
            # Limit text size
            max_text_size = 100000  # Increased to 100k characters
            if len(text) > max_text_size:
                text = text[:max_text_size]
                logger.warning(f"Text truncated to {max_text_size} characters")
            
            # Create semantic chunks
            chunks = self._semantic_chunking(
                text,
                max_chunk_size=settings.max_chunk_size,
                overlap=settings.chunk_overlap
            )
            
            # Limit number of chunks
            max_chunks = 100  # Increased limit
            if len(chunks) > max_chunks:
                # Keep chunks with better quality scores
                chunks.sort(key=lambda x: x.get('unique_word_ratio', 0), reverse=True)
                chunks = chunks[:max_chunks]
                logger.warning(f"Chunks limited to {max_chunks}, kept highest quality chunks")
            
            # Log processing stats
            logger.info(f"Document processed: {len(text)} characters, {len(chunks)} chunks")
            avg_chunk_size = sum(chunk['word_count'] for chunk in chunks) / len(chunks) if chunks else 0
            logger.info(f"Average chunk size: {avg_chunk_size:.1f} words")
            
            log_memory_usage("After document processing")
            
            return {
                "text": text,
                "chunks": chunks,
                "metadata": {
                    "url": url,
                    "file_type": file_type,
                    "text_length": len(text),
                    "chunk_count": len(chunks),
                    "avg_chunk_size": avg_chunk_size,
                    "processing_quality": "enhanced"
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise