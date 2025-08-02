import requests
import PyPDF2
from docx import Document as DocxDocument
import io
from typing import List, Dict, Any
import logging
from urllib.parse import urlparse, unquote
from app.config import settings
from app.utils.helpers import optimize_memory, log_memory_usage, chunk_text_optimized

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024  # 10MB limit
    
    def get_file_type_from_url(self, url: str) -> str:
        """Extract file type from URL, handling query parameters and URL encoding"""
        try:
            # Parse the URL to get the path
            parsed_url = urlparse(url)
            path = parsed_url.path
            
            # URL decode the path
            decoded_path = unquote(path)
            
            # Extract file extension
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
            # Check PDF signature
            if content.startswith(b'%PDF'):
                return 'pdf'
            
            # Check DOCX signature (ZIP file with specific structure)
            if content.startswith(b'PK\x03\x04'):
                # This could be a DOCX (which is a ZIP file)
                # We could do more sophisticated checking here
                return 'docx'
            
            # Check DOC signature
            if content.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
                return 'doc'
                
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error determining file type from content: {e}")
            return 'unknown'
    
    async def download_document(self, url: str) -> bytes:
        """Download document with memory optimization"""
        try:
            log_memory_usage("Before document download")
            
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check file size
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
        """Extract text from PDF with memory optimization"""
        try:
            log_memory_usage("Before PDF processing")
            
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Clean up
            pdf_file.close()
            del content
            optimize_memory()
            
            log_memory_usage("After PDF processing")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX with memory optimization"""
        try:
            log_memory_usage("Before DOCX processing")
            
            doc_file = io.BytesIO(content)
            doc = DocxDocument(doc_file)
            
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Clean up
            doc_file.close()
            del content
            optimize_memory()
            
            log_memory_usage("After DOCX processing")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise
    
    async def process_document(self, url: str) -> Dict[str, Any]:
        """Process document with memory optimization"""
        logger.info(f"Processing document: {url}")
        
        try:
            # Download document
            content = await self.download_document(url)
            
            # Determine file type from URL first, then from content
            file_type = self.get_file_type_from_url(url)
            if file_type == 'unknown':
                file_type = self.get_file_type_from_content(content)
            
            logger.info(f"Detected file type: {file_type}")
            
            # Extract text based on file type
            if file_type == 'pdf':
                text = self.extract_text_from_pdf(content)
            elif file_type in ['docx', 'doc']:
                text = self.extract_text_from_docx(content)
            else:
                raise ValueError(f"Unsupported file type: {file_type}. Only PDF and DOCX are supported.")
            
            # Limit text size to prevent memory issues
            max_text_size = 50000  # 50k characters
            if len(text) > max_text_size:
                text = text[:max_text_size]
                logger.warning(f"Text truncated to {max_text_size} characters")
            
            # Create optimized chunks
            chunks = chunk_text_optimized(
                text, 
                chunk_size=settings.max_chunk_size,
                overlap=settings.chunk_overlap
            )
            
            # Limit number of chunks
            max_chunks = 50
            if len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]
                logger.warning(f"Chunks limited to {max_chunks}")
            
            log_memory_usage("After document processing")
            
            return {
                "text": text,
                "chunks": chunks,
                "metadata": {
                    "url": url,
                    "file_type": file_type,
                    "text_length": len(text),
                    "chunk_count": len(chunks)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise