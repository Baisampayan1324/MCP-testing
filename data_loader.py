"""
Data Loader Module for Research Paper Summarization System
Handles PDF parsing and text extraction with cleaning capabilities.
"""

import fitz  # PyMuPDF
import re
from typing import List, Dict, Union, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of research papers from PDF and text files."""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt']
    
    def load_document(self, file_path: Union[str, Path]) -> Dict[str, Union[str, Dict]]:
        """
        Load a document from file path and return structured data.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document metadata and content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        if file_path.suffix.lower() == '.pdf':
            return self._load_pdf(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self._load_text(file_path)
    
    def _load_pdf(self, file_path: Path) -> Dict[str, Union[str, Dict]]:
        """
        Extract text and metadata from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            doc = fitz.open(file_path)
            text_content = ""
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'page_count': len(doc)
            }
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += page.get_text()
            
            doc.close()
            
            # Clean the extracted text
            cleaned_text = self._clean_text(text_content)
            
            return {
                'content': cleaned_text,
                'metadata': metadata,
                'file_path': str(file_path),
                'file_type': 'pdf'
            }
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise
    
    def _load_text(self, file_path: Path) -> Dict[str, Union[str, Dict]]:
        """
        Load text from plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Dictionary with text content and metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            cleaned_text = self._clean_text(content)
            
            return {
                'content': cleaned_text,
                'metadata': {
                    'title': file_path.stem,
                    'author': '',
                    'subject': '',
                    'creator': '',
                    'producer': '',
                    'page_count': 1
                },
                'file_path': str(file_path),
                'file_type': 'text'
            }
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def load_multiple_documents(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Union[str, Dict]]]:
        """
        Load multiple documents from a list of file paths.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        for file_path in file_paths:
            try:
                doc = self.load_document(file_path)
                documents.append(doc)
                logger.info(f"Successfully loaded: {file_path}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                continue
        
        return documents
    
    def validate_document(self, document: Dict[str, Union[str, Dict]]) -> bool:
        """
        Validate if a loaded document has sufficient content.
        
        Args:
            document: Loaded document dictionary
            
        Returns:
            True if document is valid, False otherwise
        """
        content = document.get('content', '')
        
        # Check if content is not empty and has minimum length
        if not content or len(content.strip()) < 100:
            return False
        
        # Check if content has meaningful text (not just whitespace/special chars)
        meaningful_chars = len(re.sub(r'\s', '', content))
        if meaningful_chars < 50:
            return False
        
        return True
