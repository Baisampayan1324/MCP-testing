"""
Text Splitter Module for Research Paper Summarization System
Handles intelligent text chunking with overlapping segments for better context preservation.
"""

import re
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    text: str
    start_index: int
    end_index: int
    chunk_id: str
    metadata: Dict


class TextSplitter:
    """Intelligent text splitting for research papers with semantic boundaries."""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 separator: str = "\n"):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            separator: Character to use for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        
        # Common sentence endings for research papers
        self.sentence_endings = ['.', '!', '?', '.\n', '!\n', '?\n']
        
        # Section headers commonly found in research papers
        self.section_headers = [
            r'^\d+\.\s+[A-Z][^.]*',  # 1. Introduction
            r'^[A-Z][A-Z\s]+\n',     # ABSTRACT, INTRODUCTION, etc.
            r'^\d+\.\d+\s+[A-Z][^.]*',  # 1.1 Subsection
            r'^[A-Z][a-z]+\s+[A-Z][^.]*',  # Method Results
        ]
    
    def split_text(self, text: str, document_id: str = "doc") -> List[TextChunk]:
        """
        Split text into overlapping chunks while preserving semantic boundaries.
        
        Args:
            text: Input text to split
            document_id: Unique identifier for the document
            
        Returns:
            List of TextChunk objects
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # First, split by paragraphs to preserve natural breaks
        paragraphs = self._split_by_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        start_index = 0
        
        for i, paragraph in enumerate(paragraphs):
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Find the best split point within the current chunk
                split_point = self._find_best_split_point(current_chunk)
                
                if split_point > 0:
                    # Create chunk up to the split point
                    chunk_text = current_chunk[:split_point].strip()
                    if chunk_text:
                        chunk = TextChunk(
                            text=chunk_text,
                            start_index=start_index,
                            end_index=start_index + len(chunk_text),
                            chunk_id=f"{document_id}_chunk_{len(chunks)}",
                            metadata={"paragraph_count": i}
                        )
                        chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[split_point - self.chunk_overlap:split_point]
                    current_chunk = overlap_text + paragraph
                    start_index = start_index + split_point - self.chunk_overlap
                else:
                    # No good split point, force split
                    chunk = TextChunk(
                        text=current_chunk.strip(),
                        start_index=start_index,
                        end_index=start_index + len(current_chunk),
                        chunk_id=f"{document_id}_chunk_{len(chunks)}",
                        metadata={"paragraph_count": i}
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk
                    current_chunk = paragraph
                    start_index = start_index + len(current_chunk) - self.chunk_overlap
            else:
                current_chunk += paragraph
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk = TextChunk(
                text=current_chunk.strip(),
                start_index=start_index,
                end_index=start_index + len(current_chunk),
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                metadata={"paragraph_count": len(paragraphs)}
            )
            chunks.append(chunk)
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs while preserving structure.
        
        Args:
            text: Input text
            
        Returns:
            List of paragraphs
        """
        # Split by double newlines first
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split long paragraphs
        final_paragraphs = []
        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                # Split long paragraphs by sentences
                sentences = self._split_by_sentences(paragraph)
                final_paragraphs.extend(sentences)
            else:
                final_paragraphs.append(paragraph)
        
        return [p.strip() for p in final_paragraphs if p.strip()]
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better chunking.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split by sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_best_split_point(self, text: str) -> int:
        """
        Find the best point to split text while preserving semantic meaning.
        
        Args:
            text: Text to find split point in
            
        Returns:
            Index of the best split point
        """
        # Prefer splitting at sentence boundaries
        for ending in self.sentence_endings:
            if ending in text:
                # Find the last occurrence within the chunk size
                last_sentence_end = text.rfind(ending)
                if last_sentence_end > 0 and last_sentence_end < self.chunk_size:
                    return last_sentence_end + len(ending)
        
        # If no sentence boundary, try paragraph breaks
        if '\n\n' in text:
            last_paragraph_break = text.rfind('\n\n')
            if last_paragraph_break > 0 and last_paragraph_break < self.chunk_size:
                return last_paragraph_break + 2
        
        # If no good break point, split at word boundary
        if ' ' in text:
            last_space = text.rfind(' ')
            if last_space > 0 and last_space < self.chunk_size:
                return last_space
        
        # If all else fails, return 0 (no good split point)
        return 0
    
    def split_document(self, document: Dict[str, Union[str, Dict]]) -> List[TextChunk]:
        """
        Split a loaded document into chunks.
        
        Args:
            document: Document dictionary from DataLoader
            
        Returns:
            List of TextChunk objects
        """
        content = str(document.get('content', ''))
        metadata = document.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}
        file_path = str(document.get('file_path', 'unknown'))
        
        # Generate document ID from file path
        doc_id = Path(file_path).stem if file_path else 'doc'
        
        chunks = self.split_text(content, doc_id)
        
        # Add document metadata to each chunk
        for chunk in chunks:
            chunk.metadata.update({
                'document_title': str(metadata.get('title', '')),
                'document_author': str(metadata.get('author', '')),
                'file_path': file_path,
                'original_metadata': metadata
            })
        
        return chunks
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict:
        """
        Get statistics about the generated chunks.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_text_length': sum(chunk_lengths)
        }


# Import Path for the split_document method
from pathlib import Path
