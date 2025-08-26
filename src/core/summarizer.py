"""
Summarizer Module for Research Paper Summarization System
Handles text summarization using transformer models with RAG techniques.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import List, Dict, Union, Optional
import logging
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Summarizer:
    """Text summarization using transformer models with RAG support."""
    
    def __init__(self, 
                 model_name: str = "facebook/bart-large-cnn",
                 device: Optional[str] = None,
                 max_length: int = 512,
                 min_length: int = 50,
                 batch_size: int = 4):
        """
        Initialize the summarizer.
        
        Args:
            model_name: Name of the summarization model
            device: Device to run the model on
            max_length: Maximum length of generated summary
            min_length: Minimum length of generated summary
            batch_size: Batch size for summarization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing summarizer: {model_name} on {self.device}")
        
        try:
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            
            # Create summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1,
                batch_size=self.batch_size
            )
            
            logger.info(f"Successfully loaded summarization model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load summarization model {model_name}: {str(e)}")
            raise
    
    def summarize_text(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Summarize a single text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary (overrides default)
            
        Returns:
            Generated summary
        """
        if not text or not text.strip():
            return ""
        
        # Clean and prepare text
        cleaned_text = self._clean_text(text)
        
        if len(cleaned_text) < 100:
            return cleaned_text  # Return original if too short
        
        try:
            # Generate summary
            summary_length = max_length or self.max_length
            
            result = self.summarizer(
                cleaned_text,
                max_length=summary_length,
                min_length=self.min_length,
                do_sample=False,
                truncation=True
            )
            
            summary = result[0]['summary_text']
            return self._post_process_summary(summary)
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return cleaned_text[:summary_length] if 'summary_length' in locals() else cleaned_text
    
    def summarize_chunks(self, chunks: List[Dict], max_length: Optional[int] = None) -> List[Dict]:
        """
        Summarize multiple text chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            max_length: Maximum length of each summary
            
        Returns:
            List of dictionaries with summaries and metadata
        """
        if not chunks:
            return []
        
        logger.info(f"Summarizing {len(chunks)} chunks")
        
        results = []
        for chunk in tqdm(chunks, desc="Summarizing chunks"):
            try:
                # Extract text from chunk
                if isinstance(chunk, dict) and 'text' in chunk:
                    text = chunk['text']
                    metadata = chunk.get('metadata', {})
                    chunk_id = chunk.get('chunk_id', '')
                elif hasattr(chunk, 'text'):
                    text = chunk.text
                    metadata = getattr(chunk, 'metadata', {})
                    chunk_id = getattr(chunk, 'chunk_id', '')
                else:
                    logger.warning(f"Skipping chunk with invalid format: {chunk}")
                    continue
                
                # Generate summary
                summary = self.summarize_text(text, max_length)
                
                result = {
                    'chunk_id': chunk_id,
                    'original_text': text,
                    'summary': summary,
                    'metadata': metadata,
                    'summary_length': len(summary),
                    'original_length': len(text)
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error summarizing chunk {chunk.get('chunk_id', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Successfully summarized {len(results)} chunks")
        return results
    
    def generate_final_summary(self, 
                              chunk_summaries: List[Dict], 
                              max_length: Optional[int] = None,
                              strategy: str = "hierarchical") -> str:
        """
        Generate a final summary from chunk summaries.
        
        Args:
            chunk_summaries: List of chunk summary dictionaries
            max_length: Maximum length of final summary
            strategy: Summarization strategy ('hierarchical', 'concatenate', 'selective')
            
        Returns:
            Final summary text
        """
        if not chunk_summaries:
            return ""
        
        if strategy == "hierarchical":
            return self._hierarchical_summarization(chunk_summaries, max_length)
        elif strategy == "concatenate":
            return self._concatenate_summaries(chunk_summaries, max_length)
        elif strategy == "selective":
            return self._selective_summarization(chunk_summaries, max_length)
        else:
            raise ValueError(f"Unknown summarization strategy: {strategy}")
    
    def _hierarchical_summarization(self, chunk_summaries: List[Dict], max_length: Optional[int]) -> str:
        """
        Hierarchical summarization: combine summaries and summarize again.
        
        Args:
            chunk_summaries: List of chunk summaries
            max_length: Maximum length of final summary
            
        Returns:
            Final summary
        """
        # Combine all summaries
        combined_text = " ".join([summary['summary'] for summary in chunk_summaries])
        
        # If combined text is short enough, return as is
        if len(combined_text) <= (max_length or self.max_length):
            return combined_text
        
        # Otherwise, summarize the combined text
        return self.summarize_text(combined_text, max_length)
    
    def _concatenate_summaries(self, chunk_summaries: List[Dict], max_length: Optional[int]) -> str:
        """
        Simple concatenation of summaries with length limit.
        
        Args:
            chunk_summaries: List of chunk summaries
            max_length: Maximum length of final summary
            
        Returns:
            Concatenated summary
        """
        summaries = [summary['summary'] for summary in chunk_summaries]
        combined_text = " ".join(summaries)
        
        max_len = max_length or self.max_length
        if len(combined_text) <= max_len:
            return combined_text
        
        # Truncate to max length
        return combined_text[:max_len] + "..."
    
    def _selective_summarization(self, chunk_summaries: List[Dict], max_length: Optional[int]) -> str:
        """
        Selective summarization: choose most important summaries.
        
        Args:
            chunk_summaries: List of chunk summaries
            max_length: Maximum length of final summary
            
        Returns:
            Selective summary
        """
        # Sort by summary length (longer summaries might be more important)
        sorted_summaries = sorted(chunk_summaries, key=lambda x: len(x['summary']), reverse=True)
        
        max_len = max_length or self.max_length
        selected_summaries = []
        current_length = 0
        
        for summary in sorted_summaries:
            summary_text = summary['summary']
            if current_length + len(summary_text) <= max_len:
                selected_summaries.append(summary_text)
                current_length += len(summary_text)
            else:
                break
        
        return " ".join(selected_summaries)
    
    def summarize_with_rag(self, 
                          query: str,
                          retrieved_chunks: List[Dict],
                          max_length: Optional[int] = None) -> str:
        """
        Generate summary using RAG (Retrieval-Augmented Generation).
        
        Args:
            query: Query or context for summarization
            retrieved_chunks: List of retrieved chunks with text and metadata
            max_length: Maximum length of summary
            
        Returns:
            RAG-based summary
        """
        if not retrieved_chunks:
            return ""
        
        # Combine query with retrieved context
        context_texts = [chunk.get('text', '') for chunk in retrieved_chunks]
        context = " ".join(context_texts)
        
        # Create prompt combining query and context
        if query:
            prompt = f"Query: {query}\n\nContext: {context}\n\nSummary:"
        else:
            prompt = f"Summarize the following text:\n\n{context}"
        
        # Generate summary
        return self.summarize_text(prompt, max_length)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and prepare text for summarization.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\n+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _post_process_summary(self, summary: str) -> str:
        """
        Post-process generated summary.
        
        Args:
            summary: Raw generated summary
            
        Returns:
            Processed summary
        """
        # Remove extra whitespace
        summary = re.sub(r'\s+', ' ', summary)
        
        # Ensure proper sentence endings
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary.strip()
    
    def get_model_info(self) -> Dict:
        """
        Get information about the summarization model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'max_length': self.max_length,
            'min_length': self.min_length,
            'batch_size': self.batch_size
        }
