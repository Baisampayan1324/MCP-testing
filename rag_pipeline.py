"""
RAG Pipeline Module for Research Paper Summarization System
Main orchestrator that combines all components for end-to-end processing.
"""

import os
import logging
from typing import List, Dict, Union, Optional
from pathlib import Path
import time

from data_loader import DataLoader
from text_splitter import TextSplitter, TextChunk
from embedding_service import EmbeddingService
from vector_store import VectorStore
from summarizer import Summarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline for research paper summarization."""
    
    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 summarization_model: str = "facebook/bart-large-cnn",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 index_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Name of the embedding model
            summarization_model: Name of the summarization model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            index_path: Path to save/load vector index
            device: Device to run models on
        """
        self.embedding_model = embedding_model
        self.summarization_model = summarization_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = index_path
        self.device = device
        
        # Initialize components
        self._initialize_components()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Initialize data loader
            self.data_loader = DataLoader()
            logger.info("Data loader initialized")
            
            # Initialize text splitter
            self.text_splitter = TextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info("Text splitter initialized")
            
            # Initialize embedding service
            self.embedding_service = EmbeddingService(
                model_name=self.embedding_model,
                device=self.device
            )
            logger.info("Embedding service initialized")
            
            # Initialize vector store
            embedding_dim = self.embedding_service.get_embedding_dimension()
            self.vector_store = VectorStore(
                index_path=self.index_path,
                embedding_dim=embedding_dim
            )
            logger.info("Vector store initialized")
            
            # Initialize summarizer
            self.summarizer = Summarizer(
                model_name=self.summarization_model,
                device=self.device
            )
            logger.info("Summarizer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def process_document(self, 
                        file_path: Union[str, Path],
                        save_index: bool = True) -> Dict:
        """
        Process a single document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            save_index: Whether to save the vector index
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Step 1: Load document
            document = self.data_loader.load_document(file_path)
            if not self.data_loader.validate_document(document):
                raise ValueError("Document validation failed")
            
            # Step 2: Split into chunks
            chunks = self.text_splitter.split_document(document)
            if not chunks:
                raise ValueError("No chunks generated from document")
            
            # Step 3: Generate embeddings
            chunk_embeddings = self.embedding_service.generate_chunk_embeddings(chunks)
            
            # Step 4: Add to vector store
            self.vector_store.add_embeddings(chunk_embeddings)
            
            # Step 5: Generate summaries
            chunk_summaries = self.summarizer.summarize_chunks(chunks)
            
            # Step 6: Generate final summary
            final_summary = self.summarizer.generate_final_summary(
                chunk_summaries,
                strategy="hierarchical"
            )
            
            # Step 7: Save index if requested
            if save_index and self.index_path:
                self.vector_store.save_index()
            
            processing_time = time.time() - start_time
            
            # Compile results
            results = {
                'file_path': str(file_path),
                'document_metadata': document.get('metadata', {}),
                'total_chunks': len(chunks),
                'chunk_summaries': chunk_summaries,
                'final_summary': final_summary,
                'processing_time': processing_time,
                'chunk_statistics': self.text_splitter.get_chunk_statistics(chunks),
                'vector_store_stats': self.vector_store.get_statistics()
            }
            
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def process_multiple_documents(self, 
                                  file_paths: List[Union[str, Path]],
                                  save_index: bool = True) -> List[Dict]:
        """
        Process multiple documents through the pipeline.
        
        Args:
            file_paths: List of document file paths
            save_index: Whether to save the vector index
            
        Returns:
            List of processing results for each document
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_document(file_path, save_index=False)
                results.append(result)
                logger.info(f"Successfully processed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                results.append({
                    'file_path': str(file_path),
                    'error': str(e),
                    'success': False
                })
        
        # Save index once after processing all documents
        if save_index and self.index_path:
            self.vector_store.save_index()
        
        return results
    
    def query_documents(self, 
                       query: str,
                       k: int = 5,
                       generate_summary: bool = True) -> Dict:
        """
        Query the processed documents using RAG.
        
        Args:
            query: Query text
            k: Number of top chunks to retrieve
            generate_summary: Whether to generate a summary from retrieved chunks
            
        Returns:
            Dictionary with query results and optional summary
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.vector_store.search_by_text(
                query, 
                self.embedding_service, 
                k=k
            )
            
            if not retrieved_chunks:
                return {
                    'query': query,
                    'retrieved_chunks': [],
                    'summary': "No relevant content found.",
                    'success': False
                }
            
            # Step 2: Generate summary if requested
            summary = ""
            if generate_summary:
                summary = self.summarizer.summarize_with_rag(
                    query, 
                    retrieved_chunks
                )
            
            results = {
                'query': query,
                'retrieved_chunks': retrieved_chunks,
                'summary': summary,
                'success': True,
                'num_chunks_retrieved': len(retrieved_chunks)
            }
            
            logger.info(f"Query processing completed")
            return results
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'query': query,
                'error': str(e),
                'success': False
            }
    
    def generate_document_summary(self, 
                                 file_path: Union[str, Path],
                                 query: Optional[str] = None) -> Dict:
        """
        Generate a summary for a document, optionally guided by a query.
        
        Args:
            file_path: Path to the document
            query: Optional query to guide summarization
            
        Returns:
            Dictionary with summary results
        """
        try:
            # Process document if not already in vector store
            if not self._document_in_store(file_path):
                self.process_document(file_path, save_index=True)
            
            if query:
                # Use RAG-based summarization
                return self.query_documents(query, k=10, generate_summary=True)
            else:
                # Generate general summary
                # Retrieve chunks from the document
                doc_chunks = self._get_document_chunks(file_path)
                if not doc_chunks:
                    return {
                        'file_path': str(file_path),
                        'summary': "No content found in document.",
                        'success': False
                    }
                
                # Generate summary from all chunks
                chunk_summaries = self.summarizer.summarize_chunks(doc_chunks)
                final_summary = self.summarizer.generate_final_summary(
                    chunk_summaries,
                    strategy="hierarchical"
                )
                
                return {
                    'file_path': str(file_path),
                    'summary': final_summary,
                    'chunk_summaries': chunk_summaries,
                    'success': True
                }
                
        except Exception as e:
            logger.error(f"Error generating summary for {file_path}: {str(e)}")
            return {
                'file_path': str(file_path),
                'error': str(e),
                'success': False
            }
    
    def _document_in_store(self, file_path: Union[str, Path]) -> bool:
        """Check if a document is already in the vector store."""
        file_path_str = str(file_path)
        
        # Check metadata store for document
        for metadata in self.vector_store.metadata_store:
            if metadata.get('metadata', {}).get('file_path') == file_path_str:
                return True
        
        return False
    
    def _get_document_chunks(self, file_path: Union[str, Path]) -> List[Dict]:
        """Get all chunks for a specific document."""
        file_path_str = str(file_path)
        chunks = []
        
        for i, metadata in enumerate(self.vector_store.metadata_store):
            if metadata.get('metadata', {}).get('file_path') == file_path_str:
                chunk_data = {
                    'text': self.vector_store.chunk_texts[i],
                    'chunk_id': metadata.get('chunk_id', ''),
                    'metadata': metadata.get('metadata', {})
                }
                chunks.append(chunk_data)
        
        return chunks
    
    def get_pipeline_info(self) -> Dict:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            'embedding_model': self.embedding_service.get_model_info(),
            'summarization_model': self.summarizer.get_model_info(),
            'text_splitter_config': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            },
            'vector_store_stats': self.vector_store.get_statistics(),
            'index_path': self.index_path,
            'device': self.device
        }
    
    def clear_index(self) -> None:
        """Clear the vector store index."""
        self.vector_store.clear()
        logger.info("Vector store index cleared")
    
    def save_index(self) -> None:
        """Save the current vector store index."""
        if self.index_path:
            self.vector_store.save_index()
            logger.info(f"Index saved to {self.index_path}")
        else:
            logger.warning("No index path specified for saving")
    
    def load_index(self) -> None:
        """Load the vector store index."""
        if self.index_path and os.path.exists(f"{self.index_path}.faiss"):
            self.vector_store.load_index(self.index_path)
            logger.info(f"Index loaded from {self.index_path}")
        else:
            logger.warning("No existing index found to load")
