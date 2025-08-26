"""
Optimized RAG Pipeline for Deployment
Lightweight version with better performance
"""

import os
import logging
from typing import List, Dict, Union, Optional
from pathlib import Path
import time
import tempfile

from data_loader import DataLoader
from text_splitter import TextSplitter, TextChunk
from embedding_service import EmbeddingService
from vector_store import VectorStore
from summarizer import Summarizer
from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedRAGPipeline:
    """Optimized RAG pipeline for deployment."""
    
    def __init__(self, **kwargs):
        """Initialize optimized pipeline with deployment config."""
        self.config = get_config()
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        
        self.data_loader = None
        self.text_splitter = None
        self.embedding_service = None
        self.vector_store = None
        self.summarizer = None
        self._index_path = kwargs.get('index_path') or tempfile.mkdtemp()
        
        logger.info("Optimized RAG Pipeline initializing...")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Data loader
            self.data_loader = DataLoader()
            logger.info("Data loader initialized")
            
            # Text splitter
            self.text_splitter = TextSplitter(
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"]
            )
            logger.info("Text splitter initialized")
            
            # Embedding service
            self.embedding_service = EmbeddingService(
                model_name=self.config["embedding_model"],
                device=self.config["device"]
            )
            logger.info("Embedding service initialized")
            
            # Vector store
            embedding_dim = self.embedding_service.get_embedding_dimension()
            self.vector_store = VectorStore(
                dimension=embedding_dim,
                index_path=self._index_path
            )
            logger.info("Vector store initialized")
            
            # Summarizer (lazy loading)
            self.summarizer = None
            logger.info("Pipeline initialized successfully (summarizer will load on demand)")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def _ensure_summarizer(self):
        """Lazy load summarizer only when needed."""
        if self.summarizer is None:
            logger.info("Loading summarizer on demand...")
            self.summarizer = Summarizer(
                model_name=self.config["summarization_model"],
                device=self.config["device"],
                max_length=self.config["max_summary_length"]
            )
            logger.info("Summarizer loaded")
    
    def process_document(self, file_path: str, title: str = None) -> Dict:
        """Process document with optimizations."""
        start_time = time.time()
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Load document
            text_content = self.data_loader.load_document(file_path)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text_content)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Generate embeddings (batch process)
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_service.encode_batch(texts)
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            
            # Add to vector store
            metadata = [{"chunk_id": i, "title": title or "document"} 
                       for i in range(len(chunks))]
            self.vector_store.add_embeddings(embeddings, texts, metadata)
            logger.info(f"Added {len(embeddings)} embeddings to vector store")
            
            # Generate summaries (only if requested)
            chunk_summaries = []
            if len(chunks) <= 10:  # Only for smaller documents
                self._ensure_summarizer()
                chunk_summaries = self.summarizer.summarize_chunks(chunks)
                logger.info(f"Generated {len(chunk_summaries)} chunk summaries")
            
            # Generate final summary
            final_summary = ""
            if chunk_summaries:
                final_summary = self.summarizer.generate_final_summary(chunk_summaries)
            else:
                # Use first few chunks for summary if too many chunks
                sample_chunks = chunks[:3]
                self._ensure_summarizer()
                final_summary = self.summarizer.generate_final_summary(
                    [chunk.content for chunk in sample_chunks]
                )
            
            processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            
            return {
                "success": True,
                "title": title or Path(file_path).stem,
                "total_chunks": len(chunks),
                "chunk_summaries": chunk_summaries,
                "final_summary": final_summary,
                "processing_time": processing_time,
                "total_vectors": self.vector_store.get_total_vectors()
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def query_documents(self, query: str, k: int = 5, generate_summary: bool = True) -> Dict:
        """Query documents with optimizations."""
        start_time = time.time()
        logger.info(f"Processing query: {query}")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode([query])[0]
            
            # Search similar chunks
            results = self.vector_store.search(query_embedding, k=k)
            
            # Prepare response
            response = {
                "success": True,
                "query": query,
                "retrieved_chunks": len(results),
                "results": results,
                "processing_time": time.time() - start_time
            }
            
            # Generate summary if requested and summarizer is available
            if generate_summary and results:
                try:
                    self._ensure_summarizer()
                    context_text = " ".join([result["text"] for result in results])
                    summary = self.summarizer.summarize_text(
                        f"Query: {query}\nContext: {context_text}"
                    )
                    response["summary"] = summary
                except Exception as e:
                    logger.warning(f"Could not generate summary: {e}")
                    response["summary"] = "Summary generation failed"
            
            logger.info("Query processing completed")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "embedding_model": self.config["embedding_model"],
            "summarization_model": self.config["summarization_model"],
            "total_vectors": self.vector_store.get_total_vectors() if self.vector_store else 0,
            "embedding_dimension": self.embedding_service.get_embedding_dimension() if self.embedding_service else 0,
            "config": self.config
        }
