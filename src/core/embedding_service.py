"""
Embedding Service Module for Research Paper Summarization System
Handles generation of vector embeddings using sentence-transformers.
"""

import numpy as np
from typing import List, Dict, Union, Optional
from sentence_transformers import SentenceTransformer
import torch
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing text embeddings."""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 batch_size: int = 32):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on ('cpu', 'cuda', etc.)
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing embedding model: {model_name} on {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            # Generate embeddings in batches
            embeddings = []
            
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings.append(batch_embeddings)
            
            # Concatenate all batches
            all_embeddings = np.vstack(embeddings)
            
            logger.info(f"Generated embeddings with shape: {all_embeddings.shape}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Numpy array of embedding with shape (embedding_dim,)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding[0]  # Return single embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_chunk_embeddings(self, chunks: List) -> List[Dict]:
        """
        Generate embeddings for text chunks with metadata.
        
        Args:
            chunks: List of TextChunk objects or dictionaries with 'text' key
            
        Returns:
            List of dictionaries containing chunk info and embeddings
        """
        if not chunks:
            return []
        
        # Extract texts from chunks
        texts = []
        chunk_metadata = []
        
        for chunk in chunks:
            if hasattr(chunk, 'text'):
                # TextChunk object
                texts.append(chunk.text)
                chunk_metadata.append({
                    'chunk_id': chunk.chunk_id,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'metadata': chunk.metadata
                })
            elif isinstance(chunk, dict) and 'text' in chunk:
                # Dictionary with text
                texts.append(chunk['text'])
                chunk_metadata.append({
                    'chunk_id': chunk.get('chunk_id', ''),
                    'metadata': chunk.get('metadata', {})
                })
            else:
                logger.warning(f"Skipping chunk with invalid format: {chunk}")
                continue
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Combine with metadata
        results = []
        for i, (embedding, metadata) in enumerate(zip(embeddings, chunk_metadata)):
            result = {
                'embedding': embedding,
                'chunk_id': metadata['chunk_id'],
                'metadata': metadata['metadata']
            }
            
            # Add position info if available
            if 'start_index' in metadata:
                result['start_index'] = metadata['start_index']
                result['end_index'] = metadata['end_index']
            
            results.append(result)
        
        logger.info(f"Generated embeddings for {len(results)} chunks")
        return results
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError("Embeddings must have the same shape")
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings generated by this model.
        
        Returns:
            Embedding dimension
        """
        # Generate a dummy embedding to get the dimension
        dummy_embedding = self.generate_embedding("test")
        return dummy_embedding.shape[0]
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dimension': self.get_embedding_dimension(),
            'batch_size': self.batch_size
        }
