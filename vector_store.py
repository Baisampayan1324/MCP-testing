"""
Vector Store Module for Research Paper Summarization System
Handles FAISS-based storage and retrieval of embeddings with metadata.
"""

import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Union, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self, 
                 index_path: Optional[str] = None,
                 embedding_dim: Optional[int] = None,
                 index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            index_path: Path to save/load the FAISS index
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Initialize FAISS index
        self.index = None
        self.metadata_store = []
        self.chunk_texts = []
        
        # Load existing index if path is provided
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
        elif embedding_dim:
            self._create_index(embedding_dim)
    
    def _create_index(self, embedding_dim: int):
        """Create a new FAISS index."""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        elif self.index_type == "ivf":
            # IVF index with 100 clusters
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(embedding_dim, 32)  # 32 neighbors
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        logger.info(f"Created {self.index_type} index with dimension {embedding_dim}")
    
    def add_embeddings(self, 
                      embeddings: List[Dict],
                      normalize: bool = True) -> None:
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: List of dictionaries with 'embedding' and metadata
            normalize: Whether to normalize embeddings before adding
        """
        if not embeddings:
            return
        
        # Extract embeddings and metadata
        embedding_vectors = []
        metadata_list = []
        text_list = []
        
        for emb_data in embeddings:
            embedding = emb_data['embedding']
            
            # Normalize if requested
            if normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            embedding_vectors.append(embedding)
            metadata_list.append({
                'chunk_id': emb_data.get('chunk_id', ''),
                'metadata': emb_data.get('metadata', {}),
                'start_index': emb_data.get('start_index', 0),
                'end_index': emb_data.get('end_index', 0)
            })
            
            # Store chunk text if available
            if 'text' in emb_data:
                text_list.append(emb_data['text'])
            else:
                text_list.append('')
        
        # Convert to numpy array
        embedding_array = np.array(embedding_vectors, dtype=np.float32)
        
        # Add to FAISS index
        if self.index is None:
            self._create_index(embedding_array.shape[1])
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embedding_array)
        
        # Add vectors to index
        self.index.add(embedding_array)
        
        # Store metadata and texts
        self.metadata_store.extend(metadata_list)
        self.chunk_texts.extend(text_list)
        
        logger.info(f"Added {len(embeddings)} embeddings to vector store")
    
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 5,
               normalize: bool = True) -> List[Dict]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            normalize: Whether to normalize query embedding
            
        Returns:
            List of dictionaries with similarity scores and metadata
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Normalize query embedding if requested
        if normalize:
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
        
        # Reshape for FAISS
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata_store):
                result = {
                    'score': float(score),
                    'index': int(idx),
                    'metadata': self.metadata_store[idx].copy(),
                    'text': self.chunk_texts[idx] if idx < len(self.chunk_texts) else ''
                }
                results.append(result)
        
        return results
    
    def search_by_text(self, 
                      query_text: str, 
                      embedding_service,
                      k: int = 5) -> List[Dict]:
        """
        Search using text query by first generating embedding.
        
        Args:
            query_text: Text query
            embedding_service: EmbeddingService instance
            k: Number of results to return
            
        Returns:
            List of dictionaries with similarity scores and metadata
        """
        # Generate embedding for query text
        query_embedding = embedding_service.generate_embedding(query_text)
        
        # Search using embedding
        return self.search(query_embedding, k)
    
    def get_all_embeddings(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Get all stored embeddings and metadata.
        
        Returns:
            Tuple of (embeddings_array, metadata_list)
        """
        if self.index is None or self.index.ntotal == 0:
            return np.array([]), []
        
        # For flat indices, we can reconstruct embeddings
        if isinstance(self.index, faiss.IndexFlat):
            embeddings = self.index.reconstruct_n(0, self.index.ntotal)
            return embeddings, self.metadata_store.copy()
        else:
            # For other index types, we can't easily reconstruct
            logger.warning("Cannot reconstruct embeddings from non-flat index")
            return np.array([]), self.metadata_store.copy()
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        if self.index is None:
            return {
                'total_vectors': 0,
                'embedding_dimension': self.embedding_dim,
                'index_type': self.index_type
            }
        
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dimension': self.index.d,
            'index_type': self.index_type,
            'is_trained': getattr(self.index, 'is_trained', True)
        }
    
    def save_index(self, path: Optional[str] = None) -> None:
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            path: Path to save the index (uses self.index_path if None)
        """
        if self.index is None:
            logger.warning("No index to save")
            return
        
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No path specified for saving index")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{save_path}.faiss")
        
        # Save metadata and texts
        metadata_path = f"{save_path}.metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata_store,
                'texts': self.chunk_texts,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)
        
        logger.info(f"Saved index to {save_path}")
    
    def load_index(self, path: str) -> None:
        """
        Load the FAISS index and metadata from disk.
        
        Args:
            path: Path to the saved index
        """
        faiss_path = f"{path}.faiss"
        metadata_path = f"{path}.metadata.pkl"
        
        if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Index files not found at {path}")
        
        # Load FAISS index
        self.index = faiss.read_index(faiss_path)
        
        # Load metadata and texts
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata_store = data['metadata']
            self.chunk_texts = data['texts']
            self.embedding_dim = data.get('embedding_dim')
            self.index_type = data.get('index_type', 'flat')
        
        logger.info(f"Loaded index from {path} with {self.index.ntotal} vectors")
    
    def clear(self) -> None:
        """Clear all stored embeddings and metadata."""
        if self.index is not None:
            # Create a new empty index
            if self.embedding_dim:
                self._create_index(self.embedding_dim)
            else:
                self.index = None
        
        self.metadata_store = []
        self.chunk_texts = []
        
        logger.info("Cleared vector store")
    
    def remove_embeddings(self, chunk_ids: List[str]) -> None:
        """
        Remove specific embeddings by chunk IDs.
        Note: This is not efficient for large indices as FAISS doesn't support deletion.
        
        Args:
            chunk_ids: List of chunk IDs to remove
        """
        if not chunk_ids:
            return
        
        # Find indices to remove
        indices_to_remove = []
        for i, metadata in enumerate(self.metadata_store):
            if metadata.get('chunk_id') in chunk_ids:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            logger.warning("No matching chunk IDs found for removal")
            return
        
        # For flat indices, we can reconstruct and rebuild
        if isinstance(self.index, faiss.IndexFlat):
            # Get all embeddings
            all_embeddings = self.index.reconstruct_n(0, self.index.ntotal)
            
            # Remove specified indices
            mask = np.ones(len(all_embeddings), dtype=bool)
            mask[indices_to_remove] = False
            
            remaining_embeddings = all_embeddings[mask]
            remaining_metadata = [self.metadata_store[i] for i in range(len(self.metadata_store)) if i not in indices_to_remove]
            remaining_texts = [self.chunk_texts[i] for i in range(len(self.chunk_texts)) if i not in indices_to_remove]
            
            # Rebuild index
            self._create_index(remaining_embeddings.shape[1])
            self.index.add(remaining_embeddings)
            self.metadata_store = remaining_metadata
            self.chunk_texts = remaining_texts
            
            logger.info(f"Removed {len(indices_to_remove)} embeddings")
        else:
            logger.warning("Cannot remove embeddings from non-flat index")
