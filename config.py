"""
Configuration file for optimized deployment
"""
import os

# Lightweight model configurations for deployment
DEPLOYMENT_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # Small but effective
    "summarization_model": "sshleifer/distilbart-cnn-6-6",      # Lightweight BART
    "chunk_size": 800,                                          # Smaller chunks
    "chunk_overlap": 100,                                       # Less overlap
    "max_summary_length": 150,                                  # Shorter summaries
    "device": "cpu",                                            # Force CPU for deployment
    "batch_size": 1,                                           # Process one at a time
}

# Production optimizations
PRODUCTION_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "summarization_model": "facebook/bart-large-cnn",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_summary_length": 512,
    "device": "cpu",
    "batch_size": 4,
}

# Get config based on environment
def get_config():
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production" or env == "deployment":
        return DEPLOYMENT_CONFIG
    else:
        return PRODUCTION_CONFIG
