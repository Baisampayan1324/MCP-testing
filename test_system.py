"""
Simple test script to verify the Research Paper Summarization System components
"""

import sys
import os

def test_imports():
    """Test if all modules can be imported successfully."""
    print("🔍 Testing module imports...")
    
    try:
        from data_loader import DataLoader
        print("✅ DataLoader imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import DataLoader: {e}")
        return False
    
    try:
        from text_splitter import TextSplitter
        print("✅ TextSplitter imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TextSplitter: {e}")
        return False
    
    try:
        from embedding_service import EmbeddingService
        print("✅ EmbeddingService imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import EmbeddingService: {e}")
        return False
    
    try:
        from vector_store import VectorStore
        print("✅ VectorStore imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import VectorStore: {e}")
        return False
    
    try:
        from summarizer import Summarizer
        print("✅ Summarizer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Summarizer: {e}")
        return False
    
    try:
        from rag_pipeline import RAGPipeline
        print("✅ RAGPipeline imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RAGPipeline: {e}")
        return False
    
    return True

def test_data_loader():
    """Test the data loader with sample text."""
    print("\n📄 Testing DataLoader...")
    
    try:
        from data_loader import DataLoader
        
        # Create sample text file
        sample_text = "This is a test document for the research paper summarization system."
        with open("test_document.txt", "w") as f:
            f.write(sample_text)
        
        # Test loading
        loader = DataLoader()
        document = loader.load_document("test_document.txt")
        
        if document and document.get('content'):
            print("✅ DataLoader works correctly")
            os.remove("test_document.txt")
            return True
        else:
            print("❌ DataLoader failed to load document")
            return False
            
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False

def test_text_splitter():
    """Test the text splitter with sample text."""
    print("\n✂️ Testing TextSplitter...")
    
    try:
        from text_splitter import TextSplitter
        
        sample_text = "This is a test document. It contains multiple sentences. We will test the text splitting functionality."
        
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_text(sample_text, "test_doc")
        
        if chunks and len(chunks) > 0:
            print(f"✅ TextSplitter created {len(chunks)} chunks")
            return True
        else:
            print("❌ TextSplitter failed to create chunks")
            return False
            
    except Exception as e:
        print(f"❌ TextSplitter test failed: {e}")
        return False

def test_embedding_service():
    """Test the embedding service."""
    print("\n🧠 Testing EmbeddingService...")
    
    try:
        from embedding_service import EmbeddingService
        
        service = EmbeddingService(device="cpu")
        
        # Test single embedding
        text = "This is a test sentence."
        embedding = service.generate_embedding(text)
        
        if embedding is not None and len(embedding) > 0:
            print(f"✅ EmbeddingService generated embedding with dimension {len(embedding)}")
            return True
        else:
            print("❌ EmbeddingService failed to generate embedding")
            return False
            
    except Exception as e:
        print(f"❌ EmbeddingService test failed: {e}")
        return False

def test_vector_store():
    """Test the vector store."""
    print("\n💾 Testing VectorStore...")
    
    try:
        from vector_store import VectorStore
        import numpy as np
        
        # Create test embeddings
        embedding_dim = 384  # MiniLM dimension
        test_embeddings = [
            {
                'embedding': np.random.rand(embedding_dim).astype(np.float32),
                'chunk_id': 'test_1',
                'metadata': {'test': True}
            }
        ]
        
        store = VectorStore(embedding_dim=embedding_dim)
        store.add_embeddings(test_embeddings)
        
        stats = store.get_statistics()
        if stats['total_vectors'] > 0:
            print(f"✅ VectorStore added {stats['total_vectors']} vectors")
            return True
        else:
            print("❌ VectorStore failed to add vectors")
            return False
            
    except Exception as e:
        print(f"❌ VectorStore test failed: {e}")
        return False

def test_summarizer():
    """Test the summarizer with sample text."""
    print("\n📝 Testing Summarizer...")
    
    try:
        from summarizer import Summarizer
        
        sample_text = "This is a test document for summarization. It contains multiple sentences that should be summarized. The summarizer should generate a concise summary of this content."
        
        summarizer = Summarizer(device="cpu")
        summary = summarizer.summarize_text(sample_text, max_length=50)
        
        if summary and len(summary) > 0:
            print(f"✅ Summarizer generated summary: {summary[:50]}...")
            return True
        else:
            print("❌ Summarizer failed to generate summary")
            return False
            
    except Exception as e:
        print(f"❌ Summarizer test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Research Paper Summarization System - Component Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_loader,
        test_text_splitter,
        test_embedding_service,
        test_vector_store,
        test_summarizer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
