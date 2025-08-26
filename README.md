# 📚 Research Paper Summarization System

AI-powered research paper analysis using **Modular Component-Based (MCB) pipelines** with **Retrieval-Augmented Generation (RAG)** techniques.

## 🎯 Features

- 📄 **Multi-format Input**: PDF and text file processing
- 🔍 **Intelligent Chunking**: Smart text segmentation with overlap
- 🧠 **Vector Embeddings**: State-of-the-art sentence transformers
- 💾 **FAISS Vector Store**: Efficient similarity search
- 📝 **Multi-level Summarization**: Chunk + document summaries
- ❓ **RAG-based Q&A**: Query documents with context
- 🎨 **Beautiful Streamlit UI**: Modern web interface

## 🏗️ Architecture

```
Data Loader → Text Splitter → Embedding Service → Vector Store
                                    ↓
Summarizer ← RAG Pipeline ← Query Interface
```

## 🛠️ Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the web app**:
```bash
streamlit run app.py
```

3. **Or run the demo**:
```bash
python example.py
```

## 🚀 Quick Start

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    summarization_model="facebook/bart-large-cnn"
)

# Process document
result = pipeline.process_document("paper.pdf")
print(result['final_summary'])

# Query documents
query_result = pipeline.query_documents("What are the main findings?")
print(query_result['summary'])
```

## 📦 Modules

- `data_loader.py`: PDF/text parsing
- `text_splitter.py`: Intelligent chunking
- `embedding_service.py`: Vector embeddings
- `vector_store.py`: FAISS storage
- `summarizer.py`: Multi-level summarization
- `rag_pipeline.py`: Main orchestrator
- `app.py`: Streamlit frontend

## 🔧 Configuration

- **Embedding Models**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`
- **Summarization Models**: `bart-large-cnn`, `flan-t5-large`
- **Chunk Size**: 500-2000 characters
- **Device**: CPU/CUDA (auto-detected)

## 📊 Performance

- **10 pages**: ~30 seconds, ~2GB RAM
- **50 pages**: ~2 minutes, ~4GB RAM
- **100 pages**: ~5 minutes, ~6GB RAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Built with ❤️ for the research community**
