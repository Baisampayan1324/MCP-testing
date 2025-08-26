# 📚 Research Paper Summarization System

A complete RAG (Retrieval-Augmented Generation) system for research paper summarization with modular architecture and cloud deployment support.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run locally (full models):**
   ```bash
   cd apps
   streamlit run app.py
   ```

3. **Run deployment version (optimized):**
   ```bash
   cd apps
   streamlit run app_deploy.py
   ```

## 📁 Project Structure

```
📁 Research Paper Summarization System/
├── 📁 src/core/           # Core RAG pipeline modules
│   ├── data_loader.py     # File processing & text extraction
│   ├── text_splitter.py   # Intelligent text chunking
│   ├── embedding_service.py # Vector embeddings
│   ├── vector_store.py    # FAISS similarity search
│   └── summarizer.py      # BART summarization
├── 📁 apps/               # Streamlit applications
│   ├── app.py            # Full-featured local app
│   └── app_deploy.py     # Lightweight deployment app
├── 📁 deployment/         # Cloud deployment configs
│   ├── Procfile          # Heroku configuration
│   ├── render.yaml       # Render.com configuration
│   ├── requirements-deploy.txt # Optimized dependencies
│   └── runtime.txt       # Python version
├── 📁 configs/            # Configuration files
├── 📁 tests/              # Unit tests
├── 📁 docs/               # Documentation
│   ├── PROJECT_STRUCTURE.md
│   └── DEPLOYMENT.md
├── requirements.txt       # Main dependencies
└── README.md             # This file
```

## ✨ Features

- 📄 PDF and text file processing
- 🔍 Advanced text chunking and embedding
- 🧠 FAISS vector store for similarity search
- 📝 BART-based summarization
- 🌐 Streamlit web interface
- ☁️ Cloud deployment ready

## 🏗️ Architecture

```
Data Loader → Text Splitter → Embedding Service → Vector Store
                                    ↓
Summarizer ← RAG Pipeline ← Query Interface
```

## 📖 Documentation

- [Project Structure](docs/PROJECT_STRUCTURE.md) - Detailed folder organization
- [Deployment Guide](docs/DEPLOYMENT.md) - Cloud deployment instructions

## ⚡ Performance

- **Local version** (`app.py`): Full models for best quality
- **Deploy version** (`app_deploy.py`): Optimized lightweight models (3x faster)

## 🛠️ Usage

1. Upload a research paper (PDF or text)
2. Wait for processing and indexing
3. Ask questions about the content
4. Get relevant summaries and answers

Choose the appropriate app version based on your needs!
