# 📁 Project Structure Guide

## 🏗️ **Organized Folder Structure**

```
MCP/                                    # Root project directory
├── 📁 src/                            # Source code (core system)
│   ├── 📄 __init__.py                 # Package initialization
│   ├── 📄 rag_pipeline.py             # Main RAG orchestrator
│   └── 📁 core/                       # Core modules
│       ├── 📄 __init__.py             # Core package init
│       ├── 📄 data_loader.py          # PDF/text loading
│       ├── 📄 text_splitter.py        # Document chunking
│       ├── 📄 embedding_service.py    # Vector embeddings
│       ├── 📄 vector_store.py         # FAISS vector storage
│       └── 📄 summarizer.py           # Text summarization
│
├── 📁 apps/                           # Applications & interfaces
│   ├── 📄 app.py                      # Main Streamlit app (full-featured)
│   └── 📄 app_deploy.py               # Deployment-optimized app
│
├── 📁 deployment/                     # Cloud deployment files
│   ├── 📄 requirements-deploy.txt     # Lightweight dependencies
│   ├── 📄 Procfile                    # Heroku configuration
│   ├── 📄 render.yaml                 # Render.com configuration
│   └── 📄 runtime.txt                 # Python version
│
├── 📁 configs/                        # Configuration files
│   └── 📄 config.py                   # Environment configurations
│
├── 📁 tests/                          # Testing & validation
│   ├── 📄 test_deployment.py          # Deployment readiness tests
│   └── 📄 test_system.py              # System functionality tests
│
├── 📁 docs/                           # Documentation
│   └── 📄 DEPLOYMENT.md               # Deployment guide
│
├── 📁 venv/                           # Virtual environment (ignored)
├── 📄 .gitignore                      # Git ignore rules
├── 📄 README.md                       # Project overview
└── 📄 requirements.txt                # Development dependencies
```

## 🎯 **Folder Purposes**

### 📁 **`src/`** - Core System
- **Purpose**: Main source code, modular and reusable
- **Contains**: Core RAG pipeline components
- **Usage**: `from src.rag_pipeline import RAGPipeline`

### 📁 **`src/core/`** - Core Modules  
- **Purpose**: Individual system components
- **Contains**: Data processing, embeddings, summarization
- **Usage**: Internal imports within the system

### 📁 **`apps/`** - User Interfaces
- **Purpose**: Different application interfaces
- **Contains**: Streamlit apps, CLI tools, APIs
- **Usage**: Run applications from here

### 📁 **`deployment/`** - Cloud Deployment
- **Purpose**: Cloud platform configurations
- **Contains**: Platform-specific config files
- **Usage**: Copy to root when deploying

### 📁 **`configs/`** - Configuration Management
- **Purpose**: Environment and model configurations
- **Contains**: Settings for different environments
- **Usage**: Centralized configuration management

### 📁 **`tests/`** - Quality Assurance
- **Purpose**: Testing and validation scripts
- **Contains**: Unit tests, integration tests
- **Usage**: Ensure system reliability

### 📁 **`docs/`** - Documentation
- **Purpose**: Project documentation and guides
- **Contains**: Deployment guides, API docs
- **Usage**: Reference and setup instructions

## 🚀 **How to Use Each Folder**

### **For Development:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run main app (from project root)
streamlit run apps/app.py

# Run deployment app
streamlit run apps/app_deploy.py
```

### **For Testing:**
```bash
# Test system functionality
python tests/test_system.py

# Test deployment readiness
python tests/test_deployment.py
```

### **For Deployment:**
```bash
# Copy deployment files to root
cp deployment/* .

# Deploy to cloud platform
# (Follow docs/DEPLOYMENT.md)
```

### **For Configuration:**
```bash
# Edit environment settings
nano configs/config.py

# Set environment variables
export ENVIRONMENT=production
```

## 📦 **Import Patterns**

### **From Root Directory:**
```python
# Main pipeline
from src.rag_pipeline import RAGPipeline

# Core components
from src.core import DataLoader, TextSplitter, EmbeddingService

# Configuration
from configs.config import get_config
```

### **Within Apps:**
```python
# In apps/app.py
import sys
sys.path.append('..')  # Add parent directory
from src.rag_pipeline import RAGPipeline
```

## 🔧 **Benefits of This Structure**

✅ **Clean Separation**: Core logic separated from apps  
✅ **Easy Deployment**: Dedicated deployment folder  
✅ **Modular Design**: Individual components in core/  
✅ **Professional Layout**: Industry-standard structure  
✅ **Easy Testing**: Dedicated tests folder  
✅ **Clear Documentation**: Centralized docs  
✅ **Configuration Management**: Environment-specific configs  

## 🎯 **Quick Start Commands**

```bash
# 1. Development
cd MCP
python -m streamlit run apps/app.py

# 2. Testing  
python tests/test_system.py

# 3. Deployment Prep
cp deployment/requirements-deploy.txt .
cp deployment/Procfile .

# 4. Deploy
git add . && git commit -m "Deploy" && git push
```

**This structure makes your project production-ready and easily maintainable!** 🌟
