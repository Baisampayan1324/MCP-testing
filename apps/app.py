import streamlit as st
import sys
import os
import tempfile
import time
import json

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_loader import DataLoader
from src.core.text_splitter import TextSplitter
from src.core.embedding_service import EmbeddingService
from src.core.vector_store import VectorStore
from src.core.summarizer import Summarizer

# Model configurations
EMBEDDING_MODELS = {
    'all-MiniLM-L6-v2': {
        'description': 'Fast & Balanced - Good speed/quality ratio',
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
    },
    'all-mpnet-base-v2': {
        'description': 'High Quality - Better accuracy, slower',
        'model_name': 'sentence-transformers/all-mpnet-base-v2'
    },
    'all-distilroberta-v1': {
        'description': 'Fast - Quick processing, good for large docs',
        'model_name': 'sentence-transformers/all-distilroberta-v1'
    },
    'paraphrase-multilingual-MiniLM-L12-v2': {
        'description': 'Multilingual - Supports 50+ languages',
        'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    }
}

SUMMARIZATION_MODELS = {
    'facebook/bart-large-cnn': {
        'description': 'High Quality - Best for detailed summaries',
        'model_name': 'facebook/bart-large-cnn'
    },
    't5-small': {
        'description': 'Fast - Quick processing, good quality',
        'model_name': 't5-small'
    },
    'google/pegasus-xsum': {
        'description': 'Concise - Very short, focused summaries',
        'model_name': 'google/pegasus-xsum'
    },
    'sshleifer/distilbart-cnn-12-6': {
        'description': 'Balanced - Good speed/quality for general use',
        'model_name': 'sshleifer/distilbart-cnn-12-6'
    },
    'philschmid/bart-large-cnn-samsum': {
        'description': 'Conversational - Best for Q&A style content',
        'model_name': 'philschmid/bart-large-cnn-samsum'
    }
}

def get_theme_styles():
    """Get CSS styles based on current theme."""
    if st.session_state.theme == 'dark':
        return """
        <style>
            /* Dark theme styles */
            .stApp {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1200px;
                background-color: #1e1e1e;
                color: #ffffff;
            }
            
            .main-header {
                font-size: 2.5rem;
                font-weight: 700;
                text-align: center;
                margin: 1rem 0;
                padding: 1rem;
                background: linear-gradient(90deg, #64b5f6 0%, #ab47bc 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            }
            
            .subtitle {
                text-align: center;
                color: #b0bec5;
                font-size: 1.1rem;
                margin-bottom: 2rem;
            }
            
            .upload-card {
                background: #2d2d2d;
                border-radius: 12px;
                padding: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                border: 1px solid #404040;
                margin: 1rem 0;
            }
            
            .result-card {
                background: #2d2d2d;
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                border-left: 4px solid #64b5f6;
                color: #ffffff;
            }
            
            .status-success {
                background: #1b5e20;
                color: #c8e6c9;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                border-left: 4px solid #4caf50;
                margin: 1rem 0;
            }
            
            .status-info {
                background: #0d47a1;
                color: #bbdefb;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                border-left: 4px solid #2196f3;
                margin: 1rem 0;
            }
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            .stButton > button {
                background: linear-gradient(90deg, #64b5f6 0%, #ab47bc 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            }
            
            /* Streamlit sidebar styling */
            .css-1d391kg {
                background-color: #2d2d2d;
            }
            
            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: #2d2d2d;
                color: #ffffff;
                border-radius: 8px;
                padding: 0.5rem 1rem;
            }
        </style>
        """
    else:
        return """
        <style>
            /* Light theme styles */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }
            
            .main-header {
                font-size: 2.5rem;
                font-weight: 700;
                text-align: center;
                margin: 1rem 0;
                padding: 1rem;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            
            .subtitle {
                text-align: center;
                color: #7f8c8d;
                font-size: 1.1rem;
                margin-bottom: 2rem;
            }
            
            .upload-card {
                background: white;
                border-radius: 12px;
                padding: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
                border: 1px solid #e9ecef;
                margin: 1rem 0;
            }
            
            .result-card {
                background: #f8f9fa;
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                border-left: 4px solid #667eea;
            }
            
            .status-success {
                background: #d1f2eb;
                color: #00695c;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                border-left: 4px solid #26a69a;
                margin: 1rem 0;
            }
            
            .status-info {
                background: #e3f2fd;
                color: #1565c0;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                border-left: 4px solid #42a5f5;
                margin: 1rem 0;
            }
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            .stButton > button {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            
            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 0.5rem 1rem;
            }
        </style>
        """

def export_data_tab():
    """Export data tab content."""
    st.markdown("### 💾 Export Your Data")
    
    if 'last_query_result' not in st.session_state or not st.session_state.get('processed_files'):
        st.info("📄 Process documents and run queries to enable data export")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 JSON Export")
        if st.button("📥 Download JSON Report", key="json_export"):
            # Create comprehensive JSON export
            export_data = {
                "export_info": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_documents": len(st.session_state.processed_files),
                    "app_version": "Research Paper Summarizer v3.0",
                    "models_used": {
                        "embedding": st.session_state.embedding_model,
                        "summarization": st.session_state.summarization_model
                    }
                },
                "processed_documents": st.session_state.processed_files,
                "query_results": st.session_state.last_query_result
            }
            
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Complete Report",
                data=json_data,
                file_name=f"research_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        st.markdown("#### 📊 CSV Export")
        if st.button("📈 Download CSV Summary", key="csv_export"):
            # Create CSV export
            csv_data = "Document,Processing_Date,Summary,Model_Used\n"
            for file_info in st.session_state.processed_files:
                summary_clean = file_info.get('summary', 'No summary').replace('\n', ' ').replace('"', '""')
                csv_data += f'"{file_info["name"]}","{file_info["timestamp"]}","{summary_clean}","{st.session_state.summarization_model}"\n'
            
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_data.encode('utf-8'),
                file_name=f"research_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def main():
    st.set_page_config(
        page_title="Research Paper Summarizer",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = 'all-MiniLM-L6-v2'
    if 'summarization_model' not in st.session_state:
        st.session_state.summarization_model = 'facebook/bart-large-cnn'
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    # Apply theme styles
    st.markdown(get_theme_styles(), unsafe_allow_html=True)
    
    # Theme toggle
    with st.container():
        col1, col2, col3 = st.columns([6, 1, 1])
        with col3:
            if st.button("🌓", help="Toggle Theme"):
                st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
                st.experimental_rerun()
    
    # Main header with better visibility
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="main-header">📚 Research Paper Summarizer</h1>
            <p class="subtitle">Upload your research papers and get AI-powered summaries and insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model selection and settings
    with st.sidebar:
        st.markdown("### ⚙️ Model Settings")
        
        # Embedding model selection
        st.markdown("#### 🔍 Embedding Model")
        selected_embedding = st.selectbox(
            "Choose embedding model:",
            options=list(EMBEDDING_MODELS.keys()),
            index=list(EMBEDDING_MODELS.keys()).index(st.session_state.embedding_model),
            format_func=lambda x: f"{x} - {EMBEDDING_MODELS[x]['description']}",
            help="Embedding models convert text to numerical representations for similarity search"
        )
        
        if selected_embedding != st.session_state.embedding_model:
            st.session_state.embedding_model = selected_embedding
            if 'vector_store' in st.session_state:
                del st.session_state.vector_store  # Reset vector store when model changes
        
        # Summarization model selection
        st.markdown("#### 📝 Summarization Model")
        selected_summarization = st.selectbox(
            "Choose summarization model:",
            options=list(SUMMARIZATION_MODELS.keys()),
            index=list(SUMMARIZATION_MODELS.keys()).index(st.session_state.summarization_model),
            format_func=lambda x: f"{x} - {SUMMARIZATION_MODELS[x]['description']}",
            help="Summarization models generate concise summaries from longer text"
        )
        
        if selected_summarization != st.session_state.summarization_model:
            st.session_state.summarization_model = selected_summarization
        
        # Model info
        st.markdown("---")
        st.markdown("#### ℹ️ Current Models")
        st.info(f"**Embedding:** {EMBEDDING_MODELS[st.session_state.embedding_model]['description']}")
        st.info(f"**Summarization:** {SUMMARIZATION_MODELS[st.session_state.summarization_model]['description']}")
        
        # Clear data option
        st.markdown("---")
        if st.button("🗑️ Clear All Data", key="sidebar_clear"):
            clear_session_data()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["📄 Document Processing", "💾 Export Data"])
    
    with tab1:
        # File upload section
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type=['pdf'], 
            accept_multiple_files=True,
            help="Upload one or more PDF research papers"
        )
        
        if uploaded_files:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Process Documents", key="process_docs"):
                    process_documents(uploaded_files)
            with col2:
                if st.button("🗑️ Clear All Data", key="main_clear"):
                    clear_session_data()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Processing status
        if uploaded_files:
            st.markdown('<div class="status-info">📁 Files ready for processing</div>', unsafe_allow_html=True)
        
        # Query section
        if 'vector_store' in st.session_state and st.session_state.vector_store is not None:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Ask Questions About Your Documents")
            
            # Query input
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., What are the main findings of this research?"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔍 Search & Summarize", key="search_summarize"):
                    if query:
                        answer_query(query)
                    else:
                        st.warning("Please enter a question first.")
            
            # Display results
            if 'last_query_result' in st.session_state:
                st.markdown("#### 📋 Results:")
                result = st.session_state.last_query_result
                
                if 'summary' in result:
                    st.markdown("**Summary:**")
                    st.write(result['summary'])
                
                if 'relevant_chunks' in result:
                    with st.expander("📄 Relevant Text Sections", expanded=False):
                        for i, chunk in enumerate(result['relevant_chunks'], 1):
                            st.markdown(f"**Section {i}:**")
                            st.write(chunk)
                            st.markdown("---")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        export_data_tab()

def process_documents(uploaded_files):
    """Process uploaded documents with improved error handling."""
    try:
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        # Initialize components with selected models
        status_placeholder.markdown('<div class="status-info">🔧 Initializing AI models...</div>', unsafe_allow_html=True)
        progress_bar.progress(10)
        
        # Load selected models
        embedding_service = EmbeddingService(model_name=EMBEDDING_MODELS[st.session_state.embedding_model]['model_name'])
        progress_bar.progress(25)
        
        # Process documents
        status_placeholder.markdown('<div class="status-info">📖 Processing documents...</div>', unsafe_allow_html=True)
        all_chunks = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Extract text from PDF
            loader = DataLoader()
            text = loader.load_pdf_from_bytes(uploaded_file.getvalue())
            
            # Split text into chunks
            splitter = TextSplitter()
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
            
            progress = 25 + (i + 1) * (50 / len(uploaded_files))
            progress_bar.progress(int(progress))
        
        # Create vector store
        status_placeholder.markdown('<div class="status-info">🔍 Creating vector database...</div>', unsafe_allow_html=True)
        vector_store = VectorStore()
        
        # Generate embeddings for all chunks
        embeddings_data = []
        for i, chunk in enumerate(all_chunks):
            # Extract text from TextChunk object
            chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            embedding = embedding_service.generate_embedding(chunk_text)
            embeddings_data.append({
                'embedding': embedding,
                'text': chunk_text,
                'chunk_id': f"chunk_{i}",
                'metadata': {'chunk_index': i}
            })
        
        # Add embeddings to vector store
        vector_store.add_embeddings(embeddings_data)
        st.session_state.vector_store = vector_store
        progress_bar.progress(90)
        
        # Store document info
        st.session_state.processed_files = [
            {
                'name': f.name,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'chunks': len(all_chunks) // len(uploaded_files)  # Approximate chunks per file
            } for f in uploaded_files
        ]
        st.session_state.total_chunks = len(all_chunks)
        
        progress_bar.progress(100)
        status_placeholder.markdown(
            f'<div class="status-success">✅ Successfully processed {len(uploaded_files)} documents into {len(all_chunks)} text chunks!</div>', 
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        st.info("💡 Try: 1) Check PDF files are not corrupted 2) Ensure sufficient memory 3) Try smaller files")
        
def answer_query(query):
    """Answer user query with better error handling."""
    try:
        if 'vector_store' not in st.session_state:
            st.error("Please process documents first!")
            return
        
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        # Search for relevant chunks
        status_placeholder.markdown('<div class="status-info">🔍 Searching relevant content...</div>', unsafe_allow_html=True)
        progress_bar.progress(30)
        
        embedding_service = EmbeddingService(model_name=EMBEDDING_MODELS[st.session_state.embedding_model]['model_name'])
        relevant_chunks = st.session_state.vector_store.search_by_text(query, embedding_service, k=5)
        progress_bar.progress(60)
        
        if not relevant_chunks:
            st.warning("No relevant information found. Try rephrasing your question or check if documents were processed correctly.")
            st.info("💡 Tips: Use specific keywords, ask about main topics, or try broader questions")
            return
        
        # Generate summary
        status_placeholder.markdown('<div class="status-info">📝 Generating summary...</div>', unsafe_allow_html=True)
        summarizer = Summarizer(model_name=SUMMARIZATION_MODELS[st.session_state.summarization_model]['model_name'])
        
        # The relevant_chunks are already in the right format (list of dicts with 'text' keys)
        summary = summarizer.summarize_with_rag(query, relevant_chunks)
        progress_bar.progress(90)
        
        # Store results
        result = {
            'query': query,
            'summary': summary,
            'relevant_chunks': relevant_chunks,
            'model_info': {
                'embedding': st.session_state.embedding_model,
                'summarization': st.session_state.summarization_model
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.last_query_result = result
        progress_bar.progress(100)
        status_placeholder.markdown('<div class="status-success">✅ Analysis complete!</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        st.info("💡 Try: 1) Check if documents are processed correctly 2) Rephrase your question 3) Use simpler queries")

def clear_session_data():
    """Clear all session data."""
    keys_to_clear = ['vector_store', 'processed_files', 'total_chunks', 'last_query_result']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.success("✅ All data cleared!")
    st.experimental_rerun()

if __name__ == "__main__":
    main()
