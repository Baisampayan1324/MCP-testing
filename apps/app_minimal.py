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
        'description': 'Fast & Balanced',
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
    },
    'all-mpnet-base-v2': {
        'description': 'High Quality',
        'model_name': 'sentence-transformers/all-mpnet-base-v2'
    },
    'all-distilroberta-v1': {
        'description': 'Fast Processing',
        'model_name': 'sentence-transformers/all-distilroberta-v1'
    },
    'paraphrase-multilingual-MiniLM-L12-v2': {
        'description': 'Multilingual Support',
        'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    }
}

SUMMARIZATION_MODELS = {
    'facebook/bart-large-cnn': {
        'description': 'High Quality Summaries',
        'model_name': 'facebook/bart-large-cnn'
    },
    't5-small': {
        'description': 'Fast Processing',
        'model_name': 't5-small'
    },
    'google/pegasus-xsum': {
        'description': 'Concise Summaries',
        'model_name': 'google/pegasus-xsum'
    },
    'sshleifer/distilbart-cnn-12-6': {
        'description': 'Balanced Performance',
        'model_name': 'sshleifer/distilbart-cnn-12-6'
    }
}

def get_minimal_styles():
    """Get clean, minimal CSS styles with subtle animations."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #fafafa;
            color: #1a1a1a;
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 800px;
            animation: fadeInUp 0.6s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-10px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        /* Clean header with subtle animation */
        .minimal-header {
            font-size: 2rem;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
            animation: fadeIn 0.8s ease-out;
        }
        
        .minimal-subtitle {
            color: #6b7280;
            font-size: 1rem;
            font-weight: 400;
            margin-bottom: 3rem;
            line-height: 1.5;
            animation: fadeIn 1s ease-out 0.2s both;
        }
        
        /* Clean cards with hover animation */
        .minimal-card {
            background: #ffffff;
            border-radius: 8px;
            padding: 1.5rem;
            border: 1px solid #e5e7eb;
            margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
            animation: slideIn 0.5s ease-out;
        }
        
        .minimal-card:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            transform: translateY(-2px);
        }
        
        /* Simple buttons with smooth animations */
        .stButton > button {
            background-color: #1a1a1a;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.5s;
        }
        
        .stButton > button:hover {
            background-color: #374151;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Clean tabs with smooth transitions */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 1px solid #e5e7eb;
            background: transparent;
            padding: 0;
            animation: fadeIn 0.6s ease-out;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #6b7280;
            border-radius: 0;
            padding: 1rem 0;
            margin-right: 2rem;
            border-bottom: 2px solid transparent;
            font-weight: 500;
            transition: all 0.3s ease;
            position: relative;
        }
        
        .stTabs [data-baseweb="tab"]::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 0;
            height: 2px;
            background-color: #1a1a1a;
            transition: width 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: #1a1a1a;
            transform: translateY(-1px);
        }
        
        .stTabs [data-baseweb="tab"]:hover::after {
            width: 50%;
        }
        
        .stTabs [aria-selected="true"] {
            color: #1a1a1a !important;
            border-bottom-color: #1a1a1a !important;
        }
        
        .stTabs [aria-selected="true"]::after {
            width: 100% !important;
        }
        
        /* Clean inputs with focus animations */
        .stTextInput > div > div > input {
            border: 1px solid #d1d5db;
            border-radius: 6px;
            padding: 0.75rem;
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #1a1a1a;
            box-shadow: 0 0 0 3px rgba(26, 26, 26, 0.1);
            transform: translateY(-1px);
        }
        
        /* Clean selectbox with hover effect */
        .stSelectbox > div > div {
            border: 1px solid #d1d5db;
            border-radius: 6px;
            transition: all 0.3s ease;
        }
        
        .stSelectbox > div > div:hover {
            border-color: #9ca3af;
            transform: translateY(-1px);
        }
        
        /* File uploader animation */
        .uploadedFile {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 1rem;
            transition: all 0.3s ease;
            animation: slideIn 0.5s ease-out;
        }
        
        .uploadedFile:hover {
            background: #f3f4f6;
            transform: translateY(-1px);
        }
        
        /* Status messages with entrance animation */
        .status-success {
            background: #f0fdf4;
            color: #166534;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid #bbf7d0;
            margin: 1rem 0;
            animation: slideIn 0.5s ease-out;
        }
        
        .status-info {
            background: #f0f9ff;
            color: #1e40af;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid #bfdbfe;
            margin: 1rem 0;
            animation: slideIn 0.5s ease-out;
        }
        
        .status-warning {
            background: #fffbeb;
            color: #92400e;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid #fed7aa;
            margin: 1rem 0;
            animation: slideIn 0.5s ease-out;
        }
        
        /* Animated progress bar */
        .stProgress > div > div {
            background: #1a1a1a;
            transition: width 0.3s ease;
        }
        
        .stProgress > div {
            border-radius: 6px;
            overflow: hidden;
        }
        
        /* Processing indicator animation */
        .processing-indicator {
            display: inline-block;
            animation: pulse 1.5s infinite;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Sidebar styling */
        .css-1d391kg {
            background: #f9fafb;
            border-right: 1px solid #e5e7eb;
        }
        
        /* Clean expander */
        .streamlit-expanderHeader {
            font-weight: 500;
            color: #1a1a1a;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 600;
            color: #1a1a1a;
            letter-spacing: -0.025em;
        }
        
        p {
            color: #4b5563;
            line-height: 1.6;
        }
        
        /* Stats grid with staggered animation */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .stat-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            animation: slideIn 0.6s ease-out;
        }
        
        .stat-card:nth-child(1) { animation-delay: 0.1s; }
        .stat-card:nth-child(2) { animation-delay: 0.2s; }
        .stat-card:nth-child(3) { animation-delay: 0.3s; }
        .stat-card:nth-child(4) { animation-delay: 0.4s; }
        
        .stat-card:hover {
            transform: translateY(-4px) scale(1.02);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 0.5rem;
            transition: all 0.3s ease;
        }
        
        .stat-card:hover .stat-number {
            transform: scale(1.1);
            color: #374151;
        }
        
        .stat-label {
            font-size: 0.875rem;
            color: #6b7280;
            font-weight: 500;
        }
        
        /* Sidebar animations */
        .css-1d391kg {
            background: #f9fafb;
            border-right: 1px solid #e5e7eb;
            animation: slideIn 0.5s ease-out;
        }
        
        /* Clean expander with hover effect */
        .streamlit-expanderHeader {
            font-weight: 500;
            color: #1a1a1a;
            transition: all 0.3s ease;
            border-radius: 6px;
            padding: 0.5rem;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #f9fafb;
            transform: translateX(5px);
        }
        
        /* Loading states */
        .loading-dots {
            display: inline-block;
        }
        
        .loading-dots::after {
            content: '';
            animation: loadingDots 1.5s infinite;
        }
        
        @keyframes loadingDots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60% { content: '...'; }
            80%, 100% { content: ''; }
        }
        
        /* Smooth transitions for all interactive elements */
        * {
            transition: color 0.3s ease, background-color 0.3s ease, border-color 0.3s ease, 
                       box-shadow 0.3s ease, transform 0.3s ease, opacity 0.3s ease;
        }
        
        /* Animation for content updates */
        .content-update {
            animation: fadeInUp 0.4s ease-out;
        }
        
        /* Hover effect for cards containing buttons */
        .minimal-card:hover .stButton > button {
            background-color: #374151;
        }
    </style>
    """

def create_simple_stats():
    """Create simple statistics display."""
    if 'processed_files' in st.session_state and st.session_state.processed_files:
        total_docs = len(st.session_state.processed_files)
        total_chunks = sum(file.get('chunks', 0) for file in st.session_state.processed_files)
        
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{total_docs}</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{total_chunks}</div>
                <div class="stat-label">Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{st.session_state.get('embedding_model', 'MiniLM')[:6]}</div>
                <div class="stat-label">Model</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def simple_sidebar():
    """Create a clean, minimal sidebar."""
    with st.sidebar:
        st.markdown("### Settings")
        
        # Model selection
        st.markdown("**Embedding Model**")
        embedding_options = list(EMBEDDING_MODELS.keys())
        selected_embedding = st.selectbox(
            "Choose embedding model:",
            options=embedding_options,
            index=0 if st.session_state.get('embedding_model') not in EMBEDDING_MODELS else embedding_options.index(st.session_state.get('embedding_model', 'all-MiniLM-L6-v2'))
        )
        
        if selected_embedding != st.session_state.get('embedding_model'):
            st.session_state.embedding_model = selected_embedding
        
        st.markdown("**Summarization Model**")
        summarization_options = list(SUMMARIZATION_MODELS.keys())
        selected_summarization = st.selectbox(
            "Choose summarization model:",
            options=summarization_options,
            index=0 if st.session_state.get('summarization_model') not in SUMMARIZATION_MODELS else summarization_options.index(st.session_state.get('summarization_model', 'facebook/bart-large-cnn'))
        )
        
        if selected_summarization != st.session_state.get('summarization_model'):
            st.session_state.summarization_model = selected_summarization
        
        st.markdown("---")
        
        # Model info
        st.markdown("**Current Setup**")
        st.markdown(f"Embedding: {EMBEDDING_MODELS[st.session_state.get('embedding_model', 'all-MiniLM-L6-v2')]['description']}")
        st.markdown(f"Summary: {SUMMARIZATION_MODELS[st.session_state.get('summarization_model', 'facebook/bart-large-cnn')]['description']}")
        
        st.markdown("---")
        
        if st.button("Clear All Data", key="sidebar_clear"):
            clear_session_data()

def export_tab():
    """Simple export functionality."""
    st.markdown('<div class="minimal-card">', unsafe_allow_html=True)
    st.markdown("### Export Data")
    
    if 'last_query_result' not in st.session_state or not st.session_state.get('processed_files'):
        st.markdown("""
        <div class="status-warning">
            No data available for export. Process documents and run queries first.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**JSON Export**")
        st.markdown("Complete analysis data in JSON format")
        
        if st.button("Download JSON", key="json_export"):
            export_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "documents": st.session_state.processed_files,
                "query_results": st.session_state.last_query_result,
                "models": {
                    "embedding": st.session_state.get('embedding_model'),
                    "summarization": st.session_state.get('summarization_model')
                }
            }
            
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download",
                data=json_data,
                file_name=f"analysis_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        st.markdown("**CSV Export**")
        st.markdown("Summary data in spreadsheet format")
        
        if st.button("Download CSV", key="csv_export"):
            csv_data = "Document,Date,Summary,Model\n"
            for file_info in st.session_state.processed_files:
                summary_clean = file_info.get('summary', '').replace('\n', ' ').replace('"', '""')
                csv_data += f'"{file_info["name"]}","{file_info["timestamp"]}","{summary_clean}","{st.session_state.get("summarization_model", "")}"\n'
            
            st.download_button(
                label="Download",
                data=csv_data.encode('utf-8'),
                file_name=f"summary_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Research Paper Summarizer",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = 'all-MiniLM-L6-v2'
    if 'summarization_model' not in st.session_state:
        st.session_state.summarization_model = 'facebook/bart-large-cnn'
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    # Apply minimal styles
    st.markdown(get_minimal_styles(), unsafe_allow_html=True)
    
    # Sidebar
    simple_sidebar()
    
    # Main header
    st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem;">
            <h1 class="minimal-header">Research Paper Summarizer</h1>
            <p class="minimal-subtitle">AI-powered document analysis with semantic search and summarization</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Stats
    create_simple_stats()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Process", "Query", "Export"])
    
    with tab1:
        st.markdown('<div class="minimal-card">', unsafe_allow_html=True)
        
        st.markdown("### Upload Documents")
        st.markdown("Select PDF files to process and analyze")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type=['pdf'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if st.button("Process Documents", key="process_docs"):
                    process_documents(uploaded_files)
            with col2:
                chunk_size = st.slider("Chunk Size", 200, 2000, 1000)
            with col3:
                if st.button("Clear All", key="main_clear"):
                    clear_session_data()
        
        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} files ready for processing**")
        
        # Show processed files
        if st.session_state.processed_files:
            st.markdown("### Processed Documents")
            
            for file_info in st.session_state.processed_files:
                with st.expander(f"{file_info['name']} • {file_info.get('chunks', 0)} chunks • {file_info['timestamp']}"):
                    st.markdown("**Summary:**")
                    st.write(file_info.get('summary', 'Processing...'))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        if not st.session_state.processed_files:
            st.markdown("""
            <div class="minimal-card" style="text-align: center; padding: 3rem;">
                <h3>No Documents Processed</h3>
                <p>Upload and process documents first to enable querying.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        st.markdown('<div class="minimal-card">', unsafe_allow_html=True)
        
        st.markdown("### Ask Questions")
        st.markdown("Enter your research question to get AI-powered answers")
        
        query = st.text_input(
            "Your question:",
            placeholder="What are the main findings of this research?"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Search & Analyze", key="search_summarize", disabled=not query):
                if query:
                    answer_query(query)
        with col2:
            k_results = st.slider("Results", 3, 10, 5)
        
        # Show results
        if 'last_query_result' in st.session_state:
            result = st.session_state.last_query_result
            
            st.markdown("---")
            st.markdown("### Results")
            
            st.markdown(f"""
            <div class="status-info">
                <strong>Query:</strong> {result.get('query', '')}<br>
                <strong>Time:</strong> {result.get('timestamp', '')}<br>
                <strong>Model:</strong> {result.get('model_info', {}).get('summarization', '')}
            </div>
            """, unsafe_allow_html=True)
            
            if 'summary' in result:
                st.markdown("**Answer:**")
                st.write(result['summary'])
            
            if 'relevant_chunks' in result:
                with st.expander("Source Context", expanded=False):
                    for i, chunk in enumerate(result['relevant_chunks'], 1):
                        score = chunk.get('score', 0) if isinstance(chunk, dict) else 0
                        text = chunk.get('text', str(chunk)) if isinstance(chunk, dict) else str(chunk)
                        
                        st.markdown(f"**Source {i}** (Score: {score:.3f})")
                        st.write(text[:500] + ('...' if len(text) > 500 else ''))
                        st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        export_tab()

def process_documents(uploaded_files):
    """Process documents with animated progress indication."""
    try:
        progress_bar = st.progress(0)
        status_container = st.container()
        
        with status_container:
            status_text = st.empty()
            
            # Animated status messages
            status_text.markdown("""
            <div class="processing-indicator" style="background: #f0f9ff; padding: 1rem; border-radius: 6px; border: 1px solid #bfdbfe;">
                🧠 <span class="loading-dots">Initializing models</span>
            </div>
            """, unsafe_allow_html=True)
            
        embedding_service = EmbeddingService(
            model_name=EMBEDDING_MODELS[st.session_state.embedding_model]['model_name']
        )
        progress_bar.progress(20)
        time.sleep(0.5)  # Brief pause for visual feedback
        
        status_text.markdown("""
        <div class="processing-indicator" style="background: #f0f9ff; padding: 1rem; border-radius: 6px; border: 1px solid #bfdbfe;">
            📄 <span class="loading-dots">Processing documents</span>
        </div>
        """, unsafe_allow_html=True)
        
        all_chunks = []
        processed_summaries = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            loader = DataLoader()
            text = loader.load_pdf_from_bytes(uploaded_file.getvalue())
            
            splitter = TextSplitter()
            chunks = splitter.split_text(text)
            chunk_texts = [chunk.text if hasattr(chunk, 'text') else str(chunk) for chunk in chunks]
            all_chunks.extend(chunk_texts)
            
            summarizer = Summarizer(
                model_name=SUMMARIZATION_MODELS[st.session_state.summarization_model]['model_name']
            )
            doc_summary = summarizer.summarize_text(text[:2000])
            
            processed_summaries.append({
                'name': uploaded_file.name,
                'summary': doc_summary,
                'chunks': len(chunk_texts),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            progress = 20 + (i + 1) * (50 / len(uploaded_files))
            progress_bar.progress(int(progress))
        
        status_text.markdown("""
        <div class="processing-indicator" style="background: #f0f9ff; padding: 1rem; border-radius: 6px; border: 1px solid #bfdbfe;">
            🔬 <span class="loading-dots">Creating vector embeddings</span>
        </div>
        """, unsafe_allow_html=True)
        
        vector_store = VectorStore()
        
        embeddings_data = []
        for i, chunk_text in enumerate(all_chunks):
            embedding = embedding_service.generate_embedding(chunk_text)
            embeddings_data.append({
                'embedding': embedding,
                'text': chunk_text,
                'chunk_id': f"chunk_{i}",
                'metadata': {'chunk_index': i}
            })
        
        vector_store.add_embeddings(embeddings_data)
        progress_bar.progress(90)
        time.sleep(0.3)
        
        st.session_state.vector_store = vector_store
        st.session_state.processed_files = processed_summaries
        st.session_state.total_chunks = len(all_chunks)
        
        progress_bar.progress(100)
        
        # Animated success message
        status_text.markdown(f"""
        <div class="content-update status-success">
            ✅ Successfully processed {len(uploaded_files)} documents into {len(all_chunks)} chunks
        </div>
        """, unsafe_allow_html=True)
        
        # Brief pause before showing success
        time.sleep(0.5)
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")

def answer_query(query):
    """Process query with animated progress indication."""
    try:
        if 'vector_store' not in st.session_state:
            st.error("Please process documents first!")
            return
        
        progress_bar = st.progress(0)
        status_container = st.container()
        
        with status_container:
            status_text = st.empty()
            
            status_text.markdown("""
            <div class="processing-indicator" style="background: #f0f9ff; padding: 1rem; border-radius: 6px; border: 1px solid #bfdbfe;">
                🔍 <span class="loading-dots">Searching documents</span>
            </div>
            """, unsafe_allow_html=True)
            
        embedding_service = EmbeddingService(
            model_name=EMBEDDING_MODELS[st.session_state.embedding_model]['model_name']
        )
        relevant_chunks = st.session_state.vector_store.search_by_text(query, embedding_service, k=5)
        progress_bar.progress(50)
        time.sleep(0.3)
        
        if not relevant_chunks:
            status_text.markdown("""
            <div class="status-warning">
                ⚠️ No relevant information found. Try rephrasing your question.
            </div>
            """, unsafe_allow_html=True)
            return
        
        status_text.markdown("""
        <div class="processing-indicator" style="background: #f0f9ff; padding: 1rem; border-radius: 6px; border: 1px solid #bfdbfe;">
            🤖 <span class="loading-dots">Generating summary</span>
        </div>
        """, unsafe_allow_html=True)
        
        summarizer = Summarizer(
            model_name=SUMMARIZATION_MODELS[st.session_state.summarization_model]['model_name']
        )
        summary = summarizer.summarize_with_rag(query, relevant_chunks)
        progress_bar.progress(90)
        time.sleep(0.3)
        
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
        
        status_text.markdown("""
        <div class="content-update status-success">
            ✅ Analysis complete! Results are ready below.
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

def clear_session_data():
    """Clear all session data."""
    keys_to_clear = ['vector_store', 'processed_files', 'total_chunks', 'last_query_result']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("All data cleared successfully!")
    time.sleep(1)
    st.experimental_rerun()

if __name__ == "__main__":
    main()
