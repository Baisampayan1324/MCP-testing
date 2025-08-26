"""
Streamlit Frontend for Research Paper Summarization System
Modern and beautiful UI for the RAG-based summarization system.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import time
import json
from typing import List, Dict
import sys
import os

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, clean styling
st.markdown("""
<style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Cards and containers */
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
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'index_path' not in st.session_state:
    st.session_state.index_path = None
if 'pipeline_initialized' not in st.session_state:
    st.session_state.pipeline_initialized = False

@st.cache(allow_output_mutation=True)
def get_pipeline():
    """Initialize and cache the RAG pipeline."""
    try:
        # Create temporary directory for index
        temp_dir = tempfile.mkdtemp()
        index_path = os.path.join(temp_dir, "research_paper_index")
        
        pipeline = RAGPipeline(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            summarization_model="facebook/bart-large-cnn",
            chunk_size=1000,
            chunk_overlap=200,
            index_path=index_path,
            device="cpu"  # Use CPU for demo
        )
        
        return pipeline, index_path
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        return None, None

def initialize_pipeline():
    """Initialize the RAG pipeline."""
    if not st.session_state.pipeline_initialized:
        pipeline, index_path = get_pipeline()
        if pipeline:
            st.session_state.pipeline = pipeline
            st.session_state.index_path = index_path
            st.session_state.pipeline_initialized = True
            return pipeline
    return st.session_state.pipeline

def main():
    """Main application function."""
    
    # Auto-initialize pipeline on first load
    if not st.session_state.pipeline_initialized:
        with st.spinner("� Initializing AI Pipeline..."):
            initialize_pipeline()
    
    # Clean header
    st.markdown('<h1 class="main-header">📚 Research Paper Summarizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered document analysis with advanced RAG technology</p>', unsafe_allow_html=True)
    
    # Show status
    if st.session_state.pipeline_initialized:
        st.markdown('<div class="status-success">✅ AI Pipeline Ready</div>', unsafe_allow_html=True)
    
    # Simplified sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Just show status, remove complex controls
        if st.session_state.pipeline:
            st.markdown("### 📊 Pipeline Status")
            st.write("🤖 **Embedding**: MiniLM-L6-v2")
            st.write("📝 **Summarizer**: BART-Large")
            st.write("💾 **Vector Store**: FAISS")
            
            if st.button("🔄 Reset Pipeline"):
                st.session_state.pipeline_initialized = False
                st.experimental_rerun()
    
    # Main content - simplified tabs
    tab1, tab2 = st.tabs(["📄 Upload & Process", "🔍 Query Documents"])
    
    with tab1:
        if not st.session_state.pipeline_initialized:
            st.info("🔄 Initializing pipeline...")
            return
            
        # Clean upload interface
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown("### Upload Research Papers")
        
        uploaded_files = st.file_uploader(
            "Choose PDF or text files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload research papers to analyze and summarize"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in [f['name'] for f in st.session_state.processed_files]:
                    with st.expander(f"� Processing: {file.name}", expanded=True):
                        with st.spinner(f"Processing {file.name}..."):
                            try:
                                # Save uploaded file temporarily
                                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1])
                                temp_file.write(file.read())
                                temp_file.close()
                                
                                # Process the document
                                result = st.session_state.pipeline.process_document(temp_file.name)
                                
                                # Store processed file info
                                st.session_state.processed_files.append({
                                    'name': file.name,
                                    'path': temp_file.name,
                                    'summary': result.get('final_summary', 'No summary available'),
                                    'chunks': len(result.get('chunk_summaries', [])),
                                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                                })
                                
                                # Clean up
                                os.unlink(temp_file.name)
                                
                                # Show results
                                st.success(f"✅ Successfully processed {file.name}")
                                
                                if 'final_summary' in result:
                                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                                    st.markdown("**📋 Document Summary:**")
                                    st.write(result['final_summary'])
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                st.info(f"📊 Processed {len(result.get('chunk_summaries', []))} text chunks")
                                
                            except Exception as e:
                                st.error(f"❌ Error processing {file.name}: {str(e)}")
        
        # Show processed files
        if st.session_state.processed_files:
            st.markdown("### � Processed Documents")
            for i, file_info in enumerate(st.session_state.processed_files):
                with st.expander(f"📄 {file_info['name']} (Processed: {file_info['timestamp']})"):
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.write(f"**Chunks:** {file_info['chunks']}")
                    st.write(f"**Summary:** {file_info['summary'][:200]}...")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Choose research paper files",
                type=['pdf', 'txt'],
                accept_multiple_files=True,
                help="Upload PDF or text files of research papers"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_files:
                st.markdown(f"**Uploaded {len(uploaded_files)} file(s):**")
                for file in uploaded_files:
                    st.write(f"📄 {file.name} ({file.size} bytes)")
                
                # Process button
                if st.button("🚀 Process Documents"):
                    with st.spinner("Processing documents..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        for i, uploaded_file in enumerate(uploaded_files):
                            status_text.text(f"Processing {uploaded_file.name}...")
                            
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            try:
                                # Process document
                                result = st.session_state.pipeline.process_document(tmp_file_path, save_index=True)
                                results.append(result)
                                
                                # Clean up temporary file
                                os.unlink(tmp_file_path)
                                
                                st.session_state.processed_files.append(uploaded_file.name)
                                
                            except Exception as e:
                                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        status_text.text("Processing completed!")
                        
                        # Display results
                        st.markdown('<h3>📋 Processing Results</h3>', unsafe_allow_html=True)
                        
                        for result in results:
                            with st.expander(f"📄 {Path(result['file_path']).name}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Chunks Generated", result['total_chunks'])
                                    st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                                
                                with col2:
                                    st.metric("Document Title", result['document_metadata'].get('title', 'Unknown'))
                                    st.metric("Pages", result['document_metadata'].get('page_count', 'Unknown'))
                                
                                # Display final summary
                                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                                st.markdown("**📝 Generated Summary:**")
                                st.write(result['final_summary'])
                                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Query and Analyze Documents</h2>', unsafe_allow_html=True)
        
        if not st.session_state.pipeline:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Pipeline Not Initialized</strong><br>
                Please initialize the pipeline and process some documents first.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Query interface
            st.markdown("### 🔍 Ask Questions About Your Documents")
            
            query = st.text_area(
                "Enter your question or topic of interest:",
                placeholder="e.g., What are the main findings about machine learning?",
                height=100
            )
            
            col1, col2 = st.columns(2)
            with col1:
                k_results = st.slider("Number of relevant chunks", 3, 15, 5)
            with col2:
                generate_summary = st.checkbox("Generate summary from results", value=True)
            
            if st.button("🔍 Search Documents") and query:
                with st.spinner("Searching documents..."):
                    try:
                        results = st.session_state.pipeline.query_documents(
                            query=query,
                            k=k_results,
                            generate_summary=generate_summary
                        )
                        
                        if results['success']:
                            # Display summary
                            if results['summary']:
                                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                                st.markdown("**📝 Generated Summary:**")
                                st.write(results['summary'])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Display retrieved chunks
                            st.markdown(f"**🔍 Retrieved {len(results['retrieved_chunks'])} relevant chunks:**")
                            
                            for i, chunk in enumerate(results['retrieved_chunks']):
                                with st.expander(f"Chunk {i+1} (Score: {chunk['score']:.3f})"):
                                    st.write("**Text:**")
                                    st.write(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                                    
                                    st.write("**Metadata:**")
                                    st.json(chunk['metadata'])
                        else:
                            st.error("No relevant content found or error occurred.")
                            
                    except Exception as e:
                        st.error(f"Error during search: {str(e)}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">System Analytics</h2>', unsafe_allow_html=True)
        
        if not st.session_state.pipeline:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Pipeline Not Initialized</strong><br>
                No analytics available without an initialized pipeline.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Pipeline information
            info = st.session_state.pipeline.get_pipeline_info()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Documents Processed", len(st.session_state.processed_files))
                st.metric("Total Vectors in Index", info['vector_store_stats']['total_vectors'])
                st.metric("Embedding Dimension", info['vector_store_stats']['embedding_dimension'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Chunk Size", info['text_splitter_config']['chunk_size'])
                st.metric("Chunk Overlap", info['text_splitter_config']['chunk_overlap'])
                st.metric("Index Type", info['vector_store_stats']['index_type'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Model information
            st.markdown("### 🤖 Model Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Embedding Model:**")
                st.json(info['embedding_model'])
            
            with col2:
                st.markdown("**Summarization Model:**")
                st.json(info['summarization_model'])
            
            # Processed files list
            if st.session_state.processed_files:
                st.markdown("### 📚 Processed Documents")
                for file in st.session_state.processed_files:
                    st.write(f"📄 {file}")
    
    with tab4:
        st.markdown('<h2 class="sub-header">About the System</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h3>🎯 Research Paper Summarization System</h3>
            <p>This system uses advanced AI techniques to analyze and summarize research papers:</p>
            <ul>
                <li><strong>RAG (Retrieval-Augmented Generation):</strong> Combines document retrieval with AI generation</li>
                <li><strong>Vector Search:</strong> Uses FAISS for efficient similarity search</li>
                <li><strong>Transformer Models:</strong> State-of-the-art language models for summarization</li>
                <li><strong>Modular Architecture:</strong> Scalable and extensible design</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h3>🔧 Technical Stack</h3>
            <ul>
                <li><strong>Frontend:</strong> Streamlit</li>
                <li><strong>AI Models:</strong> Hugging Face Transformers, SentenceTransformers</li>
                <li><strong>Vector Database:</strong> FAISS</li>
                <li><strong>PDF Processing:</strong> PyMuPDF</li>
                <li><strong>Language:</strong> Python 3.10+</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Features</h3>
            <ul>
                <li>📄 PDF and text document processing</li>
                <li>🔍 Intelligent text chunking with overlap</li>
                <li>🧠 Vector embedding generation and storage</li>
                <li>📝 Multi-level summarization (chunk + final)</li>
                <li>❓ Query-based document analysis</li>
                <li>💾 Persistent vector index storage</li>
                <li>📊 Real-time analytics and metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
