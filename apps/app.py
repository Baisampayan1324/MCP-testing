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

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .upload-area {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    
    .summary-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
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

def initialize_pipeline():
    """Initialize the RAG pipeline."""
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
        
        st.session_state.pipeline = pipeline
        st.session_state.index_path = index_path
        
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Research Paper Summarizer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Research Paper Analysis with RAG Technology</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ Configuration")
        
        # Model selection
        st.markdown("### Models")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
            index=0
        )
        
        summarization_model = st.selectbox(
            "Summarization Model",
            ["facebook/bart-large-cnn", "google/flan-t5-large"],
            index=0
        )
        
        # Pipeline parameters
        st.markdown("### Parameters")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        
        # Initialize pipeline button
        if st.button("🔄 Initialize Pipeline"):
            with st.spinner("Initializing pipeline..."):
                pipeline = initialize_pipeline()
                if pipeline:
                    st.success("Pipeline initialized successfully!")
        
        # Pipeline info
        if st.session_state.pipeline:
            st.markdown("### Pipeline Status")
            info = st.session_state.pipeline.get_pipeline_info()
            
            st.metric("Embedding Model", info['embedding_model']['model_name'].split('/')[-1])
            st.metric("Summarization Model", info['summarization_model']['model_name'].split('/')[-1])
            st.metric("Total Vectors", info['vector_store_stats']['total_vectors'])
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload & Process", "🔍 Query Documents", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Upload and Process Research Papers</h2>', unsafe_allow_html=True)
        
        if not st.session_state.pipeline:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Pipeline Not Initialized</strong><br>
                Please initialize the pipeline from the sidebar before processing documents.
            </div>
            """, unsafe_allow_html=True)
        else:
            # File upload
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
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
