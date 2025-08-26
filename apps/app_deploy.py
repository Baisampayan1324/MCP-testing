"""
Streamlit App for Deployment - Optimized Version
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import time
import json
from typing import List, Dict
import sys

# Set environment for deployment
os.environ["ENVIRONMENT"] = "deployment"

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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

@st.cache(allow_output_mutation=True)
def initialize_pipeline():
    """Initialize the RAG pipeline with caching."""
    try:
        pipeline = RAGPipeline(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            summarization_model="facebook/bart-large-cnn",  # Use standard BART
            chunk_size=800,
            chunk_overlap=100,
            device="cpu"
        )
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Research Paper Summarizer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #666;">AI-Powered Research Paper Analysis</p>', unsafe_allow_html=True)
    
    # Performance warning
    st.markdown("""
    <div class="warning-box">
        <strong>⚡ Performance Note:</strong> This is a lightweight deployment version. 
        Processing may take 30-60 seconds per document. For better performance, consider local deployment.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ Configuration")
        
        # Initialize pipeline
        if st.button("🔄 Initialize Pipeline"):
            with st.spinner("Initializing pipeline..."):
                pipeline = initialize_pipeline()
                if pipeline:
                    st.session_state.pipeline = pipeline
                    st.success("Pipeline initialized!")
        
        # Show pipeline status
        if st.session_state.pipeline:
            st.success("✅ Pipeline Ready")
        else:
            st.warning("⚠️ Pipeline Not Initialized")
    
    # Main content
    if st.session_state.pipeline is None:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Getting Started</h3>
            <p>Click <strong>"Initialize Pipeline"</strong> in the sidebar to start.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # File upload
    st.markdown("## 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose research paper files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Maximum 5MB per file for deployment version"
    )
    
    if uploaded_files:
        # File size check
        max_size = 5 * 1024 * 1024  # 5MB
        valid_files = []
        
        for file in uploaded_files:
            if file.size > max_size:
                st.error(f"❌ {file.name} is too large ({file.size/1024/1024:.1f}MB). Maximum 5MB allowed.")
            else:
                valid_files.append(file)
                st.write(f"📄 {file.name} ({file.size/1024:.1f}KB)")
        
        if valid_files and st.button("🚀 Process Documents"):
            results = []
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(valid_files):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process document
                    result = st.session_state.pipeline.process_document(tmp_file_path)
                    
                    # Add the filename to the result
                    if result.get('success', True):
                        result['title'] = uploaded_file.name
                    
                    results.append(result)
                    
                    # Clean up
                    os.unlink(tmp_file_path)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(valid_files))
            
            # Display results
            st.markdown("## 📋 Processing Results")
            for result in results:
                # The actual process_document doesn't return 'success' field, so we check if result exists
                if result and 'total_chunks' in result:
                    title = result.get('title', Path(result.get('file_path', 'Document')).stem)
                    with st.expander(f"📄 {title}", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Chunks", result['total_chunks'])
                        with col2:
                            st.metric("Time (s)", f"{result['processing_time']:.1f}")
                        with col3:
                            # Get vector count from vector_store_stats if available
                            vector_count = result.get('vector_store_stats', {}).get('total_vectors', 'N/A')
                            st.metric("Vectors", vector_count)
                        
                        if result.get('final_summary'):
                            st.markdown("**📝 Summary:**")
                            st.markdown(result['final_summary'])
                else:
                    st.error(f"❌ Failed to process document: {result.get('error', 'Unknown error')}")
            
            # Update processed files list
            successful_files = [r.get('title', Path(r.get('file_path', 'Document')).stem) 
                              for r in results if r and 'total_chunks' in r]
            st.session_state.processed_files.extend(successful_files)
    
    # Query interface
    if st.session_state.processed_files:
        st.markdown("## 🔍 Query Documents")
        
        query = st.text_input("Ask a question about your documents:")
        
        col1, col2 = st.columns(2)
        with col1:
            k_results = st.slider("Number of relevant chunks", 3, 10, 5)
        with col2:
            generate_summary = st.checkbox("Generate summary", value=True)
        
        if st.button("🔍 Search") and query:
            with st.spinner("Searching..."):
                results = st.session_state.pipeline.query_documents(
                    query=query,
                    k=k_results,
                    generate_summary=generate_summary
                )
                
                if results.get('success', True):
                    if generate_summary and 'summary' in results:
                        st.markdown("### 📝 Summary")
                        st.markdown(results['summary'])
                    
                    st.markdown("### 🔍 Retrieved Chunks")
                    retrieved_chunks = results.get('retrieved_chunks', [])
                    for i, chunk in enumerate(retrieved_chunks):
                        score = chunk.get('score', 'N/A')
                        text = chunk.get('text', chunk.get('content', 'No content'))
                        with st.expander(f"Chunk {i+1} (Score: {score:.3f})" if isinstance(score, (int, float)) else f"Chunk {i+1}"):
                            st.markdown(text)
                else:
                    st.error(f"Search failed: {results.get('error', 'Unknown error')}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and Transformers")

if __name__ == "__main__":
    main()
