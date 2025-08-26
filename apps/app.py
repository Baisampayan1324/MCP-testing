import streamlit as st
import tempfile
import os
import time
import sys
import json
import io

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'pipeline_initialized' not in st.session_state:
    st.session_state.pipeline_initialized = False
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def get_theme_styles():
    """Get CSS styles based on current theme."""
    if st.session_state.theme == 'dark':
        return """
        <style>
            /* Dark theme styles */
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 1200px;
                background-color: #1e1e1e;
                color: #ffffff;
            }
            
            .stApp {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            
            .main-header {
                font-size: 2.5rem;
                font-weight: 700;
                text-align: center;
                margin-bottom: 0.5rem;
                background: linear-gradient(90deg, #64b5f6 0%, #ab47bc 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
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
            
            .theme-toggle {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 1000;
                background: #404040;
                border-radius: 50px;
                padding: 0.5rem;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
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
            
            .download-btn {
                background: linear-gradient(90deg, #4caf50 0%, #45a049 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.4rem 0.8rem;
                font-size: 0.9rem;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .download-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 3px 6px rgba(0, 0, 0, 0.3);
            }
        </style>
        """
    else:
        return """
        <style>
            /* Light theme styles */
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }
            
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
            
            .theme-toggle {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 1000;
                background: white;
                border-radius: 50px;
                padding: 0.5rem;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
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
            
            .download-btn {
                background: linear-gradient(90deg, #4caf50 0%, #45a049 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.4rem 0.8rem;
                font-size: 0.9rem;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .download-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
            }
        </style>
        """

def create_download_data(processed_files):
    """Create downloadable data from processed files."""
    download_data = {
        "export_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_documents": len(processed_files),
            "app_version": "Research Paper Summarizer v2.0"
        },
        "documents": []
    }
    
    for file_info in processed_files:
        doc_data = {
            "filename": file_info['name'],
            "processing_timestamp": file_info['timestamp'],
            "summary": file_info['summary'],
            "total_chunks": file_info['chunks'],
            "metadata": {
                "processed_by": "RAG Pipeline",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "summarization_model": "facebook/bart-large-cnn"
            }
        }
        download_data["documents"].append(doc_data)
    
    return json.dumps(download_data, indent=2, ensure_ascii=False)

@st.cache(allow_output_mutation=True)
def get_pipeline():
    """Initialize and cache the RAG pipeline."""
    try:
        temp_dir = tempfile.mkdtemp()
        index_path = os.path.join(temp_dir, "research_paper_index")
        
        pipeline = RAGPipeline(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            summarization_model="facebook/bart-large-cnn",
            chunk_size=1000,
            chunk_overlap=200,
            index_path=index_path,
            device="cpu"
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
    
    # Apply theme styles
    st.markdown(get_theme_styles(), unsafe_allow_html=True)
    
    # Theme toggle in header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col3:
        theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
        if st.button(theme_icon, help="Toggle theme"):
            st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
            st.experimental_rerun()
    
    # Auto-initialize pipeline
    if not st.session_state.pipeline_initialized:
        with st.spinner("🚀 Initializing AI Pipeline..."):
            initialize_pipeline()
    
    # Clean header
    st.markdown('<h1 class="main-header">📚 Research Paper Summarizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered document analysis with advanced RAG technology</p>', unsafe_allow_html=True)
    
    # Status indicator
    if st.session_state.pipeline_initialized:
        st.markdown('<div class="status-success">✅ AI Pipeline Ready</div>', unsafe_allow_html=True)
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        if st.session_state.pipeline:
            st.write("🤖 **Embedding**: MiniLM-L6-v2")
            st.write("📝 **Summarizer**: BART-Large")
            st.write("💾 **Vector Store**: FAISS")
            
            st.markdown("---")
            
            # Download section
            if st.session_state.processed_files:
                st.markdown("### 📥 Export Data")
                
                download_data = create_download_data(st.session_state.processed_files)
                
                st.download_button(
                    label="📄 Download Summary Report",
                    data=download_data,
                    file_name=f"research_summary_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Download processed documents and summaries as JSON"
                )
                
                # Create CSV for simple viewing
                csv_data = "Filename,Processing Date,Chunks,Summary\n"
                for file_info in st.session_state.processed_files:
                    summary_clean = file_info['summary'].replace('\n', ' ').replace('"', '""')
                    csv_data += f'"{file_info["name"]}","{file_info["timestamp"]}",{file_info["chunks"]},"{summary_clean}"\n'
                
                st.download_button(
                    label="📊 Download CSV Report",
                    data=csv_data.encode('utf-8'),
                    file_name=f"research_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download as CSV for Excel/Sheets"
                )
            
            st.markdown("---")
            
            if st.button("🔄 Reset Pipeline"):
                st.session_state.pipeline_initialized = False
                st.experimental_rerun()
    
    # Main content - two clean tabs
    tab1, tab2 = st.tabs(["📄 Upload & Process", "🔍 Query Documents"])
    
    with tab1:
        if not st.session_state.pipeline_initialized:
            st.info("🔄 Initializing pipeline...")
            return
            
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown("### Upload Research Papers")
        
        uploaded_files = st.file_uploader(
            "Choose PDF or text files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload research papers to analyze"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in [f['name'] for f in st.session_state.processed_files]:
                    with st.expander(f"📄 Processing: {file.name}", expanded=True):
                        with st.spinner(f"Processing {file.name}..."):
                            try:
                                # Save and process file
                                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1])
                                temp_file.write(file.read())
                                temp_file.close()
                                
                                result = st.session_state.pipeline.process_document(temp_file.name)
                                
                                # Store file info
                                st.session_state.processed_files.append({
                                    'name': file.name,
                                    'summary': result.get('final_summary', 'No summary available'),
                                    'chunks': len(result.get('chunk_summaries', [])),
                                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                    'full_result': result  # Store full result for download
                                })
                                
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
        
        # Show processed files with enhanced display
        if st.session_state.processed_files:
            st.markdown("### 📚 Processed Documents")
            for i, file_info in enumerate(st.session_state.processed_files):
                with st.expander(f"📄 {file_info['name']} ({file_info['timestamp']})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.write(f"**Chunks:** {file_info['chunks']}")
                        st.write(f"**Summary:** {file_info['summary'][:200]}...")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        # Individual document download
                        if 'full_result' in file_info:
                            doc_json = json.dumps({
                                "document": file_info['name'],
                                "processed": file_info['timestamp'],
                                "summary": file_info['summary'],
                                "chunks": file_info['chunks'],
                                "full_analysis": file_info.get('full_result', {})
                            }, indent=2, ensure_ascii=False)
                            
                            st.download_button(
                                label="📥 Download",
                                data=doc_json,
                                file_name=f"{file_info['name']}_analysis.json",
                                mime="application/json",
                                key=f"download_{i}"
                            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        if not st.session_state.pipeline_initialized:
            st.info("🔄 Initializing pipeline...")
            return
            
        if not st.session_state.processed_files:
            st.markdown('<div class="status-info">📄 Upload and process documents first to enable querying</div>', unsafe_allow_html=True)
            return
        
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown("### Ask Questions About Your Documents")
        
        query = st.text_input(
            "Enter your question:",
            placeholder="What is the main contribution of this research?",
            help="Ask specific questions about the uploaded documents"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            k_results = st.slider("Number of results", 1, 10, 3)
        with col2:
            if st.button("🔍 Search", disabled=not query):
                if query:
                    with st.spinner("Searching documents..."):
                        try:
                            results = st.session_state.pipeline.query_documents(query, k=k_results)
                            
                            if results and 'results' in results:
                                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                                st.markdown(f"**🎯 Answer for: '{query}'**")
                                st.write(results['final_answer'])
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Show source chunks
                                with st.expander("📚 Source Chunks", expanded=False):
                                    for i, chunk in enumerate(results['results'][:3]):
                                        st.markdown(f"**Chunk {i+1}** (Score: {chunk['score']:.3f})")
                                        st.write(chunk['content'][:300] + "...")
                                        st.markdown("---")
                            else:
                                st.warning("No relevant information found.")
                                
                        except Exception as e:
                            st.error(f"Error during search: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
