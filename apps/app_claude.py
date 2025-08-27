"""
Improved Research Paper Summarizer
Optimized for performance, usability, and maintainability
"""

import streamlit as st
import sys
import os
import tempfile
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
import traceback

# Set environment for deployment
os.environ["ENVIRONMENT"] = "deployment"

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_loader import DataLoader
from src.core.text_splitter import TextSplitter
from src.core.embedding_service import EmbeddingService
from src.core.vector_store import VectorStore
from src.core.summarizer import Summarizer

# Streamlined Model configurations
EMBEDDING_MODELS = {
    'all-MiniLM-L6-v2': {
        'name': 'MiniLM-L6-v2',
        'description': 'Fast & Balanced - Good speed/quality ratio',
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'performance': 'Fast',
        'quality': 'Good'
    },
    'all-mpnet-base-v2': {
        'name': 'MPNet-Base-v2',
        'description': 'High Quality - Better accuracy, slower',
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'performance': 'Slow',
        'quality': 'Excellent'
    },
    'all-distilroberta-v1': {
        'name': 'DistilRoBERTa-v1',
        'description': 'Fast Processing - Quick for large documents',
        'model_name': 'sentence-transformers/all-distilroberta-v1',
        'performance': 'Very Fast',
        'quality': 'Good'
    }
}

SUMMARIZATION_MODELS = {
    'facebook/bart-large-cnn': {
        'name': 'BART-Large-CNN',
        'description': 'High Quality Summaries - Best for news/research',
        'model_name': 'facebook/bart-large-cnn',
        'performance': 'Slow',
        'quality': 'Excellent'
    },
    't5-small': {
        'name': 'T5-Small',
        'description': 'Fast Processing - Quick summaries',
        'model_name': 't5-small',
        'performance': 'Fast',
        'quality': 'Good'
    },
    'sshleifer/distilbart-cnn-12-6': {
        'name': 'DistilBART-CNN',
        'description': 'Balanced Performance - Good speed/quality',
        'model_name': 'sshleifer/distilbart-cnn-12-6',
        'performance': 'Medium',
        'quality': 'Good'
    }
}

def get_optimized_styles():
    """Get optimized CSS styles focused on performance and usability."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Header Styles */
        .app-header {
            text-align: center;
            padding: 2rem 0;
            margin-bottom: 2rem;
            border-bottom: 2px solid #e5e7eb;
        }
        
        .app-title {
            font-size: 2.5rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.5rem;
        }
        
        .app-subtitle {
            font-size: 1.1rem;
            color: #6b7280;
            margin-bottom: 0;
        }
        
        /* Card Components */
        .info-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .success-card {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-left: 4px solid #22c55e;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .warning-card {
            background: #fffbeb;
            border: 1px solid #fed7aa;
            border-left: 4px solid #f59e0b;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .error-card {
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-left: 4px solid #ef4444;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Statistics Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .stat-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: 600;
            color: #3b82f6;
            margin-bottom: 0.25rem;
        }
        
        .stat-label {
            font-size: 0.875rem;
            color: #6b7280;
            font-weight: 500;
        }
        
        /* Model Selection */
        .model-selector {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .model-info {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        
        .model-details h4 {
            margin: 0 0 0.5rem 0;
            color: #1f2937;
            font-weight: 500;
        }
        
        .model-details p {
            margin: 0;
            color: #6b7280;
            font-size: 0.875rem;
        }
        
        .model-badges {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        
        .badge {
            background: #e5e7eb;
            color: #374151;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        
        .badge-performance { background: #dbeafe; color: #1d4ed8; }
        .badge-quality { background: #dcfce7; color: #166534; }
        
        /* Buttons */
        .stButton > button {
            border-radius: 6px;
            border: none;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #3b82f6, #1d4ed8) !important;
        }
        
        /* File Upload */
        .uploadedFile {
            border: 2px dashed #d1d5db;
            border-radius: 8px;
            padding: 1rem;
        }
        
        /* Results */
        .result-section {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .result-header {
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e5e7eb;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .app-title {
                font-size: 2rem;
            }
            
            .stats-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }
            
            .model-info {
                flex-direction: column;
                gap: 0.5rem;
            }
        }
        
        /* Hide Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """

@st.cache_resource
def load_embedding_model(model_name: str):
    """Load embedding model with caching."""
    try:
        return EmbeddingService(model_name=model_name)
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

@st.cache_resource
def load_summarization_model(model_name: str):
    """Load summarization model with caching."""
    try:
        return Summarizer(model_name=model_name)
    except Exception as e:
        st.error(f"Error loading summarization model: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state with default values."""
    defaults = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'summarization_model': 'facebook/bart-large-cnn',
        'processed_files': [],
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'max_summary_length': 512,
        'vector_store': None,
        'last_query_result': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_stats_display():
    """Create statistics display."""
    if not st.session_state.processed_files:
        return
    
    total_docs = len(st.session_state.processed_files)
    total_chunks = sum(file.get('chunks', 0) for file in st.session_state.processed_files)
    avg_chunks = total_chunks // total_docs if total_docs > 0 else 0
    
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{total_docs}</div>
            <div class="stat-label">Documents</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{total_chunks}</div>
            <div class="stat-label">Total Chunks</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{avg_chunks}</div>
            <div class="stat-label">Avg per Doc</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{st.session_state.get('embedding_model', 'N/A')[:8]}</div>
            <div class="stat-label">Embedding</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create optimized sidebar."""
    with st.sidebar:
        st.markdown("## Settings")
        
        # Embedding Model Selection
        st.markdown("### Embedding Model")
        embedding_model = st.selectbox(
            "Choose embedding model:",
            options=list(EMBEDDING_MODELS.keys()),
            index=list(EMBEDDING_MODELS.keys()).index(st.session_state.embedding_model),
            format_func=lambda x: EMBEDDING_MODELS[x]['name'],
            help="Model used to create document embeddings"
        )
        st.session_state.embedding_model = embedding_model
        
        # Display model info
        model_info = EMBEDDING_MODELS[embedding_model]
        st.markdown(f"""
        <div class="model-selector">
            <p><strong>{model_info['name']}</strong></p>
            <p>{model_info['description']}</p>
            <div class="model-badges">
                <span class="badge badge-performance">{model_info['performance']}</span>
                <span class="badge badge-quality">{model_info['quality']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Summarization Model Selection
        st.markdown("### Summarization Model")
        summarization_model = st.selectbox(
            "Choose summarization model:",
            options=list(SUMMARIZATION_MODELS.keys()),
            index=list(SUMMARIZATION_MODELS.keys()).index(st.session_state.summarization_model),
            format_func=lambda x: SUMMARIZATION_MODELS[x]['name'],
            help="Model used to generate summaries"
        )
        st.session_state.summarization_model = summarization_model
        
        # Display model info
        model_info = SUMMARIZATION_MODELS[summarization_model]
        st.markdown(f"""
        <div class="model-selector">
            <p><strong>{model_info['name']}</strong></p>
            <p>{model_info['description']}</p>
            <div class="model-badges">
                <span class="badge badge-performance">{model_info['performance']}</span>
                <span class="badge badge-quality">{model_info['quality']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Advanced Settings
        with st.expander("Advanced Settings"):
            st.session_state.chunk_size = st.slider(
                "Chunk Size", 200, 2000, st.session_state.chunk_size,
                help="Size of text chunks for processing"
            )
            st.session_state.chunk_overlap = st.slider(
                "Chunk Overlap", 0, 500, st.session_state.chunk_overlap,
                help="Overlap between consecutive chunks"
            )
            st.session_state.max_summary_length = st.slider(
                "Max Summary Length", 100, 1000, st.session_state.max_summary_length,
                help="Maximum length of generated summaries"
            )
        
        st.markdown("---")
        
        # Clear data
        if st.button("Clear All Data", help="Clear all processed documents"):
            clear_session_data()

def process_documents(uploaded_files):
    """Process uploaded documents with improved error handling."""
    if not uploaded_files:
        return
    
    try:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status = st.empty()
        
        status.markdown('<div class="info-card">Initializing AI models...</div>', unsafe_allow_html=True)
        
        # Load models
        embedding_service = load_embedding_model(
            EMBEDDING_MODELS[st.session_state.embedding_model]['model_name']
        )
        if not embedding_service:
            st.error("Failed to load embedding model")
            return
            
        progress_bar.progress(20)
        
        summarizer = load_summarization_model(
            SUMMARIZATION_MODELS[st.session_state.summarization_model]['model_name']
        )
        if not summarizer:
            st.error("Failed to load summarization model")
            return
            
        progress_bar.progress(30)
        
        status.markdown('<div class="info-card">Processing documents...</div>', unsafe_allow_html=True)
        
        all_chunks = []
        processed_summaries = []
        
        # Process each document
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Validate file size (limit to 10MB)
                if uploaded_file.size > 10 * 1024 * 1024:
                    st.warning(f"File {uploaded_file.name} is too large (>10MB). Skipping.")
                    continue
                
                # Load document
                loader = DataLoader()
                # Save uploaded file temporarily for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    text = loader.load_pdf(tmp_path)
                except AttributeError:
                    # Fallback to other methods if load_pdf doesn't exist
                    try:
                        text = loader.load_document(tmp_path)
                    except AttributeError:
                        st.error(f"DataLoader doesn't support PDF loading for {uploaded_file.name}")
                        continue
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)
                
                # Handle different return types from data loader
                if isinstance(text, dict):
                    # If DataLoader returns a dict, extract the text content
                    text = text.get('content', '') or text.get('text', '') or str(text)
                
                # Ensure text is a string
                text = str(text) if text else ""
                
                if not text or len(text.strip()) < 100:
                    st.warning(f"File {uploaded_file.name} appears to be empty or too short. Skipping.")
                    continue
                
                # Split text
                splitter = TextSplitter(
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap
                )
                chunks = splitter.split_text(text)
                chunk_texts = [chunk.text if hasattr(chunk, 'text') else str(chunk) for chunk in chunks]
                all_chunks.extend(chunk_texts)
                
                # Generate summary - ensure text is string and truncated
                summary_text = str(text)[:3000] if len(str(text)) > 3000 else str(text)
                doc_summary = summarizer.summarize_text(
                    summary_text,
                    max_length=st.session_state.max_summary_length
                )
                
                processed_summaries.append({
                    'name': uploaded_file.name,
                    'summary': doc_summary,
                    'chunks': len(chunk_texts),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'size': len(text),
                    'model_used': st.session_state.summarization_model
                })
                
                # Update progress
                progress = 30 + (i + 1) * (40 / len(uploaded_files))
                progress_bar.progress(int(progress))
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        if not all_chunks:
            st.error("No documents were successfully processed")
            return
        
        status.markdown('<div class="info-card">Creating vector embeddings...</div>', unsafe_allow_html=True)
        
        # Create vector store
        vector_store = VectorStore()
        embeddings_data = []
        
        # Process embeddings in batches
        batch_size = 10
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            
            for j, chunk_text in enumerate(batch):
                try:
                    embedding = embedding_service.generate_embedding(chunk_text)
                    embeddings_data.append({
                        'embedding': embedding,
                        'text': chunk_text,
                        'chunk_id': f"chunk_{i+j}",
                        'metadata': {'chunk_index': i+j}
                    })
                except Exception as e:
                    st.warning(f"Error creating embedding for chunk {i+j}: {str(e)}")
                    continue
            
            # Update progress
            progress = 70 + (i / len(all_chunks)) * 25
            progress_bar.progress(int(progress))
        
        if embeddings_data:
            vector_store.add_embeddings(embeddings_data)
        
        # Save to session state
        st.session_state.vector_store = vector_store
        st.session_state.processed_files = processed_summaries
        
        progress_bar.progress(100)
        status.markdown(
            f'<div class="success-card"><strong>Success!</strong> Processed {len(processed_summaries)} documents into {len(all_chunks)} chunks.</div>',
            unsafe_allow_html=True
        )
        
        # Auto-rerun to update UI
        time.sleep(1)
        st.experimental_rerun()
        
    except Exception as e:
        st.error(f"Error during document processing: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")

def answer_query(query: str):
    """Process query with improved error handling."""
    if not query.strip():
        st.warning("Please enter a query")
        return
    
    if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
        st.error("Please process documents first")
        return
    
    try:
        progress_bar = st.progress(0)
        status = st.empty()
        
        status.markdown('<div class="info-card">Searching documents...</div>', unsafe_allow_html=True)
        
        # Load embedding service
        embedding_service = load_embedding_model(
            EMBEDDING_MODELS[st.session_state.embedding_model]['model_name']
        )
        if not embedding_service:
            st.error("Failed to load embedding model")
            return
        
        progress_bar.progress(30)
        
        # Search for relevant chunks
        try:
            relevant_chunks = st.session_state.vector_store.search_by_text(
                query, embedding_service, k=5
            )
        except AttributeError:
            # Fallback if search_by_text doesn't exist
            try:
                query_embedding = embedding_service.generate_embedding(query)
                relevant_chunks = st.session_state.vector_store.search(query_embedding, k=5)
            except Exception as e:
                st.error(f"Error searching documents: {str(e)}")
                return
        
        if not relevant_chunks:
            st.warning("No relevant content found for your query")
            return
        
        progress_bar.progress(60)
        
        status.markdown('<div class="info-card">Generating answer...</div>', unsafe_allow_html=True)
        
        # Load summarization model
        summarizer = load_summarization_model(
            SUMMARIZATION_MODELS[st.session_state.summarization_model]['model_name']
        )
        if not summarizer:
            st.error("Failed to load summarization model")
            return
        
        # Generate answer
        try:
            summary = summarizer.summarize_with_rag(query, relevant_chunks)
        except AttributeError:
            # Fallback if summarize_with_rag doesn't exist
            try:
                # Combine relevant chunks into context
                context = "\n".join([
                    chunk.get('text', str(chunk)) if isinstance(chunk, dict) else str(chunk) 
                    for chunk in relevant_chunks[:3]
                ])
                combined_text = f"Query: {query}\n\nContext: {context}"
                summary = summarizer.summarize_text(combined_text[:2000])
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                return
        
        progress_bar.progress(90)
        
        # Save result
        result = {
            'query': query,
            'summary': summary,
            'relevant_chunks': relevant_chunks,
            'model_info': {
                'embedding': st.session_state.embedding_model,
                'summarization': st.session_state.summarization_model
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'chunk_count': len(relevant_chunks)
        }
        st.session_state.last_query_result = result
        
        progress_bar.progress(100)
        status.markdown('<div class="success-card">Analysis complete!</div>', unsafe_allow_html=True)
        
        time.sleep(0.5)
        st.experimental_rerun()
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

def display_query_results():
    """Display query results with improved formatting."""
    if 'last_query_result' not in st.session_state:
        return
    
    result = st.session_state.last_query_result
    
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.markdown('<div class="result-header">Query Results</div>', unsafe_allow_html=True)
    
    # Query info
    st.markdown(f"""
    **Question:** {result.get('query', 'N/A')}  
    **Processed:** {result.get('timestamp', 'N/A')}  
    **Sources Used:** {result.get('chunk_count', 0)} document chunks  
    **Models:** {result.get('model_info', {}).get('embedding', 'N/A')} + {result.get('model_info', {}).get('summarization', 'N/A')}
    """)
    
    # Answer
    if 'summary' in result:
        st.markdown("### Answer")
        st.write(result['summary'])
    
    # Source chunks
    if 'relevant_chunks' in result:
        with st.expander("Source Context", expanded=False):
            for i, chunk in enumerate(result['relevant_chunks'], 1):
                if isinstance(chunk, dict):
                    score = chunk.get('score', 0)
                    text = chunk.get('text', '')
                else:
                    score = 0
                    text = str(chunk)
                
                st.markdown(f"""
                **Source {i}** (Relevance: {score:.3f})  
                {text[:300]}{'...' if len(text) > 300 else ''}
                """)
                st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_export_section():
    """Create export functionality."""
    if not st.session_state.processed_files:
        st.markdown('<div class="warning-card">No data available for export. Process documents first.</div>', unsafe_allow_html=True)
        return
    
    st.markdown("### Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**JSON Export**")
        st.markdown("Complete analysis data with metadata")
        
        if st.button("Download JSON", key="json_export"):
            export_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "models": {
                    "embedding": st.session_state.embedding_model,
                    "summarization": st.session_state.summarization_model
                },
                "settings": {
                    "chunk_size": st.session_state.chunk_size,
                    "chunk_overlap": st.session_state.chunk_overlap,
                    "max_summary_length": st.session_state.max_summary_length
                },
                "documents": st.session_state.processed_files,
                "query_results": st.session_state.get('last_query_result')
            }
            
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download Analysis (JSON)",
                data=json_data,
                file_name=f"rag_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        st.markdown("**CSV Report**")
        st.markdown("Structured data for spreadsheets")
        
        if st.button("Download CSV", key="csv_export"):
            csv_data = "Document,Timestamp,Summary,Chunks,Embedding_Model,Summarization_Model\n"
            for file_info in st.session_state.processed_files:
                summary_clean = file_info.get('summary', '').replace('\n', ' ').replace('"', '""')
                csv_data += f'"{file_info["name"]}","{file_info["timestamp"]}","{summary_clean}","{file_info.get("chunks", 0)}","{st.session_state.embedding_model}","{st.session_state.summarization_model}"\n'
            
            st.download_button(
                label="Download Report (CSV)",
                data=csv_data.encode('utf-8'),
                file_name=f"rag_report_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        st.markdown("**Markdown Report**")
        st.markdown("Human-readable formatted report")
        
        if st.button("Download Markdown", key="md_export"):
            md_data = f"""# RAG Analysis Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Embedding Model:** {st.session_state.embedding_model}  
**Summarization Model:** {st.session_state.summarization_model}  

## Documents Processed

"""
            for file_info in st.session_state.processed_files:
                md_data += f"""### {file_info['name']}
**Processed:** {file_info['timestamp']}  
**Chunks:** {file_info.get('chunks', 0)}  

**Summary:**  
{file_info.get('summary', 'No summary available')}

---

"""
            
            if st.session_state.get('last_query_result'):
                result = st.session_state.last_query_result
                md_data += f"""## Latest Query Results

**Query:** {result.get('query', 'N/A')}  
**Timestamp:** {result.get('timestamp', 'N/A')}  

**Answer:**  
{result.get('summary', 'No answer available')}
"""
            
            st.download_button(
                label="Download Report (MD)",
                data=md_data.encode('utf-8'),
                file_name=f"rag_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

def clear_session_data():
    """Clear session data with confirmation."""
    keys_to_clear = ['vector_store', 'processed_files', 'last_query_result']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("All data cleared successfully!")
    time.sleep(1)
    st.experimental_rerun()

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Research Paper Summarizer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply styles
    st.markdown(get_optimized_styles(), unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar
    create_sidebar()
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">Research Paper Summarizer</h1>
        <p class="app-subtitle">AI-powered document analysis with semantic search and summarization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics
    create_stats_display()
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["Process Documents", "Query Documents", "Export Results"])
    
    with tab1:
        st.markdown("## Document Processing")
        st.markdown("Upload PDF documents to analyze and summarize")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files (max 10MB each)"
        )
        
        if uploaded_files:
            st.info(f"Ready to process {len(uploaded_files)} files")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("Process Documents"):
                    process_documents(uploaded_files)
            
            with col2:
                if st.session_state.processed_files:
                    if st.button("Clear All Data"):
                        clear_session_data()
        
        # Display processed files
        if st.session_state.processed_files:
            st.markdown("### Processed Documents")
            
            for i, file_info in enumerate(st.session_state.processed_files):
                with st.expander(f"{file_info['name']} - {file_info.get('chunks', 0)} chunks", expanded=(i == 0)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown("**Summary:**")
                        st.write(file_info.get('summary', 'Processing...'))
                    
                    with col2:
                        st.metric("Chunks", file_info.get('chunks', 0))
                        st.metric("Size (chars)", file_info.get('size', 0))
                        st.caption(f"Processed: {file_info.get('timestamp', 'N/A')}")
    
    with tab2:
        st.markdown("## Query Your Documents")
        
        if not st.session_state.processed_files:
            st.markdown('<div class="warning-card">No documents available. Please process documents first in the "Process Documents" tab.</div>', unsafe_allow_html=True)
        else:
            st.markdown("Ask questions about your processed documents")
            
            query = st.text_area(
                "Enter your question:",
                placeholder="What are the main findings? What methodology was used? What are the key conclusions?",
                height=100
            )
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("Search and Analyze", disabled=not query.strip()):
                    if query.strip():
                        answer_query(query)
            
            with col2:
                k_value = st.slider("Number of sources to use", 3, 10, 5)
            
            # Display results
            display_query_results()
    
    with tab3:
        st.markdown("## Export Results")
        create_export_section()

if __name__ == "__main__":
    main()
