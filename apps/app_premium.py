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
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'icon': '⚡'
    },
    'all-mpnet-base-v2': {
        'description': 'High Quality - Better accuracy, slower',
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'icon': '🎯'
    },
    'all-distilroberta-v1': {
        'description': 'Fast - Quick processing, good for large docs',
        'model_name': 'sentence-transformers/all-distilroberta-v1',
        'icon': '🚀'
    },
    'paraphrase-multilingual-MiniLM-L12-v2': {
        'description': 'Multilingual - Supports 50+ languages',
        'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'icon': '🌍'
    }
}

SUMMARIZATION_MODELS = {
    'facebook/bart-large-cnn': {
        'description': 'High Quality - Best for detailed summaries',
        'model_name': 'facebook/bart-large-cnn',
        'icon': '💎'
    },
    't5-small': {
        'description': 'Fast - Quick processing, good quality',
        'model_name': 't5-small',
        'icon': '⚡'
    },
    'google/pegasus-xsum': {
        'description': 'Concise - Very short, focused summaries',
        'model_name': 'google/pegasus-xsum',
        'icon': '🎯'
    },
    'sshleifer/distilbart-cnn-12-6': {
        'description': 'Balanced - Good speed/quality for general use',
        'model_name': 'sshleifer/distilbart-cnn-12-6',
        'icon': '⚖️'
    },
    'philschmid/bart-large-cnn-samsum': {
        'description': 'Conversational - Best for Q&A style content',
        'model_name': 'philschmid/bart-large-cnn-samsum',
        'icon': '💬'
    }
}

def get_premium_styles():
    """Get premium CSS styles with animations and modern design."""
    if st.session_state.get('theme', 'dark') == 'dark':
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* Dark theme styles with premium effects */
            .stApp {
                background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
                color: #ffffff;
                font-family: 'Inter', sans-serif;
                min-height: 100vh;
            }
            
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 1400px;
                background: transparent;
            }
            
            /* Animated header with glow effect */
            .premium-header {
                font-size: 3.5rem;
                font-weight: 700;
                text-align: center;
                margin: 2rem 0;
                padding: 2rem;
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
                background-size: 300% 300%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: gradientShift 8s ease infinite;
                text-shadow: 0 0 30px rgba(255, 107, 107, 0.5);
                position: relative;
            }
            
            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            .subtitle {
                text-align: center;
                color: #a0a0a0;
                font-size: 1.2rem;
                margin-bottom: 3rem;
                font-weight: 300;
                animation: fadeInUp 1s ease-out;
            }
            
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            /* Glass morphism cards */
            .glass-card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                padding: 2rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                margin: 2rem 0;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .glass-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .glass-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
                transition: left 0.5s;
            }
            
            .glass-card:hover::before {
                left: 100%;
            }
            
            /* Neon buttons */
            .stButton > button {
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                color: white;
                border: none;
                border-radius: 15px;
                padding: 1rem 2rem;
                font-weight: 600;
                font-size: 1.1rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
                position: relative;
                overflow: hidden;
            }
            
            .stButton > button:hover {
                transform: translateY(-3px) scale(1.05);
                box-shadow: 0 8px 30px rgba(255, 107, 107, 0.5);
                filter: brightness(1.1);
            }
            
            .stButton > button:active {
                transform: translateY(-1px) scale(1.02);
            }
            
            /* Animated progress bars */
            .stProgress > div > div {
                background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
                background-size: 200% 200%;
                animation: progressShine 2s linear infinite;
            }
            
            @keyframes progressShine {
                0% { background-position: 0% 50%; }
                100% { background-position: 200% 50%; }
            }
            
            /* Status cards with glow */
            .status-success {
                background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1));
                color: #81c784;
                padding: 1rem 1.5rem;
                border-radius: 15px;
                border: 1px solid rgba(76, 175, 80, 0.3);
                margin: 1rem 0;
                animation: pulseGlow 2s infinite;
                box-shadow: 0 0 20px rgba(76, 175, 80, 0.2);
            }
            
            .status-info {
                background: linear-gradient(135deg, rgba(33, 150, 243, 0.2), rgba(33, 150, 243, 0.1));
                color: #64b5f6;
                padding: 1rem 1.5rem;
                border-radius: 15px;
                border: 1px solid rgba(33, 150, 243, 0.3);
                margin: 1rem 0;
                animation: pulseGlow 2s infinite;
                box-shadow: 0 0 20px rgba(33, 150, 243, 0.2);
            }
            
            @keyframes pulseGlow {
                0%, 100% { box-shadow: 0 0 20px rgba(255, 255, 255, 0.1); }
                50% { box-shadow: 0 0 30px rgba(255, 255, 255, 0.2); }
            }
            
            /* Sidebar styling */
            .css-1d391kg {
                background: linear-gradient(135deg, rgba(13, 13, 13, 0.9), rgba(26, 26, 46, 0.9));
                backdrop-filter: blur(20px);
            }
            
            /* Tab styling with neon effect */
            .stTabs [data-baseweb="tab-list"] {
                gap: 1rem;
                background: rgba(255, 255, 255, 0.05);
                padding: 0.5rem;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                color: #a0a0a0;
                border-radius: 12px;
                padding: 1rem 1.5rem;
                font-weight: 500;
                transition: all 0.3s ease;
                border: 1px solid transparent;
            }
            
            .stTabs [data-baseweb="tab"]:hover {
                background: rgba(255, 255, 255, 0.1);
                color: #ffffff;
                transform: translateY(-2px);
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4) !important;
                color: white !important;
                box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
            }
            
            /* File uploader styling */
            .uploadedFile {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 1rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            /* Input fields */
            .stTextInput > div > div > input {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                color: white;
                padding: 1rem;
                backdrop-filter: blur(10px);
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #ff6b6b;
                box-shadow: 0 0 20px rgba(255, 107, 107, 0.2);
            }
            
            /* Selectbox styling */
            .stSelectbox > div > div {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                backdrop-filter: blur(10px);
            }
            
            /* Hide Streamlit branding */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
        </style>
        """
    else:
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* Light theme with premium effects */
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                font-family: 'Inter', sans-serif;
                min-height: 100vh;
            }
            
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 1400px;
            }
            
            /* Animated header */
            .premium-header {
                font-size: 3.5rem;
                font-weight: 700;
                text-align: center;
                margin: 2rem 0;
                padding: 2rem;
                background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
                background-size: 300% 300%;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: gradientShift 8s ease infinite;
                text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            
            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            .subtitle {
                text-align: center;
                color: #6c757d;
                font-size: 1.2rem;
                margin-bottom: 3rem;
                font-weight: 300;
                animation: fadeInUp 1s ease-out;
            }
            
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            /* Glass morphism cards for light theme */
            .glass-card {
                background: rgba(255, 255, 255, 0.8);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                padding: 2rem;
                border: 1px solid rgba(255, 255, 255, 0.3);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                margin: 2rem 0;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .glass-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
            
            /* Premium buttons */
            .stButton > button {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                border-radius: 15px;
                padding: 1rem 2rem;
                font-weight: 600;
                font-size: 1.1rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            
            .stButton > button:hover {
                transform: translateY(-3px) scale(1.05);
                box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
            }
            
            /* Status cards */
            .status-success {
                background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(76, 175, 80, 0.05));
                color: #2e7d32;
                padding: 1rem 1.5rem;
                border-radius: 15px;
                border: 1px solid rgba(76, 175, 80, 0.2);
                margin: 1rem 0;
                box-shadow: 0 4px 20px rgba(76, 175, 80, 0.1);
            }
            
            .status-info {
                background: linear-gradient(135deg, rgba(33, 150, 243, 0.1), rgba(33, 150, 243, 0.05));
                color: #1565c0;
                padding: 1rem 1.5rem;
                border-radius: 15px;
                border: 1px solid rgba(33, 150, 243, 0.2);
                margin: 1rem 0;
                box-shadow: 0 4px 20px rgba(33, 150, 243, 0.1);
            }
            
            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 1rem;
                background: rgba(255, 255, 255, 0.5);
                padding: 0.5rem;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                color: #6c757d;
                border-radius: 12px;
                padding: 1rem 1.5rem;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(45deg, #667eea, #764ba2) !important;
                color: white !important;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            
            /* Input styling */
            .stTextInput > div > div > input {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 10px;
                padding: 1rem;
                backdrop-filter: blur(10px);
            }
            
            /* Hide Streamlit branding */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """

def create_floating_stats():
    """Create animated floating statistics."""
    if 'processed_files' in st.session_state and st.session_state.processed_files:
        total_docs = len(st.session_state.processed_files)
        total_chunks = sum(file.get('chunks', 0) for file in st.session_state.processed_files)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: rgba(255, 107, 107, 0.1); border-radius: 15px; border: 1px solid rgba(255, 107, 107, 0.2);">
                <h2 style="color: #ff6b6b; margin: 0; font-size: 2rem;">📄</h2>
                <h3 style="margin: 0.5rem 0; color: #ff6b6b;">{total_docs}</h3>
                <p style="margin: 0; color: #a0a0a0;">Documents</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: rgba(78, 205, 196, 0.1); border-radius: 15px; border: 1px solid rgba(78, 205, 196, 0.2);">
                <h2 style="color: #4ecdc4; margin: 0; font-size: 2rem;">🧩</h2>
                <h3 style="margin: 0.5rem 0; color: #4ecdc4;">{total_chunks}</h3>
                <p style="margin: 0; color: #a0a0a0;">Chunks</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: rgba(69, 183, 209, 0.1); border-radius: 15px; border: 1px solid rgba(69, 183, 209, 0.2);">
                <h2 style="color: #45b7d1; margin: 0; font-size: 2rem;">🤖</h2>
                <h3 style="margin: 0.5rem 0; color: #45b7d1;">{st.session_state.get('embedding_model', 'MiniLM')[:8]}</h3>
                <p style="margin: 0; color: #a0a0a0;">AI Model</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            query_count = 1 if 'last_query_result' in st.session_state else 0
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: rgba(150, 206, 180, 0.1); border-radius: 15px; border: 1px solid rgba(150, 206, 180, 0.2);">
                <h2 style="color: #96ceb4; margin: 0; font-size: 2rem;">🔍</h2>
                <h3 style="margin: 0.5rem 0; color: #96ceb4;">{query_count}</h3>
                <p style="margin: 0; color: #a0a0a0;">Queries</p>
            </div>
            """, unsafe_allow_html=True)

def enhanced_sidebar():
    """Create an enhanced sidebar with better styling."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="background: linear-gradient(45deg, #ff6b6b, #4ecdc4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">⚙️ AI Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Theme toggle with enhanced styling
        st.markdown("### 🎨 Theme")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌙 Dark", key="dark_theme"):
                st.session_state.theme = 'dark'
                st.experimental_rerun()
        with col2:
            if st.button("☀️ Light", key="light_theme"):
                st.session_state.theme = 'light'
                st.experimental_rerun()
        
        st.markdown("---")
        
        # Enhanced model selection
        st.markdown("### 🧠 Embedding Model")
        embedding_options = []
        for key, value in EMBEDDING_MODELS.items():
            embedding_options.append(f"{value['icon']} {key}")
        
        selected_embedding_display = st.selectbox(
            "Choose your AI brain:",
            options=embedding_options,
            index=0 if st.session_state.get('embedding_model') not in EMBEDDING_MODELS else list(EMBEDDING_MODELS.keys()).index(st.session_state.get('embedding_model', 'all-MiniLM-L6-v2')),
            help="This model converts text into mathematical representations for search"
        )
        
        # Extract the actual model name
        selected_embedding = selected_embedding_display.split(' ', 1)[1]
        if selected_embedding != st.session_state.get('embedding_model'):
            st.session_state.embedding_model = selected_embedding
        
        st.markdown("### 📝 Summarization Model")
        summarization_options = []
        for key, value in SUMMARIZATION_MODELS.items():
            summarization_options.append(f"{value['icon']} {key}")
        
        selected_summarization_display = st.selectbox(
            "Choose your writer:",
            options=summarization_options,
            index=0 if st.session_state.get('summarization_model') not in SUMMARIZATION_MODELS else list(SUMMARIZATION_MODELS.keys()).index(st.session_state.get('summarization_model', 'facebook/bart-large-cnn')),
            help="This model creates summaries from your documents"
        )
        
        selected_summarization = selected_summarization_display.split(' ', 1)[1]
        if selected_summarization != st.session_state.get('summarization_model'):
            st.session_state.summarization_model = selected_summarization
        
        st.markdown("---")
        
        # Model info with styling
        st.markdown("### 📊 Current Setup")
        st.markdown(f"""
        <div style="background: rgba(255, 107, 107, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #ff6b6b; margin: 0.5rem 0;">
            <strong>🧠 Brain:</strong><br>{EMBEDDING_MODELS[st.session_state.get('embedding_model', 'all-MiniLM-L6-v2')]['description']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: rgba(78, 205, 196, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #4ecdc4; margin: 0.5rem 0;">
            <strong>📝 Writer:</strong><br>{SUMMARIZATION_MODELS[st.session_state.get('summarization_model', 'facebook/bart-large-cnn')]['description']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced clear button
        if st.button("🗑️ Nuclear Reset", key="sidebar_clear", help="⚠️ This will destroy everything!"):
            clear_session_data()

def export_data_tab():
    """Enhanced export data tab with better UI."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💾 Export Your Research Data")
    
    if 'last_query_result' not in st.session_state or not st.session_state.get('processed_files'):
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(255, 255, 255, 0.05); border-radius: 15px; border: 2px dashed rgba(255, 255, 255, 0.2);">
            <h2 style="color: #a0a0a0; margin: 0;">📄 No Data to Export</h2>
            <p style="color: #808080;">Process documents and run queries first to enable data export</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: rgba(255, 107, 107, 0.1); border-radius: 15px; border: 1px solid rgba(255, 107, 107, 0.2); margin: 1rem 0;">
            <h3 style="color: #ff6b6b; margin: 0;">📋 JSON Export</h3>
            <p style="color: #a0a0a0;">Complete research analysis data</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Download JSON Report", key="json_export"):
            export_data = {
                "export_info": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_documents": len(st.session_state.processed_files),
                    "app_version": "Research Paper Summarizer v4.0 Premium",
                    "models_used": {
                        "embedding": st.session_state.get('embedding_model', 'all-MiniLM-L6-v2'),
                        "summarization": st.session_state.get('summarization_model', 'facebook/bart-large-cnn')
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
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: rgba(78, 205, 196, 0.1); border-radius: 15px; border: 1px solid rgba(78, 205, 196, 0.2); margin: 1rem 0;">
            <h3 style="color: #4ecdc4; margin: 0;">📊 CSV Export</h3>
            <p style="color: #a0a0a0;">Spreadsheet-ready summary data</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📈 Download CSV Summary", key="csv_export"):
            csv_data = "Document,Processing_Date,Summary,Model_Used\n"
            for file_info in st.session_state.processed_files:
                summary_clean = file_info.get('summary', 'No summary').replace('\n', ' ').replace('"', '""')
                csv_data += f'"{file_info["name"]}","{file_info["timestamp"]}","{summary_clean}","{st.session_state.get("summarization_model", "N/A")}"\n'
            
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_data.encode('utf-8'),
                file_name=f"research_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Research Paper Summarizer Pro",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = 'all-MiniLM-L6-v2'
    if 'summarization_model' not in st.session_state:
        st.session_state.summarization_model = 'facebook/bart-large-cnn'
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    # Apply premium styles
    st.markdown(get_premium_styles(), unsafe_allow_html=True)
    
    # Enhanced sidebar
    enhanced_sidebar()
    
    # Premium animated header
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
            <h1 class="premium-header">🚀 Research Paper Summarizer Pro</h1>
            <p class="subtitle">Next-generation AI-powered document analysis with premium RAG technology</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Floating stats
    create_floating_stats()
    
    # Enhanced tabs with premium styling
    tab1, tab2, tab3 = st.tabs(["🔬 AI Laboratory", "🤖 Query Engine", "💾 Data Vault"])
    
    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Enhanced file upload section
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="background: linear-gradient(45deg, #ff6b6b, #4ecdc4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📚 Document Processing Laboratory</h2>
            <p style="color: #a0a0a0;">Upload your research papers and watch AI magic happen</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "🎯 Choose your research papers", 
            type=['pdf'], 
            accept_multiple_files=True,
            help="Drag and drop multiple PDF files or click to browse"
        )
        
        if uploaded_files:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                if st.button("🚀 Process with AI", key="process_docs", help="Process documents with selected AI models"):
                    process_documents(uploaded_files)
            with col2:
                chunk_size = st.slider("📏 Chunk Size", 200, 2000, 1000, help="Size of text chunks for processing")
            with col3:
                if st.button("💥 Clear All", key="main_clear", help="Reset everything"):
                    clear_session_data()
        
        # Processing status with enhanced styling
        if uploaded_files:
            st.markdown(f"""
            <div style="background: rgba(69, 183, 209, 0.1); padding: 1rem; border-radius: 15px; border-left: 4px solid #45b7d1; margin: 1rem 0; animation: pulseGlow 2s infinite;">
                🎯 <strong>{len(uploaded_files)} files</strong> ready for AI processing
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced processed files display
        if st.session_state.processed_files:
            st.markdown("### 🏆 Successfully Processed Documents")
            
            for i, file_info in enumerate(st.session_state.processed_files):
                with st.expander(f"📄 {file_info['name']} • {file_info.get('chunks', 0)} chunks • {file_info['timestamp']}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**📝 Summary:**")
                        st.write(file_info.get('summary', 'Processing...'))
                    with col2:
                        st.markdown(f"""
                        <div style="background: rgba(150, 206, 180, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                            <h4 style="color: #96ceb4; margin: 0;">🧩 {file_info.get('chunks', 0)}</h4>
                            <p style="margin: 0; color: #a0a0a0;">Chunks</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        if not st.session_state.processed_files:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 4rem;">
                <h2 style="color: #a0a0a0;">🤖 Query Engine Offline</h2>
                <p style="color: #808080; font-size: 1.2rem;">Upload and process documents in the AI Laboratory first</p>
                <div style="margin: 2rem 0;">
                    <div style="font-size: 4rem; opacity: 0.3;">🔌</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="background: linear-gradient(45deg, #45b7d1, #96ceb4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🤖 AI Query Engine</h2>
            <p style="color: #a0a0a0;">Ask intelligent questions about your research documents</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced query input
        query = st.text_input(
            "🎯 Enter your research question:",
            placeholder="e.g., What are the key methodologies used in this research?",
            help="Ask specific, detailed questions for best results"
        )
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            if st.button("🔍 AI Search & Analyze", key="search_summarize", disabled=not query):
                if query:
                    answer_query(query)
                else:
                    st.warning("🤔 Please enter a question first.")
        
        with col2:
            k_results = st.slider("📊 Results", 3, 10, 5, help="Number of relevant chunks to analyze")
        
        with col3:
            search_depth = st.selectbox("🎯 Depth", ["Quick", "Deep", "Comprehensive"], help="Analysis depth")
        
        # Enhanced results display
        if 'last_query_result' in st.session_state:
            result = st.session_state.last_query_result
            
            st.markdown("---")
            st.markdown("### 🎯 AI Analysis Results")
            
            # Query info
            st.markdown(f"""
            <div style="background: rgba(255, 107, 107, 0.1); padding: 1rem; border-radius: 15px; border-left: 4px solid #ff6b6b; margin: 1rem 0;">
                <strong>🔍 Query:</strong> {result.get('query', 'N/A')}<br>
                <strong>⏰ Analyzed:</strong> {result.get('timestamp', 'N/A')}<br>
                <strong>🤖 Model:</strong> {result.get('model_info', {}).get('summarization', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
            
            # Summary with enhanced styling
            if 'summary' in result:
                st.markdown("#### 📝 AI Summary")
                st.markdown(f"""
                <div style="background: rgba(78, 205, 196, 0.1); padding: 2rem; border-radius: 15px; border: 1px solid rgba(78, 205, 196, 0.2); font-size: 1.1rem; line-height: 1.6;">
                    {result['summary']}
                </div>
                """, unsafe_allow_html=True)
            
            # Source chunks with better presentation
            if 'relevant_chunks' in result:
                with st.expander("🔍 Source Evidence & Context", expanded=False):
                    for i, chunk in enumerate(result['relevant_chunks'], 1):
                        score = chunk.get('score', 0) if isinstance(chunk, dict) else 0
                        text = chunk.get('text', str(chunk)) if isinstance(chunk, dict) else str(chunk)
                        
                        st.markdown(f"""
                        <div style="background: rgba(69, 183, 209, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #45b7d1; margin: 1rem 0;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <strong style="color: #45b7d1;">📄 Evidence #{i}</strong>
                                <span style="background: rgba(69, 183, 209, 0.2); padding: 0.25rem 0.5rem; border-radius: 8px; font-size: 0.9rem;">
                                    🎯 {score:.3f}
                                </span>
                            </div>
                            <p style="margin: 0; line-height: 1.6;">{text[:500]}{'...' if len(text) > 500 else ''}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        export_data_tab()

def process_documents(uploaded_files):
    """Enhanced document processing with better UI feedback."""
    try:
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            # Animated processing status
            status_placeholder.markdown("""
            <div style="background: rgba(255, 107, 107, 0.1); padding: 1rem; border-radius: 15px; border-left: 4px solid #ff6b6b; animation: pulseGlow 2s infinite;">
                🧠 Initializing AI models and neural networks...
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(15)
            
            # Initialize components with selected models
            embedding_service = EmbeddingService(
                model_name=EMBEDDING_MODELS[st.session_state.embedding_model]['model_name']
            )
            progress_bar.progress(30)
            
            status_placeholder.markdown("""
            <div style="background: rgba(78, 205, 196, 0.1); padding: 1rem; border-radius: 15px; border-left: 4px solid #4ecdc4; animation: pulseGlow 2s infinite;">
                📚 Processing documents with advanced AI algorithms...
            </div>
            """, unsafe_allow_html=True)
            
            all_chunks = []
            processed_summaries = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Extract text from PDF
                loader = DataLoader()
                text = loader.load_pdf_from_bytes(uploaded_file.getvalue())
                
                # Split text into chunks
                splitter = TextSplitter()
                chunks = splitter.split_text(text)
                
                # Extract text from TextChunk objects
                chunk_texts = [chunk.text if hasattr(chunk, 'text') else str(chunk) for chunk in chunks]
                all_chunks.extend(chunk_texts)
                
                # Generate summary
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
                
                progress = 30 + (i + 1) * (40 / len(uploaded_files))
                progress_bar.progress(int(progress))
            
            # Create vector store
            status_placeholder.markdown("""
            <div style="background: rgba(69, 183, 209, 0.1); padding: 1rem; border-radius: 15px; border-left: 4px solid #45b7d1; animation: pulseGlow 2s infinite;">
                🔬 Creating high-dimensional vector embeddings...
            </div>
            """, unsafe_allow_html=True)
            
            vector_store = VectorStore()
            
            # Generate embeddings
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
            progress_bar.progress(85)
            
            # Store in session
            st.session_state.vector_store = vector_store
            st.session_state.processed_files = processed_summaries
            st.session_state.total_chunks = len(all_chunks)
            
            progress_bar.progress(100)
            
            # Success message with celebration
            status_placeholder.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1)); color: #81c784; padding: 2rem; border-radius: 15px; border: 1px solid rgba(76, 175, 80, 0.3); text-align: center; animation: pulseGlow 2s infinite;">
                <h3 style="margin: 0; color: #4caf50;">🎉 Processing Complete!</h3>
                <p style="margin: 0.5rem 0;">Successfully processed <strong>{len(uploaded_files)}</strong> documents into <strong>{len(all_chunks)}</strong> intelligent chunks</p>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Ready for AI-powered queries and analysis!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confetti effect
            st.balloons()
            
    except Exception as e:
        st.error(f"💥 Error processing documents: {str(e)}")
        st.markdown("""
        <div style="background: rgba(244, 67, 54, 0.1); padding: 1rem; border-radius: 15px; border-left: 4px solid #f44336;">
            💡 <strong>Troubleshooting Tips:</strong><br>
            • Check if PDF files are not corrupted<br>
            • Ensure sufficient memory available<br>
            • Try processing smaller files<br>
            • Restart the application if issues persist
        </div>
        """, unsafe_allow_html=True)

def answer_query(query):
    """Enhanced query processing with better UI."""
    try:
        if 'vector_store' not in st.session_state:
            st.error("🚨 Please process documents first!")
            return
        
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        # Search animation
        status_placeholder.markdown("""
        <div style="background: rgba(255, 193, 7, 0.1); padding: 1rem; border-radius: 15px; border-left: 4px solid #ffc107; animation: pulseGlow 2s infinite;">
            🔍 AI is searching through your documents...
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(30)
        
        embedding_service = EmbeddingService(
            model_name=EMBEDDING_MODELS[st.session_state.embedding_model]['model_name']
        )
        relevant_chunks = st.session_state.vector_store.search_by_text(query, embedding_service, k=5)
        progress_bar.progress(60)
        
        if not relevant_chunks:
            st.markdown("""
            <div style="background: rgba(255, 152, 0, 0.1); padding: 2rem; border-radius: 15px; border: 1px solid rgba(255, 152, 0, 0.3); text-align: center;">
                <h3 style="color: #ff9800; margin: 0;">🤔 No Relevant Information Found</h3>
                <p style="color: #a0a0a0; margin: 1rem 0;">Try rephrasing your question or using different keywords</p>
                <div style="background: rgba(255, 152, 0, 0.05); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                    💡 <strong>Tips:</strong> Use specific keywords, ask about main topics, or try broader questions
                </div>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Generate summary animation
        status_placeholder.markdown("""
        <div style="background: rgba(156, 39, 176, 0.1); padding: 1rem; border-radius: 15px; border-left: 4px solid #9c27b0; animation: pulseGlow 2s infinite;">
            🤖 AI is generating intelligent summary...
        </div>
        """, unsafe_allow_html=True)
        
        summarizer = Summarizer(
            model_name=SUMMARIZATION_MODELS[st.session_state.summarization_model]['model_name']
        )
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
        
        # Success message
        status_placeholder.markdown("""
        <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1)); color: #81c784; padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(76, 175, 80, 0.3); text-align: center;">
            ✨ <strong>AI Analysis Complete!</strong> Scroll down to see results.
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"💥 Error processing query: {str(e)}")
        st.markdown("""
        <div style="background: rgba(244, 67, 54, 0.1); padding: 1rem; border-radius: 15px; border-left: 4px solid #f44336;">
            💡 <strong>Try:</strong> Check if documents are processed • Rephrase your question • Use simpler queries
        </div>
        """, unsafe_allow_html=True)

def clear_session_data():
    """Enhanced session clearing with confirmation."""
    keys_to_clear = ['vector_store', 'processed_files', 'total_chunks', 'last_query_result']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("🧹 All data cleared successfully!")
    st.markdown("""
    <div style="background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 15px; border-left: 4px solid #4caf50; text-align: center;">
        🔄 <strong>Fresh Start!</strong> Ready for new documents and queries.
    </div>
    """, unsafe_allow_html=True)
    time.sleep(1)
    st.experimental_rerun()

if __name__ == "__main__":
    main()
