"""
Enhanced Research Paper Summarizer - Deployment Version
Combines features from app.py, app_premium.py, and app_minimal_fixed.py
Optimized for both local and cloud deployment with theme support and advanced features.
"""

import streamlit as st
import sys
import os
import tempfile
import time
import json
from pathlib import Path
from typing import List, Dict, Optional

# Set environment for deployment
os.environ["ENVIRONMENT"] = "deployment"

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_loader import DataLoader
from src.core.text_splitter import TextSplitter
from src.core.embedding_service import EmbeddingService
from src.core.vector_store import VectorStore
from src.core.summarizer import Summarizer

# Enhanced Model configurations with icons and detailed descriptions
EMBEDDING_MODELS = {
    'all-MiniLM-L6-v2': {
        'description': 'Fast & Balanced - Good speed/quality ratio',
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'icon': '⚡',
        'performance': 'Fast',
        'quality': 'Good'
    },
    'all-mpnet-base-v2': {
        'description': 'High Quality - Better accuracy, slower',
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'icon': '🎯',
        'performance': 'Slow',
        'quality': 'Excellent'
    },
    'all-distilroberta-v1': {
        'description': 'Fast Processing - Quick for large documents',
        'model_name': 'sentence-transformers/all-distilroberta-v1',
        'icon': '🚀',
        'performance': 'Very Fast',
        'quality': 'Good'
    },
    'paraphrase-multilingual-MiniLM-L12-v2': {
        'description': 'Multilingual Support - 50+ languages',
        'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'icon': '🌍',
        'performance': 'Medium',
        'quality': 'Good'
    }
}

SUMMARIZATION_MODELS = {
    'facebook/bart-large-cnn': {
        'description': 'High Quality Summaries - Best for news/research',
        'model_name': 'facebook/bart-large-cnn',
        'icon': '🎯',
        'performance': 'Slow',
        'quality': 'Excellent'
    },
    't5-small': {
        'description': 'Fast Processing - Quick summaries',
        'model_name': 't5-small',
        'icon': '⚡',
        'performance': 'Fast',
        'quality': 'Good'
    },
    'google/pegasus-xsum': {
        'description': 'Concise Summaries - Short and precise',
        'model_name': 'google/pegasus-xsum',
        'icon': '📝',
        'performance': 'Medium',
        'quality': 'Very Good'
    },
    'sshleifer/distilbart-cnn-12-6': {
        'description': 'Balanced Performance - Good speed/quality',
        'model_name': 'sshleifer/distilbart-cnn-12-6',
        'icon': '⚖️',
        'performance': 'Medium',
        'quality': 'Good'
    }
}

# Theme configurations
THEMES = {
    'light': {
        'primary_bg': '#ffffff',
        'secondary_bg': '#f8f9fa',
        'accent_bg': '#e9ecef',
        'text_primary': '#212529',
        'text_secondary': '#6c757d',
        'border': '#dee2e6',
        'shadow': 'rgba(0,0,0,0.1)',
        'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    },
    'dark': {
        'primary_bg': '#1a1a1a',
        'secondary_bg': '#2d2d2d',
        'accent_bg': '#404040',
        'text_primary': '#ffffff',
        'text_secondary': '#b0b0b0',
        'border': '#555555',
        'shadow': 'rgba(255,255,255,0.1)',
        'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    },
    'blue': {
        'primary_bg': '#f0f8ff',
        'secondary_bg': '#e6f3ff',
        'accent_bg': '#cce7ff',
        'text_primary': '#1a365d',
        'text_secondary': '#2c5282',
        'border': '#90cdf4',
        'shadow': 'rgba(66,153,225,0.2)',
        'gradient': 'linear-gradient(135deg, #4299e1 0%, #3182ce 100%)'
    }
}

def get_enhanced_styles(theme='light'):
    """Get enhanced CSS styles with theme support and premium animations."""
    colors = THEMES[theme]
    
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {{
            --primary-bg: {colors['primary_bg']};
            --secondary-bg: {colors['secondary_bg']};
            --accent-bg: {colors['accent_bg']};
            --text-primary: {colors['text_primary']};
            --text-secondary: {colors['text_secondary']};
            --border: {colors['border']};
            --shadow: {colors['shadow']};
            --gradient: {colors['gradient']};
        }}
        
        .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--primary-bg);
            color: var(--text-primary);
            transition: all 0.3s ease;
        }}
        
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
            animation: fadeInUp 0.6s ease-out;
        }}
        
        /* Enhanced Animations */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        @keyframes slideInLeft {{
            from {{
                opacity: 0;
                transform: translateX(-30px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}
        
        @keyframes slideInRight {{
            from {{
                opacity: 0;
                transform: translateX(30px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-5px); }}
        }}
        
        @keyframes glow {{
            0%, 100% {{ box-shadow: 0 0 5px var(--shadow); }}
            50% {{ box-shadow: 0 0 20px var(--shadow), 0 0 30px var(--shadow); }}
        }}
        
        /* Premium Header */
        .premium-header {{
            background: var(--gradient);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
            animation: fadeInUp 0.8s ease-out;
            position: relative;
        }}
        
        .premium-header::after {{
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: var(--gradient);
            border-radius: 2px;
            animation: slideInLeft 1s ease-out 0.5s both;
        }}
        
        .premium-subtitle {{
            text-align: center;
            font-size: 1.2rem;
            color: var(--text-secondary);
            margin-bottom: 3rem;
            animation: fadeInUp 1s ease-out 0.3s both;
        }}
        
        /* Glass Morphism Cards */
        .glass-card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            animation: slideInUp 0.6s ease-out;
            position: relative;
            overflow: hidden;
        }}
        
        .glass-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.5s;
        }}
        
        .glass-card:hover {{
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        }}
        
        .glass-card:hover::before {{
            left: 100%;
        }}
        
        /* Floating Stats Grid */
        .floating-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}
        
        .stat-card {{
            background: var(--secondary-bg);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid var(--border);
            transition: all 0.4s ease;
            animation: float 3s ease-in-out infinite;
            position: relative;
            overflow: hidden;
        }}
        
        .stat-card:nth-child(1) {{ animation-delay: 0s; }}
        .stat-card:nth-child(2) {{ animation-delay: 0.5s; }}
        .stat-card:nth-child(3) {{ animation-delay: 1s; }}
        .stat-card:nth-child(4) {{ animation-delay: 1.5s; }}
        
        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: var(--gradient);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-15px) rotateY(5deg);
            box-shadow: 0 15px 30px var(--shadow);
        }}
        
        .stat-card:hover::before {{
            transform: scaleX(1);
        }}
        
        .stat-number {{
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--gradient);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            transition: all 0.3s ease;
        }}
        
        .stat-card:hover .stat-number {{
            transform: scale(1.1);
            animation: pulse 1s infinite;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Neon Buttons */
        .stButton > button {{
            background: var(--gradient);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }}
        
        .stButton > button::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            animation: glow 1.5s infinite;
        }}
        
        .stButton > button:hover::before {{
            left: 100%;
        }}
        
        .stButton > button:active {{
            transform: translateY(-1px);
        }}
        
        /* Enhanced Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1rem;
            background: var(--secondary-bg);
            border-radius: 15px;
            padding: 0.5rem;
            border: 1px solid var(--border);
            animation: slideInLeft 0.6s ease-out;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            color: var(--text-secondary);
            border-radius: 10px;
            padding: 1rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
            position: relative;
        }}
        
        .stTabs [data-baseweb="tab"]::before {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 0;
            height: 3px;
            background: var(--gradient);
            transition: width 0.3s ease;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background: var(--accent-bg);
            transform: translateY(-2px);
        }}
        
        .stTabs [data-baseweb="tab"]:hover::before {{
            width: 100%;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: var(--gradient) !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }}
        
        /* Model Selection Cards */
        .model-card {{
            background: var(--secondary-bg);
            border-radius: 12px;
            padding: 1rem;
            border: 2px solid var(--border);
            transition: all 0.3s ease;
            cursor: pointer;
            margin: 0.5rem 0;
        }}
        
        .model-card:hover {{
            border-color: #667eea;
            transform: translateX(5px);
            box-shadow: 0 5px 15px var(--shadow);
        }}
        
        .model-card.selected {{
            border-color: #667eea;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        }}
        
        .model-info {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .model-icon {{
            font-size: 1.5rem;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--gradient);
            border-radius: 10px;
            color: white;
        }}
        
        .model-details h4 {{
            margin: 0;
            color: var(--text-primary);
            font-weight: 600;
        }}
        
        .model-details p {{
            margin: 0.25rem 0 0 0;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        .model-badges {{
            display: flex;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }}
        
        .badge {{
            background: var(--accent-bg);
            color: var(--text-primary);
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        
        /* Progress Animations */
        .stProgress > div > div {{
            background: var(--gradient) !important;
            border-radius: 5px;
            animation: pulse 1.5s infinite;
        }}
        
        .stProgress > div {{
            border-radius: 5px;
            overflow: hidden;
        }}
        
        /* Processing Indicators */
        .processing-card {{
            background: var(--secondary-bg);
            border-radius: 15px;
            padding: 1.5rem;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
            animation: slideInRight 0.5s ease-out;
        }}
        
        .processing-icon {{
            font-size: 1.5rem;
            margin-right: 1rem;
            animation: pulse 1.5s infinite;
        }}
        
        /* Enhanced File Upload */
        .uploadedFile {{
            background: var(--secondary-bg);
            border: 2px dashed var(--border);
            border-radius: 10px;
            padding: 1.5rem;
            transition: all 0.3s ease;
            animation: slideInUp 0.5s ease-out;
        }}
        
        .uploadedFile:hover {{
            border-color: #667eea;
            background: var(--accent-bg);
            transform: scale(1.02);
        }}
        
        /* Sidebar Enhancements */
        .css-1d391kg {{
            background: var(--secondary-bg) !important;
            border-right: 1px solid var(--border);
            animation: slideInLeft 0.5s ease-out;
        }}
        
        /* Theme Toggle */
        .theme-toggle {{
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 1000;
            background: var(--secondary-bg);
            border: 1px solid var(--border);
            border-radius: 25px;
            padding: 0.5rem;
            display: flex;
            gap: 0.5rem;
            box-shadow: 0 4px 15px var(--shadow);
        }}
        
        .theme-btn {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }}
        
        .theme-btn:hover {{
            transform: scale(1.1);
        }}
        
        .theme-btn.active {{
            background: var(--gradient);
            color: white;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }}
        
        /* Results Display */
        .result-card {{
            background: var(--secondary-bg);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            margin: 1rem 0;
            animation: slideInUp 0.6s ease-out;
        }}
        
        .result-header {{
            background: var(--gradient);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }}
        
        /* Export Section */
        .export-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 1rem 0;
        }}
        
        .export-card {{
            background: var(--secondary-bg);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            transition: all 0.3s ease;
            text-align: center;
        }}
        
        .export-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px var(--shadow);
        }}
        
        .export-icon {{
            font-size: 2rem;
            margin-bottom: 1rem;
            background: var(--gradient);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .premium-header {{
                font-size: 2rem;
            }}
            
            .floating-stats {{
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }}
            
            .glass-card {{
                padding: 1rem;
            }}
            
            .theme-toggle {{
                top: 0.5rem;
                right: 0.5rem;
            }}
        }}
        
        /* Hide Streamlit Elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Enhanced Theme Toggle in Sidebar */
        .stButton > button[title*="Light"] {{
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            color: white;
        }}
        
        .stButton > button[title*="Dark"] {{
            background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
            color: white;
        }}
        
        /* Model Card Enhancements */
        .model-selector {{
            background: var(--secondary-bg);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }}
        
        .model-selector:hover {{
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px var(--shadow);
        }}
        
        .model-selector.active {{
            border-color: #667eea;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        }}
        
        /* Progress Bar Enhancement */
        .stProgress {{
            margin: 1rem 0;
        }}
        
        .stProgress > div {{
            background: var(--accent-bg);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .stProgress > div > div {{
            background: var(--gradient) !important;
            border-radius: 8px;
            position: relative;
            overflow: hidden;
        }}
        
        .stProgress > div > div::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }}
        
        @keyframes shimmer {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
        
        /* File Upload Enhancement */
        .stFileUploader {{
            padding: 1rem;
            border-radius: 10px;
            border: 2px dashed var(--border);
            transition: all 0.3s ease;
        }}
        
        .stFileUploader:hover {{
            border-color: #667eea;
            background: var(--accent-bg);
        }}
        
        /* Status Message Improvements */
        .status-success {{
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border: 1px solid #34d399;
            color: #065f46;
        }}
        
        .status-error {{
            background: linear-gradient(135deg, #fee2e2 0%, #fca5a5 100%);
            border: 1px solid #f87171;
            color: #991b1b;
        }}
        
        .status-warning {{
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border: 1px solid #f59e0b;
            color: #92400e;
        }}
        
        .status-info {{
            background: linear-gradient(135deg, #dbeafe 0%, #93c5fd 100%);
            border: 1px solid #3b82f6;
            color: #1e40af;
        }}
        
        /* Enhanced Metrics Display */
        .metric-card {{
            background: var(--secondary-bg);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid var(--border);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: var(--gradient);
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px var(--shadow);
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            background: var(--gradient);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        
        .metric-label {{
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 0.9rem;
        }}
        
        /* Enhanced Sidebar Styling */
        .sidebar-section {{
            background: var(--secondary-bg);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid var(--border);
        }}
        
        .sidebar-section h3 {{
            margin-top: 0;
            color: var(--text-primary);
            border-bottom: 2px solid var(--accent-bg);
            padding-bottom: 0.5rem;
        }}
        
        /* Loading Spinner */
        .loading-spinner {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid var(--accent-bg);
            border-radius: 50%;
            border-top-color: #667eea;
            animation: spin 1s ease-in-out infinite;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
    </style>
    """

def create_enhanced_metrics():
    """Create enhanced metrics display with better visualization."""
    if 'processed_files' in st.session_state and st.session_state.processed_files:
        total_docs = len(st.session_state.processed_files)
        total_chunks = sum(file.get('chunks', 0) for file in st.session_state.processed_files)
        avg_chunks = total_chunks // total_docs if total_docs > 0 else 0
        total_size = sum(file.get('size', 0) for file in st.session_state.processed_files)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">📚 {total_docs}</div>
                <div class="metric-label">Documents Processed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">🧩 {total_chunks}</div>
                <div class="metric-label">Text Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">📊 {avg_chunks}</div>
                <div class="metric-label">Avg Chunks/Doc</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            size_mb = total_size / (1024 * 1024) if total_size > 1024*1024 else total_size / 1024
            unit = "MB" if total_size > 1024*1024 else "KB"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">💾 {size_mb:.1f}{unit}</div>
                <div class="metric-label">Total Size</div>
            </div>
            """, unsafe_allow_html=True)

def create_status_indicator(status_type, message, details=""):
    """Create enhanced status indicators with better visual feedback."""
    icons = {
        'success': '✅',
        'error': '❌', 
        'warning': '⚠️',
        'info': 'ℹ️',
        'processing': '⏳'
    }
    
    icon = icons.get(status_type, 'ℹ️')
    
    if status_type == 'processing':
        st.markdown(f"""
        <div class="processing-card">
            <span class="processing-icon">{icon}</span>
            <strong>{message}</strong><br>
            {f'<small>{details}</small>' if details else ''}
            <div class="loading-spinner" style="margin-top: 0.5rem;"></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-{status_type}">
            <span style="margin-right: 0.5rem;">{icon}</span>
            <strong>{message}</strong>
            {f'<br><small>{details}</small>' if details else ''}
        </div>
        """, unsafe_allow_html=True)

def create_model_selector(model_type: str, models_dict: dict, session_key: str):
    """Create an enhanced model selector with cards."""
    st.markdown(f"### {model_type} Models")
    
    current_model = st.session_state.get(session_key, list(models_dict.keys())[0])
    
    cols = st.columns(2)
    
    for i, (model_key, model_info) in enumerate(models_dict.items()):
        with cols[i % 2]:
            selected = model_key == current_model
            card_class = "model-card selected" if selected else "model-card"
            
            # Create clickable model card
            if st.button(
                f"{model_info['icon']} {model_key}",
                key=f"{session_key}_{model_key}",
                help=model_info['description']
            ):
                st.session_state[session_key] = model_key
                st.experimental_rerun()
            
            # Display model info
            st.markdown(f"""
            <div class="{card_class}">
                <div class="model-info">
                    <div class="model-icon">{model_info['icon']}</div>
                    <div class="model-details">
                        <h4>{model_key}</h4>
                        <p>{model_info['description']}</p>
                        <div class="model-badges">
                            <span class="badge">⚡ {model_info['performance']}</span>
                            <span class="badge">🎯 {model_info['quality']}</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def enhanced_sidebar():
    """Create enhanced sidebar with model selection."""
    with st.sidebar:
        st.markdown("# ⚙️ Settings")
        
        # Theme selection with visual preview
        st.markdown("### 🎨 Theme")
        
        # Create theme preview cards
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("☀️ Light", key="theme_light", 
                        help="Clean, bright interface"):
                if st.session_state.get('theme') != 'light':
                    st.session_state.theme = 'light'
                    st.experimental_rerun()
        
        with col2:
            if st.button("🌙 Dark", key="theme_dark",
                        help="Easy on the eyes"):
                if st.session_state.get('theme') != 'dark':
                    st.session_state.theme = 'dark'
                    st.experimental_rerun()
        
        # Show current theme
        current_theme = st.session_state.get('theme', 'light')
        st.markdown(f"**Active:** {current_theme.title()} Theme")
        
        st.markdown("---")
        
        # Model selections
        create_model_selector("Embedding", EMBEDDING_MODELS, 'embedding_model')
        
        st.markdown("---")
        
        create_model_selector("Summarization", SUMMARIZATION_MODELS, 'summarization_model')
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 200, 2000, 1000)
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
            max_summary_length = st.slider("Max Summary Length", 100, 1000, 512)
            
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            st.session_state.max_summary_length = max_summary_length
        
        st.markdown("---")
        
        # System info
        st.markdown("### 📊 System Info")
        current_embedding = st.session_state.get('embedding_model', 'all-MiniLM-L6-v2')
        current_summarization = st.session_state.get('summarization_model', 'facebook/bart-large-cnn')
        
        st.markdown(f"""
        **Embedding:** {EMBEDDING_MODELS[current_embedding]['icon']} {current_embedding}  
        **Summary:** {SUMMARIZATION_MODELS[current_summarization]['icon']} {current_summarization}  
        **Theme:** {st.session_state.get('theme', 'light').title()}
        """)
        
        st.markdown("---")
        
        # Clear data button
        if st.button("🗑️ Clear All Data", key="sidebar_clear", help="Clear all processed data"):
            clear_session_data()

def enhanced_export_tab():
    """Enhanced export functionality with multiple formats."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("# 📤 Export Center")
    
    if 'last_query_result' not in st.session_state or not st.session_state.get('processed_files'):
        st.markdown("""
        <div class="processing-card">
            <span class="processing-icon">⚠️</span>
            <strong>No data available for export.</strong><br>
            Process documents and run queries first to unlock export features.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.markdown("### Choose your export format:")
    
    # Export options grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="export-card">
            <div class="export-icon">📄</div>
            <h4>JSON Export</h4>
            <p>Complete analysis data with metadata</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📄 Download JSON", key="json_export"):
            export_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "theme": st.session_state.get('theme', 'light'),
                "models": {
                    "embedding": st.session_state.get('embedding_model'),
                    "summarization": st.session_state.get('summarization_model')
                },
                "settings": {
                    "chunk_size": st.session_state.get('chunk_size', 1000),
                    "chunk_overlap": st.session_state.get('chunk_overlap', 200),
                    "max_summary_length": st.session_state.get('max_summary_length', 512)
                },
                "documents": st.session_state.processed_files,
                "query_results": st.session_state.last_query_result
            }
            
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="💾 Download Complete Analysis",
                data=json_data,
                file_name=f"rag_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        st.markdown("""
        <div class="export-card">
            <div class="export-icon">📊</div>
            <h4>CSV Report</h4>
            <p>Structured data for spreadsheets</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Download CSV", key="csv_export"):
            csv_data = "Document,Timestamp,Summary,Chunks,Embedding_Model,Summarization_Model\n"
            for file_info in st.session_state.processed_files:
                summary_clean = file_info.get('summary', '').replace('\n', ' ').replace('"', '""')
                csv_data += f'"{file_info["name"]}","{file_info["timestamp"]}","{summary_clean}","{file_info.get("chunks", 0)}","{st.session_state.get("embedding_model", "")}","{st.session_state.get("summarization_model", "")}"\n'
            
            st.download_button(
                label="💾 Download CSV Report",
                data=csv_data.encode('utf-8'),
                file_name=f"rag_report_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        st.markdown("""
        <div class="export-card">
            <div class="export-icon">📝</div>
            <h4>Markdown</h4>
            <p>Human-readable formatted report</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📝 Download Markdown", key="md_export"):
            md_data = f"""# RAG Analysis Report
            
**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Theme:** {st.session_state.get('theme', 'light').title()}  
**Embedding Model:** {st.session_state.get('embedding_model', 'N/A')}  
**Summarization Model:** {st.session_state.get('summarization_model', 'N/A')}  

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
            
            if 'last_query_result' in st.session_state:
                result = st.session_state.last_query_result
                md_data += f"""## Latest Query Results

**Query:** {result.get('query', 'N/A')}  
**Timestamp:** {result.get('timestamp', 'N/A')}  

**Answer:**  
{result.get('summary', 'No answer available')}

"""
            
            st.download_button(
                label="💾 Download Markdown Report",
                data=md_data.encode('utf-8'),
                file_name=f"rag_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function with enhanced UI."""
    st.set_page_config(
        page_title="RAG Research Summarizer Pro",
        page_icon="🚀",
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
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 1000
    if 'chunk_overlap' not in st.session_state:
        st.session_state.chunk_overlap = 200
    if 'max_summary_length' not in st.session_state:
        st.session_state.max_summary_length = 512
    
    # Apply theme-based styles
    current_theme = st.session_state.get('theme', 'light')
    st.markdown(get_enhanced_styles(current_theme), unsafe_allow_html=True)
    
    # Enhanced sidebar
    enhanced_sidebar()
    
    # Premium header
    st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem;">
            <h1 class="premium-header">AI Research Summarizer </h1>
            <p class="premium-subtitle">AI-powered document analysis with semantic search, advanced summarization, and beautiful UI</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced metrics
    create_enhanced_metrics()
    
    # Main tabs with enhanced styling
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Process", "🔍 Query", "📤 Export", "📊 Analytics"])
    
    with tab1:
        process_tab()
    
    with tab2:
        query_tab()
    
    with tab3:
        enhanced_export_tab()
    
    with tab4:
        analytics_tab()

def process_tab():
    """Enhanced document processing tab."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.markdown("# 📄 Document Processing Center")
    st.markdown("Upload and process your research papers with advanced AI models")
    
    # File upload with enhanced styling
    uploaded_files = st.file_uploader(
        "📁 Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload multiple PDF files for batch processing"
    )
    
    if uploaded_files:
        st.markdown(f"""
        <div class="processing-card">
            <span class="processing-icon">📚</span>
            <strong>{len(uploaded_files)} files ready for processing</strong>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Process Documents", key="process_docs", help="Start AI processing"):
                process_documents_enhanced(uploaded_files)
        
        with col2:
            processing_mode = st.selectbox(
                "Processing Mode:",
                options=["Standard", "Fast", "Detailed"],
                help="Choose processing speed vs quality"
            )
        
        with col3:
            if st.button("🗑️ Clear All", key="main_clear", help="Clear all processed data"):
                clear_session_data()
    
    # Display processed files with enhanced cards
    if st.session_state.processed_files:
        st.markdown("## 📊 Processed Documents")
        
        for i, file_info in enumerate(st.session_state.processed_files):
            with st.expander(
                f"📄 {file_info['name']} • {file_info.get('chunks', 0)} chunks • {file_info['timestamp']}",
                expanded=i == 0
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown("### 📝 Summary")
                    st.write(file_info.get('summary', 'Processing...'))
                
                with col2:
                    st.markdown("### 📊 Stats")
                    st.metric("Chunks", file_info.get('chunks', 0))
                    st.metric("Size", f"{len(file_info.get('summary', ''))}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def query_tab():
    """Enhanced query interface."""
    if not st.session_state.processed_files:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
            <h2>🔍 Query Interface</h2>
            <p>No documents processed yet. Upload and process documents first to enable querying.</p>
            <div style="font-size: 4rem; margin: 2rem 0;">📚</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.markdown("# 🔍 Intelligent Query System")
    st.markdown("Ask questions about your documents and get AI-powered answers")
    
    # Enhanced query input
    query = st.text_area(
        "💬 Your Research Question:",
        placeholder="What are the main findings of this research? What methodology was used? What are the limitations?",
        height=100,
        help="Ask detailed questions about your documents"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🧠 Analyze & Answer", key="search_summarize", disabled=not query):
            if query:
                answer_query_enhanced(query)
    
    with col2:
        k_results = st.slider("Context Sources", 3, 10, 5, help="Number of relevant chunks to use")
    
    with col3:
        response_style = st.selectbox(
            "Response Style:",
            options=["Detailed", "Concise", "Bullet Points"],
            help="Choose how you want the answer formatted"
        )
    
    # Display results with enhanced styling
    if 'last_query_result' in st.session_state:
        result = st.session_state.last_query_result
        
        st.markdown("---")
        st.markdown("## 🎯 Analysis Results")
        
        # Result header
        st.markdown(f"""
        <div class="result-card">
            <div class="result-header">📋 Query Information</div>
            <strong>Question:</strong> {result.get('query', '')}<br>
            <strong>Analyzed:</strong> {result.get('timestamp', '')}<br>
            <strong>Model:</strong> {result.get('model_info', {}).get('summarization', '')}<br>
            <strong>Sources:</strong> {len(result.get('relevant_chunks', []))} documents
        </div>
        """, unsafe_allow_html=True)
        
        # Answer section
        if 'summary' in result:
            st.markdown("### 🤖 AI Answer")
            st.markdown(f"""
            <div class="result-card">
                {result['summary']}
            </div>
            """, unsafe_allow_html=True)
        
        # Source context
        if 'relevant_chunks' in result:
            with st.expander("📚 Source Context & Evidence", expanded=False):
                for i, chunk in enumerate(result['relevant_chunks'], 1):
                    score = chunk.get('score', 0) if isinstance(chunk, dict) else 0
                    text = chunk.get('text', str(chunk)) if isinstance(chunk, dict) else str(chunk)
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-header">📖 Source {i} (Relevance: {score:.3f})</div>
                        {text[:500]}{'...' if len(text) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def analytics_tab():
    """Analytics and insights tab."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.markdown("# 📊 Analytics Dashboard")
    
    if not st.session_state.processed_files:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h3>📈 No Analytics Available</h3>
            <p>Process documents to see detailed analytics and insights.</p>
        </div>
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Document analytics
    total_docs = len(st.session_state.processed_files)
    total_chunks = sum(file.get('chunks', 0) for file in st.session_state.processed_files)
    total_text = sum(len(file.get('summary', '')) for file in st.session_state.processed_files)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Documents", total_docs)
    with col2:
        st.metric("🧩 Total Chunks", total_chunks)
    with col3:
        st.metric("📝 Avg Chunks/Doc", f"{total_chunks/total_docs:.1f}" if total_docs > 0 else "0")
    with col4:
        st.metric("📊 Total Characters", f"{total_text:,}")
    
    # Model usage
    st.markdown("### 🤖 Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        current_embedding = st.session_state.get('embedding_model', 'N/A')
        embedding_info = EMBEDDING_MODELS.get(current_embedding, {})
        st.markdown(f"""
        **Embedding Model:**  
        {embedding_info.get('icon', '🤖')} {current_embedding}  
        Performance: {embedding_info.get('performance', 'N/A')}  
        Quality: {embedding_info.get('quality', 'N/A')}
        """)
    
    with col2:
        current_summarization = st.session_state.get('summarization_model', 'N/A')
        summarization_info = SUMMARIZATION_MODELS.get(current_summarization, {})
        st.markdown(f"""
        **Summarization Model:**  
        {summarization_info.get('icon', '🤖')} {current_summarization}  
        Performance: {summarization_info.get('performance', 'N/A')}  
        Quality: {summarization_info.get('quality', 'N/A')}
        """)
    
    # Processing timeline
    if st.session_state.processed_files:
        st.markdown("### ⏱️ Processing Timeline")
        
        timeline_data = []
        for file_info in st.session_state.processed_files:
            timeline_data.append({
                'Document': file_info['name'],
                'Timestamp': file_info['timestamp'],
                'Chunks': file_info.get('chunks', 0)
            })
        
        st.dataframe(timeline_data)
    
    # System settings
    st.markdown("### ⚙️ Current Settings")
    settings_col1, settings_col2 = st.columns(2)
    
    with settings_col1:
        st.markdown(f"""
        **Processing Settings:**  
        • Chunk Size: {st.session_state.get('chunk_size', 1000)}  
        • Chunk Overlap: {st.session_state.get('chunk_overlap', 200)}  
        • Max Summary Length: {st.session_state.get('max_summary_length', 512)}
        """)
    
    with settings_col2:
        st.markdown(f"""
        **UI Settings:**  
        • Theme: {st.session_state.get('theme', 'light').title()}  
        • Layout: Wide  
        • Sidebar: Expanded
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_documents_enhanced(uploaded_files):
    """Enhanced document processing with better progress indication."""
    try:
        progress_bar = st.progress(0)
        status_container = st.container()
        
        with status_container:
            status_text = st.empty()
            
            # Enhanced status messages with new indicator
            create_status_indicator('processing', 'Initializing AI models...', 
                                  'Loading embedding and summarization models')
        
        # Initialize models
        embedding_service = EmbeddingService(
            model_name=EMBEDDING_MODELS[st.session_state.embedding_model]['model_name']
        )
        progress_bar.progress(15)
        time.sleep(0.5)
        
        summarizer = Summarizer(
            model_name=SUMMARIZATION_MODELS[st.session_state.summarization_model]['model_name']
        )
        progress_bar.progress(25)
        
        status_text.markdown("""
        <div class="processing-card">
            <span class="processing-icon">📄</span>
            <strong>Processing documents...</strong><br>
            <small>Extracting text and creating chunks</small>
        </div>
        """, unsafe_allow_html=True)
        
        all_chunks = []
        processed_summaries = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Load and process document
            loader = DataLoader()
            text = loader.load_pdf_from_bytes(uploaded_file.getvalue())
            
            splitter = TextSplitter(
                chunk_size=st.session_state.get('chunk_size', 1000),
                chunk_overlap=st.session_state.get('chunk_overlap', 200)
            )
            chunks = splitter.split_text(text)
            chunk_texts = [chunk.text if hasattr(chunk, 'text') else str(chunk) for chunk in chunks]
            all_chunks.extend(chunk_texts)
            
            # Generate summary
            max_length = st.session_state.get('max_summary_length', 512)
            doc_summary = summarizer.summarize_text(text[:2000], max_length=max_length)
            
            processed_summaries.append({
                'name': uploaded_file.name,
                'summary': doc_summary,
                'chunks': len(chunk_texts),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'size': len(text),
                'model_used': st.session_state.summarization_model
            })
            
            progress = 25 + (i + 1) * (50 / len(uploaded_files))
            progress_bar.progress(int(progress))
        
        status_text.markdown("""
        <div class="processing-card">
            <span class="processing-icon">🔬</span>
            <strong>Creating vector embeddings...</strong><br>
            <small>Generating semantic representations</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Create vector store
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
            
            if i % 10 == 0:  # Update progress every 10 chunks
                progress = 75 + (i / len(all_chunks)) * 20
                progress_bar.progress(int(progress))
        
        vector_store.add_embeddings(embeddings_data)
        progress_bar.progress(95)
        
        # Save to session state
        st.session_state.vector_store = vector_store
        st.session_state.processed_files = processed_summaries
        st.session_state.total_chunks = len(all_chunks)
        
        progress_bar.progress(100)
        
        # Success message
        status_text.markdown(f"""
        <div class="processing-card" style="border-left-color: #22c55e;">
            <span class="processing-icon">✅</span>
            <strong>Processing complete!</strong><br>
            <small>Successfully processed {len(uploaded_files)} documents into {len(all_chunks)} chunks</small>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1)
        st.experimental_rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing documents: {str(e)}")
        st.markdown("""
        <div class="processing-card" style="border-left-color: #ef4444;">
            <span class="processing-icon">❌</span>
            <strong>Processing failed</strong><br>
            <small>Please check your documents and try again</small>
        </div>
        """, unsafe_allow_html=True)

def answer_query_enhanced(query):
    """Enhanced query processing with better progress indication."""
    try:
        if 'vector_store' not in st.session_state:
            st.error("Please process documents first!")
            return
        
        progress_bar = st.progress(0)
        status_container = st.container()
        
        with status_container:
            status_text = st.empty()
            
            status_text.markdown("""
            <div class="processing-card">
                <span class="processing-icon">🔍</span>
                <strong>Searching documents...</strong><br>
                <small>Finding relevant content using semantic search</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Perform semantic search
        embedding_service = EmbeddingService(
            model_name=EMBEDDING_MODELS[st.session_state.embedding_model]['model_name']
        )
        relevant_chunks = st.session_state.vector_store.search_by_text(query, embedding_service, k=5)
        progress_bar.progress(50)
        time.sleep(0.3)
        
        if not relevant_chunks:
            status_text.markdown("""
            <div class="processing-card" style="border-left-color: #f59e0b;">
                <span class="processing-icon">⚠️</span>
                <strong>No relevant content found</strong><br>
                <small>Try rephrasing your question or using different keywords</small>
            </div>
            """, unsafe_allow_html=True)
            return
        
        status_text.markdown("""
        <div class="processing-card">
            <span class="processing-icon">🤖</span>
            <strong>Generating AI answer...</strong><br>
            <small>Analyzing content and creating response</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate summary
        summarizer = Summarizer(
            model_name=SUMMARIZATION_MODELS[st.session_state.summarization_model]['model_name']
        )
        summary = summarizer.summarize_with_rag(query, relevant_chunks)
        progress_bar.progress(90)
        time.sleep(0.3)
        
        # Save results
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
        
        status_text.markdown("""
        <div class="processing-card" style="border-left-color: #22c55e;">
            <span class="processing-icon">✅</span>
            <strong>Analysis complete!</strong><br>
            <small>Your answer is ready below</small>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(0.5)
        
    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")

def clear_session_data():
    """Clear all session data with confirmation."""
    keys_to_clear = ['vector_store', 'processed_files', 'total_chunks', 'last_query_result']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("✅ All data cleared successfully!")
    time.sleep(1)
    st.experimental_rerun()

if __name__ == "__main__":
    main()