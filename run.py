#!/usr/bin/env python3
"""
Quick setup and run script for the RAG System
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_main_app():
    """Run the main Streamlit application"""
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", f"{project_root}/apps/app.py"])

def run_deploy_app():
    """Run the deployment-optimized application"""
    import subprocess
    os.environ["ENVIRONMENT"] = "deployment"
    subprocess.run([sys.executable, "-m", "streamlit", "run", f"{project_root}/apps/app_deploy.py"])

def main():
    """Main script entry point"""
    print("🚀 RAG System Quick Start")
    print("=" * 30)
    print("1. Run Main App (Full Features)")
    print("2. Run Deploy App (Optimized)")
    print("3. Exit")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == "1":
        print("🖥️  Starting main application...")
        run_main_app()
    elif choice == "2":
        print("🚀 Starting deployment application...")
        run_deploy_app()
    elif choice == "3":
        print("👋 Goodbye!")
        sys.exit(0)
    else:
        print("❌ Invalid choice. Please select 1-3.")
        main()

if __name__ == "__main__":
    main()
