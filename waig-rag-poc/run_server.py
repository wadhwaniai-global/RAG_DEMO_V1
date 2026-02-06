#!/usr/bin/env python3
"""
Simple startup script for the Advanced RAG System API server.
This script handles environment setup and starts the FastAPI server.
"""

import os
import sys
from pathlib import Path
import subprocess
from dotenv import load_dotenv

def check_environment():
    """Check if the environment is properly configured."""
    print("🔍 Checking environment configuration...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found!")
        print("📝 Creating .env from template...")
        
        # Copy from env.example if it exists
        example_file = Path("env.example")
        if example_file.exists():
            with open(example_file, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            print("✅ Created .env file from template")
            print("🔧 Please edit .env file with your API keys before proceeding")
            return False
        else:
            print("❌ No env.example file found")
            return False
    
    # Load and check required environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY', 'LLAMA_CLOUD_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("🔧 Please set these in your .env file")
        return False
    
    print("✅ Environment configuration looks good!")
    return True


def check_qdrant():
    """Check if Qdrant is running."""
    print("🔍 Checking Qdrant connection...")
    
    try:
        import httpx
        qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        
        with httpx.Client() as client:
            response = client.get(f"{qdrant_url}/collections", timeout=5)
            if response.status_code == 200:
                print("✅ Qdrant is running and accessible!")
                return True
            else:
                print(f"⚠️  Qdrant responded with status {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Cannot connect to Qdrant: {str(e)}")
        print("\n🐳 To start Qdrant with Docker:")
        print("   docker run -p 6333:6333 qdrant/qdrant")
        print("\n📖 Or see README.md for other installation options")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("📦 Checking dependencies...")
    
    try:
        # Try importing key dependencies
        import fastapi
        import qdrant_client
        import langfuse
        from langfuse.openai import openai as _openai  # noqa: F401
        print("✅ Core dependencies are installed")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Installing dependencies...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False


def main():
    """Main startup function."""
    print("🚀 Advanced RAG System - Server Startup")
    print("=" * 40)
    
    # Check requirements in order
    if not install_dependencies():
        sys.exit(1)
    
    if not check_environment():
        sys.exit(1)
    
    if not check_qdrant():
        print("⚠️  Continuing without Qdrant check...")
        print("    The server will try to connect on startup")
    
    # Start the server
    print("\n🌟 Starting the RAG API server...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Import and run the server
        from config import settings
        import uvicorn
        load_dotenv()      
        uvicorn.run(
            "api.main:app",  # Use import string instead of app object
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.api_reload,
            log_level=settings.log_level.lower(),
            workers=1 if not settings.api_reload else None  # workers=1 incompatible with reload
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Server startup failed: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check that all API keys are properly set in .env")
        print("2. Ensure Qdrant is running and accessible")
        print("3. Verify that port 8000 is not already in use")
        print("4. Check the logs above for specific error details")
        sys.exit(1)


if __name__ == "__main__":
    main() 