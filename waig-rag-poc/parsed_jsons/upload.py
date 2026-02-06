#!/usr/bin/env python3
"""
Script to upload JSON pages to the RAG system using the hierarchical chunking endpoint.
Reads a JSON file containing page data and uploads it to the /documents/upload-json endpoint.
"""

import json
import requests
import time
import argparse
import sys
from typing import Dict, Any, List
from pathlib import Path

class JSONPagesUploader:
    """Upload JSON pages to the RAG system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the uploader with the API base URL."""
        self.base_url = base_url
        self.upload_endpoint = f"{base_url}/documents/upload-json"
        self.status_endpoint = f"{base_url}/documents/status"
        self.health_endpoint = f"{base_url}/health"
    
    def check_server_health(self) -> bool:
        """Check if the server is running and healthy."""
        try:
            response = requests.get(self.health_endpoint, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Server is healthy: {health_data.get('status', 'unknown')}")
                print(f"   Pipeline initialized: {health_data.get('pipeline_initialized', False)}")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    
    def validate_pages_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate and extract pages data from JSON."""
        if 'pages' not in data:
            print("❌ JSON file must contain a 'pages' key")
            sys.exit(1)
        
        pages = data['pages']
        if not isinstance(pages, list):
            print("❌ 'pages' must be an array")
            sys.exit(1)
        
        if not pages:
            print("❌ No pages found in the JSON file")
            sys.exit(1)
        
        # Validate page structure
        for i, page in enumerate(pages):
            if not isinstance(page, dict):
                print(f"❌ Page {i} is not a valid object")
                sys.exit(1)
            
            if 'page' not in page:
                print(f"❌ Page {i} missing 'page' field")
                sys.exit(1)
        
        print(f"✅ Found {len(pages)} pages in JSON file")
        return pages
    
    def upload_pages(self, pages: List[Dict[str, Any]], document_name: str) -> str:
        """Upload pages to the server."""
        payload = {
            "pages": pages,
            "document_name": document_name
        }
        
        try:
            print(f"📤 Uploading {len(pages)} pages to server...")
            response = requests.post(
                self.upload_endpoint,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result['task_id']
                print(f"✅ Upload successful!")
                print(f"   Task ID: {task_id}")
                print(f"   Message: {result['message']}")
                print(f"   Pages received: {result['pages_received']}")
                print(f"   Document name: {result['document_name']}")
                return task_id
            else:
                print(f"❌ Upload failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text}")
                sys.exit(1)
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Upload request failed: {e}")
            sys.exit(1)
    
    def monitor_processing(self, task_id: str, max_wait_time: int = 300) -> bool:
        """Monitor the processing status."""
        print(f"⏳ Monitoring processing status (max wait: {max_wait_time}s)...")
        
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(f"{self.status_endpoint}/{task_id}", timeout=10)
                
                if response.status_code == 200:
                    status_data = response.json()
                    current_status = status_data['status']
                    
                    # Print status updates
                    if current_status != last_status:
                        print(f"📊 Status: {current_status}")
                        print(f"   Message: {status_data['message']}")
                        
                        if 'progress' in status_data:
                            progress = status_data['progress']
                            print(f"   Progress: {progress}")
                        
                        last_status = current_status
                    
                    # Check if completed
                    if current_status == "completed":
                        print("✅ Processing completed successfully!")
                        return True
                    elif current_status == "failed":
                        print("❌ Processing failed!")
                        print(f"   Error: {status_data['message']}")
                        return False
                    
                    # Still processing
                    time.sleep(5)  # Check every 5 seconds
                else:
                    print(f"❌ Status check failed: {response.status_code}")
                    return False
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Status check request failed: {e}")
                return False
        
        print(f"⏰ Processing timeout after {max_wait_time}s")
        return False
    
    def upload_from_file(self, file_path: str, document_name: str = None, monitor: bool = True, max_wait_time: int = 300):
        """Complete workflow: load file, upload, and monitor."""
        print(f"🚀 Starting JSON pages upload from: {file_path}")
        
        # Check server health
        if not self.check_server_health():
            sys.exit(1)
        
        # Load and validate JSON file
        data = self.load_json_file(file_path)
        pages = self.validate_pages_data(data)
        
        # Use document name from file or provided name
        if document_name is None:
            document_name = data.get('document_name', Path(file_path).stem)
        
        # Upload pages
        task_id = self.upload_pages(pages, document_name)
        
        # Monitor processing if requested
        if monitor:
            success = self.monitor_processing(task_id, max_wait_time)
            if success:
                print("🎉 Upload and processing completed successfully!")
            else:
                print("💥 Upload or processing failed!")
                sys.exit(1)
        else:
            print(f"📋 Task ID for manual monitoring: {task_id}")
            print(f"   Check status: GET {self.status_endpoint}/{task_id}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Upload JSON pages to RAG system with hierarchical chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_json_pages.py pages.json
  python upload_json_pages.py pages.json --document-name "Medical Manual"
  python upload_json_pages.py pages.json --no-monitor
  python upload_json_pages.py pages.json --server http://localhost:8000
        """
    )
    
    parser.add_argument(
        'json_file',
        help='Path to JSON file containing pages data'
    )
    
    parser.add_argument(
        '--document-name',
        help='Name for the document (default: extracted from JSON or filename)'
    )
    
    parser.add_argument(
        '--server',
        default='http://localhost:8000',
        help='Server URL (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--no-monitor',
        action='store_true',
        help='Do not monitor processing status'
    )
    
    parser.add_argument(
        '--max-wait-time',
        type=int,
        default=300,
        help='Maximum time to wait for processing (seconds, default: 300)'
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    if not Path(args.json_file).exists():
        print(f"❌ File not found: {args.json_file}")
        sys.exit(1)
    
    # Create uploader and run
    uploader = JSONPagesUploader(args.server)
    uploader.upload_from_file(
        file_path=args.json_file,
        document_name=args.document_name,
        monitor=not args.no_monitor,
        max_wait_time=args.max_wait_time
    )


if __name__ == "__main__":
    main()
