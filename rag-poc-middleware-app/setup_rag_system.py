#!/usr/bin/env python3
"""
Setup script for the RAG system.
This script helps set up the necessary users and tokens for the RAG chat system.
"""

import requests
import json
import sys
import os
from typing import Dict, Any, Optional

API_BASE_URL = "http://localhost:8000/api/v1"


def make_request(method: str, endpoint: str, data: Optional[Dict[Any, Any]] = None, 
                headers: Optional[Dict[str, str]] = None) -> Optional[Dict[Any, Any]]:
    """Make HTTP request to the API"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers)
        elif method.upper() == "GET":
            response = requests.get(url, headers=headers)
        else:
            print(f"Unsupported method: {method}")
            return None
            
        if response.status_code in [200, 201]:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to {url}. Make sure the server is running.")
        return None
    except Exception as e:
        print(f"Request error: {e}")
        return None


def create_superuser() -> Optional[str]:
    """Create a superuser and return user ID"""
    print("Creating superuser...")
    
    user_data = {
        "name": "superuser",
        "email": "admin@example.com",
        "password": "securepassword123",
        "user_type": "human",
        "is_superuser": True
    }
    
    result = make_request("POST", "/users/", user_data)
    if result:
        user_id = result.get("id")
        print(f"✓ Superuser created with ID: {user_id}")
        return user_id
    else:
        print("✗ Failed to create superuser")
        return None


def generate_superuser_token(superuser_name: str = "superuser", password: str = "securepassword123") -> Optional[str]:
    """Generate token for superuser"""
    print("Generating superuser token...")
    
    token_data = {
        "name": superuser_name,
        "password": password
    }
    
    result = make_request("POST", "/users/generate-token", token_data)
    if result:
        token = result.get("access_token")
        print(f"✓ Superuser token generated")
        return token
    else:
        print("✗ Failed to generate superuser token")
        return None


def create_bot_user(superuser_token: str) -> Optional[str]:
    """Create a bot user and return user ID"""
    print("Creating bot user...")
    
    headers = {"Authorization": f"Bearer {superuser_token}"}
    bot_data = {
        "name": "rag-bot",
        "description": "Intelligent RAG-powered assistant",
        "user_type": "bot"
    }
    
    result = make_request("POST", "/users/", bot_data, headers)
    if result:
        bot_id = result.get("id")
        print(f"✓ Bot user created with ID: {bot_id}")
        return bot_id
    else:
        print("✗ Failed to create bot user")
        return None


def create_test_human_user(superuser_token: str) -> Optional[str]:
    """Create a test human user for testing"""
    print("Creating test human user...")
    
    headers = {"Authorization": f"Bearer {superuser_token}"}
    user_data = {
        "name": "test_human",
        "email": "test@example.com",
        "password": "testpassword123",
        "user_type": "human"
    }
    
    result = make_request("POST", "/users/", user_data, headers)
    if result:
        user_id = result.get("id")
        print(f"✓ Test human user created with ID: {user_id}")
        return user_id
    else:
        print("✗ Failed to create test human user")
        return None


def generate_test_user_token() -> Optional[str]:
    """Generate token for test user"""
    print("Generating test user token...")
    
    token_data = {
        "name": "test_human",
        "password": "testpassword123"
    }
    
    result = make_request("POST", "/users/generate-token", token_data)
    if result:
        token = result.get("access_token")
        print(f"✓ Test user token generated")
        return token
    else:
        print("✗ Failed to generate test user token")
        return None


def update_env_file(superuser_token: str):
    """Update .env file with superuser token"""
    env_file_path = ".env"
    
    if os.path.exists(env_file_path):
        # Read existing .env file
        with open(env_file_path, 'r') as f:
            lines = f.readlines()
        
        # Update or add SUPERUSER_TOKEN
        token_line = f"SUPERUSER_TOKEN={superuser_token}\n"
        found_token_line = False
        
        for i, line in enumerate(lines):
            if line.startswith("SUPERUSER_TOKEN="):
                lines[i] = token_line
                found_token_line = True
                break
        
        if not found_token_line:
            lines.append(token_line)
        
        # Write back to .env file
        with open(env_file_path, 'w') as f:
            f.writelines(lines)
        
        print(f"✓ Updated {env_file_path} with superuser token")
    else:
        print(f"✗ {env_file_path} not found. Please create it manually and add:")
        print(f"SUPERUSER_TOKEN={superuser_token}")


def main():
    """Main setup function"""
    print("🚀 Setting up RAG Chat System")
    print("=" * 50)
    
    # Check if server is running
    print("Checking if server is running...")
    health_check = make_request("GET", "/../../health")
    if not health_check:
        print("✗ Server is not running. Please start the server first:")
        print("  python main.py")
        sys.exit(1)
    print("✓ Server is running")
    
    # Create superuser
    superuser_id = create_superuser()
    if not superuser_id:
        print("Setup failed: Could not create superuser")
        sys.exit(1)
    
    # Generate superuser token
    superuser_token = generate_superuser_token()
    if not superuser_token:
        print("Setup failed: Could not generate superuser token")
        sys.exit(1)
    
    # Update .env file
    update_env_file(superuser_token)
    
    # Create bot user
    bot_id = create_bot_user(superuser_token)
    if not bot_id:
        print("Setup failed: Could not create bot user")
        sys.exit(1)
    
    # Create test human user
    test_user_id = create_test_human_user(superuser_token)
    test_user_token = None
    if test_user_id:
        test_user_token = generate_test_user_token()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nUser Information:")
    print(f"  Superuser ID: {superuser_id}")
    print(f"  Bot User ID: {bot_id}")
    if test_user_id:
        print(f"  Test Human ID: {test_user_id}")
    
    print("\nTokens:")
    print(f"  Superuser Token: {superuser_token[:20]}...")
    if test_user_token:
        print(f"  Test User Token: {test_user_token[:20]}...")
    
    if test_user_id and bot_id and test_user_token:
        print("\n🧪 Test the system with:")
        print(f"""curl -X POST "http://localhost:8000/api/v1/chat/" \\
  -H "Authorization: Bearer {test_user_token}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "sender_id": "{test_user_id}",
    "receiver_id": "{bot_id}",
    "message": "What is the summary of Kenya'\''s health policy?"
  }}'""")
    
    print(f"\n📝 Remember to restart the server to load the new SUPERUSER_TOKEN from .env")


if __name__ == "__main__":
    main()