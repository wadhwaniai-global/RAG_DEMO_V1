"""
Automated Query Testing Script for RAG Chat System
===================================================
This script automates the process of:
1. Authenticating with the middleware API
2. Reading queries from queries.txt or command line
3. Sending each query to the bot
4. Waiting for and collecting responses
5. Saving all results to a structured JSON file
"""

import requests
import time
import json
import sys
import io
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import argparse

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class AuthenticationManager:
    """Handles authentication and token management"""

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip('/')
        self.token = None
        self.user_id = None
        self.bot_id = None

    def login(self, username: str, password: str) -> bool:
        """
        Authenticate with the API and get a token.

        Args:
            username: Username for authentication
            password: Password for authentication

        Returns:
            True if authentication successful, False otherwise
        """
        url = f"{self.api_url}/api/v1/users/generate-token"

        payload = {
            "name": username,
            "password": password
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            self.token = data.get('access_token')
            self.user_id = data.get('user_id')

            print(f"✅ Authenticated as user: {username} (ID: {self.user_id})")
            return True

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"❌ Invalid credentials for user: {username}")
            else:
                print(f"❌ Authentication failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False

    def get_bot_user(self) -> Optional[str]:
        """
        Get the bot user ID from the API.

        Returns:
            Bot user ID or None if not found
        """
        if not self.token:
            print("❌ No authentication token available")
            return None

        url = f"{self.api_url}/api/v1/users/"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            users = response.json()

            # Find the bot user
            for user in users:
                if user.get('user_type') == 'bot':
                    self.bot_id = user.get('id')
                    print(f"✅ Found bot user: {user.get('name')} (ID: {self.bot_id})")
                    return self.bot_id

            print("❌ No bot user found in the system")
            return None

        except Exception as e:
            print(f"❌ Error fetching users: {e}")
            return None

    def create_default_users(self) -> bool:
        """
        Create default users if they don't exist.

        Returns:
            True if users created successfully, False otherwise
        """
        print("\n🔧 Setting up default users...")

        # Try to create a test user
        url = f"{self.api_url}/api/v1/users/"

        test_user_data = {
            "name": "test_user",
            "email": "test_user@example.com",
            "password": "testpassword123",
            "user_type": "human"
        }

        try:
            response = requests.post(url, json=test_user_data)
            if response.status_code == 201:
                data = response.json()
                self.user_id = data.get('id')
                print(f"✅ Created test user with ID: {self.user_id}")

                # Now login to get token
                return self.login("test_user", "testpassword123")
            elif response.status_code == 400:
                # User might already exist, try to login
                print("ℹ️ Test user might already exist, attempting login...")
                return self.login("test_user", "testpassword123")
            else:
                print(f"❌ Failed to create test user: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error creating test user: {e}")
            return False


class RAGQueryTester:
    """Main class for automated query testing"""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        timeout: int = 60,
        poll_interval: int = 2
    ):
        """
        Initialize the query tester.

        Args:
            api_url: Base URL of the middleware API
            timeout: Maximum seconds to wait for bot response
            poll_interval: Seconds between polling for new messages
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.poll_interval = poll_interval

        self.auth = AuthenticationManager(api_url)
        self.results = []
        self.start_time = None

    def authenticate(self, username: str = None, password: str = None) -> bool:
        """
        Authenticate with the API.

        Args:
            username: Username (if None, will prompt or use default)
            password: Password (if None, will prompt or use default)

        Returns:
            True if authentication successful
        """
        # Try provided credentials first
        if username and password:
            if self.auth.login(username, password):
                return self.auth.get_bot_user() is not None

        # Try default credentials
        print("\n🔐 Attempting authentication with default credentials...")

        # Try common default users
        default_users = [
            ("test_human", "testpassword123"),
            ("test_user", "testpassword123"),
            ("superuser", "securepassword123"),
        ]

        for user, pwd in default_users:
            print(f"   Trying user: {user}")
            if self.auth.login(user, pwd):
                if self.auth.get_bot_user():
                    return True

        # If all fail, try to create default users
        print("\n⚠️ No valid credentials found. Creating default users...")
        if self.auth.create_default_users():
            return self.auth.get_bot_user() is not None

        return False

    def read_queries_from_file(self, file_path: str) -> List[str]:
        """
        Read queries from a text file.

        Args:
            file_path: Path to the queries file

        Returns:
            List of queries
        """
        queries = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        queries.append(line)

            print(f"✅ Loaded {len(queries)} queries from {file_path}")
            return queries

        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return []
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return []

    def send_message(self, message: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Send a message to the bot.

        Args:
            message: The query/message to send

        Returns:
            Tuple of (success, response_data)
        """
        url = f"{self.api_url}/api/v1/chats/"

        headers = {
            "Authorization": f"Bearer {self.auth.token}",
            "Content-Type": "application/json"
        }

        payload = {
            "sender_id": self.auth.user_id,
            "receiver_id": self.auth.bot_id,
            "message": message
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return True, response.json()
        except requests.exceptions.HTTPError as e:
            print(f"   ❌ HTTP Error: {e.response.status_code} - {e.response.text[:100]}")
            return False, None
        except Exception as e:
            print(f"   ❌ Error sending message: {e}")
            return False, None

    def get_bot_response(self, offset_after: int = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest bot response.

        Args:
            offset_after: Only get messages with offset greater than this

        Returns:
            Bot response or None
        """
        url = f"{self.api_url}/api/v1/chats/messages/"

        headers = {
            "Authorization": f"Bearer {self.auth.token}"
        }

        # Get messages sent BY the bot TO the user
        params = {
            "sender_id": self.auth.bot_id,
            "receiver_id": self.auth.user_id,
            "offset": offset_after if offset_after else 0,
            "limit": 20,
            "before": False
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            messages = data.get('messages', [])

            # Filter messages to only those after the initial offset
            if offset_after is not None:
                messages = [m for m in messages if m.get('offset', 0) > offset_after]

            if messages:
                # Sort by offset to get messages in order
                messages.sort(key=lambda x: x.get('offset', 0))
                # Return the first bot message after the offset
                for msg in messages:
                    # Ensure it's from the bot and has content
                    if msg.get('sender_id') == self.auth.bot_id:
                        message_content = msg.get('message', {})
                        # Make sure it has actual content (not empty)
                        if isinstance(message_content, dict) and message_content.get('text'):
                            return msg
                        elif isinstance(message_content, str) and message_content:
                            return msg

            return None
        except Exception as e:
            print(f"   ❌ Error getting response: {e}")
            return None

    def wait_for_response(self, initial_offset: int = None) -> Optional[Dict[str, Any]]:
        """
        Wait for bot response with timeout.

        Args:
            initial_offset: The offset before sending query

        Returns:
            Bot response or None if timeout
        """
        start_time = time.time()
        dots = 0

        print("   ⏳ Waiting for response", end="", flush=True)

        while time.time() - start_time < self.timeout:
            response = self.get_bot_response(offset_after=initial_offset)

            if response:
                message_content = response.get('message', {})

                # Check if this is a real RAG response
                if isinstance(message_content, dict):
                    # New format: message is a dict with 'text' and other fields
                    if 'text' in message_content and message_content.get('text'):
                        # Check if this is not just an echo (has additional RAG metadata)
                        has_rag_metadata = any(k in message_content for k in [
                            'confidence_score', 'sources', 'retrieval_metadata',
                            'processing_time', 'status'
                        ])

                        # If it has RAG metadata or sufficient text, it's a real response
                        if has_rag_metadata or len(message_content.get('text', '')) > 20:
                            print(" ✓")
                            return message_content
                    # Legacy check for other possible fields
                    elif any(k in message_content for k in ['answer', 'response']):
                        print(" ✓")
                        return message_content
                elif isinstance(message_content, str) and len(message_content) > 0:
                    # Legacy string format
                    print(" ✓")
                    return {"text": message_content}

            # Show progress dots
            dots = (dots + 1) % 4
            print("\r   ⏳ Waiting for response" + "." * dots + " " * (3 - dots), end="", flush=True)
            time.sleep(self.poll_interval)

        print(" ⏱️ Timeout!")
        return None

    def get_current_offset(self) -> int:
        """Get the current maximum offset in conversations."""
        url = f"{self.api_url}/api/v1/chats/conversations/"

        headers = {"Authorization": f"Bearer {self.auth.token}"}

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                conversations = data.get('conversations', [])

                for conv in conversations:
                    if conv.get('participant', {}).get('id') == self.auth.bot_id:
                        last_msg = conv.get('last_message')
                        if last_msg:
                            return last_msg.get('offset', 0)
            return 0
        except:
            return 0

    def process_query(self, query: str, index: int = None, total: int = None) -> Dict[str, Any]:
        """
        Process a single query and get response.

        Args:
            query: The query to process
            index: Query index (for display)
            total: Total number of queries (for display)

        Returns:
            Dictionary with query, response, and metadata
        """
        # Display query
        if index is not None and total is not None:
            print(f"\n{'='*70}")
            print(f"📝 Query {index + 1}/{total}: {query[:100]}{'...' if len(query) > 100 else ''}")
            print(f"{'='*70}")
        else:
            print(f"\n📝 Processing: {query[:100]}{'...' if len(query) > 100 else ''}")

        # Get current offset
        current_offset = self.get_current_offset()

        # Send query
        print("   📤 Sending query...")
        success, send_result = self.send_message(query)

        if not success:
            return {
                "query": query,
                "response": None,
                "error": "Failed to send message",
                "timestamp": datetime.now().isoformat(),
                "success": False
            }

        print("   ✅ Query sent")

        # Wait for response
        bot_response = self.wait_for_response(initial_offset=current_offset)

        if bot_response:
            # Extract response text and metadata properly
            if isinstance(bot_response, dict):
                # Primary text extraction
                response_text = (
                    bot_response.get('text') or
                    bot_response.get('answer') or
                    bot_response.get('response') or
                    str(bot_response)
                )

                # Extract all metadata fields
                confidence_score = bot_response.get('confidence_score')
                sources = bot_response.get('sources')
                processing_time = bot_response.get('processing_time')
                status = bot_response.get('status')

                # Handle retrieval_metadata specially
                retrieval_metadata = bot_response.get('retrieval_metadata')
            else:
                response_text = str(bot_response)
                confidence_score = None
                sources = None
                processing_time = None
                status = None
                retrieval_metadata = None

            # Display snippet of response
            snippet = response_text[:150] + ('...' if len(response_text) > 150 else '')
            print(f"   ✅ Response received: {snippet}")

            return {
                "query": query,
                "response": response_text,
                "full_response": bot_response if isinstance(bot_response, dict) else {"text": response_text},
                "confidence_score": confidence_score,
                "sources": sources,
                "retrieval_metadata": retrieval_metadata,
                "processing_time": processing_time,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
        else:
            return {
                "query": query,
                "response": None,
                "error": "No response received within timeout",
                "timestamp": datetime.now().isoformat(),
                "success": False
            }

    def process_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries sequentially.

        Args:
            queries: List of queries to process

        Returns:
            List of results
        """
        self.start_time = time.time()
        results = []

        print(f"\n{'='*70}")
        print(f"🚀 Starting Automated Query Testing")
        print(f"{'='*70}")
        print(f"📊 Total queries: {len(queries)}")
        print(f"⏱️ Timeout per query: {self.timeout}s")
        print(f"🔄 Poll interval: {self.poll_interval}s")

        for i, query in enumerate(queries):
            result = self.process_query(query, index=i, total=len(queries))
            results.append(result)

            # Small delay between queries
            if i < len(queries) - 1:
                time.sleep(1)

        self.results = results
        return results

    def save_results(self, output_file: str = None) -> str:
        """
        Save results to JSON file.

        Args:
            output_file: Path to output file (auto-generated if None)

        Returns:
            Path to saved file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"query_test_results_{timestamp}.json"

        # Calculate statistics
        successful = sum(1 for r in self.results if r.get('success'))
        failed = sum(1 for r in self.results if not r.get('success'))
        total_time = time.time() - self.start_time if self.start_time else 0

        # Prepare output
        output_data = {
            "metadata": {
                "total_queries": len(self.results),
                "successful_queries": successful,
                "failed_queries": failed,
                "success_rate": f"{(successful/len(self.results)*100):.1f}%" if self.results else "0%",
                "total_time_seconds": round(total_time, 2),
                "average_time_per_query": round(total_time / len(self.results), 2) if self.results else 0,
                "generated_at": datetime.now().isoformat(),
                "api_url": self.api_url,
                "timeout_seconds": self.timeout
            },
            "results": self.results
        }

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_file}")
        return output_file

    def print_summary(self):
        """Print a summary of results."""
        if not self.results:
            print("\n⚠️ No results to summarize")
            return

        successful = sum(1 for r in self.results if r.get('success'))
        failed = sum(1 for r in self.results if not r.get('success'))

        print(f"\n{'='*70}")
        print(f"📊 SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Successful: {successful}/{len(self.results)} ({successful/len(self.results)*100:.1f}%)")
        print(f"❌ Failed: {failed}/{len(self.results)}")

        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"⏱️ Total time: {total_time:.1f}s")
            print(f"⚡ Average time per query: {total_time/len(self.results):.1f}s")

        # Show failed queries
        if failed > 0:
            print(f"\n⚠️ Failed queries:")
            for i, result in enumerate(self.results):
                if not result.get('success'):
                    query = result['query'][:50] + ('...' if len(result['query']) > 50 else '')
                    print(f"   {i + 1}. {query}")
                    print(f"      Error: {result.get('error', 'Unknown')}")

        # Show low confidence responses
        low_confidence = [
            (i, r) for i, r in enumerate(self.results)
            if r.get('success') and r.get('confidence_score') and r['confidence_score'] < 0.5
        ]

        if low_confidence:
            print(f"\n⚠️ Low confidence responses (< 0.5):")
            for i, result in low_confidence:
                query = result['query'][:50] + ('...' if len(result['query']) > 50 else '')
                print(f"   {i + 1}. {query}")
                print(f"      Confidence: {result['confidence_score']:.2f}")


def main():
    """Main execution function."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Automated Query Testing for RAG Chat System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Use queries.txt with default settings
  %(prog)s --file custom_queries.txt  # Use custom query file
  %(prog)s --username test_user --password pass123
  %(prog)s --api-url http://localhost:8080 --timeout 90
  %(prog)s --output results.json
        """
    )

    parser.add_argument(
        '--file', '-f',
        default='queries.txt',
        help='Path to queries file (default: queries.txt)'
    )

    parser.add_argument(
        '--api-url',
        default='http://localhost:8000',
        help='API base URL (default: http://localhost:8000)'
    )

    parser.add_argument(
        '--username', '-u',
        help='Username for authentication'
    )

    parser.add_argument(
        '--password', '-p',
        help='Password for authentication'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Timeout per query in seconds (default: 60)'
    )

    parser.add_argument(
        '--poll-interval',
        type=int,
        default=2,
        help='Poll interval in seconds (default: 2)'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: auto-generated)'
    )

    args = parser.parse_args()

    # Initialize tester
    tester = RAGQueryTester(
        api_url=args.api_url,
        timeout=args.timeout,
        poll_interval=args.poll_interval
    )

    # Authenticate
    print(f"\n🔐 Authenticating with {args.api_url}...")
    if not tester.authenticate(username=args.username, password=args.password):
        print("\n❌ Authentication failed. Please check your credentials and try again.")
        print("\nTip: You can run the setup script to create default users:")
        print("  cd rag-poc-middleware-app && python setup_rag_system.py")
        sys.exit(1)

    # Load queries
    queries = tester.read_queries_from_file(args.file)

    if not queries:
        # If no file, use default queries
        print(f"\n⚠️ No queries found in {args.file}")
        print("Using default test queries...")

        queries = [
            "How should I screen household contacts of a confirmed TB patient?",
            "What tests should I request first when I suspect pulmonary TB?",
            "What is the standard TB regimen for a newly diagnosed adult?",
        ]

    # Process queries
    results = tester.process_queries(queries)

    # Save results
    output_file = tester.save_results(args.output)

    # Print summary
    tester.print_summary()

    print(f"\n✨ Testing complete!")
    print(f"📄 View detailed results in: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)