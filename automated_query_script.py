"""
Automated Query Script for RAG Chat System

This script sends queries to the middleware API, waits for bot responses,
and saves all Q&A pairs in a structured JSON format.
"""

import requests
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class RAGQueryAutomation:
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        bearer_token: str = None,
        sender_id: str = None,
        receiver_id: str = None,
        timeout: int = 120,
        poll_interval: int = 2
    ):
        """
        Initialize the automation script.

        Args:
            api_url: Base URL of the middleware API
            bearer_token: JWT authentication token
            sender_id: ID of the human user (sender)
            receiver_id: ID of the bot user (receiver)
            timeout: Maximum seconds to wait for bot response
            poll_interval: Seconds between polling for new messages
        """
        self.api_url = api_url.rstrip('/')
        self.bearer_token = bearer_token
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.timeout = timeout
        self.poll_interval = poll_interval

        self.headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        self.results = []

    def send_message(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Send a message to the bot.

        Args:
            message: The query/question to send

        Returns:
            Response data from the API or None if failed
        """
        url = f"{self.api_url}/api/v1/chats/"

        payload = {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message": message
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error sending message: {e}")
            return None

    def get_latest_bot_response(self, offset_after: int = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest bot response from the conversation.

        Args:
            offset_after: Only get messages with offset greater than this

        Returns:
            Latest bot message or None
        """
        url = f"{self.api_url}/api/v1/chats/messages/"

        # Default offset to 0 if not provided
        reference_offset = offset_after if offset_after is not None else 0

        params = {
            "sender_id": self.receiver_id,  # Bot is sender
            "receiver_id": self.sender_id,   # Human is receiver
            "offset": reference_offset,
            "limit": 50,
            "before": False  # Get messages after the offset
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            messages = data.get('messages', [])

            # The API returns messages after the offset, sorted
            # Get the latest (last in list if sorted ascending)
            if messages:
                # Sort by offset to ensure we get the latest
                messages.sort(key=lambda x: x.get('offset', 0), reverse=True)
                return messages[0]

            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting messages: {e}")
            return None

    def wait_for_bot_response(self, initial_offset: int = None) -> Optional[Dict[str, Any]]:
        """
        Wait for the bot to respond to the latest query.

        Args:
            initial_offset: The offset before sending the query

        Returns:
            Bot's full message object with metadata or None if timeout
        """
        start_time = time.time()

        print("⏳ Waiting for bot response", end="", flush=True)

        while time.time() - start_time < self.timeout:
            bot_message = self.get_latest_bot_response(offset_after=initial_offset)

            if bot_message:
                message_content = bot_message.get('message', {})

                # Check if this is a real RAG response (has processing_time or status field)
                # Skip echo responses that just repeat the query
                if isinstance(message_content, dict):
                    # A real RAG response will have at least one of these fields set
                    has_processing_time = message_content.get('processing_time') is not None
                    has_status = message_content.get('status') is not None
                    has_retrieval_metadata = message_content.get('retrieval_metadata') is not None

                    if has_processing_time or has_status or has_retrieval_metadata:
                        print(" ✓")
                        return message_content

            print(".", end="", flush=True)
            time.sleep(self.poll_interval)

        print(" ⏱️ Timeout!")
        return None

    def get_current_max_offset(self) -> int:
        """
        Get the current maximum offset in the conversation.

        Returns:
            Maximum offset value
        """
        # Use conversations endpoint to get the last message offset
        url = f"{self.api_url}/api/v1/chats/conversations/"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            conversations = data.get('conversations', [])

            # Find the conversation with the bot
            for conv in conversations:
                if conv.get('participant', {}).get('id') == self.receiver_id:
                    last_message = conv.get('last_message')
                    if last_message:
                        return last_message.get('offset', 0)

            return 0
        except:
            return 0

    def process_query(self, query: str, query_index: int = None) -> Dict[str, Any]:
        """
        Process a single query and get the bot's response.

        Args:
            query: The question to ask
            query_index: Index of the query in the batch

        Returns:
            Dictionary with query, response, and metadata
        """
        if query_index is not None:
            print(f"\n{'='*60}")
            print(f"Query #{query_index + 1}: {query}")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")

        # Get current offset before sending
        current_offset = self.get_current_max_offset()

        # Send the query
        print("📤 Sending query...")
        send_result = self.send_message(query)

        if not send_result:
            return {
                "query": query,
                "response": None,
                "error": "Failed to send message",
                "timestamp": datetime.now().isoformat(),
                "success": False
            }

        print("✅ Query sent successfully")

        # Wait for bot response
        bot_response = self.wait_for_bot_response(initial_offset=current_offset)

        if bot_response:
            response_text = bot_response.get('text', '') if isinstance(bot_response, dict) else str(bot_response)
            print(f"✅ Bot responded: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")

            return {
                "query": query,
                "response": response_text,
                "confidence_score": bot_response.get('confidence_score') if isinstance(bot_response, dict) else None,
                "sources": bot_response.get('sources') if isinstance(bot_response, dict) else None,
                "retrieval_metadata": bot_response.get('retrieval_metadata') if isinstance(bot_response, dict) else None,
                "processing_time": bot_response.get('processing_time') if isinstance(bot_response, dict) else None,
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
        Process a list of queries sequentially.

        Args:
            queries: List of questions to ask

        Returns:
            List of results with queries and responses
        """
        print(f"\n🚀 Starting automated query processing")
        print(f"📝 Total queries: {len(queries)}")
        print(f"⏱️  Timeout per query: {self.timeout}s")
        print(f"🔄 Poll interval: {self.poll_interval}s")

        results = []

        for i, query in enumerate(queries):
            result = self.process_query(query, query_index=i)
            results.append(result)

            # Small delay between queries to avoid overwhelming the system
            if i < len(queries) - 1:
                time.sleep(1)

        self.results = results
        return results

    def save_results(self, output_file: str = None) -> str:
        """
        Save results to a JSON file.

        Args:
            output_file: Path to output file. If None, auto-generate filename.

        Returns:
            Path to the saved file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"rag_queries_results_{timestamp}.json"

        # Prepare output data
        output_data = {
            "metadata": {
                "total_queries": len(self.results),
                "successful_queries": sum(1 for r in self.results if r.get('success')),
                "failed_queries": sum(1 for r in self.results if not r.get('success')),
                "generated_at": datetime.now().isoformat(),
                "api_url": self.api_url,
                "sender_id": self.sender_id,
                "receiver_id": self.receiver_id
            },
            "results": self.results
        }

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_file}")
        return output_file

    def print_summary(self):
        """Print a summary of the results."""
        successful = sum(1 for r in self.results if r.get('success'))
        failed = sum(1 for r in self.results if not r.get('success'))

        print(f"\n{'='*60}")
        print(f"📊 SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful queries: {successful}/{len(self.results)}")
        print(f"❌ Failed queries: {failed}/{len(self.results)}")

        if failed > 0:
            print(f"\n⚠️  Failed queries:")
            for i, result in enumerate(self.results):
                if not result.get('success'):
                    print(f"   {i + 1}. {result['query']}")
                    print(f"      Error: {result.get('error', 'Unknown')}")


def main():
    """Main execution function."""

    # Configuration - UPDATE THESE VALUES
    API_URL = "http://localhost:8000"
    BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNjhlNGEyMGU3YzdjODgzNGZhYzkwM2U3IiwibmFtZSI6InN1cGVydXNlciIsInVzZXJfdHlwZSI6Imh1bWFuIiwiZXhwIjoxNzYyNDQwMzUyfQ.wfkKoJXsWpK8Dx2o9o5Uod5SUUh0D2m97Zx1av1kXxM"
    SENDER_ID = "68e4a20e7c7c8834fac903e7"  # Human user ID
    RECEIVER_ID = "68e4a41a7c7c8834fac903ef"  # Bot user ID

    # List of queries - ADD YOUR QUESTIONS HERE
    queries = [
        "How should I screen household contacts of a confirmed TB patient??",
        "What tests should I request first when I suspect pulmonary TB?",
        "What is the standard TB regimen for a newly diagnosed adult?",
        "How long is the continuation phase of treatment for DS-TB?",
        "How do I diagnose TB in a child who cannot produce sputum?",
        "What are the weight-based drug dosages for children under 5 years?",
        "Which drugs are used in the short regimen for MDR-TB?",
        "Which patients are eligible for TB preventive therapy?",
        "Can PLHIV on ART also take TPT? If yes, which regimen is preferred?",
        "How do I manage TB in a pregnant woman?",
        "What precautions are needed when a TB patient also has diabetes?",
        "What triage measures should be in place at a TB clinic to reduce infection risk?",
        "What type of mask should healthcare workers use when attending TB patients?",
        "What nutrition advice should I give to a TB patient?",
        "Which forms should I fill after diagnosing a new TB case?",
        "How do I record a patient who was lost to follow-up and later returned to treatment?"
    ]

    # Initialize automation
    automation = RAGQueryAutomation(
        api_url=API_URL,
        bearer_token=BEARER_TOKEN,
        sender_id=SENDER_ID,
        receiver_id=RECEIVER_ID,
        timeout=60,  # Wait up to 60 seconds for each response
        poll_interval=2  # Check for response every 2 seconds
    )

    # Process all queries
    results = automation.process_queries(queries)

    # Save results
    output_file = automation.save_results()

    # Print summary
    automation.print_summary()

    print(f"\n✨ Processing complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
