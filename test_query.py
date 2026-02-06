"""Quick test script to verify the automated query system works"""

from automated_query_script import RAGQueryAutomation

# Configuration
API_URL = "http://localhost:8000"
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNjhlNGEyMGU3YzdjODgzNGZhYzkwM2U3IiwibmFtZSI6InN1cGVydXNlciIsInVzZXJfdHlwZSI6Imh1bWFuIiwiZXhwIjoxNzYyNDQwMzUyfQ.wfkKoJXsWpK8Dx2o9o5Uod5SUUh0D2m97Zx1av1kXxM"
SENDER_ID = "68e4a20e7c7c8834fac903e7"  # Human user ID
RECEIVER_ID = "68e4a41a7c7c8834fac903ef"  # Bot user ID

# Initialize automation
automation = RAGQueryAutomation(
    api_url=API_URL,
    bearer_token=BEARER_TOKEN,
    sender_id=SENDER_ID,
    receiver_id=RECEIVER_ID,
    timeout=60,
    poll_interval=2
)

# Test with a single query
test_query = "What are the first-line drugs for treating tuberculosis?"

print("\nTesting automated query system...")
print(f"Query: {test_query}\n")

result = automation.process_query(test_query)

print("\n" + "="*60)
print("RESULT:")
print("="*60)
print(f"Success: {result['success']}")
if result['success']:
    print(f"Response: {result['response'][:200]}...")
    print(f"Confidence: {result.get('confidence_score', 'N/A')}")
    print(f"Processing Time: {result.get('processing_time', 'N/A')}s")
    print(f"Sources: {len(result.get('sources', []))} documents")
else:
    print(f"Error: {result.get('error', 'Unknown error')}")

print("\nTest complete!")
