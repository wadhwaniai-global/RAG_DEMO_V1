import requests
import json

# Configuration
USER_ID = "690390a6b3750101f0b11fda"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNjkwMzkwYTZiMzc1MDEwMWYwYjExZmRhIiwibmFtZSI6ImFzaGEiLCJ1c2VyX3R5cGUiOiJodW1hbiIsImV4cCI6MTc2NDQzODgzMn0.0wmZVYB-EVl8Sf48sj_fxGNiAuqMwx4pUU2uDuTiIJM"
API_URL = "http://localhost:8080/api/v1/users/"

# All document names from both directories
document_filters = [
    # From Parsed_Englsih_Bot
    "ASHA Incentives 2024-2025",
    "ASHA Activities Guide",
    "ASHA Incentives April 2024",
    "ASHA NCD Module",
    "ASHA Booklet 2022",
    "asha_v1",
    "ENT Care Training Manual for MPW",
    "Eye Care Training Manual for ASHA",
    "FAQ_on_Immunization_for_Health_Workers-English",

    # From New_Parsed_Json
    "ASHA_Handbook-Mobilizing_for_Action_on_Violence_against_Women_English",
    "ASHA_Induction_Module_English",
    "book-no-1",
    "book-no-2",
    "book-no-3",
    "book-no-4",
    "book-no-5",
    "book-no-6",
    "book-no-7",
    "Reaching_The_Unreached_Brochure_for_ASHA"
]

# Request payload
payload = {
    "name": "Asha-bot",
    "email": "asha@example.com",
    "user_type": "bot",
    "document_filter": document_filters
}

# Headers
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def create_bot_user():
    """Create bot user with document filters"""
    print("Creating bot user with all document filters...")
    print(f"Total documents in filter: {len(document_filters)}")
    print("\nDocument filters:")
    for i, doc in enumerate(document_filters, 1):
        print(f"  {i}. {doc}")

    try:
        response = requests.post(API_URL,
                                headers=headers,
                                json=payload)

        if response.status_code == 200 or response.status_code == 201:
            print("\n✓ Bot user created successfully!")
            print("\nResponse:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"\n✗ Failed to create bot user")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("\n✗ Connection Error: Could not connect to the API")
        print("   Make sure the server is running on http://localhost:8080")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")

if __name__ == "__main__":
    create_bot_user()