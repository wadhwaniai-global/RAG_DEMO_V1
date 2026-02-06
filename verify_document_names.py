import json
import os
from pathlib import Path

def verify_json_files():
    """
    Verify that all JSON files have the document_name field
    """
    json_dir = Path("D:/WAIG_RAG/New_Parsed_Json")

    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist!")
        return

    json_files = list(json_dir.glob("*.json"))

    print(f"Verifying {len(json_files)} JSON files...\n")
    print("-" * 80)

    all_valid = True

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            filename = json_file.stem

            if 'document_name' not in data:
                print(f"[MISSING] {filename}.json - No document_name field!")
                all_valid = False
            elif data['document_name'] != filename:
                print(f"[MISMATCH] {filename}.json - document_name is '{data['document_name']}', expected '{filename}'")
                all_valid = False
            else:
                print(f"[OK] {filename}.json - document_name: '{data['document_name']}'")

        except Exception as e:
            print(f"[ERROR] {json_file.name} - {str(e)}")
            all_valid = False

    print("-" * 80)
    if all_valid:
        print("\nSUCCESS: All files have correct document_name fields!")
    else:
        print("\nWARNING: Some files have issues. Please check the output above.")

    return all_valid

if __name__ == "__main__":
    verify_json_files()