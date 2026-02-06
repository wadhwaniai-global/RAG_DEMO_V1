import json
import os
from pathlib import Path

def add_document_name_to_json(json_file_path):
    """
    Add document_name field to a JSON file based on its filename
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get the filename without extension
        filename = Path(json_file_path).stem

        # Check if document_name already exists
        if 'document_name' in data:
            print(f"[OK] {filename}.json already has document_name field")
            return False

        # Create new structure with document_name at the top
        new_data = {
            "document_name": filename,
            **data  # Unpack the rest of the data
        }

        # Write the updated JSON back to file
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

        print(f"[UPDATED] Added document_name to {filename}.json")
        return True

    except Exception as e:
        print(f"[ERROR] Error processing {json_file_path}: {str(e)}")
        return False

def main():
    # Directory containing JSON files
    json_dir = Path("D:/WAIG_RAG/New_Parsed_Json")

    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist!")
        return

    # Get all JSON files in the directory
    json_files = list(json_dir.glob("*.json"))

    if not json_files:
        print("No JSON files found in the directory!")
        return

    print(f"Found {len(json_files)} JSON files to process\n")

    updated_count = 0
    for json_file in json_files:
        if add_document_name_to_json(json_file):
            updated_count += 1

    print(f"\n{'='*50}")
    print(f"Summary: Updated {updated_count} out of {len(json_files)} files")

if __name__ == "__main__":
    main()