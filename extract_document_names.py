import json
import os
from pathlib import Path

def extract_document_names():
    """
    Extract document_name values from all JSON files in both directories
    """
    # Directories to process
    directories = {
        "Parsed_Englsih_Bot": Path("D:/WAIG_RAG/Parsed_Englsih_Bot"),
        "New_Parsed_Json": Path("D:/WAIG_RAG/New_Parsed_Json")
    }

    all_document_names = []

    for dir_name, dir_path in directories.items():
        print(f"\nProcessing {dir_name}...")
        print("-" * 50)

        if not dir_path.exists():
            print(f"  Directory {dir_path} does not exist!")
            continue

        json_files = sorted(list(dir_path.glob("*.json")))

        if not json_files:
            print(f"  No JSON files found in {dir_name}")
            continue

        dir_document_names = []

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'document_name' in data:
                    doc_name = data['document_name']
                    dir_document_names.append(doc_name)
                    print(f"  [OK] {json_file.name}: '{doc_name}'")
                else:
                    print(f"  [MISSING] {json_file.name}: No document_name field")
                    # Use filename without extension as fallback
                    fallback_name = json_file.stem
                    dir_document_names.append(f"{fallback_name} (filename-based)")

            except Exception as e:
                print(f"  [ERROR] {json_file.name}: {str(e)}")

        # Add section header and names to the main list
        if dir_document_names:
            all_document_names.append(f"\n{'='*60}")
            all_document_names.append(f"{dir_name} ({len(dir_document_names)} files)")
            all_document_names.append("="*60)
            all_document_names.extend([f"{i+1}. {name}" for i, name in enumerate(dir_document_names)])

    # Write to text file
    output_file = Path("D:/WAIG_RAG/document_names_list.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DOCUMENT NAMES FROM ALL JSON FILES\n")
        f.write("Generated from Parsed_Englsih_Bot and New_Parsed_Json directories\n")
        f.write("="*60 + "\n")

        for item in all_document_names:
            f.write(item + "\n")

        f.write("\n" + "="*60 + "\n")
        f.write(f"Total files processed: {len([x for x in all_document_names if x.startswith((' ', '1', '2', '3', '4', '5', '6', '7', '8', '9'))])}\n")

    print(f"\n{'='*60}")
    print(f"Document names have been saved to: {output_file}")

    return output_file

if __name__ == "__main__":
    extract_document_names()