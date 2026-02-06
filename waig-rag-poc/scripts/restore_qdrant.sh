#!/bin/bash

# Qdrant Restore Script
# This script restores Qdrant data from a backup

set -e

# Configuration
QDRANT_URL=${QDRANT_URL:-"http://localhost:6333"}
QDRANT_API_KEY=${QDRANT_API_KEY:-""}
BACKUP_FILE=${1:-""}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ -z "$BACKUP_FILE" ]; then
    echo -e "${RED}❌ Please provide backup file path${NC}"
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}❌ Backup file not found: ${BACKUP_FILE}${NC}"
    exit 1
fi

echo -e "${GREEN}🚀 Starting Qdrant Restore${NC}"
echo "Backup file: ${BACKUP_FILE}"

# Extract backup
TEMP_DIR=$(mktemp -d)
echo -e "${YELLOW}📦 Extracting backup...${NC}"
tar -xzf "${BACKUP_FILE}" -C "${TEMP_DIR}"

# Find the extracted directory
EXTRACTED_DIR=$(find "${TEMP_DIR}" -type d -name "qdrant_backup_*" | head -1)
if [ -z "$EXTRACTED_DIR" ]; then
    echo -e "${RED}❌ Invalid backup format${NC}"
    rm -rf "${TEMP_DIR}"
    exit 1
fi

echo "Extracted to: ${EXTRACTED_DIR}"

# Check if Qdrant is accessible
echo -e "${YELLOW}🔍 Checking Qdrant connection...${NC}"
if ! curl -f "${QDRANT_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to Qdrant at ${QDRANT_URL}${NC}"
    rm -rf "${TEMP_DIR}"
    exit 1
fi
echo -e "${GREEN}✅ Qdrant is accessible${NC}"

# Read backup info
if [ -f "${EXTRACTED_DIR}/backup_info.json" ]; then
    echo -e "${YELLOW}📋 Backup information:${NC}"
    cat "${EXTRACTED_DIR}/backup_info.json" | jq '.'
fi

# Find JSON files (collections)
JSON_FILES=$(find "${EXTRACTED_DIR}" -name "*.json" -not -name "backup_info.json")

if [ -z "$JSON_FILES" ]; then
    echo -e "${YELLOW}⚠️  No collection files found in backup${NC}"
    rm -rf "${TEMP_DIR}"
    exit 0
fi

echo "Found collection files: $(echo $JSON_FILES | tr ' ' '\n' | xargs -I {} basename {} .json | tr '\n' ' ')"

# Restore each collection
for json_file in $JSON_FILES; do
    collection_name=$(basename "$json_file" .json)
    echo -e "${YELLOW}📦 Restoring collection: ${collection_name}${NC}"
    
    # Import collection using Python script
    python3 scripts/migrate_qdrant.py \
        --action import \
        --target-url "${QDRANT_URL}" \
        --target-api-key "${QDRANT_API_KEY}" \
        --file "${json_file}" \
        --collection "${collection_name}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully restored ${collection_name}${NC}"
    else
        echo -e "${RED}❌ Failed to restore ${collection_name}${NC}"
    fi
done

# Cleanup
rm -rf "${TEMP_DIR}"

echo -e "${GREEN}✅ Restore completed successfully!${NC}"
