#!/bin/bash

# Qdrant Backup Script
# This script creates a backup of Qdrant data and collections

set -e

# Configuration
QDRANT_URL=${QDRANT_URL:-"http://localhost:6333"}
QDRANT_API_KEY=${QDRANT_API_KEY:-""}
BACKUP_DIR=${BACKUP_DIR:-"./backups"}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="qdrant_backup_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Qdrant Backup${NC}"
echo "Backup directory: ${BACKUP_DIR}"
echo "Timestamp: ${TIMESTAMP}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Check if Qdrant is accessible
echo -e "${YELLOW}🔍 Checking Qdrant connection...${NC}"
if ! curl -f "${QDRANT_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to Qdrant at ${QDRANT_URL}${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Qdrant is accessible${NC}"

# Get list of collections
echo -e "${YELLOW}📋 Getting collection list...${NC}"
COLLECTIONS=$(curl -s "${QDRANT_URL}/collections" | jq -r '.result.collections[].name' 2>/dev/null || echo "")

if [ -z "$COLLECTIONS" ]; then
    echo -e "${YELLOW}⚠️  No collections found${NC}"
    exit 0
fi

echo "Found collections: $COLLECTIONS"

# Backup each collection
for collection in $COLLECTIONS; do
    echo -e "${YELLOW}📦 Backing up collection: ${collection}${NC}"
    
    # Export collection using Python script
    python3 scripts/migrate_qdrant.py \
        --action export \
        --source-url "${QDRANT_URL}" \
        --source-api-key "${QDRANT_API_KEY}" \
        --collection "${collection}" \
        --file "${BACKUP_DIR}/${BACKUP_NAME}/${collection}.json"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully backed up ${collection}${NC}"
    else
        echo -e "${RED}❌ Failed to backup ${collection}${NC}"
    fi
done

# Create backup metadata
cat > "${BACKUP_DIR}/${BACKUP_NAME}/backup_info.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "qdrant_url": "${QDRANT_URL}",
    "collections": [$(echo $COLLECTIONS | sed 's/ /", "/g' | sed 's/^/"/' | sed 's/$/"/')],
    "backup_version": "1.0"
}
EOF

# Create compressed archive
echo -e "${YELLOW}🗜️  Creating compressed archive...${NC}"
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
rm -rf "${BACKUP_NAME}"

echo -e "${GREEN}✅ Backup completed successfully!${NC}"
echo "Backup file: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo "Size: $(du -h "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | cut -f1)"
