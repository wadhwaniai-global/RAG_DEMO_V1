#!/bin/bash

# Migration Script: Local Qdrant to Docker Qdrant
# This script helps migrate data from a local Qdrant instance to the Docker container

set -e

# Configuration
LOCAL_QDRANT_URL=${LOCAL_QDRANT_URL:-"http://localhost:6333"}
DOCKER_QDRANT_URL=${DOCKER_QDRANT_URL:-"http://localhost:6333"}
LOCAL_API_KEY=${LOCAL_API_KEY:-""}
DOCKER_API_KEY=${DOCKER_API_KEY:-""}
BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 Qdrant Migration: Local to Docker${NC}"
echo "====================================="

# Check if local Qdrant is running
echo -e "${YELLOW}🔍 Checking local Qdrant connection...${NC}"
if ! curl -f "${LOCAL_QDRANT_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to local Qdrant at ${LOCAL_QDRANT_URL}${NC}"
    echo "Please ensure your local Qdrant instance is running"
    exit 1
fi
echo -e "${GREEN}✅ Local Qdrant is accessible${NC}"

# Check if Docker Qdrant is running
echo -e "${YELLOW}🔍 Checking Docker Qdrant connection...${NC}"
if ! curl -f "${DOCKER_QDRANT_URL}/health" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Docker Qdrant is not running. Starting Docker services...${NC}"
    docker-compose up -d qdrant
    
    # Wait for Qdrant to be ready
    echo -e "${YELLOW}⏳ Waiting for Docker Qdrant to be ready...${NC}"
    for i in {1..30}; do
        if curl -f "${DOCKER_QDRANT_URL}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Docker Qdrant is ready${NC}"
            break
        fi
        sleep 2
        echo -n "."
    done
    
    if ! curl -f "${DOCKER_QDRANT_URL}/health" > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker Qdrant failed to start${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Docker Qdrant is accessible${NC}"
fi

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Get list of collections from local Qdrant
echo -e "${YELLOW}📋 Getting collections from local Qdrant...${NC}"
COLLECTIONS=$(curl -s "${LOCAL_QDRANT_URL}/collections" | jq -r '.result.collections[].name' 2>/dev/null || echo "")

if [ -z "$COLLECTIONS" ]; then
    echo -e "${YELLOW}⚠️  No collections found in local Qdrant${NC}"
    exit 0
fi

echo "Found collections: $COLLECTIONS"

# Export collections from local Qdrant
echo -e "${YELLOW}📦 Exporting collections from local Qdrant...${NC}"
for collection in $COLLECTIONS; do
    echo -e "${YELLOW}  Exporting: ${collection}${NC}"
    
    python3 scripts/migrate_qdrant.py \
        --action export \
        --source-url "${LOCAL_QDRANT_URL}" \
        --source-api-key "${LOCAL_API_KEY}" \
        --collection "${collection}" \
        --file "${BACKUP_DIR}/${collection}_${TIMESTAMP}.json"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✅ Exported ${collection}${NC}"
    else
        echo -e "${RED}  ❌ Failed to export ${collection}${NC}"
    fi
done

# Import collections to Docker Qdrant
echo -e "${YELLOW}📥 Importing collections to Docker Qdrant...${NC}"
for collection in $COLLECTIONS; do
    backup_file="${BACKUP_DIR}/${collection}_${TIMESTAMP}.json"
    
    if [ -f "$backup_file" ]; then
        echo -e "${YELLOW}  Importing: ${collection}${NC}"
        
        python3 scripts/migrate_qdrant.py \
            --action import \
            --target-url "${DOCKER_QDRANT_URL}" \
            --target-api-key "${DOCKER_API_KEY}" \
            --file "${backup_file}" \
            --collection "${collection}"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✅ Imported ${collection}${NC}"
        else
            echo -e "${RED}  ❌ Failed to import ${collection}${NC}"
        fi
    else
        echo -e "${RED}  ❌ Backup file not found: ${backup_file}${NC}"
    fi
done

# Verify migration
echo -e "${YELLOW}🔍 Verifying migration...${NC}"
DOCKER_COLLECTIONS=$(curl -s "${DOCKER_QDRANT_URL}/collections" | jq -r '.result.collections[].name' 2>/dev/null || echo "")

echo "Collections in Docker Qdrant: $DOCKER_COLLECTIONS"

# Compare collection counts
for collection in $COLLECTIONS; do
    if echo "$DOCKER_COLLECTIONS" | grep -q "$collection"; then
        # Get point counts
        LOCAL_COUNT=$(curl -s "${LOCAL_QDRANT_URL}/collections/${collection}" | jq -r '.result.points_count' 2>/dev/null || echo "0")
        DOCKER_COUNT=$(curl -s "${DOCKER_QDRANT_URL}/collections/${collection}" | jq -r '.result.points_count' 2>/dev/null || echo "0")
        
        if [ "$LOCAL_COUNT" = "$DOCKER_COUNT" ]; then
            echo -e "${GREEN}  ✅ ${collection}: ${DOCKER_COUNT} points migrated successfully${NC}"
        else
            echo -e "${YELLOW}  ⚠️  ${collection}: Local=${LOCAL_COUNT}, Docker=${DOCKER_COUNT}${NC}"
        fi
    else
        echo -e "${RED}  ❌ ${collection}: Not found in Docker Qdrant${NC}"
    fi
done

# Create migration summary
cat > "${BACKUP_DIR}/migration_summary_${TIMESTAMP}.txt" << EOF
Qdrant Migration Summary
=======================
Timestamp: ${TIMESTAMP}
Source: ${LOCAL_QDRANT_URL}
Target: ${DOCKER_QDRANT_URL}

Collections migrated:
$(echo $COLLECTIONS | tr ' ' '\n' | sed 's/^/- /')

Backup files created:
$(ls -1 ${BACKUP_DIR}/*_${TIMESTAMP}.json | sed 's/^/- /')
EOF

echo -e "${GREEN}🎉 Migration completed!${NC}"
echo -e "${BLUE}📄 Summary saved to: ${BACKUP_DIR}/migration_summary_${TIMESTAMP}.txt${NC}"
echo -e "${BLUE}💾 Backup files saved in: ${BACKUP_DIR}/${NC}"
