#!/bin/bash

# Docker Setup Script for WAIG RAG POC
# This script sets up the containerized environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 WAIG RAG POC - Docker Setup${NC}"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are available${NC}"

# Create necessary directories
echo -e "${YELLOW}📁 Creating necessary directories...${NC}"
mkdir -p data logs backups qdrant_config ssl

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found${NC}"
    if [ -f "docker.env.example" ]; then
        echo -e "${YELLOW}📝 Creating .env from docker.env.example...${NC}"
        cp docker.env.example .env
        echo -e "${YELLOW}🔧 Please edit .env file with your API keys before proceeding${NC}"
        echo -e "${YELLOW}   Required: OPENAI_API_KEY, LLAMA_CLOUD_API_KEY${NC}"
    else
        echo -e "${RED}❌ No docker.env.example file found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ .env file found${NC}"
fi

# Create Qdrant configuration
echo -e "${YELLOW}⚙️  Creating Qdrant configuration...${NC}"
cat > qdrant_config/config.yaml << EOF
# Qdrant Configuration
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334

storage:
  # Use persistent storage
  storage_path: /qdrant/storage
  
  # Performance settings
  wal:
    wal_capacity_mb: 32
    wal_segments_ahead: 0
  
  # Optimization settings
  performance:
    max_optimization_threads: 2
    max_compaction_threads: 2

# Logging
log_level: INFO

# Cluster settings (for future scaling)
cluster:
  enabled: false
EOF

# Create nginx configuration
echo -e "${YELLOW}🌐 Creating Nginx configuration...${NC}"
cat > nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream rag_app {
        server rag-app:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://rag_app;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Health check endpoint
        location /health {
            proxy_pass http://rag_app/health;
            access_log off;
        }
    }
}
EOF

echo -e "${GREEN}✅ Configuration files created${NC}"

# Build and start services
echo -e "${YELLOW}🔨 Building Docker images...${NC}"
docker-compose build

echo -e "${YELLOW}🚀 Starting services...${NC}"
docker-compose up -d

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 10

# Check service health
echo -e "${YELLOW}🔍 Checking service health...${NC}"

# Check Qdrant
if curl -f http://localhost:6333/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Qdrant is running${NC}"
else
    echo -e "${RED}❌ Qdrant is not responding${NC}"
fi

# Check RAG App
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ RAG App is running${NC}"
else
    echo -e "${RED}❌ RAG App is not responding${NC}"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is running${NC}"
else
    echo -e "${RED}❌ Redis is not responding${NC}"
fi

echo -e "${GREEN}🎉 Setup completed!${NC}"
echo ""
echo -e "${BLUE}📖 Access URLs:${NC}"
echo "  - RAG API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Qdrant: http://localhost:6333"
echo "  - Nginx: http://localhost:80"
echo ""
echo -e "${BLUE}🔧 Management Commands:${NC}"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Restart services: docker-compose restart"
echo "  - Update services: docker-compose pull && docker-compose up -d"
echo ""
echo -e "${BLUE}📊 Monitoring:${NC}"
echo "  - Service status: docker-compose ps"
echo "  - Resource usage: docker stats"
