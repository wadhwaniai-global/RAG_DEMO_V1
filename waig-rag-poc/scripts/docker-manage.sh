#!/bin/bash

# Docker Management Script for WAIG RAG POC
# This script provides easy management commands for the containerized environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo -e "${BLUE}WAIG RAG POC - Docker Management${NC}"
    echo "=================================="
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  logs        Show logs (use -f for follow)"
    echo "  build       Build/rebuild images"
    echo "  update      Update and restart services"
    echo "  clean       Clean up containers and images"
    echo "  backup      Backup Qdrant data"
    echo "  restore     Restore Qdrant data from backup"
    echo "  shell       Open shell in RAG app container"
    echo "  qdrant      Open Qdrant shell"
    echo "  health      Check service health"
    echo "  migrate     Migrate data between Qdrant instances"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 logs -f"
    echo "  $0 backup"
    echo "  $0 restore backup_file.tar.gz"
    echo "  $0 migrate --source-url http://old-qdrant:6333 --target-url http://new-qdrant:6333"
}

# Function to check if services are running
check_services() {
    if ! docker-compose ps | grep -q "Up"; then
        echo -e "${YELLOW}⚠️  No services are currently running${NC}"
        return 1
    fi
    return 0
}

# Function to show service status
show_status() {
    echo -e "${BLUE}📊 Service Status${NC}"
    echo "=================="
    docker-compose ps
    echo ""
    
    if check_services; then
        echo -e "${BLUE}🔍 Health Checks${NC}"
        echo "================="
        
        # Check Qdrant
        if curl -f http://localhost:6333/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Qdrant: Healthy${NC}"
        else
            echo -e "${RED}❌ Qdrant: Unhealthy${NC}"
        fi
        
        # Check RAG App
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ RAG App: Healthy${NC}"
        else
            echo -e "${RED}❌ RAG App: Unhealthy${NC}"
        fi
        
        # Check Redis (if service exists)
        if docker-compose ps | grep -q "redis"; then
            if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
                echo -e "${GREEN}✅ Redis: Healthy${NC}"
            else
                echo -e "${RED}❌ Redis: Unhealthy${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️  Redis: Not running (optional service)${NC}"
        fi
    fi
}

# Function to show logs
show_logs() {
    if [ "$1" = "-f" ]; then
        echo -e "${BLUE}📋 Following logs (Ctrl+C to stop)${NC}"
        docker-compose logs -f
    else
        echo -e "${BLUE}📋 Recent logs${NC}"
        docker-compose logs --tail=50
    fi
}

# Function to backup Qdrant
backup_qdrant() {
    echo -e "${BLUE}💾 Creating Qdrant backup...${NC}"
    ./scripts/backup_qdrant.sh
}

# Function to restore Qdrant
restore_qdrant() {
    if [ -z "$1" ]; then
        echo -e "${RED}❌ Please provide backup file path${NC}"
        echo "Usage: $0 restore <backup_file.tar.gz>"
        exit 1
    fi
    
    echo -e "${BLUE}📦 Restoring Qdrant from backup...${NC}"
    ./scripts/restore_qdrant.sh "$1"
}

# Function to migrate data
migrate_data() {
    echo -e "${BLUE}🔄 Starting data migration...${NC}"
    python3 scripts/migrate_qdrant.py "$@"
}

# Main command handling
case "$1" in
    start)
        echo -e "${GREEN}🚀 Starting services...${NC}"
        docker-compose up -d
        sleep 5
        show_status
        ;;
    
    stop)
        echo -e "${YELLOW}🛑 Stopping services...${NC}"
        docker-compose down
        ;;
    
    restart)
        echo -e "${YELLOW}🔄 Restarting services...${NC}"
        docker-compose restart
        sleep 5
        show_status
        ;;
    
    status)
        show_status
        ;;
    
    logs)
        show_logs "$2"
        ;;
    
    build)
        echo -e "${BLUE}🔨 Building images...${NC}"
        docker-compose build --no-cache
        ;;
    
    update)
        echo -e "${BLUE}🔄 Updating services...${NC}"
        docker-compose pull
        docker-compose up -d
        sleep 5
        show_status
        ;;
    
    clean)
        echo -e "${YELLOW}🧹 Cleaning up...${NC}"
        docker-compose down -v
        docker system prune -f
        echo -e "${GREEN}✅ Cleanup completed${NC}"
        ;;
    
    backup)
        backup_qdrant
        ;;
    
    restore)
        restore_qdrant "$2"
        ;;
    
    shell)
        echo -e "${BLUE}🐚 Opening shell in RAG app container...${NC}"
        docker-compose exec rag-app /bin/bash
        ;;
    
    qdrant)
        echo -e "${BLUE}🐚 Opening Qdrant shell...${NC}"
        docker-compose exec qdrant /bin/bash
        ;;
    
    health)
        show_status
        ;;
    
    migrate)
        shift
        migrate_data "$@"
        ;;
    
    *)
        show_usage
        ;;
esac
