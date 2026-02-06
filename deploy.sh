#!/bin/bash
# WAIG RAG System - Automated Deployment Script
# Usage: ./deploy.sh [command]
# Commands: setup, start, stop, restart, logs, status, backup, clean

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

check_requirements() {
    echo "Checking requirements..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is installed"

    # Check .env file
    if [ ! -f .env ]; then
        print_warning ".env file not found. Copying from .env.example..."
        cp .env.example .env
        print_info "Please edit .env file with your actual credentials before continuing."
        print_info "Required: OPENAI_API_KEY, LLAMA_CLOUD_API_KEY, SUPERUSER_TOKEN"
        exit 1
    fi
    print_success ".env file exists"
}

check_env_variables() {
    echo "Checking environment variables..."

    source .env

    missing_vars=()

    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-your-openai-api-key-here" ]; then
        missing_vars+=("OPENAI_API_KEY")
    fi

    if [ -z "$LLAMA_CLOUD_API_KEY" ] || [ "$LLAMA_CLOUD_API_KEY" = "llx-your-llamaparse-api-key-here" ]; then
        missing_vars+=("LLAMA_CLOUD_API_KEY")
    fi

    if [ -z "$SECRET_KEY" ] || [ "$SECRET_KEY" = "your-super-secret-jwt-key-change-this-in-production" ]; then
        missing_vars+=("SECRET_KEY")
    fi

    if [ ${#missing_vars[@]} -gt 0 ]; then
        print_error "Missing or invalid environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        print_info "Please update .env file with actual values"
        exit 1
    fi

    print_success "All required environment variables are set"
}

setup() {
    echo "========================================="
    echo "  WAIG RAG System - Initial Setup"
    echo "========================================="
    echo ""

    check_requirements
    check_env_variables

    echo ""
    echo "Building Docker images (this may take 10-15 minutes)..."
    docker-compose build

    print_success "Build complete!"
    echo ""
    print_info "Next steps:"
    echo "  1. Generate SUPERUSER_TOKEN:"
    echo "     cd rag-poc-middleware-app"
    echo "     python3 -m venv venv && source venv/bin/activate"
    echo "     pip install -r requirements.txt && python setup_rag_system.py"
    echo "     # Copy SUPERUSER_TOKEN to .env"
    echo ""
    echo "  2. Start services:"
    echo "     ./deploy.sh start"
}

start() {
    echo "Starting all services..."
    check_requirements

    docker-compose up -d

    echo ""
    echo "Waiting for services to be healthy..."
    sleep 10

    docker-compose ps

    echo ""
    print_success "All services started!"
    echo ""
    echo "Access your application at:"
    echo "  - Frontend: http://$(curl -s ifconfig.me):3000"
    echo "  - API: http://$(curl -s ifconfig.me):8080"
    echo "  - RabbitMQ UI: http://$(curl -s ifconfig.me):15672"
    echo ""
    print_info "Run './deploy.sh logs' to view logs"
}

stop() {
    echo "Stopping all services..."
    docker-compose down
    print_success "All services stopped"
}

restart() {
    echo "Restarting all services..."
    docker-compose restart
    print_success "All services restarted"
}

logs() {
    echo "Showing logs (Ctrl+C to exit)..."
    docker-compose logs -f
}

status() {
    echo "========================================="
    echo "  Service Status"
    echo "========================================="
    docker-compose ps

    echo ""
    echo "========================================="
    echo "  Resource Usage"
    echo "========================================="
    docker stats --no-stream

    echo ""
    echo "========================================="
    echo "  Disk Usage"
    echo "========================================="
    docker system df
}

backup() {
    BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"

    echo "Creating backup in $BACKUP_DIR..."

    # Backup MongoDB
    echo "Backing up MongoDB..."
    docker exec waig-mongodb mongodump --out /backup
    docker cp waig-mongodb:/backup "$BACKUP_DIR/mongodb"
    print_success "MongoDB backup complete"

    # Backup Qdrant
    echo "Backing up Qdrant..."
    docker exec waig-qdrant tar -czf /backup.tar.gz /qdrant/storage
    docker cp waig-qdrant:/backup.tar.gz "$BACKUP_DIR/qdrant.tar.gz"
    print_success "Qdrant backup complete"

    # Backup .env
    echo "Backing up .env..."
    cp .env "$BACKUP_DIR/.env"
    print_success ".env backup complete"

    echo ""
    print_success "Backup complete: $BACKUP_DIR"
}

clean() {
    echo "========================================="
    echo "  WARNING: This will delete all data!"
    echo "========================================="
    read -p "Are you sure you want to clean up everything? (yes/no): " confirm

    if [ "$confirm" = "yes" ]; then
        echo "Stopping and removing all containers, networks, and volumes..."
        docker-compose down -v

        echo "Removing unused Docker resources..."
        docker system prune -a -f

        print_success "Cleanup complete"
    else
        print_info "Cleanup cancelled"
    fi
}

health_check() {
    echo "========================================="
    echo "  Health Check"
    echo "========================================="

    services=("rag-backend:8000" "rag-middleware:8080" "qdrant:6333")

    for service in "${services[@]}"; do
        name="${service%%:*}"
        port="${service##*:}"

        echo -n "Checking $name... "
        if curl -f -s "http://localhost:$port/health" > /dev/null 2>&1; then
            print_success "Healthy"
        else
            print_error "Unhealthy"
        fi
    done

    echo ""
    echo -n "Checking frontend... "
    if curl -f -s "http://localhost:3000/" > /dev/null 2>&1; then
        print_success "Healthy"
    else
        print_error "Unhealthy"
    fi
}

usage() {
    echo "WAIG RAG System - Deployment Script"
    echo ""
    echo "Usage: ./deploy.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup       - Initial setup (build images)"
    echo "  start       - Start all services"
    echo "  stop        - Stop all services"
    echo "  restart     - Restart all services"
    echo "  logs        - View logs (real-time)"
    echo "  status      - Show service status and resource usage"
    echo "  health      - Check health of all services"
    echo "  backup      - Create backup of MongoDB and Qdrant"
    echo "  clean       - Stop and remove all containers, volumes, and images"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh setup       # First-time setup"
    echo "  ./deploy.sh start       # Start services"
    echo "  ./deploy.sh logs        # View logs"
    echo "  ./deploy.sh backup      # Create backup"
}

# Main
case "$1" in
    setup)
        setup
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    health)
        health_check
        ;;
    backup)
        backup
        ;;
    clean)
        clean
        ;;
    *)
        usage
        exit 1
        ;;
esac
