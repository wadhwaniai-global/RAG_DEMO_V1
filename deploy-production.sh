#!/bin/bash
# WAIG RAG System - Production Deployment Script
# Uses EXISTING infrastructure (MongoDB, RabbitMQ, Qdrant)
# Deploys only application services (Backend, Middleware, Frontend)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Docker Compose file
COMPOSE_FILE="docker-compose.production.yml"

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
    echo -e "${BLUE}ℹ $1${NC}"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi

    print_success "Docker and Docker Compose are installed"
}

check_infrastructure() {
    print_header "Checking Existing Infrastructure"

    local missing=0

    # Check MongoDB
    if docker ps --format '{{.Names}}' | grep -q "my-mongo"; then
        print_success "MongoDB container (my-mongo) is running"
    else
        print_error "MongoDB container (my-mongo) is NOT running"
        missing=1
    fi

    # Check RabbitMQ
    if docker ps --format '{{.Names}}' | grep -q "my-rabbitmq"; then
        print_success "RabbitMQ container (my-rabbitmq) is running"
    else
        print_error "RabbitMQ container (my-rabbitmq) is NOT running"
        missing=1
    fi

    # Check Qdrant
    if docker ps --format '{{.Names}}' | grep -q "qdrant"; then
        print_success "Qdrant container (qdrant) is running"
    else
        print_error "Qdrant container (qdrant) is NOT running"
        missing=1
    fi

    if [ $missing -eq 1 ]; then
        print_error "Required infrastructure containers are not running!"
        print_info "Please start MongoDB, RabbitMQ, and Qdrant containers first"
        exit 1
    fi
}

check_networks() {
    print_header "Checking Docker Networks"

    if docker network ls | grep -q "rag_network"; then
        print_success "Network 'rag_network' exists"
    else
        print_warning "Network 'rag_network' does not exist"
        read -p "Create network 'rag_network'? (y/n): " create_net
        if [ "$create_net" = "y" ]; then
            docker network create rag_network --subnet 172.19.0.0/16
            print_success "Created network 'rag_network'"
        else
            print_error "Network required for deployment"
            exit 1
        fi
    fi

    if docker network ls | grep -q "ragnet"; then
        print_success "Network 'ragnet' exists"
    else
        print_warning "Network 'ragnet' does not exist"
        read -p "Create network 'ragnet'? (y/n): " create_net
        if [ "$create_net" = "y" ]; then
            docker network create ragnet --subnet 172.18.0.0/16
            print_success "Created network 'ragnet'"
        else
            print_error "Network required for deployment"
            exit 1
        fi
    fi
}

check_env() {
    if [ ! -f .env ]; then
        print_error ".env file not found!"
        print_info "Copy .env.production.example to .env and configure it"
        print_info "Command: cp .env.production.example .env"
        exit 1
    fi

    source .env

    local missing_vars=()

    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-proj-your-openai-api-key-here" ]; then
        missing_vars+=("OPENAI_API_KEY")
    fi

    if [ -z "$LLAMA_CLOUD_API_KEY" ] || [ "$LLAMA_CLOUD_API_KEY" = "llx-your-llamaparse-api-key-here" ]; then
        missing_vars+=("LLAMA_CLOUD_API_KEY")
    fi

    if [ -z "$SECRET_KEY" ] || [ "$SECRET_KEY" = "change-me-in-production-use-openssl-rand-hex-32" ]; then
        missing_vars+=("SECRET_KEY")
    fi

    if [ ${#missing_vars[@]} -gt 0 ]; then
        print_error "Missing or invalid environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        print_info "Please update .env file"
        exit 1
    fi

    print_success ".env file configured"
}

setup() {
    print_header "WAIG RAG Production Setup"

    check_docker
    check_infrastructure
    check_networks
    check_env

    echo ""
    print_info "Building Docker images..."
    docker-compose -f $COMPOSE_FILE build

    print_success "Setup complete!"
    echo ""
    print_info "Next: Run './deploy-production.sh start' to start services"
}

start() {
    print_header "Starting Application Services"

    check_infrastructure
    check_networks

    docker-compose -f $COMPOSE_FILE up -d

    echo ""
    print_info "Waiting for services to be healthy..."
    sleep 15

    docker-compose -f $COMPOSE_FILE ps

    echo ""
    print_success "Services started!"
    echo ""
    print_info "Access your application:"
    echo "  - Frontend: http://$(curl -s ifconfig.me 2>/dev/null || echo 'localhost'):3000"
    echo "  - API: http://$(curl -s ifconfig.me 2>/dev/null || echo 'localhost'):8080"
    echo "  - RabbitMQ UI: http://$(curl -s ifconfig.me 2>/dev/null || echo 'localhost'):15672"
}

stop() {
    print_header "Stopping Application Services"

    docker-compose -f $COMPOSE_FILE down

    print_success "Application services stopped"
    print_info "Infrastructure containers (MongoDB, RabbitMQ, Qdrant) are still running"
}

restart() {
    print_header "Restarting Application Services"

    docker-compose -f $COMPOSE_FILE restart

    print_success "Services restarted"
}

logs() {
    docker-compose -f $COMPOSE_FILE logs -f
}

status() {
    print_header "Service Status"

    echo ""
    echo "=== Infrastructure Containers ==="
    docker ps --filter "name=my-mongo" --filter "name=my-rabbitmq" --filter "name=qdrant" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

    echo ""
    echo "=== Application Containers ==="
    docker-compose -f $COMPOSE_FILE ps

    echo ""
    print_header "Resource Usage"
    docker stats --no-stream waig-rag-backend waig-rag-middleware waig-rag-frontend 2>/dev/null || echo "No application containers running"
}

health() {
    print_header "Health Checks"

    echo ""
    echo "=== Infrastructure Health ==="

    # MongoDB
    echo -n "MongoDB (my-mongo): "
    if docker exec my-mongo mongosh --eval "db.adminCommand('ping')" &>/dev/null; then
        print_success "Healthy"
    else
        print_error "Unhealthy"
    fi

    # RabbitMQ
    echo -n "RabbitMQ (my-rabbitmq): "
    if docker exec my-rabbitmq rabbitmqctl status &>/dev/null; then
        print_success "Healthy"
    else
        print_error "Unhealthy"
    fi

    # Qdrant
    echo -n "Qdrant: "
    if curl -f -s http://localhost:6333/health &>/dev/null; then
        print_success "Healthy"
    else
        print_error "Unhealthy"
    fi

    echo ""
    echo "=== Application Health ==="

    # RAG Backend
    echo -n "RAG Backend: "
    if curl -f -s http://localhost:8000/health &>/dev/null; then
        print_success "Healthy"
    else
        print_error "Unhealthy or not running"
    fi

    # Middleware
    echo -n "Middleware: "
    if curl -f -s http://localhost:8080/health &>/dev/null; then
        print_success "Healthy"
    else
        print_error "Unhealthy or not running"
    fi

    # Frontend
    echo -n "Frontend: "
    if curl -f -s http://localhost:3000/ &>/dev/null; then
        print_success "Healthy"
    else
        print_error "Unhealthy or not running"
    fi
}

verify_worker() {
    print_header "Worker Verification"

    echo "Checking worker logs..."
    docker logs waig-rag-middleware 2>&1 | grep -i worker | tail -10

    echo ""
    echo "Checking RabbitMQ queue..."
    docker exec my-rabbitmq rabbitmqctl list_queues
}

usage() {
    cat << EOF
WAIG RAG System - Production Deployment Script
Uses existing infrastructure containers (MongoDB, RabbitMQ, Qdrant)

Usage: ./deploy-production.sh [command]

Commands:
  setup       - Check prerequisites and build images
  start       - Start application services
  stop        - Stop application services (keeps infrastructure running)
  restart     - Restart application services
  logs        - View logs (real-time)
  status      - Show service status and resource usage
  health      - Check health of all services
  worker      - Verify worker is running and processing

Examples:
  ./deploy-production.sh setup     # First-time setup
  ./deploy-production.sh start     # Start services
  ./deploy-production.sh health    # Check health
  ./deploy-production.sh logs      # View logs

Infrastructure:
  This script manages application services only.
  Existing infrastructure containers must be running:
  - my-mongo (MongoDB)
  - my-rabbitmq (RabbitMQ)
  - qdrant (Vector Store)

EOF
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
        health
        ;;
    worker)
        verify_worker
        ;;
    *)
        usage
        exit 1
        ;;
esac
