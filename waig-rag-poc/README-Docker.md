# WAIG RAG POC - Docker Setup Guide

This guide explains how to containerize and run the WAIG RAG POC application using Docker and Docker Compose.

## 🏗️ Architecture

The containerized setup includes:

- **RAG Application**: FastAPI-based RAG system
- **Qdrant**: Vector database for embeddings storage
- **Redis**: Optional caching layer
- **Nginx**: Reverse proxy and load balancer

## 📋 Prerequisites

- Docker (20.10+)
- Docker Compose (2.0+)
- Python 3.9+ (for migration scripts)
- jq (for JSON processing in scripts)

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Clone and navigate to the project
cd waig-rag-poc

# Run the setup script
./scripts/docker-setup.sh
```

This script will:
- Create necessary directories
- Set up configuration files
- Create `.env` file from template
- Build Docker images
- Start all services

### 2. Configure Environment

Edit the `.env` file with your API keys:

```bash
# Required API keys
OPENAI_API_KEY=your_openai_api_key_here
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here

# Other configurations (optional)
QDRANT_URL=http://qdrant:6333
API_PORT=8000
```

### 3. Start Services

```bash
# Start all services
./scripts/docker-manage.sh start

# Or use docker-compose directly
docker-compose up -d
```

## 🔧 Management Commands

Use the management script for easy operations:

```bash
# Service management
./scripts/docker-manage.sh start      # Start services
./scripts/docker-manage.sh stop       # Stop services
./scripts/docker-manage.sh restart    # Restart services
./scripts/docker-manage.sh status     # Check status
./scripts/docker-manage.sh logs -f    # View logs

# Maintenance
./scripts/docker-manage.sh build      # Rebuild images
./scripts/docker-manage.sh update     # Update and restart
./scripts/docker-manage.sh clean      # Clean up containers

# Data management
./scripts/docker-manage.sh backup     # Backup Qdrant data
./scripts/docker-manage.sh restore    # Restore from backup
./scripts/docker-manage.sh migrate    # Migrate between instances

# Access
./scripts/docker-manage.sh shell      # Open app container shell
./scripts/docker-manage.sh qdrant     # Open Qdrant container shell
```

## 📊 Access URLs

- **RAG API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Nginx Proxy**: http://localhost:80

## 💾 Data Migration

### From Local Qdrant to Docker

If you have an existing local Qdrant instance with data:

```bash
# Migrate all collections from local to Docker
./scripts/migrate_from_local.sh

# Or migrate specific collections
python3 scripts/migrate_qdrant.py \
    --action export \
    --source-url http://localhost:6333 \
    --collection your_collection_name \
    --file backup.json

python3 scripts/migrate_qdrant.py \
    --action import \
    --target-url http://localhost:6333 \
    --file backup.json
```

### Backup and Restore

```bash
# Create backup
./scripts/docker-manage.sh backup

# Restore from backup
./scripts/docker-manage.sh restore backups/qdrant_backup_20240101_120000.tar.gz
```

## 🔍 Monitoring and Debugging

### Health Checks

```bash
# Check all services
./scripts/docker-manage.sh health

# Check individual services
curl http://localhost:8000/health    # RAG App
curl http://localhost:6333/health    # Qdrant
docker-compose exec redis redis-cli ping  # Redis
```

### Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f rag-app
docker-compose logs -f qdrant
docker-compose logs -f redis
```

### Container Access

```bash
# Access RAG app container
./scripts/docker-manage.sh shell

# Access Qdrant container
./scripts/docker-manage.sh qdrant

# Execute commands in containers
docker-compose exec rag-app python -c "print('Hello from container')"
```

## 🗂️ Data Persistence

Data is persisted using Docker volumes:

- **Qdrant Data**: `qdrant_data` volume
- **Redis Data**: `redis_data` volume
- **Application Data**: `./data` directory (mounted)
- **Logs**: `./logs` directory (mounted)

## 🔧 Configuration

### Qdrant Configuration

Qdrant configuration is in `qdrant_config/config.yaml`:

```yaml
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334

storage:
  storage_path: /qdrant/storage
  wal:
    wal_capacity_mb: 32
```

### Nginx Configuration

Nginx configuration is in `nginx.conf` for reverse proxy setup.

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8000, 6333, 6379 are available
2. **Permission issues**: Check file permissions in mounted directories
3. **Memory issues**: Increase Docker memory limits if needed
4. **API key errors**: Verify `.env` file has correct API keys

### Debug Commands

```bash
# Check container status
docker-compose ps

# Check resource usage
docker stats

# Check container logs
docker-compose logs rag-app

# Restart specific service
docker-compose restart rag-app

# Rebuild specific service
docker-compose build --no-cache rag-app
```

### Reset Everything

```bash
# Stop and remove everything
docker-compose down -v
docker system prune -f

# Start fresh
./scripts/docker-setup.sh
```

## 📈 Scaling

### Horizontal Scaling

To scale the RAG application:

```yaml
# In docker-compose.yml
rag-app:
  # ... existing config
  deploy:
    replicas: 3
```

### Load Balancing

Nginx is configured for load balancing multiple RAG app instances.

## 🔒 Security

### Production Considerations

1. **API Keys**: Use Docker secrets or external secret management
2. **Network**: Use custom networks and restrict access
3. **Volumes**: Use named volumes instead of bind mounts
4. **Images**: Use specific image tags, not `latest`
5. **SSL**: Configure SSL certificates in nginx

### Environment Variables

Never commit `.env` files with real API keys. Use `.env.example` as template.

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Support

For issues and questions:
1. Check the logs: `./scripts/docker-manage.sh logs`
2. Verify configuration: `./scripts/docker-manage.sh status`
3. Check this documentation
4. Create an issue in the repository
