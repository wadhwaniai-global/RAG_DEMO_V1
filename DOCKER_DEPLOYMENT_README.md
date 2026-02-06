# 🚀 WAIG RAG System - Docker Deployment

Complete Docker setup for deploying the WAIG RAG Healthcare Assistant System on AWS EC2.

## 📋 Overview

This Docker Compose setup includes:
- **RAG Backend** (waig-rag-poc) - Document processing & query engine
- **Middleware** (rag-poc-middleware-app) - Chat orchestration with RabbitMQ worker
- **Frontend** (rag-poc-user-app) - React/TypeScript chat interface
- **MongoDB** - User & message database
- **Qdrant** - Vector store for semantic search
- **RabbitMQ** - Message queue for async processing

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS EC2 Instance                      │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Frontend   │───▶│  Middleware  │───▶│ RAG Backend  │  │
│  │   (React)    │    │   (FastAPI)  │    │  (FastAPI)   │  │
│  │   Port 3000  │    │   Port 8080  │    │  Port 8000   │  │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘  │
│                              │                    │          │
│                      ┌───────▼────────┐   ┌──────▼───────┐ │
│                      │    RabbitMQ    │   │   Qdrant     │ │
│                      │  (Message Q)   │   │(Vector Store)│ │
│                      └────────────────┘   └──────────────┘ │
│                              │                              │
│                      ┌───────▼────────┐                     │
│                      │    MongoDB     │                     │
│                      │   (Database)   │                     │
│                      └────────────────┘                     │
│                                                               │
│                     rag_network (172.20.0.0/16)              │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Quick Start

### 1. Prerequisites

- AWS EC2 instance (t3.xlarge or better, Ubuntu 22.04)
- Docker & Docker Compose installed
- OpenAI API key
- LlamaParse API key

### 2. Setup

```bash
# Clone repository
git clone <your-repo> waig-rag
cd waig-rag

# Copy environment template
cp .env.example .env

# Edit with your actual credentials
nano .env
```

### 3. Configure Environment

**Minimum Required Variables:**

```bash
# In .env file
OPENAI_API_KEY=sk-your-key-here
LLAMA_CLOUD_API_KEY=llx-your-key-here
MONGO_ROOT_PASSWORD=strong-password
RABBITMQ_PASSWORD=strong-password
SECRET_KEY=$(openssl rand -hex 32)
REACT_APP_API_URL=http://<EC2_PUBLIC_IP>:8080
```

**Generate Superuser Token:**

```bash
cd rag-poc-middleware-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup_rag_system.py
# Copy SUPERUSER_TOKEN to .env
deactivate
cd ..
```

### 4. Deploy

```bash
# Build images (takes ~10 minutes)
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 5. Verify Deployment

```bash
# All services should show "Up (healthy)"
docker-compose ps

# Check logs
docker-compose logs -f

# Test endpoints
curl http://localhost:8000/health  # RAG Backend
curl http://localhost:8080/health  # Middleware
curl http://localhost:3000/        # Frontend
```

### 6. Upload Documents

```bash
# Upload JSON documents
curl -X POST "http://localhost:8000/documents/upload-json" \
  -H "Content-Type: application/json" \
  -d @Parsed_Englsih_Bot/FAQ_on_Immunization_for_Health_Workers-English.json

# Or upload PDFs
curl -X POST "http://localhost:8000/documents/upload" \
  -F "files=@document.pdf"
```

### 7. Access Application

- **Frontend:** `http://<EC2_PUBLIC_IP>:3000`
- **API:** `http://<EC2_PUBLIC_IP>:8080`
- **RabbitMQ UI:** `http://<EC2_PUBLIC_IP>:15672` (admin / password)

## 📦 Services & Ports

| Service      | Port  | External Access | Description                    |
|--------------|-------|-----------------|--------------------------------|
| Frontend     | 3000  | ✅ Public       | React chat interface           |
| Middleware   | 8080  | ✅ Public       | FastAPI + RabbitMQ worker      |
| RAG Backend  | 8000  | ⚠️ Optional     | Document processing & queries  |
| MongoDB      | 27017 | ❌ Internal     | User & message database        |
| Qdrant       | 6333  | ❌ Internal     | Vector store                   |
| RabbitMQ     | 5672  | ❌ Internal     | Message queue                  |
| RabbitMQ UI  | 15672 | ⚠️ Admin only   | Queue management interface     |

## 🔧 Common Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f rag-backend

# Restart a service
docker-compose restart rag-middleware

# Rebuild after code changes
docker-compose build && docker-compose up -d

# Check resource usage
docker stats

# Access service shell
docker exec -it waig-rag-backend bash

# Clean up (WARNING: deletes volumes/data)
docker-compose down -v
```

## 🔍 Health Checks

All services have built-in health checks:

```bash
# Check all service health
docker-compose ps

# Manual health checks
curl http://localhost:8000/health  # RAG Backend
curl http://localhost:8080/health  # Middleware
curl http://localhost:3000/        # Frontend
curl http://localhost:6333/health  # Qdrant
```

## 🛡️ Security Checklist

- [ ] Change default passwords in `.env`
- [ ] Generate strong `SECRET_KEY`
- [ ] Restrict Security Group rules (AWS Console)
- [ ] Close unnecessary ports to public
- [ ] Enable UFW firewall on EC2
- [ ] Set up SSL/TLS with Nginx (see AWS_DEPLOYMENT_GUIDE.md)
- [ ] Regular backups of MongoDB & Qdrant
- [ ] Monitor logs for suspicious activity

## 📊 Monitoring

### Resource Usage

```bash
# Check container resources
docker stats

# Check system resources
htop

# Check disk usage
df -h
docker system df
```

### Logs

```bash
# Real-time logs (all services)
docker-compose logs -f

# Specific service logs
docker-compose logs -f rag-backend
docker-compose logs -f rag-middleware | grep worker

# Last 100 lines
docker-compose logs --tail=100 rag-frontend
```

## 🔄 Backup & Restore

### Backup

```bash
# MongoDB backup
docker exec waig-mongodb mongodump --out /backup
docker cp waig-mongodb:/backup ./mongodb-backup-$(date +%Y%m%d)

# Qdrant backup
docker exec waig-qdrant tar -czf /backup.tar.gz /qdrant/storage
docker cp waig-qdrant:/backup.tar.gz ./qdrant-backup-$(date +%Y%m%d).tar.gz
```

### Restore

```bash
# MongoDB restore
docker cp ./mongodb-backup/ waig-mongodb:/restore
docker exec waig-mongodb mongorestore /restore

# Qdrant restore
docker cp ./qdrant-backup.tar.gz waig-qdrant:/restore.tar.gz
docker exec waig-qdrant tar -xzf /restore.tar.gz -C /
docker-compose restart qdrant
```

## 🐛 Troubleshooting

### Services Won't Start

```bash
# Check for port conflicts
sudo netstat -tulpn | grep LISTEN

# Check logs for errors
docker-compose logs

# Restart services
docker-compose restart
```

### Worker Not Processing Messages

```bash
# Check worker logs
docker-compose logs -f rag-middleware | grep worker

# Check RabbitMQ queue
docker exec waig-rabbitmq rabbitmqctl list_queues

# Verify SUPERUSER_TOKEN in .env
cat .env | grep SUPERUSER_TOKEN
```

### Out of Memory

```bash
# Check memory usage
free -h

# Add swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Frontend Can't Reach API

```bash
# Check REACT_APP_API_URL
docker exec waig-rag-frontend env | grep REACT_APP

# Should point to middleware URL
# Correct: http://<EC2_IP>:8080
# Wrong: http://localhost:8080 (won't work from browser)
```

## 📖 Documentation

- **Full AWS Deployment Guide:** See `AWS_DEPLOYMENT_GUIDE.md`
- **Architecture & Development:** See `CLAUDE.md`
- **ASHA Bot Setup:** See `ASHA_BOT_SETUP.md`
- **Environment Variables:** See `.env.example`

## 🎓 Environment Variables Reference

### Required

| Variable               | Description                          | Example                              |
|------------------------|--------------------------------------|--------------------------------------|
| `OPENAI_API_KEY`       | OpenAI API key                       | `sk-...`                             |
| `LLAMA_CLOUD_API_KEY`  | LlamaParse API key                   | `llx-...`                            |
| `SECRET_KEY`           | JWT signing key                      | Generate with `openssl rand -hex 32` |
| `SUPERUSER_TOKEN`      | Bot authentication token             | From `setup_rag_system.py`           |
| `REACT_APP_API_URL`    | Middleware URL for frontend          | `http://<EC2_IP>:8080`               |

### Optional (with defaults)

| Variable                  | Default                    | Description                      |
|---------------------------|----------------------------|----------------------------------|
| `MONGO_ROOT_USERNAME`     | `admin`                    | MongoDB admin username           |
| `MONGO_ROOT_PASSWORD`     | `changeme123`              | MongoDB admin password           |
| `RABBITMQ_USER`           | `admin`                    | RabbitMQ username                |
| `RABBITMQ_PASSWORD`       | `changeme123`              | RabbitMQ password                |
| `OPENAI_MODEL`            | `gpt-4o-mini-2024-07-18`   | GPT model for generation         |
| `OPENAI_EMBEDDING_MODEL`  | `text-embedding-3-large`   | Embedding model                  |
| `RAG_WORKERS`             | `4`                        | Number of Uvicorn workers        |
| `LOG_LEVEL`               | `INFO`                     | Logging level                    |

## 💰 Cost Estimation (AWS us-east-1)

**Monthly Costs:**
- EC2 t3.xlarge (on-demand): ~$120
- Storage (50GB gp3): ~$4
- Data transfer (100GB): ~$9
- **Total:** ~$133/month

**Cost Optimization:**
- Reserved Instance (1 year): Save 40% (~$80/month)
- Spot Instance: Save up to 90% (~$13/month) *for dev/test*

## 🚨 Production Checklist

Before going to production:

- [ ] **Security**
  - [ ] Changed all default passwords
  - [ ] Generated strong SECRET_KEY
  - [ ] Restricted security group rules
  - [ ] Enabled firewall (UFW)
  - [ ] Set up SSL/TLS certificates

- [ ] **Performance**
  - [ ] Right-sized EC2 instance (based on load testing)
  - [ ] Configured resource limits in docker-compose.yml
  - [ ] Set up CloudWatch monitoring
  - [ ] Configured log rotation

- [ ] **Reliability**
  - [ ] Set up automated backups
  - [ ] Configured health checks
  - [ ] Set up alerting (CloudWatch Alarms)
  - [ ] Tested disaster recovery

- [ ] **Operations**
  - [ ] Documented deployment process
  - [ ] Set up CI/CD pipeline
  - [ ] Configured log aggregation
  - [ ] Created runbooks for common issues

## 📞 Support

For detailed deployment instructions, troubleshooting, and architecture details:
- See `AWS_DEPLOYMENT_GUIDE.md`
- See `CLAUDE.md` for system architecture
- Check logs: `docker-compose logs -f`

---

**Happy Deploying! 🎉**
