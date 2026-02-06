# 🎉 Docker Deployment Complete!

## 📦 What Was Created

All necessary files for AWS EC2 deployment have been created:

### Core Deployment Files
- ✅ **docker-compose.yml** - Complete multi-service orchestration
- ✅ **.env.example** - Environment variable template
- ✅ **deploy.sh** - Automated deployment script

### Documentation
- ✅ **AWS_DEPLOYMENT_GUIDE.md** - Comprehensive AWS EC2 setup guide
- ✅ **DOCKER_DEPLOYMENT_README.md** - Quick-start Docker guide
- ✅ **DEPLOYMENT_SUMMARY.md** - This file

### Existing Dockerfiles (Already Present)
- ✅ **waig-rag-poc/Dockerfile** - RAG backend
- ✅ **rag-poc-middleware-app/Dockerfile** - Middleware + worker
- ✅ **rag-poc-user-app/Dockerfile** - React frontend
- ✅ **.dockerignore files** - All three services

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          AWS EC2 Instance (t3.xlarge)                    │
│                                                                           │
│  Internet ──▶ [Security Group] ──▶ Ports: 3000, 8080, 8000 (optional)   │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Docker Network: rag_network                   │    │
│  │                        (172.20.0.0/16)                           │    │
│  │                                                                   │    │
│  │  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐ │   │
│  │  │   Frontend   │──────▶│  Middleware  │──────▶│ RAG Backend  │ │   │
│  │  │              │       │              │       │              │ │   │
│  │  │  React App   │       │  FastAPI +   │       │   FastAPI    │ │   │
│  │  │  (Node.js)   │       │  Worker      │       │   +OpenAI    │ │   │
│  │  │              │       │  (Python)    │       │   +LlamaParse│ │   │
│  │  │  Port: 3000  │       │  Port: 8080  │       │  Port: 8000  │ │   │
│  │  │              │       │              │       │              │ │   │
│  │  └──────────────┘       └────┬─────────┘       └──────┬───────┘ │   │
│  │                              │                         │         │   │
│  │                              │ Publishes               │ Queries │   │
│  │                              │ Messages                │         │   │
│  │                              ▼                         ▼         │   │
│  │                    ┌──────────────┐         ┌──────────────┐    │   │
│  │                    │  RabbitMQ    │         │   Qdrant     │    │   │
│  │                    │              │         │              │    │   │
│  │                    │ Message      │         │ Vector Store │    │   │
│  │                    │ Queue        │         │ (Hybrid      │    │   │
│  │                    │              │         │  Search)     │    │   │
│  │                    │ Port: 5672   │         │ Port: 6333   │    │   │
│  │                    │ UI: 15672    │         │              │    │   │
│  │                    └──────┬───────┘         └──────────────┘    │   │
│  │                           │                                      │   │
│  │                           │ Stores user data                     │   │
│  │                           ▼                                      │   │
│  │                    ┌──────────────┐                              │   │
│  │                    │   MongoDB    │                              │   │
│  │                    │              │                              │   │
│  │                    │   NoSQL DB   │                              │   │
│  │                    │   (Users,    │                              │   │
│  │                    │   Messages)  │                              │   │
│  │                    │ Port: 27017  │                              │   │
│  │                    └──────────────┘                              │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  Persistent Volumes:                                                     │
│  • mongodb_data     - User and message data                              │
│  • qdrant_data      - Vector embeddings and documents                    │
│  • rabbitmq_data    - Message queue state                                │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Deployment Steps

### Option 1: Using Deploy Script (Recommended)

```bash
# 1. Copy .env template and configure
cp .env.example .env
nano .env  # Add your API keys

# 2. Generate superuser token
cd rag-poc-middleware-app
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python setup_rag_system.py
# Copy SUPERUSER_TOKEN to .env file
deactivate && cd ..

# 3. Setup and build
./deploy.sh setup

# 4. Start all services
./deploy.sh start

# 5. Check status
./deploy.sh status
./deploy.sh health
```

### Option 2: Manual Deployment

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your credentials

# 2. Build images
docker-compose build

# 3. Start services
docker-compose up -d

# 4. Check status
docker-compose ps
```

---

## 📋 Pre-Deployment Checklist

### Required Configuration

- [ ] **OPENAI_API_KEY** - Get from https://platform.openai.com/api-keys
- [ ] **LLAMA_CLOUD_API_KEY** - Get from https://cloud.llamaindex.ai
- [ ] **SECRET_KEY** - Generate with: `openssl rand -hex 32`
- [ ] **SUPERUSER_TOKEN** - Generate with `setup_rag_system.py`
- [ ] **REACT_APP_API_URL** - Set to `http://<EC2_PUBLIC_IP>:8080`

### AWS EC2 Configuration

- [ ] Instance type: **t3.xlarge or better** (4 vCPU, 16 GB RAM)
- [ ] Storage: **50 GB SSD (gp3)**
- [ ] OS: **Ubuntu 22.04 LTS**
- [ ] Security Group ports opened:
  - [ ] 22 (SSH)
  - [ ] 3000 (Frontend)
  - [ ] 8080 (Middleware)
  - [ ] 15672 (RabbitMQ UI - optional)

### Software Installation

- [ ] Docker installed
- [ ] Docker Compose installed
- [ ] Git installed (for cloning repo)

---

## 🔒 Security Notes

### Credentials to Change

All default passwords MUST be changed in production:

```bash
# In .env file:
MONGO_ROOT_PASSWORD=changeme123          # ❌ Change this!
RABBITMQ_PASSWORD=changeme123            # ❌ Change this!
SECRET_KEY=your-super-secret-jwt-key...  # ❌ Change this!
```

### Security Group Rules

**Recommended Settings:**

| Port  | Source         | Purpose                |
|-------|----------------|------------------------|
| 22    | Your IP only   | SSH access             |
| 3000  | 0.0.0.0/0      | Public frontend        |
| 8080  | 0.0.0.0/0      | Public API             |
| 8000  | ❌ Closed      | Internal RAG API       |
| 27017 | ❌ Closed      | Internal MongoDB       |
| 5672  | ❌ Closed      | Internal RabbitMQ      |
| 6333  | ❌ Closed      | Internal Qdrant        |
| 15672 | Your IP only   | RabbitMQ admin (opt)   |

---

## 📊 Service Details

### Frontend (Port 3000)
- **Technology:** React 18 + TypeScript + Material-UI
- **Features:**
  - Chat interface with bot-specific welcome chips
  - Auto-sequential question flow for ASHA bot
  - Resizable sidebar
  - Mobile responsive
- **Container:** waig-rag-frontend
- **Health check:** HTTP GET `/`

### Middleware (Port 8080)
- **Technology:** FastAPI + MongoEngine + Pika (RabbitMQ)
- **Features:**
  - User authentication (JWT)
  - Chat message management
  - Background worker for bot responses
  - WebSocket support (future)
- **Container:** waig-rag-middleware
- **Health check:** HTTP GET `/health`

### RAG Backend (Port 8000)
- **Technology:** FastAPI + Qdrant + OpenAI + LlamaParse
- **Features:**
  - Document upload and processing
  - Hybrid search (vector + lexical)
  - Query expansion and reranking
  - Hierarchical chunking
- **Container:** waig-rag-backend
- **Health check:** HTTP GET `/health`

### Infrastructure Services

| Service   | Port  | Purpose                          | Data Persistence |
|-----------|-------|----------------------------------|------------------|
| MongoDB   | 27017 | User and message database        | ✅ Volume        |
| Qdrant    | 6333  | Vector store for embeddings      | ✅ Volume        |
| RabbitMQ  | 5672  | Message queue for async tasks    | ✅ Volume        |

---

## 🧪 Testing Deployment

### 1. Health Checks

```bash
# Using deploy script
./deploy.sh health

# Manual checks
curl http://localhost:8000/health  # RAG Backend
curl http://localhost:8080/health  # Middleware
curl http://localhost:3000/        # Frontend
curl http://localhost:6333/health  # Qdrant
```

### 2. Upload Test Document

```bash
# Upload JSON document
curl -X POST "http://localhost:8000/documents/upload-json" \
  -H "Content-Type: application/json" \
  -d @Parsed_Englsih_Bot/FAQ_on_Immunization_for_Health_Workers-English.json

# Check documents
curl http://localhost:8000/documents
```

### 3. Test RAG Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the vaccination schedule for newborns?",
    "use_query_expansion": true
  }'
```

### 4. Access Frontend

1. Open browser: `http://<EC2_PUBLIC_IP>:3000`
2. Login with credentials
3. Select ASHA bot
4. Click "Maternal Health" chip
5. Watch auto-sequential questions!

---

## 📈 Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f rag-backend
docker-compose logs -f rag-middleware | grep worker
docker-compose logs -f rag-frontend
```

### Resource Usage

```bash
# Container stats
docker stats

# System resources
htop

# Disk usage
df -h
docker system df
```

### RabbitMQ Management UI

Access: `http://<EC2_PUBLIC_IP>:15672`
- Username: `admin` (from .env)
- Password: (from .env `RABBITMQ_PASSWORD`)

View:
- Queue depth
- Message rates
- Worker status

---

## 💾 Backup & Restore

### Create Backup

```bash
# Using deploy script
./deploy.sh backup

# Manual MongoDB backup
docker exec waig-mongodb mongodump --out /backup
docker cp waig-mongodb:/backup ./mongodb-backup-$(date +%Y%m%d)

# Manual Qdrant backup
docker exec waig-qdrant tar -czf /backup.tar.gz /qdrant/storage
docker cp waig-qdrant:/backup.tar.gz ./qdrant-backup-$(date +%Y%m%d).tar.gz
```

### Restore from Backup

```bash
# MongoDB restore
docker cp ./mongodb-backup/ waig-mongodb:/restore
docker exec waig-mongodb mongorestore /restore

# Qdrant restore
docker cp ./qdrant-backup.tar.gz waig-qdrant:/restore.tar.gz
docker exec waig-qdrant tar -xzf /restore.tar.gz -C /
docker-compose restart qdrant
```

---

## 🔧 Maintenance

### Update Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose build
docker-compose up -d
```

### Scale Services

```bash
# Edit docker-compose.yml to add replicas:
services:
  rag-backend:
    deploy:
      replicas: 2

# Restart
docker-compose up -d --scale rag-backend=2
```

### Clean Up

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes (WARNING: deletes data)
docker volume prune

# Complete cleanup (WARNING: deletes everything)
./deploy.sh clean
```

---

## 🚨 Troubleshooting

### Services Won't Start

```bash
# Check logs for errors
docker-compose logs

# Check port conflicts
sudo netstat -tulpn | grep LISTEN

# Restart all services
docker-compose restart
```

### Worker Not Processing

```bash
# Check worker logs
docker-compose logs -f rag-middleware | grep worker

# Check RabbitMQ queue
docker exec waig-rabbitmq rabbitmqctl list_queues

# Verify SUPERUSER_TOKEN
cat .env | grep SUPERUSER_TOKEN
```

### Out of Memory

```bash
# Check memory
free -h

# Add swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Frontend Can't Reach API

```bash
# Check REACT_APP_API_URL
docker exec waig-rag-frontend env | grep REACT_APP

# Should be: http://<EC2_PUBLIC_IP>:8080
# NOT: http://localhost:8080
```

---

## 💰 Cost Estimation

**AWS EC2 (us-east-1) - Monthly:**

| Resource              | Spec            | Cost/Month |
|-----------------------|-----------------|------------|
| EC2 t3.xlarge         | On-Demand       | ~$120      |
| Storage (50GB gp3)    | SSD             | ~$4        |
| Data Transfer (100GB) | Out to Internet | ~$9        |
| **Total**             |                 | **~$133**  |

**Cost Optimization:**
- **Reserved Instance (1 year):** Save 40% → ~$80/month
- **Spot Instance:** Save 90% → ~$13/month (for dev/test)

---

## 📚 Documentation Reference

- **Complete AWS Guide:** `AWS_DEPLOYMENT_GUIDE.md`
- **Docker Quick Start:** `DOCKER_DEPLOYMENT_README.md`
- **System Architecture:** `CLAUDE.md`
- **ASHA Bot Setup:** `ASHA_BOT_SETUP.md`
- **Environment Variables:** `.env.example`

---

## 🎯 Quick Command Reference

```bash
# Deploy script commands
./deploy.sh setup      # Initial setup
./deploy.sh start      # Start services
./deploy.sh stop       # Stop services
./deploy.sh restart    # Restart services
./deploy.sh logs       # View logs
./deploy.sh status     # Check status
./deploy.sh health     # Health checks
./deploy.sh backup     # Create backup
./deploy.sh clean      # Remove all

# Docker Compose commands
docker-compose up -d           # Start
docker-compose down            # Stop
docker-compose ps              # Status
docker-compose logs -f         # Logs
docker-compose restart         # Restart
docker-compose build           # Rebuild

# Container access
docker exec -it waig-rag-backend bash
docker exec -it waig-rag-middleware bash
docker exec -it waig-mongodb mongosh
```

---

## ✅ Deployment Complete!

Your WAIG RAG Healthcare Assistant System is now ready for production deployment on AWS EC2.

**Next Steps:**
1. Review security checklist
2. Test all endpoints
3. Upload production documents
4. Monitor logs and performance
5. Set up SSL/TLS (optional but recommended)

**Support:**
- Check logs: `docker-compose logs -f`
- Review docs: `AWS_DEPLOYMENT_GUIDE.md`
- Health check: `./deploy.sh health`

🎉 **Happy Deploying!**
