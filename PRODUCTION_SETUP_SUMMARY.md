# 🎉 Production Docker Setup Complete!

## Overview

Your production deployment setup is ready! This setup **reuses your existing infrastructure containers** and deploys only the three application services.

---

## 📦 What Was Created

### Production-Specific Files

1. **docker-compose.production.yml** ✅
   - Deploys ONLY application services (Backend, Middleware, Frontend)
   - Connects to existing MongoDB, RabbitMQ, and Qdrant containers
   - Uses existing networks: `rag_network` and `ragnet`
   - No new infrastructure containers

2. **.env.production.example** ✅
   - Production environment template
   - Configured for your existing setup
   - Uses same container names: `my-mongo`, `my-rabbitmq`, `qdrant`
   - Includes Langfuse observability keys

3. **deploy-production.sh** ✅ (Executable)
   - Production deployment automation
   - Validates existing infrastructure
   - Commands: setup, start, stop, status, health, worker

4. **PRODUCTION_DEPLOYMENT_GUIDE.md** ✅
   - Complete deployment guide
   - Troubleshooting for existing infrastructure
   - Network connectivity verification
   - Worker setup and monitoring

---

## 🏗️ Architecture Comparison

### Your Current Setup (Existing Infrastructure)

```
Existing Containers (Must Keep Running):
┌─────────────────────────────────────┐
│  my-mongo      → Port 27017         │
│  my-rabbitmq   → Ports 5672, 15672  │
│  qdrant        → Port 6333          │
└─────────────────────────────────────┘
       Networks: rag_network + ragnet
```

### New Application Services (To Deploy)

```
New Containers (Deploy with docker-compose.production.yml):
┌─────────────────────────────────────┐
│  waig-rag-backend    → Port 8000    │
│  waig-rag-middleware → Port 8080    │
│  waig-rag-frontend   → Port 3000    │
└─────────────────────────────────────┘
       Networks: rag_network + ragnet
```

### Complete System

```
┌──────────────────────────────────────────────────────────┐
│                    AWS EC2 Instance                       │
│                                                            │
│  ┌──────── NEW APPLICATION SERVICES ─────────┐           │
│  │                                            │           │
│  │  Frontend  →  Middleware  →  RAG Backend  │           │
│  │  :3000         :8080          :8000        │           │
│  │                                            │           │
│  └──────────────────┬─────────────────────────┘           │
│                     │                                      │
│                     ↓                                      │
│  ┌──────── EXISTING INFRASTRUCTURE ────────┐             │
│  │                                          │             │
│  │  my-mongo      → Port 27017              │             │
│  │  my-rabbitmq   → Ports 5672, 15672       │             │
│  │  qdrant        → Port 6333               │             │
│  │                                          │             │
│  └──────────────────────────────────────────┘             │
│                                                            │
│  Networks: rag_network (172.19.0.0/16)                    │
│            ragnet (172.18.0.0/16)                         │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### Step 1: Verify Existing Infrastructure

```bash
# Check existing containers are running
docker ps

# Expected output should show:
# - my-mongo
# - my-rabbitmq
# - qdrant
```

### Step 2: Configure Environment

```bash
# Copy production template
cp .env.production.example .env

# Edit with your actual keys
nano .env
```

**Update these values:**

```bash
# From RAG Backend env
OPENAI_API_KEY=sk-proj-********************************************************************************************************************************
LLAMA_CLOUD_API_KEY=llx-pRroMia6sPDppIQvPGlUSJbu7hQjTk82Tjb8zpinOIUjJVOR

# Langfuse keys (Backend)
LANGFUSE_SECRET_KEY=sk-lf-be61bc33-40a2-4ee0-b49a-7fc59e4675fd
LANGFUSE_PUBLIC_KEY=pk-lf-e0bdc66b-7ea0-420a-8a43-23025bb181d4
LANGFUSE_HOST=https://langfuse.wadhwaniaiglobal.com

# Langfuse keys (Middleware)
LANGFUSE_SECRET_KEY_MIDDLEWARE=sk-lf-b1a2e117-7eae-405a-aff8-5d20eb309873
LANGFUSE_PUBLIC_KEY_MIDDLEWARE=pk-lf-63c5abb7-c34d-4e01-96ee-8855c1cb6c56

# From Middleware env (Keep these)
DATABASE_NAME=fastapi_db
RABBITMQ_USERNAME=guest
RABBITMQ_PASSWORD=guest
CHAT_QUEUE_NAME=chat_processing_queue
QDRANT_COLLECTION_NAME=rag_documents

# Generate new SECRET_KEY
SECRET_KEY=$(openssl rand -hex 32)

# Set your EC2 public IP
REACT_APP_API_URL=http://<YOUR_EC2_PUBLIC_IP>:8080

# From Middleware env
SUPERUSER_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNjg5MjU2M2ViMzhlMTRhY2I5NzAyNDQ4IiwibmFtZSI6ImtlbnlhX2JvdCIsInVzZXJfdHlwZSI6ImJvdCIsImV4cCI6MTc1NzAxNDUzOH0.JGnp0S-ixlacUrgazKQW4quRvlUrVXTy5AzaBQqwZ9w
```

### Step 3: Deploy

```bash
# Using deploy script (recommended)
./deploy-production.sh setup
./deploy-production.sh start

# Or using docker-compose directly
docker-compose -f docker-compose.production.yml build
docker-compose -f docker-compose.production.yml up -d
```

### Step 4: Verify

```bash
# Check status
./deploy-production.sh status

# Health checks
./deploy-production.sh health

# Check worker
./deploy-production.sh worker

# View logs
./deploy-production.sh logs
```

---

## 🔑 Key Differences from Standalone Setup

### Original docker-compose.yml (Standalone)
- Includes MongoDB, RabbitMQ, Qdrant
- Creates new volumes
- Creates new network (172.20.0.0/16)
- **Use for:** Fresh deployment or development

### docker-compose.production.yml (Production)
- **Only** application services (Backend, Middleware, Frontend)
- **No** infrastructure containers
- **Uses** existing networks (rag_network, ragnet)
- **Connects to** existing containers (my-mongo, my-rabbitmq, qdrant)
- **Use for:** AWS EC2 production deployment

---

## 📊 Configuration Mapping

### Database Configuration

| Setting | Value | Container |
|---------|-------|-----------|
| `DATABASE_URL` | `mongodb://my-mongo:27017` | Existing MongoDB |
| `DATABASE_NAME` | `fastapi_db` | Same as middleware |

### RabbitMQ Configuration

| Setting | Value | Container |
|---------|-------|-----------|
| `RABBITMQ_HOST` | `my-rabbitmq` | Existing RabbitMQ |
| `RABBITMQ_PORT` | `5672` | AMQP port |
| `CHAT_QUEUE_NAME` | `chat_processing_queue` | Same as middleware |

### Qdrant Configuration

| Setting | Value | Container |
|---------|-------|-----------|
| `QDRANT_URL` | `http://qdrant:6333` | Existing Qdrant |
| `QDRANT_COLLECTION_NAME` | `rag_documents` | Same as middleware |

---

## 🔍 Verification Checklist

### Pre-Deployment

- [ ] Existing containers running (my-mongo, my-rabbitmq, qdrant)
- [ ] Networks exist (rag_network, ragnet)
- [ ] Existing containers connected to networks
- [ ] `.env` file configured with actual keys
- [ ] `SUPERUSER_TOKEN` set from middleware
- [ ] `REACT_APP_API_URL` set with EC2 public IP

### Post-Deployment

- [ ] All 3 application containers running and healthy
- [ ] Backend can connect to Qdrant
- [ ] Middleware can connect to MongoDB
- [ ] Middleware can connect to RabbitMQ
- [ ] Worker is processing messages
- [ ] Frontend can reach middleware API
- [ ] Health checks pass for all services

---

## 🛠️ Common Commands

### Deployment

```bash
# First-time setup
./deploy-production.sh setup

# Start services
./deploy-production.sh start

# Stop services (keeps infrastructure running)
./deploy-production.sh stop

# Restart services
./deploy-production.sh restart
```

### Monitoring

```bash
# Check status
./deploy-production.sh status

# Health checks
./deploy-production.sh health

# Verify worker
./deploy-production.sh worker

# View logs
./deploy-production.sh logs

# View specific service logs
docker logs waig-rag-backend
docker logs waig-rag-middleware
docker logs waig-rag-frontend
```

### Troubleshooting

```bash
# Check network connectivity
docker exec waig-rag-backend curl http://qdrant:6333/health
docker exec waig-rag-middleware curl http://my-mongo:27017

# Check environment variables
docker exec waig-rag-middleware env | grep DATABASE
docker exec waig-rag-middleware env | grep RABBITMQ
docker exec waig-rag-backend env | grep QDRANT

# Restart specific service
docker-compose -f docker-compose.production.yml restart rag-middleware

# View last 100 log lines
docker logs waig-rag-middleware --tail=100
```

---

## 🚨 Important Notes

### ⚠️ Do NOT Stop Infrastructure

```bash
# ❌ WRONG - This will stop EVERYTHING including infrastructure
docker stop $(docker ps -q)

# ✅ CORRECT - This stops ONLY application services
./deploy-production.sh stop
# or
docker-compose -f docker-compose.production.yml down
```

### ⚠️ Network Requirements

Your existing containers MUST be on both networks:

```bash
# Verify network connections
docker network inspect rag_network | grep -E "my-mongo|my-rabbitmq|qdrant"
docker network inspect ragnet | grep -E "my-mongo|my-rabbitmq|qdrant"

# If missing, connect containers
docker network connect rag_network my-mongo
docker network connect rag_network my-rabbitmq
docker network connect rag_network qdrant
```

### ⚠️ Port Conflicts

If ports 3000, 8000, or 8080 are already in use:

```bash
# Check existing containers on these ports
docker ps | grep -E ":3000|:8000|:8080"

# Stop conflicting containers
docker stop <container-name>

# Or change ports in docker-compose.production.yml
```

---

## 📈 Next Steps

### 1. Upload Documents

```bash
# Upload JSON documents
curl -X POST "http://localhost:8000/documents/upload-json" \
  -H "Content-Type: application/json" \
  -d @Parsed_Englsih_Bot/FAQ_on_Immunization_for_Health_Workers-English.json

# Verify documents
curl http://localhost:8000/documents
```

### 2. Test Bot Responses

1. Access frontend: `http://<EC2_PUBLIC_IP>:3000`
2. Login with user credentials
3. Select ASHA bot conversation
4. Click "Maternal Health" chip
5. Watch auto-sequential questions!

### 3. Monitor Performance

```bash
# Watch resource usage
docker stats waig-rag-backend waig-rag-middleware waig-rag-frontend

# Monitor queue depth
docker exec my-rabbitmq rabbitmqctl list_queues

# Check Langfuse traces
# Visit: https://langfuse.wadhwaniaiglobal.com
```

---

## 💰 Cost Impact

### No Additional Infrastructure Costs
- Using existing MongoDB, RabbitMQ, Qdrant
- Only adds 3 application containers
- Minimal memory overhead (~2-3GB total for apps)

### Current EC2 Usage
Your current setup can handle the additional load:
- Existing: Infrastructure containers
- Adding: 3 application containers
- Total: Should fit in current instance

---

## 📚 Documentation Reference

### Production Deployment
- **Main Guide:** `PRODUCTION_DEPLOYMENT_GUIDE.md`
- **Environment:** `.env.production.example`
- **Deploy Script:** `deploy-production.sh`
- **Compose File:** `docker-compose.production.yml`

### General Documentation
- **Architecture:** `CLAUDE.md`
- **ASHA Bot:** `ASHA_BOT_SETUP.md`
- **Standalone Setup:** `docker-compose.yml` + `AWS_DEPLOYMENT_GUIDE.md`

---

## ✅ Deployment Ready!

You have **two deployment options**:

### Option 1: Production (Uses Existing Infrastructure) ⭐ **RECOMMENDED FOR YOU**
```bash
./deploy-production.sh setup
./deploy-production.sh start
```

### Option 2: Standalone (Fresh Deployment)
```bash
./deploy.sh setup
./deploy.sh start
```

---

## 🎯 Quick Decision Guide

**Use Production Setup (`docker-compose.production.yml`) if:**
- ✅ You have existing MongoDB, RabbitMQ, Qdrant containers
- ✅ You want to keep current data and configuration
- ✅ You're deploying to your current AWS EC2 instance
- ✅ **This is your case!**

**Use Standalone Setup (`docker-compose.yml`) if:**
- You're starting fresh with no existing containers
- You want completely isolated environment
- You're setting up development environment

---

**Production deployment is ready! 🚀**

Next step: Run `./deploy-production.sh setup`
