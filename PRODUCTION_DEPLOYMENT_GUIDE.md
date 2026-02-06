# 🚀 Production Deployment Guide - Using Existing Infrastructure

## Overview

This guide deploys **only the application services** (RAG Backend, Middleware, Frontend) and connects them to your **existing infrastructure containers** already running on AWS EC2.

### Existing Infrastructure (Already Running)
✅ MongoDB (`my-mongo`) - Port 27017
✅ RabbitMQ (`my-rabbitmq`) - Ports 5672, 15672
✅ Qdrant (`qdrant`) - Port 6333

### New Application Services (To Deploy)
🆕 RAG Backend (`waig-rag-backend`) - Port 8000
🆕 Middleware (`waig-rag-middleware`) - Port 8080
🆕 Frontend (`waig-rag-frontend`) - Port 3000

---

## 📋 Prerequisites Check

### 1. Verify Existing Containers

```bash
# Check all running containers
docker ps

# Expected output should show:
# - my-mongo (MongoDB)
# - my-rabbitmq (RabbitMQ)
# - qdrant (Vector Store)
```

**Verify Networks:**
```bash
# Check existing networks
docker network ls

# Expected networks:
# - rag_network (172.19.0.0/16)
# - ragnet (172.18.0.0/16)
```

**Check Network Connections:**
```bash
# Check which containers are on which networks
docker network inspect rag_network
docker network inspect ragnet
```

### 2. Verify Container Health

```bash
# MongoDB health check
docker exec my-mongo mongosh --eval "db.adminCommand('ping')"

# RabbitMQ health check
docker exec my-rabbitmq rabbitmqctl status

# Qdrant health check
curl http://localhost:6333/health
```

---

## 🔧 Configuration

### Step 1: Create Environment File

```bash
# Copy the production template
cp .env.production.example .env

# Edit with your actual values
nano .env
```

### Step 2: Required Environment Variables

**CRITICAL - Must Update These:**

```bash
# 1. OpenAI API Key (REQUIRED)
OPENAI_API_KEY=sk-proj-your-actual-key-here

# 2. LlamaParse API Key (REQUIRED)
LLAMA_CLOUD_API_KEY=llx-your-actual-key-here

# 3. Langfuse Keys (REQUIRED for observability)
LANGFUSE_SECRET_KEY=sk-lf-your-backend-key
LANGFUSE_PUBLIC_KEY=pk-lf-your-backend-key
LANGFUSE_SECRET_KEY_MIDDLEWARE=sk-lf-your-middleware-key
LANGFUSE_PUBLIC_KEY_MIDDLEWARE=pk-lf-your-middleware-key

# 4. JWT Secret (REQUIRED - Generate new one!)
SECRET_KEY=$(openssl rand -hex 32)

# 5. Frontend API URL (REQUIRED)
REACT_APP_API_URL=http://<YOUR_EC2_PUBLIC_IP>:8080

# 6. Superuser Token (REQUIRED - Generate with setup script)
SUPERUSER_TOKEN=<from-setup-script>
```

**Optional - Can Keep Defaults:**

```bash
# Database (existing my-mongo container)
DATABASE_NAME=fastapi_db
DATABASE_USERNAME=
DATABASE_PASSWORD=

# RabbitMQ (existing my-rabbitmq container)
RABBITMQ_USERNAME=guest
RABBITMQ_PASSWORD=guest
CHAT_QUEUE_NAME=chat_processing_queue

# Qdrant (existing qdrant container)
QDRANT_COLLECTION_NAME=rag_documents
```

### Step 3: Generate Superuser Token

```bash
# Navigate to middleware directory
cd rag-poc-middleware-app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup_rag_system.py

# Output will show:
# ========================================
# Setup Complete!
# ========================================
# Superuser Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
#
# Copy this token to .env file as SUPERUSER_TOKEN

# Deactivate virtual environment
deactivate

# Go back to root
cd ..
```

**Update .env file:**
```bash
nano .env
# Paste the token as: SUPERUSER_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## 🚀 Deployment

### Option 1: Using docker-compose (Recommended)

```bash
# 1. Build images
docker-compose -f docker-compose.production.yml build

# 2. Start services
docker-compose -f docker-compose.production.yml up -d

# 3. Check status
docker-compose -f docker-compose.production.yml ps
```

**Expected Output:**
```
NAME                   STATUS              PORTS
waig-rag-frontend      Up (healthy)        0.0.0.0:3000->3000/tcp
waig-rag-middleware    Up (healthy)        0.0.0.0:8080->8080/tcp
waig-rag-backend       Up (healthy)        0.0.0.0:8000->8000/tcp
```

### Option 2: Using deploy.sh Script

```bash
# Make script executable
chmod +x deploy-production.sh

# Deploy
./deploy-production.sh start

# Check status
./deploy-production.sh status
```

---

## ✅ Verification

### 1. Check Service Health

```bash
# Check all services
docker-compose -f docker-compose.production.yml ps

# Health check endpoints
curl http://localhost:8000/health  # RAG Backend
curl http://localhost:8080/health  # Middleware
curl http://localhost:3000/        # Frontend
```

### 2. Check Network Connectivity

```bash
# Backend should connect to Qdrant
docker exec waig-rag-backend curl -f http://qdrant:6333/health

# Middleware should connect to MongoDB
docker exec waig-rag-middleware curl -f http://my-mongo:27017

# Middleware should connect to RabbitMQ
docker exec waig-rag-middleware curl -f http://my-rabbitmq:15672
```

### 3. Check Logs

```bash
# View all logs
docker-compose -f docker-compose.production.yml logs -f

# View specific service
docker-compose -f docker-compose.production.yml logs -f rag-backend
docker-compose -f docker-compose.production.yml logs -f rag-middleware
docker-compose -f docker-compose.production.yml logs -f rag-frontend
```

### 4. Verify Worker is Running

```bash
# Check middleware logs for worker
docker-compose -f docker-compose.production.yml logs -f rag-middleware | grep -i worker

# Expected output:
# INFO:     Worker starting...
# INFO:     Connected to RabbitMQ: my-rabbitmq:5672
# INFO:     Worker polling queue: chat_processing_queue
```

---

## 📊 Network Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AWS EC2 Instance                            │
│                                                                   │
│  ┌─────────────────────── NEW SERVICES ──────────────────────┐  │
│  │                                                             │  │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │  │
│  │  │  Frontend    │──▶│ Middleware   │──▶│ RAG Backend  │  │  │
│  │  │  Port 3000   │   │  Port 8080   │   │  Port 8000   │  │  │
│  │  └──────────────┘   └──────┬───────┘   └──────┬───────┘  │  │
│  │                             │                   │          │  │
│  └─────────────────────────────┼───────────────────┼──────────┘  │
│                                │                   │              │
│  ┌──────────── EXISTING INFRASTRUCTURE ────────────┼──────────┐  │
│  │                            │                   │          │  │
│  │                     ┌──────▼────────┐   ┌──────▼───────┐ │  │
│  │                     │  RabbitMQ     │   │   Qdrant     │ │  │
│  │                     │ my-rabbitmq   │   │   qdrant     │ │  │
│  │                     │  Port 5672    │   │  Port 6333   │ │  │
│  │                     └──────┬────────┘   └──────────────┘ │  │
│  │                            │                              │  │
│  │                     ┌──────▼────────┐                     │  │
│  │                     │   MongoDB     │                     │  │
│  │                     │   my-mongo    │                     │  │
│  │                     │  Port 27017   │                     │  │
│  │                     └───────────────┘                     │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Networks:                                                        │
│  • rag_network (172.19.0.0/16) - Primary                         │
│  • ragnet (172.18.0.0/16) - Secondary                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Troubleshooting

### Issue 1: Services Can't Connect to Existing Containers

**Problem:** New services can't reach MongoDB, RabbitMQ, or Qdrant

**Solution:**
```bash
# 1. Check networks exist
docker network ls | grep -E "rag_network|ragnet"

# 2. Verify existing containers are on these networks
docker network inspect rag_network
docker network inspect ragnet

# 3. If networks missing, create them
docker network create rag_network --subnet 172.19.0.0/16
docker network create ragnet --subnet 172.18.0.0/16

# 4. Connect existing containers to networks
docker network connect rag_network my-mongo
docker network connect rag_network my-rabbitmq
docker network connect rag_network qdrant

docker network connect ragnet my-mongo
docker network connect ragnet my-rabbitmq
docker network connect ragnet qdrant
```

### Issue 2: Port Conflicts

**Problem:** Ports 3000, 8000, or 8080 already in use

**Solution:**
```bash
# Check what's using the ports
sudo netstat -tulpn | grep -E ":3000|:8000|:8080"

# Stop conflicting containers
docker ps | grep -E "3000|8000|8080"
docker stop <container-id>

# Or change ports in docker-compose.production.yml
ports:
  - "3001:3000"  # Frontend on 3001
  - "8001:8000"  # Backend on 8001
  - "8081:8080"  # Middleware on 8081
```

### Issue 3: Worker Not Processing Messages

**Problem:** Bot not responding to user messages

**Checklist:**
```bash
# 1. Check SUPERUSER_TOKEN is set correctly
docker exec waig-rag-middleware env | grep SUPERUSER_TOKEN

# 2. Check worker is enabled
docker exec waig-rag-middleware env | grep WORKER_ENABLED

# 3. Check RabbitMQ connection
docker exec waig-rag-middleware curl -f http://my-rabbitmq:15672

# 4. Check RabbitMQ queue
docker exec my-rabbitmq rabbitmqctl list_queues

# 5. Check worker logs
docker logs waig-rag-middleware 2>&1 | grep -i worker
```

### Issue 4: Frontend Shows "Network Error"

**Problem:** Frontend can't connect to middleware API

**Solution:**
```bash
# 1. Check REACT_APP_API_URL in frontend
docker exec waig-rag-frontend env | grep REACT_APP_API_URL

# 2. Should be EC2 public IP, not localhost
# Wrong: http://localhost:8080
# Correct: http://<EC2_PUBLIC_IP>:8080

# 3. Rebuild frontend with correct URL
nano .env
# Update: REACT_APP_API_URL=http://<EC2_PUBLIC_IP>:8080

docker-compose -f docker-compose.production.yml build rag-frontend
docker-compose -f docker-compose.production.yml up -d rag-frontend
```

### Issue 5: MongoDB Authentication Failed

**Problem:** Middleware can't connect to MongoDB

**Solution:**
```bash
# 1. Check MongoDB is running
docker exec my-mongo mongosh --eval "db.adminCommand('ping')"

# 2. Check database name
docker exec waig-rag-middleware env | grep DATABASE_NAME

# 3. If authentication required, update .env
DATABASE_USERNAME=your_mongo_user
DATABASE_PASSWORD=your_mongo_password
DATABASE_URL=mongodb://your_mongo_user:your_mongo_password@my-mongo:27017

# 4. Restart middleware
docker-compose -f docker-compose.production.yml restart rag-middleware
```

---

## 🔄 Updates & Maintenance

### Update Application Code

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose -f docker-compose.production.yml build
docker-compose -f docker-compose.production.yml up -d

# Check logs
docker-compose -f docker-compose.production.yml logs -f
```

### View Logs

```bash
# All services
docker-compose -f docker-compose.production.yml logs -f

# Specific service with timestamps
docker-compose -f docker-compose.production.yml logs -f --timestamps rag-middleware

# Last 100 lines
docker-compose -f docker-compose.production.yml logs --tail=100 rag-backend
```

### Restart Services

```bash
# Restart all
docker-compose -f docker-compose.production.yml restart

# Restart specific service
docker-compose -f docker-compose.production.yml restart rag-middleware
```

### Stop Services

```bash
# Stop application services (keeps infrastructure running)
docker-compose -f docker-compose.production.yml down

# Note: This does NOT stop MongoDB, RabbitMQ, or Qdrant
```

---

## 📈 Monitoring

### Check Resource Usage

```bash
# Container stats
docker stats waig-rag-backend waig-rag-middleware waig-rag-frontend

# System resources
htop
df -h
```

### RabbitMQ Management UI

Access: `http://<EC2_PUBLIC_IP>:15672`
- Username: `guest` (or from .env)
- Password: `guest` (or from .env)

Monitor:
- Queue depth
- Message rates
- Worker connections

### Health Endpoints

```bash
# RAG Backend
curl http://localhost:8000/health
curl http://localhost:8000/documents  # List documents

# Middleware
curl http://localhost:8080/health
curl http://localhost:8080/api/v1/users/bots  # List bots

# Frontend
curl http://localhost:3000/
```

---

## 🔐 Security Notes

### Security Group Rules (AWS EC2)

**Required Open Ports:**

| Port  | Source         | Purpose                |
|-------|----------------|------------------------|
| 22    | Your IP        | SSH access             |
| 3000  | 0.0.0.0/0      | Frontend (public)      |
| 8080  | 0.0.0.0/0      | Middleware API (public)|
| 8000  | ❌ Close       | RAG Backend (internal) |
| 27017 | ❌ Close       | MongoDB (internal)     |
| 5672  | ❌ Close       | RabbitMQ (internal)    |
| 6333  | ❌ Close       | Qdrant (internal)      |
| 15672 | Your IP (opt)  | RabbitMQ UI (admin)    |

### Change Default Credentials

```bash
# In .env file, update:
SECRET_KEY=<generate-with-openssl-rand-hex-32>
RABBITMQ_PASSWORD=<strong-password>
# Add MongoDB auth if needed
```

---

## 📞 Quick Commands Reference

```bash
# Deploy
docker-compose -f docker-compose.production.yml up -d

# Stop
docker-compose -f docker-compose.production.yml down

# Logs
docker-compose -f docker-compose.production.yml logs -f

# Status
docker-compose -f docker-compose.production.yml ps

# Rebuild
docker-compose -f docker-compose.production.yml build

# Restart
docker-compose -f docker-compose.production.yml restart

# Health checks
curl http://localhost:8000/health
curl http://localhost:8080/health
curl http://localhost:3000/
```

---

## ✅ Deployment Checklist

- [ ] Verified existing containers running (MongoDB, RabbitMQ, Qdrant)
- [ ] Verified networks exist (rag_network, ragnet)
- [ ] Created `.env` from `.env.production.example`
- [ ] Updated `OPENAI_API_KEY`
- [ ] Updated `LLAMA_CLOUD_API_KEY`
- [ ] Updated Langfuse keys
- [ ] Generated and set `SECRET_KEY`
- [ ] Generated and set `SUPERUSER_TOKEN`
- [ ] Set `REACT_APP_API_URL` with EC2 public IP
- [ ] Built Docker images
- [ ] Started services
- [ ] Verified health checks pass
- [ ] Tested worker is processing messages
- [ ] Uploaded test documents
- [ ] Tested frontend access

---

**Deployment Complete! 🎉**

Your application services are now running and connected to existing infrastructure.

Access:
- **Frontend:** `http://<EC2_PUBLIC_IP>:3000`
- **API:** `http://<EC2_PUBLIC_IP>:8080`
- **RabbitMQ UI:** `http://<EC2_PUBLIC_IP>:15672`
