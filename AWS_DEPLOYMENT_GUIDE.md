# WAIG RAG System - AWS EC2 Deployment Guide

Complete guide for deploying the WAIG RAG Healthcare Assistant system on AWS EC2.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [EC2 Instance Setup](#ec2-instance-setup)
3. [Docker Installation](#docker-installation)
4. [Application Deployment](#application-deployment)
5. [Configuration](#configuration)
6. [SSL/TLS Setup (Optional)](#ssltls-setup-optional)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### AWS Requirements
- AWS account with EC2 access
- Basic knowledge of AWS Console, SSH, and Linux commands

### Minimum EC2 Instance Specifications
- **Instance Type:** `t3.xlarge` or better (4 vCPU, 16 GB RAM)
- **Storage:** 50 GB SSD (gp3)
- **Operating System:** Ubuntu 22.04 LTS (recommended)
- **Region:** Choose closest to your users

### Required API Keys
- OpenAI API key (for GPT-4 and embeddings)
- LlamaParse API key (for document processing)

---

## EC2 Instance Setup

### Step 1: Launch EC2 Instance

1. **Log in to AWS Console** → Navigate to EC2

2. **Launch Instance**
   - Name: `waig-rag-production`
   - AMI: Ubuntu Server 22.04 LTS
   - Instance type: `t3.xlarge`
   - Key pair: Create or select existing (save .pem file securely)

3. **Configure Storage**
   - Root volume: 50 GB gp3
   - IOPS: 3000
   - Throughput: 125 MB/s

4. **Network Settings**
   - VPC: Default or custom
   - Auto-assign public IP: **Enable**
   - Security group: Create new with following rules:

   | Type       | Protocol | Port Range | Source    | Description           |
   |------------|----------|------------|-----------|-----------------------|
   | SSH        | TCP      | 22         | My IP     | SSH access            |
   | HTTP       | TCP      | 80         | 0.0.0.0/0 | HTTP (optional)       |
   | HTTPS      | TCP      | 443        | 0.0.0.0/0 | HTTPS (optional)      |
   | Custom TCP | TCP      | 3000       | 0.0.0.0/0 | React Frontend        |
   | Custom TCP | TCP      | 8080       | 0.0.0.0/0 | Middleware API        |
   | Custom TCP | TCP      | 8000       | 0.0.0.0/0 | RAG Backend (optional)|
   | Custom TCP | TCP      | 15672      | My IP     | RabbitMQ Management   |

5. **Launch Instance**

### Step 2: Connect to EC2 Instance

```bash
# Make key file read-only
chmod 400 your-key.pem

# SSH into instance
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### Step 3: Update System

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git vim htop
```

---

## Docker Installation

### Install Docker Engine

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu

# Log out and back in for group changes
exit
# SSH back in
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# Verify Docker installation
docker --version
docker ps
```

### Install Docker Compose

```bash
# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

---

## Application Deployment

### Step 1: Clone Repository

```bash
# Clone your repository
cd ~
git clone <your-repo-url> waig-rag
cd waig-rag

# Or upload via SCP from local machine
# From your local machine:
# scp -i your-key.pem -r D:\WAIG_RAG ubuntu@<EC2_PUBLIC_IP>:~/waig-rag
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit environment file
nano .env
```

**Required Environment Variables:**

```bash
# OpenAI (REQUIRED)
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4o-mini-2024-07-18
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# LlamaParse (REQUIRED)
LLAMA_CLOUD_API_KEY=llx-your-actual-key-here

# MongoDB Credentials
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=<strong-password-here>

# RabbitMQ Credentials
RABBITMQ_USER=admin
RABBITMQ_PASSWORD=<strong-password-here>

# JWT Secret (Generate with: openssl rand -hex 32)
SECRET_KEY=<generate-strong-secret-key>

# Frontend API URL (use your EC2 public IP)
REACT_APP_API_URL=http://<EC2_PUBLIC_IP>:8080
```

**Generate Superuser Token:**

```bash
cd rag-poc-middleware-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup_rag_system.py
# Copy the generated SUPERUSER_TOKEN to .env file
deactivate
cd ..
```

### Step 3: Build and Start Services

```bash
# Build Docker images (this may take 10-15 minutes)
docker-compose build

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

**Expected Output:**
```
NAME                STATUS              PORTS
waig-mongodb        Up (healthy)        0.0.0.0:27017->27017/tcp
waig-qdrant         Up (healthy)        0.0.0.0:6333->6333/tcp
waig-rabbitmq       Up (healthy)        0.0.0.0:5672->5672/tcp, 0.0.0.0:15672->15672/tcp
waig-rag-backend    Up (healthy)        0.0.0.0:8000->8000/tcp
waig-rag-middleware Up (healthy)        0.0.0.0:8080->8080/tcp
waig-rag-frontend   Up (healthy)        0.0.0.0:3000->3000/tcp
```

### Step 4: Upload Documents

```bash
# Upload JSON documents to RAG backend
curl -X POST "http://localhost:8000/documents/upload-json" \
  -H "Content-Type: application/json" \
  -d @Parsed_Englsih_Bot/FAQ_on_Immunization_for_Health_Workers-English.json

# Or upload PDFs
curl -X POST "http://localhost:8000/documents/upload" \
  -F "files=@document.pdf"

# Check document status
curl http://localhost:8000/documents
```

### Step 5: Create Bot User

```bash
# Access middleware container
docker exec -it waig-rag-middleware bash

# Run setup script
python setup_rag_system.py

# Exit container
exit
```

---

## Configuration

### Environment-Specific Settings

**Development:**
```bash
REACT_APP_API_URL=http://localhost:8080
LOG_LEVEL=DEBUG
```

**Production:**
```bash
REACT_APP_API_URL=http://<EC2_PUBLIC_IP>:8080
LOG_LEVEL=INFO
```

**Production with Domain:**
```bash
REACT_APP_API_URL=https://api.yourdomain.com
LOG_LEVEL=WARN
```

### Service Ports

| Service      | Internal Port | External Port | Access              |
|--------------|---------------|---------------|---------------------|
| Frontend     | 3000          | 3000          | Public              |
| Middleware   | 8080          | 8080          | Public              |
| RAG Backend  | 8000          | 8000          | Internal/Optional   |
| MongoDB      | 27017         | 27017         | Internal only       |
| RabbitMQ     | 5672          | 5672          | Internal only       |
| RabbitMQ UI  | 15672         | 15672         | Admin access only   |
| Qdrant       | 6333          | 6333          | Internal only       |

---

## SSL/TLS Setup (Optional)

### Using Nginx Reverse Proxy

```bash
# Install Nginx
sudo apt install -y nginx certbot python3-certbot-nginx

# Create Nginx config
sudo nano /etc/nginx/sites-available/waig-rag
```

**Nginx Configuration:**

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Middleware API
    location /api/ {
        proxy_pass http://localhost:8080/;
        proxy_http_version 1.1;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Host $host;
    }
}
```

**Enable Nginx:**

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/waig-rag /etc/nginx/sites-enabled/

# Test config
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Enable SSL with Let's Encrypt
sudo certbot --nginx -d yourdomain.com
```

---

## Monitoring & Maintenance

### Check Service Health

```bash
# Check all services
docker-compose ps

# Check specific service logs
docker-compose logs -f rag-backend
docker-compose logs -f rag-middleware
docker-compose logs -f rag-frontend

# Check resource usage
docker stats
```

### Backup Data

```bash
# Backup MongoDB
docker exec waig-mongodb mongodump --out /backup
docker cp waig-mongodb:/backup ./mongodb-backup-$(date +%Y%m%d)

# Backup Qdrant vector store
docker exec waig-qdrant tar -czf /qdrant-backup.tar.gz /qdrant/storage
docker cp waig-qdrant:/qdrant-backup.tar.gz ./qdrant-backup-$(date +%Y%m%d).tar.gz
```

### Update Application

```bash
# Pull latest code
cd ~/waig-rag
git pull

# Rebuild and restart services
docker-compose down
docker-compose build
docker-compose up -d
```

### View Resource Usage

```bash
# System resources
htop

# Disk usage
df -h

# Docker disk usage
docker system df

# Clean up unused Docker resources
docker system prune -a
```

---

## Troubleshooting

### Common Issues

#### 1. Services Not Starting

```bash
# Check logs for errors
docker-compose logs

# Check if ports are already in use
sudo netstat -tulpn | grep LISTEN

# Restart services
docker-compose restart
```

#### 2. MongoDB Connection Failed

```bash
# Check MongoDB health
docker exec waig-mongodb mongosh --eval "db.adminCommand('ping')"

# Check credentials in .env file
cat .env | grep MONGO
```

#### 3. RabbitMQ Worker Not Processing

```bash
# Check RabbitMQ status
docker exec waig-rabbitmq rabbitmqctl status

# Check queue
docker exec waig-rabbitmq rabbitmqctl list_queues

# Check worker logs
docker-compose logs -f rag-middleware | grep worker
```

#### 4. Frontend Can't Connect to API

```bash
# Check middleware is running
curl http://localhost:8080/health

# Check CORS settings in middleware
docker-compose logs rag-middleware | grep CORS

# Verify REACT_APP_API_URL in .env
docker exec waig-rag-frontend env | grep REACT_APP
```

#### 5. Out of Memory

```bash
# Check memory usage
free -h
docker stats

# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 6. RAG Not Returning Results

```bash
# Check Qdrant has documents
curl http://localhost:6333/collections

# Check OpenAI API key
docker exec waig-rag-backend python -c "import os; print(os.getenv('OPENAI_API_KEY'))"

# Test RAG query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is vaccination schedule?"}'
```

### Logs Location

```bash
# Application logs (inside containers)
docker exec waig-rag-backend ls /app/logs
docker exec waig-rag-middleware ls /app/logs

# System logs
sudo journalctl -u docker
```

### Performance Tuning

**For t3.xlarge (16GB RAM):**

```yaml
# docker-compose.yml - Add resource limits
services:
  rag-backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

---

## Security Best Practices

1. **Change Default Passwords**
   - Update MongoDB, RabbitMQ credentials in .env
   - Use strong passwords (16+ characters)

2. **Restrict Security Group**
   - Limit SSH access to your IP only
   - Close ports 8000, 27017, 5672, 6333 to public

3. **Enable Firewall**
   ```bash
   sudo ufw enable
   sudo ufw allow 22
   sudo ufw allow 80
   sudo ufw allow 443
   sudo ufw allow 3000
   sudo ufw allow 8080
   ```

4. **Regular Updates**
   ```bash
   # Update system weekly
   sudo apt update && sudo apt upgrade -y

   # Update Docker images monthly
   docker-compose pull
   docker-compose up -d
   ```

5. **Monitor Logs**
   ```bash
   # Set up log rotation
   sudo nano /etc/docker/daemon.json
   ```

   ```json
   {
     "log-driver": "json-file",
     "log-opts": {
       "max-size": "10m",
       "max-file": "3"
     }
   }
   ```

---

## Cost Optimization

### EC2 Instance Pricing (us-east-1)
- t3.xlarge: ~$0.1664/hour (~$120/month)
- Storage (50GB gp3): ~$4/month

### Reduce Costs:
1. **Use Reserved Instances** - Save up to 72%
2. **Spot Instances** - Save up to 90% (for non-critical workloads)
3. **Auto-scaling** - Scale down during off-hours
4. **Monitoring** - Use CloudWatch to identify unused resources

---

## Support & Resources

- **Documentation:** See `CLAUDE.md` for architecture details
- **ASHA Bot Setup:** See `ASHA_BOT_SETUP.md`
- **Docker Compose Reference:** `docker-compose.yml`
- **Environment Template:** `.env.example`

---

## Quick Reference Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# Restart a service
docker-compose restart rag-backend

# Rebuild after code changes
docker-compose build && docker-compose up -d

# Check service health
docker-compose ps

# Access service shell
docker exec -it waig-rag-backend bash

# Clean up everything (WARNING: deletes volumes)
docker-compose down -v
```

---

**Deployment Complete!**

Access your application at:
- Frontend: `http://<EC2_PUBLIC_IP>:3000`
- API: `http://<EC2_PUBLIC_IP>:8080`
- RabbitMQ UI: `http://<EC2_PUBLIC_IP>:15672` (admin / password)

For questions or issues, refer to the troubleshooting section above.
