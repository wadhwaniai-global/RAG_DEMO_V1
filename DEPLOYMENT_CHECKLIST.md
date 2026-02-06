# Production Deployment Checklist

## Pre-Deployment Verification

### 1. Verify Existing Infrastructure Containers Are Running

Run these commands to check:

```bash
# Check MongoDB
docker ps --filter "name=my-mongo" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check RabbitMQ
docker ps --filter "name=my-rabbitmq" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check Qdrant
docker ps --filter "name=qdrant" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

**Expected Results:**
- `my-mongo` - Running on port 27017
- `my-rabbitmq` - Running on ports 5672, 15672
- `qdrant` - Running on port 6333

### 2. Verify Docker Networks Exist

```bash
docker network ls | grep -E "rag_network|ragnet"
```

**Expected Output:**
```
xxxxxxxx   rag_network   bridge    local
xxxxxxxx   ragnet        bridge    local
```

If networks don't exist, create them:
```bash
docker network create rag_network --subnet 172.19.0.0/16
docker network create ragnet --subnet 172.18.0.0/16
```

### 3. Configure Environment Variables

```bash
# Copy the production environment template
cp .env.production.example .env

# Edit .env file
nano .env  # or use your preferred editor
```

**Required Changes in .env:**

1. **OpenAI Keys** (REQUIRED):
   ```bash
   OPENAI_API_KEY=sk-proj-your-actual-key-here
   ```

2. **LlamaParse Key** (REQUIRED):
   ```bash
   LLAMA_CLOUD_API_KEY=llx-your-actual-key-here
   ```

3. **Secret Key** (REQUIRED - Generate new one):
   ```bash
   SECRET_KEY=$(openssl rand -hex 32)
   ```

4. **EC2 Public IP** (REQUIRED for frontend):
   ```bash
   REACT_APP_API_URL=http://YOUR_EC2_PUBLIC_IP:8080
   EC2_PUBLIC_IP=YOUR_EC2_PUBLIC_IP
   ```

5. **Superuser Token** (Use existing or generate new):
   ```bash
   # If you need to generate a new one, run setup script in middleware:
   cd rag-poc-middleware-app
   python setup_rag_system.py
   # Copy the SUPERUSER_TOKEN from output to .env
   ```

**Optional Changes:**
- Langfuse keys are already configured (can keep or update)
- Database, RabbitMQ settings match your existing setup (no changes needed)

### 4. Verify Docker Compose File

```bash
# Validate the compose file syntax
docker-compose -f docker-compose.production.yml config
```

This should output the parsed configuration without errors.

## Deployment Steps

### Step 1: Initial Setup (First Time Only)

```bash
chmod +x deploy-production.sh
./deploy-production.sh setup
```

This will:
- Check Docker and Docker Compose are installed
- Verify existing infrastructure containers are running
- Verify networks exist
- Validate .env configuration
- Build Docker images for the three application services

**Expected Output:**
```
✓ Docker and Docker Compose are installed
✓ MongoDB container (my-mongo) is running
✓ RabbitMQ container (my-rabbitmq) is running
✓ Qdrant container (qdrant) is running
✓ Network 'rag_network' exists
✓ Network 'ragnet' exists
✓ .env file configured
✓ Setup complete!
```

### Step 2: Start Application Services

```bash
./deploy-production.sh start
```

This will:
- Start RAG Backend (waig-rag-backend)
- Start Middleware (waig-rag-middleware)
- Start Frontend (waig-rag-frontend)
- Wait for services to become healthy
- Display service status

**Expected Output:**
```
✓ Services started!

Access your application:
  - Frontend: http://YOUR_IP:3000
  - API: http://YOUR_IP:8080
  - RabbitMQ UI: http://YOUR_IP:15672
```

### Step 3: Verify Health

```bash
./deploy-production.sh health
```

**Expected Output:**
All services should show "Healthy":
```
=== Infrastructure Health ===
MongoDB (my-mongo): ✓ Healthy
RabbitMQ (my-rabbitmq): ✓ Healthy
Qdrant: ✓ Healthy

=== Application Health ===
RAG Backend: ✓ Healthy
Middleware: ✓ Healthy
Frontend: ✓ Healthy
```

### Step 4: Verify Worker is Processing

```bash
./deploy-production.sh worker
```

This checks that the RabbitMQ worker in middleware is running and processing messages.

## Post-Deployment Verification

### 1. Test Frontend Access

```bash
curl http://YOUR_EC2_IP:3000/
```

Should return the React app HTML.

### 2. Test Middleware API

```bash
curl http://YOUR_EC2_IP:8080/health
```

Should return: `{"status":"healthy"}`

### 3. Test RAG Backend

```bash
curl http://YOUR_EC2_IP:8000/health
```

Should return health status JSON.

### 4. Test End-to-End Flow

1. Open browser: `http://YOUR_EC2_IP:3000`
2. Login with test user credentials
3. Send a message to the bot
4. Verify bot responds (check worker is processing)

## Monitoring Commands

### View Logs (Real-time)

```bash
# All services
./deploy-production.sh logs

# Specific service
docker logs -f waig-rag-backend
docker logs -f waig-rag-middleware
docker logs -f waig-rag-frontend
```

### Check Service Status

```bash
./deploy-production.sh status
```

Shows:
- Infrastructure container status
- Application container status
- Resource usage (CPU, Memory)

### View RabbitMQ Queue Status

```bash
docker exec my-rabbitmq rabbitmqctl list_queues
```

Should show `chat_processing_queue` with message counts.

## Common Issues and Solutions

### Issue: Infrastructure containers not running

**Error:**
```
✗ MongoDB container (my-mongo) is NOT running
```

**Solution:**
```bash
# Start the missing container
docker start my-mongo
# or
docker start my-rabbitmq
# or
docker start qdrant
```

### Issue: Networks don't exist

**Error:**
```
⚠ Network 'rag_network' does not exist
```

**Solution:**
```bash
docker network create rag_network --subnet 172.19.0.0/16
docker network create ragnet --subnet 172.18.0.0/16
```

### Issue: Application containers can't connect to infrastructure

**Symptoms:**
- Middleware shows database connection errors
- Worker can't connect to RabbitMQ
- RAG backend can't reach Qdrant

**Solution:**
```bash
# Verify infrastructure containers are on the correct networks
docker network inspect rag_network
docker network inspect ragnet

# Reconnect containers to networks if needed
docker network connect rag_network my-mongo
docker network connect rag_network my-rabbitmq
docker network connect rag_network qdrant
docker network connect ragnet my-mongo
docker network connect ragnet my-rabbitmq
docker network connect ragnet qdrant
```

### Issue: Bot not responding to messages

**Solution:**
```bash
# Check worker logs
docker logs waig-rag-middleware | grep -i worker

# Verify SUPERUSER_TOKEN is correct
grep SUPERUSER_TOKEN .env

# Verify RAG backend is accessible from middleware
docker exec waig-rag-middleware curl http://rag-backend:8000/health

# Check RabbitMQ queue
docker exec my-rabbitmq rabbitmqctl list_queues
```

### Issue: Frontend can't connect to middleware

**Symptoms:**
- Login fails
- API calls timeout in browser console

**Solution:**
```bash
# Verify REACT_APP_API_URL in .env matches your EC2 public IP
grep REACT_APP_API_URL .env

# Should be: http://YOUR_EC2_PUBLIC_IP:8080

# Rebuild frontend with correct API URL
docker-compose -f docker-compose.production.yml build rag-frontend
./deploy-production.sh restart
```

### Issue: Port conflicts

**Error:**
```
Error: port is already allocated
```

**Solution:**
```bash
# Check what's using the port
netstat -ano | findstr :3000
# or
netstat -ano | findstr :8080
# or
netstat -ano | findstr :8000

# Stop conflicting service or change port in docker-compose.production.yml
```

## Security Hardening (Production)

### 1. AWS EC2 Security Group

Configure inbound rules:
```
Port 22   (SSH)      - Your IP only
Port 80   (HTTP)     - 0.0.0.0/0
Port 443  (HTTPS)    - 0.0.0.0/0
Port 3000 (Frontend) - 0.0.0.0/0
Port 8080 (API)      - 0.0.0.0/0
```

### 2. Environment Variables

```bash
# Ensure .env has strong keys
openssl rand -hex 32  # Use for SECRET_KEY

# Restrict .env file permissions
chmod 600 .env
```

### 3. CORS Configuration

Update `.env`:
```bash
# Replace * with specific domains in production
BACKEND_CORS_ORIGINS=https://yourdomain.com,http://YOUR_EC2_IP:3000
```

### 4. SSL/TLS Setup (Recommended)

Use nginx reverse proxy with Let's Encrypt:
```bash
# Install nginx
sudo apt-get install nginx certbot python3-certbot-nginx

# Configure nginx (see AWS_DEPLOYMENT_GUIDE.md for details)
```

## Maintenance Commands

### Stop Services

```bash
./deploy-production.sh stop
```

**Note:** This only stops application containers. Infrastructure containers keep running.

### Restart Services

```bash
./deploy-production.sh restart
```

### Update Application Code

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose -f docker-compose.production.yml build
./deploy-production.sh restart
```

### Clean Up Old Images

```bash
# Remove unused images
docker image prune -a
```

## Emergency Procedures

### Complete System Restart

```bash
# Stop applications
./deploy-production.sh stop

# Verify infrastructure is still running
docker ps --filter "name=my-mongo" --filter "name=my-rabbitmq" --filter "name=qdrant"

# Start applications
./deploy-production.sh start
```

### Rollback Deployment

```bash
# Stop current deployment
./deploy-production.sh stop

# Checkout previous version
git checkout <previous-commit>

# Rebuild and deploy
./deploy-production.sh setup
./deploy-production.sh start
```

### Check Disk Space

```bash
# Check Docker disk usage
docker system df

# Clean up if needed
docker system prune -a --volumes
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `./deploy-production.sh setup` | Initial setup and build |
| `./deploy-production.sh start` | Start all application services |
| `./deploy-production.sh stop` | Stop application services |
| `./deploy-production.sh restart` | Restart services |
| `./deploy-production.sh logs` | View real-time logs |
| `./deploy-production.sh status` | Check service status |
| `./deploy-production.sh health` | Run health checks |
| `./deploy-production.sh worker` | Verify worker processing |

## Support Contacts

- Frontend: http://YOUR_EC2_IP:3000
- Middleware API: http://YOUR_EC2_IP:8080
- RAG Backend: http://YOUR_EC2_IP:8000
- RabbitMQ Management: http://YOUR_EC2_IP:15672 (guest/guest)
- Qdrant Dashboard: http://YOUR_EC2_IP:6333/dashboard

## Next Steps After Deployment

1. **Test the complete flow:**
   - User registration
   - Login
   - Send message to bot
   - Verify bot response

2. **Upload documents to RAG:**
   ```bash
   curl -X POST "http://YOUR_EC2_IP:8000/documents/upload" \
     -F "files=@document.pdf"
   ```

3. **Monitor Langfuse:**
   - Check traces at: https://langfuse.wadhwaniaiglobal.com
   - Verify both backend and middleware traces appear

4. **Set up monitoring:**
   - Consider adding Prometheus + Grafana
   - Set up log aggregation (ELK stack)
   - Configure alerting for service health

5. **Domain setup (optional):**
   - Point domain to EC2 IP
   - Set up SSL with certbot
   - Update REACT_APP_API_URL to use domain
