# RAG Chat System Installation Guide

This guide will help you set up the RAG (Retrieval-Augmented Generation) chat system step by step.

## Prerequisites

1. **Python 3.8+** installed
2. **MongoDB** running on localhost:27017
3. **RabbitMQ** running on localhost:5672
4. **RAG API** running on localhost:8888

## Quick Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Required Services

#### MongoDB
```bash
# On macOS with Homebrew
brew services start mongodb-community

# On Ubuntu/Debian
sudo systemctl start mongod

# Or run directly
mongod
```

#### RabbitMQ
```bash
# On macOS with Homebrew
brew services start rabbitmq

# On Ubuntu/Debian
sudo systemctl start rabbitmq-server

# Or run directly
rabbitmq-server
```

#### Your RAG API
Make sure your RAG API is running on `http://localhost:8888` with the `/query` endpoint.

### 3. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings (MongoDB, RabbitMQ URLs, etc.)
# Note: SUPERUSER_TOKEN will be set automatically by the setup script
```

### 4. Start the Application

```bash
python main.py
```

The server will start on `http://localhost:8000`

### 5. Run Setup Script

In a new terminal, run the setup script to create necessary users and tokens:

```bash
python setup_rag_system.py
```

This script will:
- Create a superuser account
- Generate a superuser token
- Update your .env file with the token
- Create a bot user for RAG responses
- Create a test human user
- Provide test commands

### 6. Restart the Server

After the setup script updates your .env file, restart the server:

```bash
# Stop the server (Ctrl+C) and restart
python main.py
```

## Testing the System

### 1. Test Chat Flow

Use the curl command provided by the setup script, or:

```bash
# Create a human user first
curl -X POST "http://localhost:8000/api/v1/users/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "human_user",
    "email": "user@example.com",
    "password": "userpassword123",
    "user_type": "human"
  }'

# Generate token for human user
curl -X POST "http://localhost:8000/api/v1/users/generate-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "human_user",
    "password": "userpassword123"
  }'

# Send message to bot (replace USER_TOKEN, HUMAN_ID, BOT_ID with actual values)
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "Authorization: Bearer USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "sender_id": "HUMAN_ID",
    "receiver_id": "BOT_ID",
    "message": "What is the summary of Kenya'\''s health policy?"
  }'
```

### 2. Check Logs

Monitor the application logs to see:
- Message publishing to RabbitMQ
- Worker processing messages
- RAG API calls
- Bot response creation

### 3. View API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## Troubleshooting

### Common Issues

1. **Import errors for pika/httpx**
   - Run `pip install -r requirements.txt`

2. **MongoDB connection failed**
   - Ensure MongoDB is running: `mongod`
   - Check DATABASE_URL in .env

3. **RabbitMQ connection failed**
   - Ensure RabbitMQ is running: `rabbitmq-server`
   - Check RABBITMQ_URL in .env

4. **RAG API connection failed**
   - Ensure your RAG API is running on localhost:8888
   - Check RAG_API_URL in .env

5. **Worker not starting**
   - Check SUPERUSER_TOKEN is set in .env
   - Check WORKER_ENABLED=true in .env
   - Restart the server after updating .env

6. **Bot not responding**
   - Check worker logs for errors
   - Verify RAG API is accessible
   - Ensure superuser token is valid

### Environment Variables Reference

```bash
# Required
SUPERUSER_TOKEN=jwt-token-from-setup-script

# Optional (with defaults)
RABBITMQ_URL=amqp://localhost:5672
RAG_API_URL=http://localhost:8888
WORKER_ENABLED=true
RAG_USE_QUERY_EXPANSION=true
RAG_USE_RERANKING=true
```

## Production Deployment

For production deployment:

1. Set strong passwords and tokens
2. Use proper MongoDB authentication
3. Use RabbitMQ with authentication
4. Set DEBUG=false
5. Use HTTPS
6. Set up proper logging
7. Use a process manager like systemd or supervisor

## Architecture

```
Human User -> FastAPI -> MongoDB (save message)
                      -> RabbitMQ (queue message)
                                 -> Worker -> RAG API -> Worker -> FastAPI -> MongoDB (bot response)
```

The system provides asynchronous processing of human messages through RabbitMQ, allowing for scalable RAG-powered chat responses.