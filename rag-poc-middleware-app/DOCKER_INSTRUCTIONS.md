# Docker Containerization Instructions

This guide explains how to containerize and run the FastAPI RAG POC Middleware Application using Docker.

## Prerequisites

- Docker installed on your system
- MongoDB running on your server (accessible from the container)
- RabbitMQ running on your server (accessible from the container)
- OpenAI API key for Whisper transcription functionality

## Quick Start

### 1. Environment Setup

Copy the environment template and configure your settings:

```bash
cp env.example .env
```

Edit `.env` file with your configuration:

```bash
# Required: Update these values
DATABASE_URL=mongodb://your-mongodb-host:27017
RABBITMQ_HOST=your-rabbitmq-host
OPENAI_API_KEY=your-actual-openai-api-key
SECRET_KEY=your-secure-secret-key

# Optional: Update if different from defaults
DATABASE_NAME=your-database-name
RABBITMQ_USERNAME=your-rabbitmq-username
RABBITMQ_PASSWORD=your-rabbitmq-password
```

### 2. Build the Docker Image

```bash
docker build -t rag-poc-middleware .
```

### 3. Run the Container

#### Option A: Using Environment File
```bash
docker run -d \
  --name rag-poc-app \
  --env-file .env \
  -p 8000:8000 \
  --network host \
  rag-poc-middleware
```

#### Option B: Using Individual Environment Variables
```bash
docker run -d \
  --name rag-poc-app \
  -e DATABASE_URL=mongodb://your-mongodb-host:27017 \
  -e RABBITMQ_HOST=your-rabbitmq-host \
  -e OPENAI_API_KEY=your-openai-api-key \
  -e SECRET_KEY=your-secret-key \
  -p 8000:8000 \
  --network host \
  rag-poc-middleware
```

## Configuration Details

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | MongoDB connection string | `mongodb://localhost:27017` | Yes |
| `DATABASE_NAME` | MongoDB database name | `fastapi_db` | No |
| `RABBITMQ_HOST` | RabbitMQ host | `localhost` | Yes |
| `RABBITMQ_PORT` | RabbitMQ port | `5672` | No |
| `RABBITMQ_USERNAME` | RabbitMQ username | `guest` | No |
| `RABBITMQ_PASSWORD` | RabbitMQ password | `guest` | No |
| `OPENAI_API_KEY` | OpenAI API key for Whisper | - | Yes |
| `SECRET_KEY` | JWT secret key | - | Yes |
| `HOST` | Application host | `0.0.0.0` | No |
| `PORT` | Application port | `8000` | No |
| `WORKER_ENABLED` | Enable background worker | `true` | No |

### Network Configuration

The application uses `--network host` to access your existing MongoDB and RabbitMQ services. This allows the container to connect to services running on the host machine.

If your services are on different hosts, you can:
1. Use specific host IPs in your environment variables
2. Use Docker networks to connect containers
3. Use port mapping with specific host addresses

## Container Features

### What's Included

- **FastAPI Application**: Main API server with all endpoints
- **Background Worker**: Processes chat messages from RabbitMQ
- **Health Checks**: Built-in health monitoring
- **Security**: Non-root user execution
- **Optimized Build**: Multi-stage build with minimal dependencies

### Application Structure

The container runs both:
1. **FastAPI Server**: Handles HTTP requests and API endpoints
2. **Chat Worker**: Background thread that processes messages from RabbitMQ queue

Both components start automatically when the container starts.

## Monitoring and Management

### Check Container Status
```bash
docker ps
```

### View Logs
```bash
# View all logs
docker logs rag-poc-app

# Follow logs in real-time
docker logs -f rag-poc-app
```

### Health Check
```bash
# Check if application is healthy
curl http://localhost:8000/health

# Check API documentation
curl http://localhost:8000/docs
```

### Stop and Remove
```bash
# Stop the container
docker stop rag-poc-app

# Remove the container
docker rm rag-poc-app
```

## Troubleshooting

### Common Issues

1. **Connection to MongoDB/RabbitMQ fails**
   - Ensure services are running on the host
   - Check network connectivity
   - Verify host addresses in environment variables

2. **OpenAI API errors**
   - Verify your API key is correct
   - Check API key permissions
   - Ensure you have credits in your OpenAI account

3. **Worker not processing messages**
   - Check RabbitMQ connection
   - Verify queue exists
   - Check worker logs for errors

### Debug Mode

To run in debug mode with more verbose logging:

```bash
docker run -d \
  --name rag-poc-app \
  --env-file .env \
  -e DEBUG=true \
  -p 8000:8000 \
  --network host \
  rag-poc-middleware
```

### Access Container Shell

```bash
docker exec -it rag-poc-app /bin/bash
```

## Production Considerations

### Security
- Change default `SECRET_KEY` to a secure random string
- Use strong passwords for database and message queue
- Consider using Docker secrets for sensitive data
- Regularly update base images

### Performance
- Monitor resource usage with `docker stats`
- Consider resource limits for production
- Use proper logging configuration
- Set up monitoring and alerting

### Scaling
- The application can be scaled horizontally
- Each instance will process messages independently
- Consider using a load balancer for multiple instances

## API Endpoints

Once running, the application provides:

- **Health Check**: `GET /health`
- **API Documentation**: `GET /docs`
- **OpenAPI Schema**: `GET /api/v1/openapi.json`
- **Chat Endpoints**: `/api/v1/chat/*`
- **User Endpoints**: `/api/v1/users/*`
- **Whisper Transcription**: `/api/v1/whisper/transcribe`

## Support

For issues or questions:
1. Check the application logs
2. Verify your environment configuration
3. Ensure external services (MongoDB, RabbitMQ) are accessible
4. Review the API documentation at `/docs`
