# RAG POC Middleware Application

A modern, fast web API for chat-based RAG (Retrieval-Augmented Generation) system built with FastAPI, MongoEngine, RabbitMQ, and Pydantic.

## Features

- **FastAPI**: Modern, fast web framework for building APIs
- **MongoEngine**: Document-Object Mapping for MongoDB
- **Pydantic**: Data validation and serialization using Python type annotations
- **RabbitMQ Integration**: Asynchronous message queue processing
- **RAG API Integration**: External RAG API integration for intelligent responses
- **Chat System**: Human-to-bot messaging with automatic responses
- **Automatic API Documentation**: Interactive API docs with Swagger UI
- **CORS Support**: Cross-Origin Resource Sharing configuration
- **Security**: Password hashing and JWT token authentication
- **Environment Configuration**: Flexible configuration management
- **Worker System**: Background worker for processing chat messages

## Project Structure

```
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── app/
│   ├── __init__.py
│   ├── core/              # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py      # Configuration settings
│   │   └── security.py    # Security utilities
│   ├── models/            # MongoEngine models
│   │   ├── __init__.py
│   │   ├── user.py        # User model
│   │   └── chat.py        # Chat model
│   ├── schemas/           # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── user.py        # User schemas
│   │   └── chat.py        # Chat schemas
│   ├── services/          # Business logic services
│   │   ├── __init__.py
│   │   ├── rabbitmq.py    # RabbitMQ service
│   │   ├── rag_api.py     # RAG API client
│   │   └── worker.py      # Background worker
│   └── api/               # API routes
│       ├── __init__.py
│       └── v1/
│           ├── __init__.py
│           ├── api.py     # API router
│           └── endpoints/
│               ├── __init__.py
│               ├── users.py    # User endpoints
│               └── chat.py     # Chat endpoints
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-poc-middleware-app
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start MongoDB and RabbitMQ**
   Make sure both MongoDB and RabbitMQ are running on your system
   ```bash
   # Start MongoDB (varies by installation)
   mongod
   
   # Start RabbitMQ (varies by installation)
   rabbitmq-server
   ```

## Configuration

Copy `.env.example` to `.env` and update the values:

### Basic Configuration
- `DATABASE_URL`: MongoDB connection string
- `DATABASE_NAME`: Database name
- `SECRET_KEY`: Secret key for JWT tokens (change in production!)
- `BACKEND_CORS_ORIGINS`: Allowed CORS origins

### RabbitMQ Configuration
- `RABBITMQ_URL`: RabbitMQ connection string (default: amqp://localhost:5672)
- `CHAT_QUEUE_NAME`: Queue name for chat processing (default: chat_processing_queue)

### RAG API Configuration
- `RAG_API_URL`: URL of the external RAG API (default: http://localhost:8888)
- `RAG_USE_QUERY_EXPANSION`: Enable query expansion (default: true)
- `RAG_USE_RERANKING`: Enable reranking (default: true)

### Worker Configuration
- `SUPERUSER_TOKEN`: JWT token for superuser (required for worker to create bot responses)
- `WORKER_ENABLED`: Enable/disable the background worker (default: true)

## Running the Application

```bash
# Development mode with auto-reload
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- Main application: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

## API Endpoints

### Users
- `POST /api/v1/users/` - Create a new user (human or bot)
- `GET /api/v1/users/` - Get all users (with pagination and filtering)
- `GET /api/v1/users/{user_id}` - Get user by ID
- `PUT /api/v1/users/{user_id}` - Update user
- `DELETE /api/v1/users/{user_id}` - Delete user
- `POST /api/v1/users/generate-token` - Generate JWT token for authentication

### Chat
- `POST /api/v1/chat/` - Create a new chat message (auto-triggers RAG processing for human messages)
- `GET /api/v1/chat/messages/` - Get chat messages for a conversation with pagination

## Models

### User Model
- `name`: Unique name
- `email`: Email address (required for human users)
- `description`: User description (optional)
- `user_type`: Either 'human' or 'bot'
- `hashed_password`: Hashed password (only for human users)
- `is_active`: Account status
- `is_superuser`: Admin privileges (only for human users)
- `created_at`, `updated_at`: Timestamps

### Chat Model
- `sender_id`: ID of the message sender
- `receiver_id`: ID of the message receiver
- `message`: The chat message content
- `offset`: Sequential number for conversation ordering
- `is_read`, `is_delivered`, `is_seen`: Message status flags
- `is_deleted`, `is_archived`, `is_pinned`: Message state flags
- `created_at`, `updated_at`: Timestamps

## How the RAG System Works

1. **Human Message Creation**: When a human user sends a message to a bot user, the message is saved to MongoDB
2. **Queue Publishing**: The message is automatically published to a RabbitMQ queue for asynchronous processing
3. **Worker Processing**: The background worker picks up the message and:
   - Extracts the message text as a query
   - Sends it to the external RAG API
   - Receives the intelligent response
4. **Bot Response**: Using the superuser token, the worker creates a new chat message with:
   - Sender: The original receiver (bot)
   - Receiver: The original sender (human)
   - Message: The RAG API response

## Setup Instructions

### 1. Create a Superuser
First, create a superuser account:
```bash
curl -X POST "http://localhost:8000/api/v1/users/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "superuser",
    "email": "admin@example.com",
    "password": "securepassword123",
    "user_type": "human",
    "is_superuser": true
  }'
```

### 2. Generate Superuser Token
Generate a token for the superuser:
```bash
curl -X POST "http://localhost:8000/api/v1/users/generate-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "superuser",
    "password": "securepassword123"
  }'
```

### 3. Update Environment
Add the token to your `.env` file:
```bash
SUPERUSER_TOKEN=your-generated-jwt-token-here
```

### 4. Create Bot User
Create a bot user that will respond to human messages:
```bash
curl -X POST "http://localhost:8000/api/v1/users/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "rag-bot",
    "description": "Intelligent RAG-powered assistant",
    "user_type": "bot"
  }'
```

## Development

### Adding New Models

1. Create the MongoEngine model in `app/models/`
2. Create Pydantic schemas in `app/schemas/`
3. Add API endpoints in `app/api/v1/endpoints/`
4. Register the router in `app/api/v1/api.py`

### Testing the Chat Flow

1. Create a human user and get their authentication token
2. Send a message from human to bot:
```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "Authorization: Bearer YOUR_USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "sender_id": "human_user_id",
    "receiver_id": "bot_user_id",
    "message": "What is the summary of Kenya'\''s health policy?"
  }'
```
3. The bot will automatically respond via the RAG worker

## Security

- Passwords are hashed using bcrypt
- JWT tokens for authentication
- CORS middleware for cross-origin requests
- Input validation with Pydantic

## Production Deployment

1. Set `DEBUG=False` in environment
2. Use a strong `SECRET_KEY`
3. Configure proper MongoDB credentials
4. Use a production WSGI server like Gunicorn
5. Set up reverse proxy (Nginx)
6. Configure proper CORS origins

```bash
# Example production run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```