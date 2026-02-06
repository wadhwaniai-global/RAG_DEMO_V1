# Advanced RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system built with state-of-the-art components for document processing, vector search, and intelligent querying.

## Features

### 🚀 Core Capabilities
- **LlamaParse Integration**: Advanced PDF parsing with structure preservation
- **Hybrid Search**: Combines dense vector search with lexical search using Qdrant
- **Page-Level Chunking**: Intelligent document segmentation with metadata preservation
- **Query Expansion**: Multiple retrieval strategies including HyDE (Hypothetical Document Embeddings)
- **Hallucination Prevention**: Advanced prompting and confidence scoring
- **Metadata Augmentation**: Enhanced embeddings with document structure information
- **Langfuse Observability**: Drop-in tracing for every OpenAI call with query-level metadata

### 🛠 Technology Stack
- **LLM**: OpenAI GPT-4o Mini (configurable)
- **Embeddings**: OpenAI text-embedding-3-large
- **Vector Database**: Qdrant (with hybrid search support)
- **Document Processing**: LlamaParse
- **API Framework**: FastAPI
- **Language**: Python 3.8+

### 🔍 Advanced Retrieval Features
1. **Query Expansion**: Automatically generates alternative query phrasings
2. **HyDE (Hypothetical Document Embeddings)**: Creates hypothetical answers to improve retrieval
3. **Multi-Query Retrieval**: Uses ensemble methods for better recall
4. **Semantic Re-ranking**: Post-retrieval ranking using semantic similarity
5. **Confidence Scoring**: Multi-factor confidence assessment
6. **Hybrid Search**: Combines vector similarity with lexical matching

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd waig-rag-poc

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Copy the example environment file and configure your API keys:

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here

# Qdrant Configuration (local by default)
QDRANT_URL=http://localhost:6333

# Optional: Customize models
OPENAI_MODEL=gpt-4o-mini-2024-07-18
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Optional: Langfuse tracing
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_HOST=https://langfuse.wadhwaniaiglobal.com
# LANGFUSE_DEBUG=false
# LANGFUSE_SAMPLE_RATE=1.0
# LANGFUSE_TRACING_ENABLED=true
```

### 3. Start Qdrant Vector Database

Using Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Using Docker Compose:
```bash
# Create docker-compose.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

```bash
docker-compose up -d
```

### 4. Start the API Server

```bash
# Development mode
python -m api.main

# Or using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Usage

### Upload Documents

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

Response:
```json
{
  "task_id": "uuid-here",
  "message": "Received 2 files for processing",
  "files_received": 2
}
```

### Check Processing Status

```bash
curl "http://localhost:8000/documents/status/{task_id}"
```

### Query Documents

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I configure authentication?",
    "use_query_expansion": true,
    "use_reranking": true
  }'
```

Response:
```json
{
  "status": "success",
  "answer": "According to the documentation...",
  "confidence_score": 0.85,
  "sources": [
    {
      "document_name": "auth_guide",
      "page_number": 5,
      "section": "Configuration",
      "relevance_score": 0.92
    }
  ],
  "retrieval_metadata": {
    "total_documents_searched": 3,
    "query_expansion_used": true,
    "hybrid_search_used": true
  }
}
```

## Python SDK Usage

### Basic RAG Pipeline

```python
import asyncio
from core.rag_pipeline import create_rag_pipeline

async def main():
    # Initialize pipeline
    pipeline = await create_rag_pipeline()
    
    # Process documents
    file_paths = ["document1.pdf", "document2.pdf"]
    result = await pipeline.process_documents(file_paths)
    print(f"Processed {result['total_chunks']} chunks from {result['successful_files']} files")
    
    # Query documents
    rag_result = await pipeline.query(
        query="How do I configure authentication?",
        use_query_expansion=True,
        use_reranking=True
    )
    
    print(f"Answer: {rag_result.answer}")
    print(f"Confidence: {rag_result.confidence_score:.3f}")
    print(f"Sources: {len(rag_result.sources)}")

# Run the example
asyncio.run(main())
```

### Document Processing Only

```python
from core.document_processor import DocumentProcessor
from core.embeddings import OpenAIEmbeddings
from core.vector_store import QdrantVectorStore

async def process_documents():
    # Initialize components
    processor = DocumentProcessor()
    embeddings = OpenAIEmbeddings()
    vector_store = QdrantVectorStore()
    
    # Process a PDF
    chunks = await processor.process_pdf("document.pdf")
    
    # Generate embeddings
    embedding_results = await embeddings.embed_chunks(chunks)
    embeddings_list = [result.embedding for result in embedding_results]
    
    # Store in vector database
    success = await vector_store.add_chunks(chunks, embeddings_list)
    print(f"Stored {len(chunks)} chunks: {success}")
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `LLAMA_CLOUD_API_KEY` | Yes | - | LlamaParse API key |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | No | - | Qdrant API key (for cloud) |
| `OPENAI_MODEL` | No | `gpt-4o-mini-2024-07-18` | OpenAI model for generation |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-large` | OpenAI embedding model |
| `LANGFUSE_SECRET_KEY` | No | - | Langfuse secret key for authenticated tracing |
| `LANGFUSE_PUBLIC_KEY` | No | - | Langfuse public key for dashboard access |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com` | Langfuse host (set to your self-hosted URL if applicable) |
| `LANGFUSE_SAMPLE_RATE` | No | `1.0` | Sampling rate for Langfuse traces |
| `LANGFUSE_TRACING_ENABLED` | No | `true` | Toggle Langfuse tracing without code changes |
| `MAX_CHUNK_SIZE` | No | `1000` | Maximum tokens per chunk |
| `CHUNK_OVERLAP` | No | `200` | Token overlap between chunks |
| `MAX_RETRIEVALS` | No | `10` | Maximum search results |

### Advanced Configuration

The system can be customized through the `config.py` file:

```python
from config import settings

# Modify chunk size
settings.max_chunk_size = 1500

# Change retrieval settings
settings.max_retrievals = 15

# Update model
settings.openai_model = "gpt-4o-2024-08-06"
```

## Advanced Features

### 1. Query Expansion Strategies

The system implements multiple query expansion techniques:

- **Semantic Expansion**: Generate semantically related queries
- **HyDE**: Create hypothetical documents that would answer the query
- **Multi-Query**: Ensemble multiple query representations

### 2. Hybrid Search

Combines two search approaches:
- **Dense Vector Search**: Semantic similarity using embeddings
- **Lexical Search**: Keyword-based matching
- **Weighted Combination**: Configurable alpha parameter (default: 0.7 vector, 0.3 lexical)

### 3. Metadata Augmentation

Embeddings are enhanced with document structure:
```
Document: user_manual | Page Title: Authentication Setup | Section: Configuration | Page: 5

Your original content here...
```

### 4. Confidence Scoring

Multi-factor confidence assessment based on:
- Search result relevance scores
- Number of supporting sources
- Answer completeness
- Uncertainty indicators
- Citation presence

### 5. Hallucination Prevention

- **Strict Source Attribution**: Only use provided documents
- **Uncertainty Expression**: Explicitly state when information is insufficient
- **Citation Requirements**: Force specific source citations
- **Confidence Penalties**: Lower scores for uncertain answers

### 6. Langfuse Observability

- **Drop-In Instrumentation**: The project imports `langfuse.openai` as a replacement for the standard OpenAI SDK, providing automatic tracing without additional code.
- **Rich Metadata**: Each embedding and completion call annotates traces with component names, batch sizes, and hashed query identifiers for safe debugging.
- **Centralised Client Management**: A shared Langfuse client registers a flush hook to capture events from short-lived jobs.
- **Configuration Driven**: Toggle tracing or adjust sampling directly from environment variables without redeployment.

## API Endpoints

### Document Management
- `POST /documents/upload` - Upload PDF documents
- `GET /documents/status/{task_id}` - Check processing status
- `GET /documents` - List all documents
- `DELETE /documents/{document_name}` - Delete a document

### Querying
- `POST /query` - Query the RAG system
- `GET /schema/response` - Get response JSON schema

### System
- `GET /health` - Health check
- `GET /system/stats` - System statistics

## Response Format

All query responses follow a standardized JSON schema:

```json
{
  "status": "success|no_relevant_documents|insufficient_information",
  "answer": "Generated answer based on documents",
  "confidence_score": 0.85,
  "sources": [
    {
      "document_name": "filename",
      "page_number": 5,
      "section": "Section Title",
      "relevance_score": 0.92
    }
  ],
  "retrieval_metadata": {
    "total_documents_searched": 10,
    "query_expansion_used": true,
    "hybrid_search_used": true,
    "reranking_used": true
  }
}
```

## Performance Optimization

### 1. Caching

The system includes multiple caching layers:
- **Embedding Cache**: In-memory caching for repeated texts
- **Vector Store**: Optimized Qdrant configuration
- **Query Results**: Can be extended with Redis caching

### 2. Batch Processing

- Documents are processed in parallel where possible
- Embeddings are generated in optimized batches
- Background processing for large document sets

### 3. Qdrant Optimization

The vector database is configured with:
- HNSW indexing for fast similarity search
- Quantization for memory efficiency
- Payload indexing for fast filtering
- Optimized segment configuration

## Troubleshooting

### Common Issues

1. **"RAG pipeline not initialized"**
   - Check that all required API keys are set
   - Ensure Qdrant is running and accessible
   - Check logs for initialization errors

2. **"Failed to connect to Qdrant"**
   - Verify Qdrant is running on the specified URL
   - Check firewall settings
   - Ensure correct port (default: 6333)

3. **"LlamaParse API error"**
   - Verify your LlamaParse API key is valid
   - Check API quota and usage limits
   - Ensure PDF files are not corrupted

4. **Low confidence scores**
   - Try enabling query expansion
   - Check if relevant documents are uploaded
   - Review document quality and relevance

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:
```bash
export LOG_LEVEL=DEBUG
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Security Considerations

1. **API Keys**: Use secure secret management
2. **CORS**: Configure appropriate origins for production
3. **Rate Limiting**: Implement rate limiting for API endpoints
4. **Authentication**: Add authentication middleware
5. **File Validation**: Strict file type and size validation

### Scaling

- **Horizontal Scaling**: Run multiple API instances behind a load balancer
- **Vector Database**: Use Qdrant Cloud or cluster setup
- **Caching**: Implement Redis for distributed caching
- **Background Processing**: Use Celery for document processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## License

[License information here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Open an issue on GitHub 