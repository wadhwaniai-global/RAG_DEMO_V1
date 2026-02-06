import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini-2024-07-18", env="OPENAI_MODEL")
    openai_embedding_model: str = Field("text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")

    # Langfuse Configuration
    langfuse_secret_key: Optional[str] = Field(None, env="LANGFUSE_SECRET_KEY")
    langfuse_public_key: Optional[str] = Field(None, env="LANGFUSE_PUBLIC_KEY")
    langfuse_host: Optional[str] = Field(None, env="LANGFUSE_HOST")
    langfuse_debug: bool = Field(False, env="LANGFUSE_DEBUG")
    langfuse_sample_rate: Optional[float] = Field(None, env="LANGFUSE_SAMPLE_RATE")
    langfuse_tracing_enabled: Optional[bool] = Field(None, env="LANGFUSE_TRACING_ENABLED")
    
    # LlamaParse Configuration
    llama_cloud_api_key: str = Field(..., env="LLAMA_CLOUD_API_KEY")
    
    # Qdrant Configuration
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field("rag_documents", env="QDRANT_COLLECTION_NAME")
    
    # Application Configuration
    app_env: str = Field("development", env="APP_ENV")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_chunk_size: int = Field(1000, env="MAX_CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    max_retrievals: int = Field(10, env="MAX_RETRIEVALS")
    
    # Optional: Redis for caching
    redis_url: Optional[str] = Field("redis://localhost:6379/0", env="REDIS_URL")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_reload: bool = Field(True, env="API_RELOAD")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def langfuse_enabled(self) -> bool:
        """Return True when Langfuse credentials are available."""
        return bool(self.langfuse_secret_key and self.langfuse_public_key)


# Global settings instance
settings = Settings()


# JSON Schema for RAG responses
RAG_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["success", "no_relevant_documents", "insufficient_information"]
        },
        "answer": {
            "type": "string",
            "description": "The answer to the user's question based on the documents"
        },
        "confidence_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence level of the answer (0-1)"
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document_name": {"type": "string"},
                    "page_number": {"type": "integer"},
                    "section": {"type": "string"},
                    "relevance_score": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
        },
        "retrieval_metadata": {
            "type": "object",
            "properties": {
                "total_documents_searched": {"type": "integer"},
                "query_expansion_used": {"type": "boolean"},
                "hybrid_search_used": {"type": "boolean"}
            }
        }
    },
    "required": ["status", "answer", "confidence_score", "sources"]
}

# Response for unrelated queries
UNRELATED_QUERY_RESPONSE = {
    "status": "no_relevant_documents",
    "answer": "I can only answer questions based on the documents in my knowledge base. Your query doesn't appear to be related to the available documentation.",
    "confidence_score": 0.0,
    "sources": [],
    "retrieval_metadata": {
        "total_documents_searched": 0,
        "query_expansion_used": False,
        "hybrid_search_used": False
    }
} 