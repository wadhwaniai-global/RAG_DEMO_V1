from pydantic_settings import BaseSettings
from typing import List, Optional
from pydantic import Field, field_validator


class Settings(BaseSettings):
    PROJECT_NAME: str = Field(default="FastAPI MongoEngine App")
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Server settings
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=True)
    
    # Database settings
    DATABASE_URL: str = Field(default="mongodb://localhost:27017")
    DATABASE_NAME: str = Field(default="fastapi_db")
    DATABASE_USERNAME: Optional[str] = Field(default=None)
    DATABASE_PASSWORD: Optional[str] = Field(default=None)
    
    # CORS settings
    BACKEND_CORS_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:8080")
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Convert CORS origins string to list"""
        return [i.strip() for i in self.BACKEND_CORS_ORIGINS.split(",")]
    
    # Security settings
    SECRET_KEY: str = Field(default="your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    LONG_ACCESS_TOKEN_EXPIRE_DAYS: int = Field(default=30)
    
    # RabbitMQ settings
    RABBITMQ_HOST: str = Field(default="localhost")
    RABBITMQ_PORT: int = Field(default=5672)
    RABBITMQ_USERNAME: str = Field(default="guest")
    RABBITMQ_PASSWORD: str = Field(default="guest")
    RABBITMQ_VIRTUAL_HOST: str = Field(default="/")
    CHAT_QUEUE_NAME: str = Field(default="chat_processing_queue")
    
    @property
    def RABBITMQ_URL(self) -> str:
        """Construct RabbitMQ URL from components"""
        return f"amqp://{self.RABBITMQ_USERNAME}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}{self.RABBITMQ_VIRTUAL_HOST}"
    
    # External RAG API settings
    RAG_API_URL: str = Field(default="http://localhost:8888")
    RAG_API_ENDPOINT: str = "/query"
    RAG_USE_QUERY_EXPANSION: bool = Field(default=False)
    RAG_USE_RERANKING: bool = Field(default=False)
    
    # Worker settings
    SUPERUSER_TOKEN: str = Field(default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNjg5MjU2M2ViMzhlMTRhY2I5NzAyNDQ4IiwibmFtZSI6ImtlbnlhX2JvdCIsInVzZXJfdHlwZSI6ImJvdCIsImV4cCI6MTc1NzAxNDUzOH0.JGnp0S-ixlacUrgazKQW4quRvlUrVXTy5AzaBQqwZ9w")
    WORKER_ENABLED: bool = Field(default=True)

    class Config:
        case_sensitive = True


settings = Settings()