from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mongoengine import connect, disconnect
import uvicorn
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.v1.api import api_router
from app.services.worker import chat_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Connect to MongoDB
    connect(
        db=settings.DATABASE_NAME,
        host=settings.DATABASE_URL,
        username=settings.DATABASE_USERNAME,
        password=settings.DATABASE_PASSWORD,
    )
    logger.info("Connected to MongoDB")
    
    # Start the chat worker
    chat_worker.start()
    logger.info("Chat worker started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    chat_worker.stop()
    disconnect()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.cors_origins_list],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {"ok": True}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )