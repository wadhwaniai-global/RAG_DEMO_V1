"""
FastAPI backend for the Advanced RAG system.
Provides REST API endpoints for document processing and querying.
"""

import logging
import os
import tempfile
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config import settings, RAG_RESPONSE_SCHEMA
from core.rag_pipeline import AdvancedRAGPipeline, create_rag_pipeline, validate_query
from core.document_processor import validate_file_type

# Setup logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    global rag_pipeline
    try:
        rag_pipeline = await create_rag_pipeline()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
        # Comment out the raise to see the actual error
        # raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down RAG API")

# Initialize FastAPI app
app = FastAPI(
    title="Advanced RAG System API",
    description="Comprehensive RAG system with LlamaParse, Qdrant, and OpenAI integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # Add this line
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag_pipeline: Optional[AdvancedRAGPipeline] = None

# Background processing tracking
processing_tasks: Dict[str, Dict[str, Any]] = {}


# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., min_length=3, max_length=1000, description="User query")
    use_query_expansion: bool = Field(True, description="Enable query expansion techniques")
    use_reranking: bool = Field(True, description="Enable semantic re-ranking")
    document_filter: Optional[Union[str, List[str]]] = Field(None, description="Filter by specific document name(s)")
    page_filter: Optional[int] = Field(None, description="Filter by specific page number")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do I configure the authentication system?",
                "use_query_expansion": True,
                "use_reranking": True,
                "document_filter": None,
                "page_filter": None
            }
        }


class JSONPagesUploadRequest(BaseModel):
    """Request model for JSON pages upload."""
    pages: List[Dict[str, Any]] = Field(..., description="Array of JSON page objects")
    document_name: str = Field("medical_manual", description="Name of the document")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pages": [
                    {
                        "page": 1,
                        "text": "Page content...",
                        "items": [
                            {
                                "type": "heading",
                                "value": "Section Title",
                                "lvl": 1
                            }
                        ]
                    }
                ],
                "document_name": "Ethiopian-Primary-Health-Care-Clinical-Guidelines"
            }
        }


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    status: str = Field(..., description="Response status")
    answer: str = Field(..., description="Generated answer")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    retrieval_metadata: Dict[str, Any] = Field(..., description="Retrieval metadata")
    processing_time: Optional[float] = Field(None, description="Query processing time in seconds")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    task_id: str = Field(..., description="Background task ID")
    message: str = Field(..., description="Status message")
    files_received: int = Field(..., description="Number of files received")


class JSONPagesUploadResponse(BaseModel):
    """Response model for JSON pages upload."""
    task_id: str = Field(..., description="Background task ID")
    message: str = Field(..., description="Status message")
    pages_received: int = Field(..., description="Number of pages received")
    document_name: str = Field(..., description="Name of the document")


class ProcessingStatus(BaseModel):
    """Processing status response."""
    task_id: str
    status: str  # pending, processing, completed, failed
    message: str
    progress: Dict[str, Any]
    completed_at: Optional[str] = None


class SystemStats(BaseModel):
    """System statistics response."""
    vector_store: Dict[str, Any]
    total_documents: int
    document_list: List[str]
    pipeline_config: Dict[str, Any]


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Advanced RAG System",
        "version": "1.0.0",
        "pipeline_initialized": rag_pipeline is not None
    }


# Main query endpoint
@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a natural language question.
    
    Returns a structured response with answer, confidence score, and sources.
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )
    
    # Validate query
    is_valid, error_message = validate_query(request.query)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid query: {error_message}"
        )
    
    try:
        # Execute RAG query
        result = await rag_pipeline.query(
            query=request.query,
            use_query_expansion=request.use_query_expansion,
            use_reranking=request.use_reranking,
            document_filter=request.document_filter,
            page_filter=request.page_filter
        )
        
        # Format response
        response = rag_pipeline.format_response(result)
        response["processing_time"] = result.processing_time
        
        return QueryResponse(**response)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


# Document upload endpoint
@app.post("/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="PDF files to upload and process")
):
    """
    Upload and process PDF documents for the RAG system.
    
    Files are processed in the background. Use the returned task_id to check status.
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )
    
    # Validate files and read content immediately
    valid_files = []
    invalid_files = []
    
    for file in files:
        if file.filename and validate_file_type(file.filename):
            # Read file content immediately while the stream is still open
            try:
                content = await file.read()
                valid_files.append({
                    'filename': file.filename,
                    'content': content
                })
            except Exception as e:
                logger.error(f"Error reading file {file.filename}: {str(e)}")
                invalid_files.append(file.filename)
        else:
            invalid_files.append(file.filename)
    
    if not valid_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No valid PDF files provided. Invalid files: {invalid_files}"
        )
    
    if invalid_files:
        logger.warning(f"Skipping invalid files: {invalid_files}")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    processing_tasks[task_id] = {
        "status": "pending",
        "message": "Files received, processing will start shortly",
        "progress": {
            "total_files": len(valid_files),
            "processed_files": 0,
            "total_chunks": 0,
            "failed_files": []
        },
        "created_at": str(asyncio.get_event_loop().time())
    }
    
    # Start background processing with file data instead of UploadFile objects
    background_tasks.add_task(
        process_documents_background,
        task_id,
        valid_files
    )
    
    return DocumentUploadResponse(
        task_id=task_id,
        message=f"Received {len(valid_files)} files for processing",
        files_received=len(valid_files)
    )


# JSON pages upload endpoint
@app.post("/documents/upload-json", response_model=JSONPagesUploadResponse, tags=["Documents"])
async def upload_json_pages(
    background_tasks: BackgroundTasks,
    request: JSONPagesUploadRequest
):
    """
    Upload and process JSON pages for the RAG system using hierarchical chunking.
    
    Pages are processed in the background. Use the returned task_id to check status.
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )
    
    # Validate JSON pages
    if not request.pages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pages provided"
        )
    
    # Validate page structure
    for i, page in enumerate(request.pages):
        if not isinstance(page, dict) or 'page' not in page:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid page structure at index {i}. Each page must have a 'page' field."
            )
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    processing_tasks[task_id] = {
        "status": "pending",
        "message": "JSON pages received, processing will start shortly",
        "progress": {
            "total_pages": len(request.pages),
            "processed_pages": 0,
            "total_chunks": 0,
            "hierarchical_levels": 0,
            "chunk_types": []
        },
        "created_at": str(asyncio.get_event_loop().time())
    }
    
    # Start background processing
    background_tasks.add_task(
        process_json_pages_background,
        task_id,
        request.pages,
        request.document_name
    )
    
    return JSONPagesUploadResponse(
        task_id=task_id,
        message=f"Received {len(request.pages)} pages for processing",
        pages_received=len(request.pages),
        document_name=request.document_name
    )


# Check processing status
@app.get("/documents/status/{task_id}", response_model=ProcessingStatus, tags=["Documents"])
async def get_processing_status(task_id: str):
    """Get the status of a document processing task."""
    if task_id not in processing_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task_info = processing_tasks[task_id]
    
    return ProcessingStatus(
        task_id=task_id,
        status=task_info["status"],
        message=task_info["message"],
        progress=task_info["progress"],
        completed_at=task_info.get("completed_at")
    )


# List documents
@app.get("/documents", tags=["Documents"])
async def list_documents():
    """List all documents in the system."""
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )
    
    try:
        documents = await rag_pipeline.vector_store.list_documents()
        return {
            "total_documents": len(documents),
            "documents": documents
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


# Delete document
@app.delete("/documents/{document_name}", tags=["Documents"])
async def delete_document(document_name: str):
    """Delete a specific document from the system."""
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )
    
    try:
        success = await rag_pipeline.vector_store.delete_document(document_name)
        
        if success:
            return {"message": f"Document '{document_name}' deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_name}' not found or could not be deleted"
            )
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


# System statistics
@app.get("/system/stats", response_model=SystemStats, tags=["System"])
async def get_system_stats():
    """Get system statistics and configuration."""
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )
    
    try:
        stats = await rag_pipeline.get_system_stats()
        return SystemStats(**stats)
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system stats: {str(e)}"
        )


# API schema endpoint
@app.get("/schema/response", tags=["Schema"])
async def get_response_schema():
    """Get the JSON schema for RAG responses."""
    return {
        "rag_response_schema": RAG_RESPONSE_SCHEMA,
        "description": "JSON schema for all RAG query responses"
    }


# Background processing function
async def process_documents_background(task_id: str, file_data: List[Dict[str, Any]]):
    """
    Background task for processing uploaded documents.
    
    Args:
        task_id: Task identifier
        file_data: List of dictionaries containing filename and content
    """
    try:
        # Update status to processing
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["message"] = "Processing documents..."
        
        # Save files to temporary directory
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        for file_info in file_data:
            file_path = os.path.join(temp_dir, file_info['filename'])
            
            # Save file content directly (content is already bytes)
            with open(file_path, "wb") as buffer:
                buffer.write(file_info['content'])
            
            file_paths.append(file_path)
        
        # Process documents
        result = await rag_pipeline.process_documents(file_paths)
        
        # Update progress
        processing_tasks[task_id]["progress"].update({
            "processed_files": result["successful_files"],
            "total_chunks": result["total_chunks"],
            "failed_files": result["failed_files"],
            "processing_time": result["processing_time"]
        })
        
        # Clean up temporary files
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except:
                pass
        
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        # Update final status
        if result["failed_files"]:
            processing_tasks[task_id]["status"] = "completed_with_errors"
            processing_tasks[task_id]["message"] = f"Processing completed with {len(result['failed_files'])} errors"
        else:
            processing_tasks[task_id]["status"] = "completed"
            processing_tasks[task_id]["message"] = "All documents processed successfully"
        
        processing_tasks[task_id]["completed_at"] = str(asyncio.get_event_loop().time())
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error in background processing: {str(e)}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["message"] = f"Processing failed: {str(e)}"
        processing_tasks[task_id]["completed_at"] = str(asyncio.get_event_loop().time())


# Background processing function for JSON pages
async def process_json_pages_background(task_id: str, json_pages: List[Dict[str, Any]], document_name: str):
    """
    Background task for processing JSON pages with hierarchical chunking.
    
    Args:
        task_id: Task identifier
        json_pages: List of JSON page objects
        document_name: Name of the document
    """
    try:
        # Update status to processing
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["message"] = "Processing JSON pages with hierarchical chunking..."
        
        # Process JSON pages
        result = await rag_pipeline.process_json_pages(json_pages, document_name)
        
        # Update progress
        processing_tasks[task_id]["progress"].update({
            "processed_pages": result["total_pages"],
            "total_chunks": result["total_chunks"],
            "hierarchical_levels": result.get("hierarchical_levels", 0),
            "chunk_types": result.get("chunk_types", []),
            "processing_time": result["processing_time"]
        })
        
        # Update final status
        if result["success"]:
            processing_tasks[task_id]["status"] = "completed"
            processing_tasks[task_id]["message"] = f"Successfully processed {result['total_pages']} pages into {result['total_chunks']} hierarchical chunks"
        else:
            processing_tasks[task_id]["status"] = "failed"
            processing_tasks[task_id]["message"] = f"Processing failed: {result.get('error', 'Unknown error')}"
        
        processing_tasks[task_id]["completed_at"] = str(asyncio.get_event_loop().time())
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error in JSON pages background processing: {str(e)}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["message"] = f"Processing failed: {str(e)}"
        processing_tasks[task_id]["completed_at"] = str(asyncio.get_event_loop().time())


# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "message": "An internal server error occurred",
            "status_code": 500
        }
    )


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    ) 