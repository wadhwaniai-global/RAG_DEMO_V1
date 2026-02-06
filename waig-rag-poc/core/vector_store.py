"""
Vector database integration with Qdrant for hybrid search capabilities.
Supports both dense vector search and lexical search for optimal retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
from dataclasses import asdict

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    VectorParams, Distance, CollectionInfo, PointStruct,
    Filter, FieldCondition, PayloadField, SearchRequest,
    ScrollRequest, UpdateStatus, SearchParams, HnswConfigDiff,
    MatchAny
)
import numpy as np

from config import settings
from core.document_processor import DocumentChunk

# Setup logging
logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant vector store with hybrid search capabilities.
    Supports dense vector search and sparse vector search for lexical matching.
    """
    
    def __init__(self):
        """Initialize Qdrant client and configuration."""
        self.collection_name = settings.qdrant_collection_name
        
        # Initialize sync client for management operations
        if settings.qdrant_api_key:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
            self.async_client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
            self.client = QdrantClient(url=settings.qdrant_url)
            self.async_client = AsyncQdrantClient(url=settings.qdrant_url)
        
        # Vector configuration
        self.vector_size = 3072  # OpenAI text-embedding-3-large dimension
        self.distance_metric = Distance.COSINE
        
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self) -> bool:
        """Ensure the collection exists with proper configuration."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self._create_collection()
                return True
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                return False
                
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            raise
    
    def _create_collection(self) -> None:
        """Create a new collection with optimized configuration."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric,
                        hnsw_config=HnswConfigDiff(
                            m=16,  # Number of bi-directional links for each new element
                            ef_construct=200,  # Size of dynamic candidate list  
                            full_scan_threshold=10000,  # Use exact search for small datasets
                        )
                    )
                },
                # Enable payload indexing for fast filtering
                optimizers_config={
                    "default_segment_number": 2,
                    "max_segment_size": 20000,
                    "memmap_threshold": 20000,
                    "indexing_threshold": 20000,
                    "flush_interval_sec": 5,
                },
                # Configure quantization for memory efficiency
                quantization_config={
                    "scalar": {
                        "type": "int8",
                        "quantile": 0.99,
                        "always_ram": True
                    }
                }
            )
            
            # Create payload field indexes for fast filtering
            self._create_payload_indexes()
            
            logger.info(f"Successfully created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def _create_payload_indexes(self) -> None:
        """Create indexes on payload fields for fast filtering."""
        try:
            # Index document name for filtering by document
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_name",
                field_schema="keyword"
            )
            
            # Index page number for filtering by page
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="page_number",
                field_schema="integer"
            )
            
            # Index section titles for filtering by section
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="section_title",
                field_schema="text"
            )
            
            # Index chunk content for lexical search
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="content",
                field_schema="text"
            )
            
            logger.info("Created payload indexes for fast filtering")
            
        except Exception as e:
            logger.error(f"Error creating payload indexes: {str(e)}")
            # Non-critical error, continue execution
    
    async def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]], batch_size: int = 100) -> bool:
        """
        Add document chunks with their embeddings to the vector store in batches.
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: List of embedding vectors
            batch_size: Number of chunks to process in each batch
            
        Returns:
            True if successful, False otherwise
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            total_chunks = len(chunks)
            successful_batches = 0
            failed_batches = 0
            
            logger.info(f"Processing {total_chunks} chunks in batches of {batch_size}")
            
            # Process chunks in batches
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                try:
                    points = []
                    
                    for chunk, embedding in zip(batch_chunks, batch_embeddings):
                        # Create point with vector and payload
                        point = PointStruct(
                            id=str(uuid.uuid4()),  # Generate unique ID
                            vector={"dense": embedding},
                            payload={
                                # Core content
                                "content": chunk.content,
                                "enhanced_content": chunk.get_enhanced_content(),
                                
                                # Document metadata
                                "document_name": chunk.document_name,
                                "page_number": chunk.page_number,
                                "chunk_id": chunk.chunk_id,
                                "document_type": chunk.document_type,
                                
                                # Section metadata
                                "section_title": chunk.section_title,
                                "subsection_title": chunk.subsection_title,
                                "page_title": chunk.page_title,
                                
                                # Chunk metadata
                                "chunk_index": chunk.chunk_index,
                                "total_chunks": chunk.total_chunks,
                                
                                # Full chunk data for retrieval
                                "chunk_data": chunk.to_dict()
                            }
                        )
                        points.append(point)
                    
                    # Upload batch
                    result = await self.async_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    
                    if result.status == UpdateStatus.COMPLETED:
                        successful_batches += 1
                        logger.info(f"Successfully added batch {i//batch_size + 1}: {len(batch_chunks)} chunks")
                    else:
                        failed_batches += 1
                        logger.error(f"Failed to add batch {i//batch_size + 1}. Status: {result.status}")
                        
                except Exception as batch_error:
                    failed_batches += 1
                    logger.error(f"Error processing batch {i//batch_size + 1}: {str(batch_error)}")
            
            # Check overall success
            if failed_batches == 0:
                logger.info(f"Successfully added all {total_chunks} chunks to vector store in {successful_batches} batches")
                return True
            elif successful_batches > 0:
                logger.warning(f"Partially successful: {successful_batches} batches succeeded, {failed_batches} batches failed")
                return True  # Partial success is still considered successful
            else:
                logger.error(f"Failed to add any chunks. All {failed_batches} batches failed")
                return False
                
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {str(e)}")
            return False
    
    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        document_filter: Optional[Union[str, List[str]]] = None,
        page_filter: Optional[int] = None,
        section_filter: Optional[str] = None,
        hybrid_alpha: float = 0.7,  # Weight for vector vs lexical search
        return_embeddings: bool = False,  # Whether to return embeddings for reranking
        hierarchical_level: Optional[int] = None  # Filter by hierarchical level
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense vector search and lexical search.
        
        Args:
            query_embedding: Dense vector representation of query
            query_text: Original query text for lexical search
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            document_filter: Filter by specific document name
            page_filter: Filter by specific page number
            section_filter: Filter by section title
            hybrid_alpha: Weight for combining vector and lexical scores (0.0-1.0)
            
        Returns:
            List of search results with metadata
        """
        try:
            # Build filter conditions
            filter_conditions = []

            if document_filter:
                if isinstance(document_filter, list):
                    # Use MatchAny for multiple document filters
                    filter_conditions.append(
                        FieldCondition(
                            key="document_name",
                            match=MatchAny(any=document_filter)
                        )
                    )
                else:
                    # Single document filter (backward compatibility)
                    filter_conditions.append(
                        FieldCondition(
                            key="document_name",
                            match={"value": document_filter}
                        )
                    )
            
            if page_filter is not None:
                filter_conditions.append(
                    FieldCondition(
                        key="page_number",
                        match={"value": page_filter}
                    )
                )
            
            if section_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="section_title",
                        match={"text": section_filter}
                    )
                )
            
            # Create filter
            search_filter = None
            if filter_conditions:
                search_filter = Filter(must=filter_conditions)
            
            # Perform dense vector search
            vector_results = await self.async_client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_embedding),
                query_filter=search_filter,
                limit=limit * 2,  # Get more results for re-ranking
                score_threshold=score_threshold * 0.8,  # Lower threshold for initial search
                with_payload=True,
                with_vectors=return_embeddings  # Return vectors if needed for reranking
            )
            
            # Perform lexical search if query text provided
            lexical_results = []
            if query_text.strip():
                lexical_results = await self._lexical_search(
                    query_text, 
                    search_filter, 
                    limit * 2
                )
            
            # Combine and re-rank results
            combined_results = self._combine_hybrid_results(
                vector_results, 
                lexical_results, 
                hybrid_alpha,
                limit
            )
            
            # Format results
            formatted_results = []
            for result in combined_results:
                formatted_result = {
                    "content": result.payload["content"],
                    "enhanced_content": result.payload["enhanced_content"],
                    "score": result.score,
                    "document_name": result.payload["document_name"],
                    "page_number": result.payload["page_number"],
                    "section_title": result.payload.get("section_title"),
                    "subsection_title": result.payload.get("subsection_title"),
                    "page_title": result.payload.get("page_title"),
                    "chunk_id": result.payload["chunk_id"],
                    "metadata": {
                        "chunk_index": result.payload["chunk_index"],
                        "total_chunks": result.payload["total_chunks"],
                        "document_type": result.payload["document_type"]
                    }
                }
                # Include embedding if requested and available
                if return_embeddings and hasattr(result, 'vector') and result.vector:
                    if isinstance(result.vector, dict) and "dense" in result.vector:
                        formatted_result["content_embedding"] = result.vector["dense"]
                    elif isinstance(result.vector, list):
                        formatted_result["content_embedding"] = result.vector

                formatted_results.append(formatted_result)
            
            logger.info(
                f"Hybrid search returned {len(formatted_results)} results "
                f"(vector: {len(vector_results)}, lexical: {len(lexical_results)})"
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    async def _lexical_search(
        self, 
        query_text: str, 
        search_filter: Optional[Filter], 
        limit: int
    ) -> List[Any]:
        """
        Perform lexical search using Qdrant's text matching capabilities.
        
        Args:
            query_text: Query text for lexical matching
            search_filter: Additional filters to apply
            limit: Maximum number of results
            
        Returns:
            List of lexical search results
        """
        try:
            # Create text match conditions for different fields
            query_words = query_text.lower().split()
            
            text_conditions = []
            
            # First, try exact phrase matching
            if len(query_text.strip()) > 3:
                text_conditions.append(
                    FieldCondition(
                        key="content",
                        match={"text": query_text.lower()}
                    )
                )
                text_conditions.append(
                    FieldCondition(
                        key="enhanced_content",
                        match={"text": query_text.lower()}
                    )
                )
            
            # Then, individual word matching
            for word in query_words:
                if len(word) > 2:  # Skip very short words
                    # Search in content
                    text_conditions.append(
                        FieldCondition(
                            key="content",
                            match={"text": word}
                        )
                    )
                    # Search in enhanced content
                    text_conditions.append(
                        FieldCondition(
                            key="enhanced_content",
                            match={"text": word}
                        )
                    )
            
            if not text_conditions:
                return []
            
            # Combine with existing filters
            all_conditions = []
            if search_filter and search_filter.must:
                all_conditions.extend(search_filter.must)
            
            # Use 'should' for text matching (OR logic)
            lexical_filter = Filter(
                must=all_conditions,
                should=text_conditions
            )
            
            # Scroll through results since we're doing text matching
            results = await self.async_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=lexical_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Add lexical scores based on text match quality
            scored_results = []
            for point in results[0]:  # results is tuple (points, next_page_offset)
                lexical_score = self._calculate_lexical_score(
                    query_text, 
                    point.payload.get("content", "")
                )
                
                # Create result object with lexical score
                point.score = lexical_score
                scored_results.append(point)
            
            # Sort by lexical score
            scored_results.sort(key=lambda x: x.score, reverse=True)
            
            return scored_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in lexical search: {str(e)}")
            return []
    
    def _calculate_lexical_score(self, query: str, content: str) -> float:
        """
        Calculate lexical similarity score between query and content.
        
        Args:
            query: Search query
            content: Document content
            
        Returns:
            Lexical similarity score (0.0 to 1.0)
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        jaccard_score = len(intersection) / len(union) if union else 0.0
        
        # Boost score for exact phrase matches
        phrase_boost = 1.0
        if query.lower() in content.lower():
            phrase_boost = 1.5
        
        return min(jaccard_score * phrase_boost, 1.0)
    
    def _combine_hybrid_results(
        self, 
        vector_results: List[Any], 
        lexical_results: List[Any],
        alpha: float,
        limit: int
    ) -> List[Any]:
        """
        Combine and re-rank vector and lexical search results.
        
        Args:
            vector_results: Results from vector search
            lexical_results: Results from lexical search
            alpha: Weight for vector search (1-alpha for lexical)
            limit: Maximum number of final results
            
        Returns:
            Combined and re-ranked results
        """
        # Create mapping by chunk_id to avoid duplicates
        results_map = {}
        
        # Add vector results
        for result in vector_results:
            chunk_id = result.payload["chunk_id"]
            vector_score = result.score
            
            results_map[chunk_id] = {
                "result": result,
                "vector_score": vector_score,
                "lexical_score": 0.0,
                "combined_score": alpha * vector_score
            }
        
        # Add/update with lexical results
        for result in lexical_results:
            chunk_id = result.payload["chunk_id"]
            lexical_score = result.score
            
            if chunk_id in results_map:
                # Update existing result
                results_map[chunk_id]["lexical_score"] = lexical_score
                results_map[chunk_id]["combined_score"] = (
                    alpha * results_map[chunk_id]["vector_score"] + 
                    (1 - alpha) * lexical_score
                )
            else:
                # Add new result (lexical only)
                results_map[chunk_id] = {
                    "result": result,
                    "vector_score": 0.0,
                    "lexical_score": lexical_score,
                    "combined_score": (1 - alpha) * lexical_score
                }
        
        # Sort by combined score and prepare final results
        sorted_results = sorted(
            results_map.values(), 
            key=lambda x: x["combined_score"], 
            reverse=True
        )
        
        final_results = []
        for item in sorted_results[:limit]:
            result = item["result"]
            result.score = item["combined_score"]  # Update with combined score
            final_results.append(result)
        
        return final_results
    
    async def delete_document(self, document_name: str) -> bool:
        """
        Delete all chunks belonging to a specific document.
        
        Args:
            document_name: Name of document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create filter for document
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_name",
                        match={"value": document_name}
                    )
                ]
            )
            
            # Delete points matching filter
            result = await self.async_client.delete(
                collection_name=self.collection_name,
                points_selector=delete_filter
            )
            
            if result.status == UpdateStatus.COMPLETED:
                logger.info(f"Successfully deleted document: {document_name}")
                return True
            else:
                logger.error(f"Failed to delete document. Status: {result.status}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {document_name}: {str(e)}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = await self.async_client.get_collection(self.collection_name)
            return {
                "name": info.config.name,
                "vector_size": info.config.params.vectors["dense"].size,
                "distance": info.config.params.vectors["dense"].distance,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
    async def list_documents(self) -> List[str]:
        """Get list of all document names in the collection."""
        try:
            # Scroll through all points to get unique document names
            documents = set()
            offset = None
            
            while True:
                results = await self.async_client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["document_name"],
                    with_vectors=False
                )
                
                points, next_offset = results
                
                for point in points:
                    if "document_name" in point.payload:
                        documents.add(point.payload["document_name"])
                
                if next_offset is None:
                    break
                    
                offset = next_offset
            
            return sorted(list(documents))
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return [] 