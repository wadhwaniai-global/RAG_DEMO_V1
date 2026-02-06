"""
Embedding system with OpenAI integration and metadata augmentation.
Provides enhanced embeddings for better retrieval performance.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import hashlib
from dataclasses import dataclass

from langfuse.openai import openai
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from core.document_processor import DocumentChunk
from core.langfuse_utils import get_langfuse_client

# Setup logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai.api_key = settings.openai_api_key
get_langfuse_client()


@dataclass
class EmbeddingResult:
    """Result of embedding operation with metadata."""
    embedding: List[float]
    text: str
    token_count: int
    processing_time: float
    chunk_id: Optional[str] = None


class OpenAIEmbeddings:
    """
    OpenAI embeddings integration with caching and batch processing.
    Supports metadata-augmented embeddings for improved retrieval.
    """
    
    def __init__(self):
        """Initialize OpenAI embeddings service."""
        self.model = settings.openai_embedding_model
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.cache = {}  # Simple in-memory cache
        self.max_batch_size = 100
        self.max_tokens_per_request = 8000
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text with retry logic.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Check cache first
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.cache:
                logger.debug(f"Cache hit for text hash: {text_hash[:8]}")
                return self.cache[text_hash]
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float",
                metadata={
                    "langfuse_tags": ["embedding", "single"],
                    "component": "OpenAIEmbeddings.get_embedding"
                }
            )
            
            embedding = response.data[0].embedding
            
            # Cache the result
            self.cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Get embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of EmbeddingResult objects
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
            
            # Add small delay to avoid rate limiting
            if i + self.max_batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return results
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _process_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Process a single batch of texts."""
        start_time = time.time()
        
        try:
            # Filter out cached results
            uncached_texts = []
            uncached_indices = []
            cached_results = {}
            
            for idx, text in enumerate(texts):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in self.cache:
                    cached_results[idx] = self.cache[text_hash]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(idx)
            
            # Get embeddings for uncached texts
            if uncached_texts:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=uncached_texts,
                    encoding_format="float",
                    metadata={
                        "langfuse_tags": ["embedding", "batch"],
                        "component": "OpenAIEmbeddings.get_embeddings_batch",
                        "batch_size": len(uncached_texts)
                    }
                )
                
                # Cache new results
                for text, embedding_data in zip(uncached_texts, response.data):
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    self.cache[text_hash] = embedding_data.embedding
            
            # Combine results in original order
            results = []
            processing_time = time.time() - start_time
            
            for idx, text in enumerate(texts):
                if idx in cached_results:
                    embedding = cached_results[idx]
                else:
                    # Find corresponding embedding from response
                    uncached_pos = uncached_indices.index(idx)
                    embedding = response.data[uncached_pos].embedding
                
                result = EmbeddingResult(
                    embedding=embedding,
                    text=text,
                    token_count=response.usage.total_tokens if uncached_texts else 0,
                    processing_time=processing_time / len(texts)
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise
    
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[EmbeddingResult]:
        """
        Generate embeddings for document chunks with metadata augmentation.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of EmbeddingResult objects
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Prepare enhanced texts for embedding
        enhanced_texts = []
        for chunk in chunks:
            enhanced_text = chunk.get_enhanced_content()
            enhanced_texts.append(enhanced_text)
        
        # Get embeddings in batches
        results = await self.get_embeddings_batch(enhanced_texts)
        
        # Add chunk IDs to results
        for result, chunk in zip(results, chunks):
            result.chunk_id = chunk.chunk_id
        
        logger.info(f"Successfully generated {len(results)} embeddings")
        return results


class QueryExpansionEmbeddings:
    """
    Advanced embedding service with query expansion capabilities.
    Implements multiple retrieval augmentation strategies.
    """
    
    def __init__(self):
        """Initialize query expansion service."""
        self.base_embeddings = OpenAIEmbeddings()
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        
    async def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        Expand query using LLM to generate related questions and phrasings.
        
        Args:
            query: Original query
            num_expansions: Number of expanded queries to generate
            
        Returns:
            List of expanded queries including the original
        """
        try:
            expansion_prompt = f"""
            Given the following query, generate {num_expansions} alternative ways to ask the same question.
            Focus on different phrasings, synonyms, and related concepts that might appear in technical documentation.
            
            Original query: {query}
            
            Generate {num_expansions} alternative queries (one per line):
            """
            
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert at generating query variations for document search."},
                    {"role": "user", "content": expansion_prompt}
                ],
                max_tokens=200,
                temperature=0.7,
                metadata={
                    "langfuse_tags": ["query-expansion"],
                    "component": "QueryExpansionEmbeddings.expand_query",
                    "num_expansions": num_expansions
                }
            )
            
            expanded_queries = [query]  # Include original
            
            content = response.choices[0].message.content.strip()
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and len(line) > 10:
                    # Clean up numbered lists, bullets, etc.
                    cleaned_line = line.lstrip('0123456789.- ')
                    if cleaned_line:
                        expanded_queries.append(cleaned_line)
            
            # Limit to requested number + original
            return expanded_queries[:num_expansions + 1]
            
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            return [query]  # Fallback to original query
    
    async def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query (HyDE).
        This technique improves retrieval by embedding potential answers.
        
        Args:
            query: User query
            
        Returns:
            Hypothetical document text
        """
        try:
            hyde_prompt = f"""
            Generate a detailed, technical answer to the following question as it might appear in documentation:
            
            Question: {query}
            
            Write a comprehensive answer that includes:
            - Technical details and explanations
            - Relevant terminology and concepts
            - Implementation details where applicable
            - Examples or use cases
            
            Answer:
            """
            
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a technical documentation expert. Generate detailed, accurate answers as they would appear in professional documentation."},
                    {"role": "user", "content": hyde_prompt}
                ],
                max_tokens=500,
                temperature=0.3,
                metadata={
                    "langfuse_tags": ["hyde"],
                    "component": "QueryExpansionEmbeddings.generate_hypothetical_document"
                }
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {str(e)}")
            return query  # Fallback to original query
    
    async def multi_query_embeddings(self, query: str) -> Tuple[List[str], List[List[float]]]:
        """
        Generate multiple query representations for improved retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Tuple of (expanded_queries, embeddings)
        """
        # Expand query into multiple variations
        expanded_queries = await self.expand_query(query, num_expansions=2)
        
        # Generate hypothetical document
        hypothetical_doc = await self.generate_hypothetical_document(query)
        
        # Combine all query representations
        all_queries = expanded_queries + [hypothetical_doc]
        
        # Get embeddings for all queries
        embedding_results = await self.base_embeddings.get_embeddings_batch(all_queries)
        embeddings = [result.embedding for result in embedding_results]
        
        logger.info(f"Generated {len(embeddings)} query embeddings from: {len(expanded_queries)} expansions + HyDE")
        
        return all_queries, embeddings
    
    async def get_enhanced_query_embedding(self, query: str) -> Dict[str, Any]:
        """
        Get enhanced query embedding with multiple representation strategies.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with multiple embeddings and metadata
        """
        # Get base query embedding
        base_embedding = await self.base_embeddings.get_embedding(query)
        
        # Get multi-query embeddings
        expanded_queries, expanded_embeddings = await self.multi_query_embeddings(query)
        
        # Calculate average embedding (ensemble approach)
        ensemble_embedding = np.mean(expanded_embeddings, axis=0).tolist()
        
        return {
            "base_embedding": base_embedding,
            "expanded_queries": expanded_queries,
            "expanded_embeddings": expanded_embeddings,
            "ensemble_embedding": ensemble_embedding,
            "original_query": query
        }


class SemanticSimilarityRanker:
    """
    Advanced semantic similarity ranker for re-ranking search results.
    """
    
    def __init__(self):
        """Initialize semantic ranker."""
        self.embeddings = OpenAIEmbeddings()
    
    def calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def rerank_results(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        query_embedding: List[float],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank search results using semantic similarity.
        OPTIMIZED: Uses pre-computed embeddings from vector store when available.

        Args:
            query: Original query
            search_results: List of search results with embeddings
            query_embedding: Pre-computed query embedding
            top_k: Number of top results to return

        Returns:
            Re-ranked search results
        """
        if not search_results:
            return []

        # Use provided query embedding to avoid redundant API calls

        # Calculate semantic similarities for re-ranking
        embeddings_generated = 0
        embeddings_reused = 0

        for result in search_results:
            # OPTIMIZATION: Check if embedding is already in the result (from vector store)
            if "content_embedding" not in result:
                # Only generate embedding if not present
                content_embedding = await self.embeddings.get_embedding(result["content"])
                result["content_embedding"] = content_embedding
                embeddings_generated += 1
            else:
                # Reuse existing embedding from vector store
                embeddings_reused += 1

            # Calculate semantic similarity
            semantic_score = self.calculate_cosine_similarity(
                query_embedding,
                result["content_embedding"]
            )

            # Combine with original score (weighted average)
            original_score = result.get("score", 0.0)
            combined_score = 0.7 * semantic_score + 0.3 * original_score

            result["semantic_score"] = semantic_score
            result["combined_score"] = combined_score

        # Log optimization stats
        if embeddings_generated > 0 or embeddings_reused > 0:
            logger.info(
                f"Reranking optimization: {embeddings_reused} embeddings reused from vector store, "
                f"{embeddings_generated} new embeddings generated"
            )

        # Sort by combined score
        search_results.sort(key=lambda x: x["combined_score"], reverse=True)

        # Return top_k results
        if top_k:
            return search_results[:top_k]

        return search_results


# Utility functions
async def embed_document_batch(chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], List[List[float]]]:
    """
    Utility function to embed a batch of document chunks.
    
    Args:
        chunks: List of DocumentChunk objects
        
    Returns:
        Tuple of (chunks, embeddings)
    """
    embeddings_service = OpenAIEmbeddings()
    embedding_results = await embeddings_service.embed_chunks(chunks)
    
    embeddings = [result.embedding for result in embedding_results]
    
    return chunks, embeddings


def calculate_embedding_stats(embeddings: List[List[float]]) -> Dict[str, Any]:
    """
    Calculate statistics for a set of embeddings.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Dictionary with embedding statistics
    """
    if not embeddings:
        return {}
    
    embeddings_np = np.array(embeddings)
    
    return {
        "count": len(embeddings),
        "dimension": embeddings_np.shape[1],
        "mean_norm": np.mean(np.linalg.norm(embeddings_np, axis=1)),
        "std_norm": np.std(np.linalg.norm(embeddings_np, axis=1)),
        "mean_values": np.mean(embeddings_np, axis=0).tolist()[:10],  # First 10 dimensions
        "std_values": np.std(embeddings_np, axis=0).tolist()[:10]     # First 10 dimensions
    } 