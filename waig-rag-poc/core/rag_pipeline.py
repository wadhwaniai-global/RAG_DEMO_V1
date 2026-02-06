"""
Main RAG pipeline integrating document processing, vector search, and LLM generation.
Implements advanced retrieval strategies and hallucination prevention.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import hashlib

from langfuse.openai import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings, RAG_RESPONSE_SCHEMA, UNRELATED_QUERY_RESPONSE
from core.document_processor import DocumentProcessor, DocumentChunk
from core.vector_store import QdrantVectorStore
from core.embeddings import OpenAIEmbeddings, QueryExpansionEmbeddings, SemanticSimilarityRanker
from core.enhanced_lexical_search import EnhancedLexicalSearch
from core.langfuse_utils import get_langfuse_client

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result of RAG pipeline execution."""
    answer: str
    confidence_score: float
    sources: List[Dict[str, Any]]
    retrieval_metadata: Dict[str, Any]
    status: str
    processing_time: float

class LLMResponseParseError(Exception):
    """Custom exception for failures when parsing the LLM's response."""
    pass

class AdvancedRAGPipeline:
    """
    Advanced RAG pipeline with multiple retrieval strategies and hallucination prevention.
    """
    
    def __init__(self, max_context_length: int = 16000, max_sources: int = 10, max_tokens: int = 3000):
        """Initialize the RAG pipeline components."""
        self.document_processor = DocumentProcessor()
        self.vector_store = QdrantVectorStore()
        self.embeddings = OpenAIEmbeddings()
        self.query_expander = QueryExpansionEmbeddings()
        self.reranker = SemanticSimilarityRanker()
        self.enhanced_lexical = EnhancedLexicalSearch()
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.langfuse_client = get_langfuse_client()
        
        # Pipeline configuration
        self.max_context_length = max_context_length  # Configurable context length
        self.min_confidence_threshold = 0.3
        self.max_sources = max_sources  # Configurable number of sources
        self.max_tokens = max_tokens  # Configurable response length
        
        # Medical domain keywords for enhanced retrieval
        self.medical_keywords = [
            'dosing', 'schedule', 'treatment', 'medication', 'regimen',
            'therapy', 'dose', 'dosage', 'administration', 'prescription',
            'drug', 'pharmaceutical', 'clinical', 'patient', 'diagnosis'
        ]

    def _build_langfuse_metadata(
        self,
        operation: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Construct consistent Langfuse metadata payloads."""
        metadata: Dict[str, Any] = {
            "langfuse_tags": ["rag", operation],
            "component": f"AdvancedRAGPipeline.{operation}"
        }

        if query:
            metadata["query_hash"] = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]

        if extra:
            metadata.update(extra)

        return metadata
        
    async def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process and index documents into the vector store.
        
        Args:
            file_paths: List of document file paths to process
            
        Returns:
            Processing result metadata
        """
        logger.info(f"Processing {len(file_paths)} documents")
        start_time = time.time()
        
        total_chunks = 0
        successful_files = 0
        failed_files = []
        
        for file_path in file_paths:
            try:
                # Process document into chunks
                chunks = await self.document_processor.process_pdf(file_path)
                
                if chunks:
                    # Generate embeddings for chunks
                    embedding_results = await self.embeddings.embed_chunks(chunks)
                    embeddings = [result.embedding for result in embedding_results]
                    
                    # Store in vector database
                    success = await self.vector_store.add_chunks(chunks, embeddings)
                    
                    if success:
                        total_chunks += len(chunks)
                        successful_files += 1
                        logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
                    else:
                        failed_files.append(f"{file_path}: Failed to store in vector database")
                else:
                    failed_files.append(f"{file_path}: No chunks extracted")
                    
            except Exception as e:
                error_msg = f"{file_path}: {str(e)}"
                failed_files.append(error_msg)
                logger.error(f"Failed to process {file_path}: {str(e)}")
        
        processing_time = time.time() - start_time
        
        result = {
            "total_files": len(file_paths),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_chunks": total_chunks,
            "processing_time": processing_time,
            "average_chunks_per_file": total_chunks / successful_files if successful_files > 0 else 0
        }
        
        logger.info(f"Document processing completed: {result}")
        return result
    
    async def process_json_pages(self, json_pages: List[Dict[str, Any]], document_name: str = "medical_manual") -> Dict[str, Any]:
        """
        Process JSON pages using hierarchical chunking for medical documents.
        
        Args:
            json_pages: List of JSON page objects
            document_name: Name of the document
            
        Returns:
            Processing result metadata
        """
        logger.info(f"Processing {len(json_pages)} JSON pages with hierarchical chunking")
        start_time = time.time()
        
        try:
            # Process JSON pages with hierarchical chunking
            chunks = await self.document_processor.process_json_pages(json_pages, document_name)
            
            if chunks:
                # Generate embeddings for chunks
                embedding_results = await self.embeddings.embed_chunks(chunks)
                embeddings = [result.embedding for result in embedding_results]
                
                # Store in vector database with hierarchical metadata (with batch processing)
                success = await self.vector_store.add_chunks(chunks, embeddings, batch_size=50)
                
                if success:
                    processing_time = time.time() - start_time
                    
                    result = {
                        "total_pages": len(json_pages),
                        "total_chunks": len(chunks),
                        "hierarchical_levels": len(set(chunk.hierarchical_level for chunk in chunks)),
                        "chunk_types": list(set(chunk.chunk_type for chunk in chunks)),
                        "processing_time": processing_time,
                        "success": True
                    }
                    
                    logger.info(f"JSON pages processing completed: {result}")
                    return result
                else:
                    return {
                        "total_pages": len(json_pages),
                        "total_chunks": 0,
                        "processing_time": time.time() - start_time,
                        "success": False,
                        "error": "Failed to store in vector database"
                    }
            else:
                return {
                    "total_pages": len(json_pages),
                    "total_chunks": 0,
                    "processing_time": time.time() - start_time,
                    "success": False,
                    "error": "No chunks extracted"
                }
                
        except Exception as e:
            logger.error(f"Error processing JSON pages: {str(e)}")
            return {
                "total_pages": len(json_pages),
                "total_chunks": 0,
                "processing_time": time.time() - start_time,
                "success": False,
                "error": str(e)
            }
    
    async def query(
        self,
        query: str,
        use_query_expansion: bool = True,
        use_reranking: bool = True,
        document_filter: Optional[Union[str, List[str]]] = None,
        page_filter: Optional[int] = None
    ) -> RAGResult:
        """
        Execute RAG query with advanced retrieval and generation.
        
        Args:
            query: User query
            use_query_expansion: Whether to use query expansion techniques
            use_reranking: Whether to use semantic re-ranking
            document_filter: Filter by specific document
            page_filter: Filter by specific page
            
        Returns:
            RAGResult with answer and metadata
        """
        start_time = time.time()
        logger.info(f"Processing query: {query}")
        
        # Generate query embedding once for all operations
        query_embedding = await self.embeddings.get_embedding(query)

        # Always enforce reranking for consistent retrieval quality
        use_reranking = False
        
        try:
            # Step 1: Query relevance check
            if not await self._is_query_relevant(query):
                return RAGResult(
                    answer=UNRELATED_QUERY_RESPONSE["answer"],
                    confidence_score=0.0,
                    sources=[],
                    retrieval_metadata=UNRELATED_QUERY_RESPONSE["retrieval_metadata"],
                    status="no_relevant_documents",
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Enhanced retrieval
            search_results = await self._enhanced_retrieval(
                query,
                query_embedding,
                use_query_expansion,
                document_filter,
                page_filter,
                use_reranking  # Pass reranking flag to retrieval
            )

            initial_results = [result.copy() for result in search_results]
            
            if not search_results:
                return RAGResult(
                    answer="I couldn't find relevant information in the available documents to answer your question.",
                    confidence_score=0.0,
                    sources=[],
                    retrieval_metadata={
                        "total_documents_searched": 0,
                        "query_expansion_used": use_query_expansion,
                        "hybrid_search_used": True
                    },
                    status="no_relevant_documents",
                    processing_time=time.time() - start_time
                )
            
            # Step 3: Re-ranking (if enabled)
            if use_reranking:
                search_results = await self.reranker.rerank_results(
                    query, 
                    search_results, 
                    query_embedding,  # Pass the pre-computed embedding
                    top_k=self.max_sources * 2
                )
            
            # Step 4: Context preparation
            context = self._prepare_context(search_results[:self.max_sources])
            
            # Step 5: Generate answer
            text, confidence, json_sources = await self._generate_answer(
                query,
                context,
                search_results,
                initial_results,
                use_reranking
            )
            
            # Step 6: Prepare sources - use JSON sources if available, otherwise fallback to prepared sources
            if json_sources is not None:
                sources = json_sources
                logger.info(f"Using {len(sources)} sources from JSON response")
            else:
                sources = self._prepare_sources(search_results[:self.max_sources])
                logger.info(f"Using {len(sources)} fallback sources from search results")
            
            processing_time = time.time() - start_time
            
            result = RAGResult(
                answer=text,
                confidence_score=confidence,
                sources=sources,
                retrieval_metadata={
                    "total_documents_searched": len(search_results),
                    "query_expansion_used": use_query_expansion,
                    "hybrid_search_used": True,
                    "reranking_used": use_reranking
                },
                status="success" if confidence >= self.min_confidence_threshold else "insufficient_information",
                processing_time=processing_time
            )
            
            logger.info(f"Query completed in {processing_time:.2f}s with confidence {confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return RAGResult(
                answer="An error occurred while processing your query. Please try again.",
                confidence_score=0.0,
                sources=[],
                retrieval_metadata={},
                status="error",
                processing_time=time.time() - start_time
            )
    
    async def _is_query_relevant(self, query: str) -> bool:
        """
        Check if the query is relevant to the document collection.
        
        Args:
            query: User query
            
        Returns:
            True if query appears relevant, False otherwise
        """
        try:
            #relevance_prompt = f"""
            #Analyze the following query and determine if it's asking for information that would typically be found in technical documentation, manuals, or knowledge base articles.
            
            #Query: "{query}"
            
            #Return only "YES" if the query is asking for factual information, technical details, procedures, or explanations that would be found in documentation.
            #Return only "NO" if the query is:
            #- A greeting or casual conversation
            #- Asking for personal opinions
            #- Requesting actions outside of information retrieval
            #- About topics clearly unrelated to documentation (e.g., weather, personal life, current events)
            #- Inappropriate or harmful content
            
            #Response (YES/NO):
            #"""
            
            #response = await self.client.chat.completions.create(
            #    model=settings.openai_model,
            #    messages=[
            #        {"role": "system", "content": "You are a query relevance classifier for a documentation system."},
            #        {"role": "user", "content": relevance_prompt}
            #    ],
            #    max_tokens=10,
            #    temperature=0.0
            #)
            
            #result = response.choices[0].message.content.strip().upper()
            #return result == "YES"
            return True
            
        except Exception as e:
            logger.error(f"Error checking query relevance: {str(e)}")
            return True  # Default to relevant if error occurs
    
    async def _enhanced_retrieval(
        self,
        query: str,
        query_embedding: List[float],
        use_expansion: bool,
        document_filter: Optional[Union[str, List[str]]] = None,
        page_filter: Optional[int] = None,
        use_reranking: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform enhanced retrieval with multiple strategies including hierarchical and lexical search.
        
        Args:
            query: User query
            use_expansion: Whether to use query expansion
            document_filter: Document name filter
            page_filter: Page number filter
            
        Returns:
            List of search results
        """
        all_results = []
        
        # Strategy 1: Enhanced lexical search
        enhanced_query = self.enhanced_lexical.enhance_query(query)
        lexical_queries = self.enhanced_lexical.create_lexical_queries(enhanced_query)
        
        # Search with lexical queries (use original query embedding for all lexical variations)
        for lexical_query in lexical_queries:
            lexical_results = await self.vector_store.hybrid_search(
                query_embedding=query_embedding,  # Use original query embedding
                query_text=lexical_query,
                limit=settings.max_retrievals,
                document_filter=document_filter,
                page_filter=page_filter,
                hybrid_alpha=0.3,  # More lexical
                return_embeddings=use_reranking  # Get embeddings if we'll rerank
            )
            all_results.extend(lexical_results)
        
        # Strategy 2: Hierarchical search
        hierarchical_results = await self._search_hierarchical(query, query_embedding, document_filter, page_filter, use_reranking)
        all_results.extend(hierarchical_results)

        # Strategy 3: Cross-reference search
        cross_ref_results = await self._search_cross_references(query, query_embedding, document_filter, use_reranking)
        all_results.extend(cross_ref_results)
        
        # Strategy 4: Standard semantic search
        # Use lower score threshold for medical queries
        is_medical_query = any(keyword in query.lower() for keyword in self.medical_keywords)
        score_threshold = 0.5 if is_medical_query else 0.7
        
        if is_medical_query:
            logger.info(f"Medical query detected: '{query}' - using lower score threshold: {score_threshold}")
        
        results = await self.vector_store.hybrid_search(
            query_embedding=query_embedding,  # Use the same embedding we already have
            query_text=query,
            limit=settings.max_retrievals,
            document_filter=document_filter,
            page_filter=page_filter,
            hybrid_alpha=0.7,
            score_threshold=score_threshold,
            return_embeddings=use_reranking  # Get embeddings if we'll rerank
        )
        all_results.extend(results)
        
        # Strategy 5: Query expansion (only if enabled)
        if use_expansion:
            # Enhanced query with expansion and HyDE
            query_data = await self.query_expander.get_enhanced_query_embedding(query)
            
            # Search with ensemble embedding (best overall performance)
            expansion_results = await self.vector_store.hybrid_search(
                query_embedding=query_data["ensemble_embedding"],
                query_text=query,
                limit=settings.max_retrievals,
                document_filter=document_filter,
                page_filter=page_filter,
                hybrid_alpha=0.7,
                return_embeddings=use_reranking  # Get embeddings if we'll rerank
            )
            all_results.extend(expansion_results)
        
        # Remove duplicates based on chunk_id
        seen_chunks = set()
        unique_results = []
        
        for result in all_results:
            chunk_id = result.get("chunk_id")
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        # Sort by score and return top results
        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return unique_results[:settings.max_retrievals * 2]  # Return more results for better coverage
    
    async def _search_hierarchical(self, query: str, query_embedding: List[float], document_filter: Optional[Union[str, List[str]]] = None, page_filter: Optional[int] = None, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """Search using hierarchical relationships."""
        try:
            # Use provided query embedding to avoid redundant API calls
            
            # Search for parent chunks first
            parent_results = await self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                limit=settings.max_retrievals,
                document_filter=document_filter,
                page_filter=page_filter,
                hybrid_alpha=0.7,
                hierarchical_level=1,  # Document/Section level
                return_embeddings=use_reranking  # Get embeddings if we'll rerank
            )

            # Search for child chunks
            child_results = await self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                limit=settings.max_retrievals,
                document_filter=document_filter,
                page_filter=page_filter,
                hybrid_alpha=0.7,
                hierarchical_level=2,  # Subsection/Procedure level
                return_embeddings=use_reranking  # Get embeddings if we'll rerank
            )
            
            # Combine and deduplicate
            all_results = parent_results + child_results
            seen_chunks = set()
            unique_results = []
            
            for result in all_results:
                chunk_id = result.get("chunk_id")
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    unique_results.append(result)
            
            return unique_results[:settings.max_retrievals]
            
        except Exception as e:
            logger.error(f"Error in hierarchical search: {str(e)}")
            return []
    
    async def _search_cross_references(self, query: str, query_embedding: List[float], document_filter: Optional[Union[str, List[str]]] = None, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """Search for cross-referenced content."""
        try:
            # Extract cross-references from query
            cross_refs = self.enhanced_lexical.enhance_query(query)['cross_references']
            
            if not cross_refs:
                return []
            
            all_results = []
            
            for ref in cross_refs:
                # Search for content mentioning this reference
                ref_query = f"section {ref} OR page {ref}"
                # Use the same query embedding to avoid redundant API calls
                
                ref_results = await self.vector_store.hybrid_search(
                    query_embedding=query_embedding,  # Use the same embedding
                    query_text=ref_query,
                    limit=5,  # Fewer results per reference
                    document_filter=document_filter,
                    hybrid_alpha=0.3,  # More lexical for references
                    return_embeddings=use_reranking  # Get embeddings if we'll rerank
                )
                all_results.extend(ref_results)
            
            return all_results[:settings.max_retrievals // 2]
            
        except Exception as e:
            logger.error(f"Error in cross-reference search: {str(e)}")
            return []
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Prepare context from search results for LLM generation.
        
        Args:
            search_results: Search results from vector store
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for idx, result in enumerate(search_results, 1):
            # Format each source with metadata
            source_context = f"""
Source {idx}:
Document: {result['document_name']}
Page: {result['page_number']}
"""
            
            if result.get('section_title'):
                source_context += f"Section: {result['section_title']}\n"
            
            if result.get('page_title'):
                source_context += f"Page Title: {result['page_title']}\n"
            
            source_context += f"Content: {result['content']}\n"
            source_context += f"Relevance Score: {result['score']:.3f}\n"
            
            context_parts.append(source_context)
        
        return "\n---\n".join(context_parts)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _generate_answer(
        self, 
        query: str, 
        context: str, 
        search_results: List[Dict[str, Any]],
        initial_results: List[Dict[str, Any]],
        reranking_enabled: bool
    ) -> Tuple[str, float]:
        """
        Generate answer using OpenAI with hallucination prevention.
        
        Args:
            query: User query
            context: Retrieved context
            search_results: Search results for confidence calculation
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        # Advanced prompt with hallucination prevention
        system_prompt = """You are a medical AI assistant specializing in tuberculosis management. Answer questions using ONLY information explicitly found in the provided source documents.

CRITICAL RULES:
1. Base answers EXCLUSIVELY on provided sources - no external knowledge
2. DO NOT include inline citations (Page X, Document Name, etc.) in the text field
3. Sources will be automatically displayed in the detailed metadata section
4. If sources are incomplete, explicitly state what's missing
5. Never infer, assume, or add information not in sources
6. If sources conflict, acknowledge both perspectives (without citing page numbers in text)
7. Return ONLY valid JSON - no additional text before or after
8. Answers must be concise and to the point.
9. Format your answer using Markdown DYNAMICALLY based on content type. Vary the format to keep responses interesting:

   **For comparisons or multiple options:** Use TABLES to compare features/options side-by-side
   **For step-by-step procedures:** Use NUMBERED lists (1., 2., 3.)
   **For lists of items/symptoms:** Use BULLET points (- or *)
   **For single focused answers:** Use a brief paragraph with **bold** key terms
   **For warnings/critical info:** Use > blockquotes
   **For dosages/measurements:** Use `code blocks`
   **For complex topics:** Use ### headings to organize sections
   **For related info:** Use horizontal rules (---) to separate distinct sections

   IMPORTANT: Don't always use the same format! Adapt based on what makes the information clearest:
   - If comparing drugs/treatments → Use a TABLE
   - If explaining a process → Use numbered steps OR a flowchart description
   - If listing symptoms/criteria → Use bullets OR a simple paragraph
   - If showing schedules → Use a TABLE with time/dosage columns
   - Mix formats within one answer when it improves clarity

RESPONSE FORMAT (strict JSON):
{
    "text": "your answer here WITHOUT inline citations or page numbers",
    "confidence": 0.85,
    "sources": [
        {"document_name": "exact document name", "page_number": 46, "relevance_score": 0.177}
    ]
}

CONFIDENCE SCORING:
- 0.9-1.0: Complete answer with multiple corroborating sources
- 0.7-0.89: Good answer but missing some details or single source
- 0.5-0.69: Partial answer, significant gaps in information
- <0.5: Insufficient sources to answer confidently

RESPONSE RULES:
- "text" field: Natural language answer WITHOUT any citations, page numbers, or document references
- "confidence" field: Float between 0 and 1 (not string)
- "sources" field: Array of source objects (these will be shown separately in the UI)
- "page_number": Integer (not string)
- "relevance_score": Float between 0 and 1

---

EXAMPLE 1 - Using TABLE format for comparison:

Query: "What are the different TB treatment regimens for drug-susceptible TB?"

Retrieved Sources:
[Source 1: Page 52] "Standard regimen: 2 months RHZE (intensive phase) followed by 4 months RH (continuation phase)"
[Source 2: Page 53] "Alternative regimen for streptomycin allergy: 2RHZE/4RH"
[Source 3: Page 52] "Dosages: Rifampicin 10mg/kg, Isoniazid 5mg/kg, Pyrazinamide 25mg/kg, Ethambutol 15mg/kg"

Response:
{
    "text": "## TB Treatment Regimens (Drug-Susceptible)\n\n| **Phase** | **Duration** | **Drugs** | **Dosage** |\n|-----------|--------------|-----------|------------|\n| Intensive | 2 months | **RHZE** | R: `10mg/kg`, H: `5mg/kg`, Z: `25mg/kg`, E: `15mg/kg` |\n| Continuation | 4 months | **RH** | R: `10mg/kg`, H: `5mg/kg` |\n\n**Alternative for Streptomycin Allergy:**\nSame regimen (2RHZE/4RH) can be used\n\n> Treatment duration: **6 months total**",
    "confidence": 0.92,
    "sources": [
        {"document_name": "TB Guidelines", "page_number": 52, "relevance_score": 0.91},
        {"document_name": "TB Guidelines", "page_number": 53, "relevance_score": 0.85},
        {"document_name": "TB Guidelines", "page_number": 52, "relevance_score": 0.79}
    ]
}



PENALTIES:
- Adding information not in sources: REJECTED
- Including inline citations or page numbers in text field: REJECTED
- Outputting non-JSON text: REJECTED
- Using string for confidence/page_number/relevance_score: REJECTED
- Confidence score not matching answer completeness: HEAVILY PENALIZED
"""

        user_prompt = f"""
Based on the following sources, answer this question: {query}

Sources:
{context}

Return ONLY valid JSON following the specified format. Cite sources inline within your text field.
"""
        try:
            retrieval_metadata = {
                "initial_retrieval_results": [
                    {
                        "chunk_id": result.get("chunk_id"),
                        "document_name": result.get("document_name"),
                        "page_number": result.get("page_number"),
                        "score": result.get("score"),
                        "content": result.get("content")
                    }
                    for result in initial_results
                ]
            }

            if reranking_enabled:
                retrieval_metadata["reranked_results"] = [
                    {
                        "chunk_id": result.get("chunk_id"),
                        "document_name": result.get("document_name"),
                        "page_number": result.get("page_number"),
                        "score": result.get("score"),
                        "content": result.get("content")
                    }
                    for result in search_results
                ]

            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.1,
                presence_penalty=0.0,
                frequency_penalty=0.1,
                name="rag-generate-answer",
                metadata=self._build_langfuse_metadata(
                    "generate_answer",
                    query=query,
                    extra={
                        "source_count": len(search_results),
                        "context_length_chars": len(context),
                        "retrieval_details": retrieval_metadata
                    }
                )
            )
            
            raw_response = response.choices[0].message.content.strip()

            try:
                # Extract only JSON part in case GPT adds extra text
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                json_str = raw_response[json_start:json_end]

                parsed_response = json.loads(json_str)

                # Extract answer, confidence, and sources
                text = parsed_response.get("text", "")
                confidence = float(parsed_response.get("confidence", 0.0))
                json_sources = parsed_response.get("sources", [])

                confidence = max(0.0, min(1.0, confidence))

                logger.info(f"Successfully parsed JSON response. Answer: {text}, Confidence: {confidence}, Sources: {len(json_sources)}")

                # If text is empty, treat as malformed
                if not text:
                    return "Structure is not correct, kindly recheck the JSON format returned by LLM.", 0.0, []

                return text, confidence, json_sources

            except (json.JSONDecodeError, ValueError, TypeError) as json_error:
                logger.error(f"CRITICAL: Failed to parse LLM response as JSON. Error: {json_error}")
                logger.error(f"Raw response from LLM was: {raw_response}")

                # Return a clear message for frontend
                return "Structure is not correct, kindly recheck the JSON format returned by LLM.", 0.0, []

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            # Return a fallback message for frontend instead of crashing
            return "Failed to generate answer due to backend error.", 0.0, []
    
    def _calculate_confidence(
        self, 
        answer: str, 
        search_results: List[Dict[str, Any]], 
        query: str
    ) -> float:
        """
        Calculate confidence score for the generated answer.
        
        Args:
            answer: Generated answer
            search_results: Search results used
            query: Original query
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        factors = []
        
        # Factor 1: Search result quality (average relevance score)
        if search_results:
            avg_relevance = sum(r.get("score", 0) for r in search_results) / len(search_results)
            factors.append(avg_relevance)
        else:
            factors.append(0.0)
        
        # Factor 2: Number of supporting sources
        num_sources = len(search_results)
        source_factor = min(num_sources / 3.0, 1.0)  # Normalize to max 3 sources
        factors.append(source_factor)
        
        # Factor 3: Answer completeness (simple heuristic)
        answer_length = len(answer.split())
        completeness_factor = min(answer_length / 100.0, 1.0)  # Normalize to ~100 words
        factors.append(completeness_factor)
        
        # Factor 4: Uncertainty indicators (penalty for uncertainty expressions)
        uncertainty_phrases = [
            "i don't know", "not sure", "unclear", "cannot determine",
            "insufficient information", "not enough", "unable to find"
        ]
        uncertainty_penalty = 0.0
        answer_lower = answer.lower()
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                uncertainty_penalty += 0.2
        
        uncertainty_factor = max(0.0, 1.0 - uncertainty_penalty)
        factors.append(uncertainty_factor)
        
        # Factor 5: Citation presence (bonus for citing sources)
        citation_indicators = ["source", "according to", "page", "document"]
        citation_bonus = 0.0
        for indicator in citation_indicators:
            if indicator.lower() in answer.lower():
                citation_bonus += 0.1
        
        citation_factor = min(1.0, 0.5 + citation_bonus)  # Base 0.5 + bonuses
        factors.append(citation_factor)
        
        # Weighted average of factors
        weights = [0.3, 0.2, 0.1, 0.2, 0.2]  # Emphasize relevance and uncertainty
        confidence = sum(f * w for f, w in zip(factors, weights))
        
        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    
    def _prepare_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare source information for the response.
        
        Args:
            search_results: Search results from vector store
            
        Returns:
            List of formatted source dictionaries
        """
        sources = []
        
        for result in search_results:
            source = {
                "document_name": result["document_name"],
                "page_number": result["page_number"],
                "section": result.get("section_title", "Unknown"),
                "relevance_score": round(result["score"], 3)
            }
            sources.append(source)
        
        return sources
    
    def format_response(self, rag_result: RAGResult) -> Dict[str, Any]:
        """
        Format RAG result according to the predefined JSON schema.
        
        Args:
            rag_result: RAG pipeline result
            
        Returns:
            Formatted response dictionary
        """
        return {
            "status": rag_result.status,
            "answer": rag_result.answer,
            "confidence_score": round(rag_result.confidence_score, 3),
            "sources": rag_result.sources,
            "retrieval_metadata": rag_result.retrieval_metadata
        }
    
    def configure_for_summarization(self, context_length: int = 20000, sources: int = 15, tokens: int = 4000):
        """
        Configure pipeline for better summarization with more content.
        
        Args:
            context_length: Maximum context length (default: 20000)
            sources: Maximum number of sources (default: 15)
            tokens: Maximum response tokens (default: 4000)
        """
        self.max_context_length = context_length
        self.max_sources = sources
        self.max_tokens = tokens
        logger.info(f"Pipeline configured for summarization: context={context_length}, sources={sources}, tokens={tokens}")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and health information."""
        try:
            collection_info = await self.vector_store.get_collection_info()
            documents = await self.vector_store.list_documents()
            
            return {
                "vector_store": collection_info,
                "total_documents": len(documents),
                "document_list": documents,
                "pipeline_config": {
                    "max_context_length": self.max_context_length,
                    "min_confidence_threshold": self.min_confidence_threshold,
                    "max_sources": self.max_sources,
                    "max_tokens": self.max_tokens,
                    "model": settings.openai_model,
                    "embedding_model": settings.openai_embedding_model
                }
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {"error": str(e)}


# Utility functions
async def create_rag_pipeline() -> AdvancedRAGPipeline:
    """Create and initialize RAG pipeline."""
    pipeline = AdvancedRAGPipeline()
    logger.info("RAG pipeline initialized successfully")
    return pipeline


def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validate user query for basic requirements.
    
    Args:
        query: User query string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    if len(query.strip()) < 3:
        return False, "Query is too short (minimum 3 characters)"
    
    if len(query) > 1000:
        return False, "Query is too long (maximum 1000 characters)"
    
    return True, "" 