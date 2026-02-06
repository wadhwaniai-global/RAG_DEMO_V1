import httpx
import logging
from typing import Dict, Any, Optional, List, Union
from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGAPIClient:
    """Client for interacting with the external RAG API"""
    
    def __init__(self):
        self.base_url = settings.RAG_API_URL.rstrip('/')
        self.endpoint = settings.RAG_API_ENDPOINT
        self.timeout = 30.0  # 30 seconds timeout
    
    async def query(self, query_text: str, document_filter: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Send a query to the RAG API and return the response

        Args:
            query_text: The text query to send to the RAG API
            document_filter: Optional list of document filters to limit search scope

        Returns:
            Dict containing the API response or None if failed
        """
        url = f"{self.base_url}{self.endpoint}"

        payload = {
            "query": query_text,
            "use_query_expansion": settings.RAG_USE_QUERY_EXPANSION,
            "use_reranking": settings.RAG_USE_RERANKING
        }

        # Add document filter if provided
        if document_filter:
            payload["document_filter"] = document_filter

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

        logger.info(f"Sending query to RAG API: {url}")
        logger.debug(f"Payload: {payload}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url=url,
                    json=payload,
                    headers=headers
                )
        except httpx.TimeoutException as exc:
            logger.error(f"RAG API request timed out after {self.timeout} seconds")
            raise RAGAPIError(408, "RAG API request timed out", retriable=True) from exc
        except httpx.RequestError as exc:
            logger.error(f"RAG API request failed: {exc}")
            raise RAGAPIError(503, str(exc), retriable=True) from exc
        except Exception as exc:
            logger.error(f"Unexpected transport error in RAG API call: {exc}")
            raise RAGAPIError(500, str(exc), retriable=True) from exc

        if response.status_code == 200:
            result = response.json()
            logger.info("RAG API query successful")
            return result

        try:
            detail = response.json()
        except Exception:
            detail = response.text

        retriable = response.status_code not in {400, 401, 403, 404, 422}
        if response.status_code == 422:
            retriable = False

        logger.error(f"RAG API returned status {response.status_code}: {detail}")
        raise RAGAPIError(response.status_code, detail, retriable)
    
    def extract_answer(self, api_response: Dict[str, Any]) -> str:
        """
        Extract the answer text from the RAG API response
        
        Args:
            api_response: The response dict from the RAG API
            
        Returns:
            The extracted answer text
        """
        try:
            # Try common response formats
            if 'answer' in api_response:
                return str(api_response['answer'])
            elif 'response' in api_response:
                return str(api_response['response'])
            elif 'result' in api_response:
                return str(api_response['result'])
            elif 'text' in api_response:
                return str(api_response['text'])
            else:
                # If no standard key found, return the whole response as string
                logger.warning("No standard answer key found in API response, returning full response")
                return str(api_response)
                
        except Exception as e:
            logger.error(f"Error extracting answer from API response: {e}")
            return f"Error processing response: {str(e)}"
    
    def extract_structured_content(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured message content from the RAG API response
        
        Args:
            api_response: The response dict from the RAG API
            
        Returns:
            Dict containing structured message content for MessageContent creation
        """
        try:
            # Extract the main text (answer)
            text = self.extract_answer(api_response)
            
            content = {
                "text": text
            }
            
            # Extract optional fields if they exist
            if 'confidence_score' in api_response:
                content['confidence_score'] = float(api_response['confidence_score'])
            
            if 'sources' in api_response and api_response['sources']:
                sources = []
                for source in api_response['sources']:
                    source_data = {
                        'document_name': str(source.get('document_name', ''))
                    }
                    if 'page_number' in source:
                        source_data['page_number'] = int(source['page_number'])
                    if 'relevance_score' in source:
                        source_data['relevance_score'] = float(source['relevance_score'])
                    sources.append(source_data)
                content['sources'] = sources
            
            if 'retrieval_metadata' in api_response and api_response['retrieval_metadata']:
                metadata = api_response['retrieval_metadata']
                retrieval_metadata = {}
                if 'total_documents_searched' in metadata:
                    retrieval_metadata['total_documents_searched'] = int(metadata['total_documents_searched'])
                if 'query_expansion_used' in metadata:
                    retrieval_metadata['query_expansion_used'] = bool(metadata['query_expansion_used'])
                if 'hybrid_search_used' in metadata:
                    retrieval_metadata['hybrid_search_used'] = bool(metadata['hybrid_search_used'])
                if 'reranking_used' in metadata:
                    retrieval_metadata['reranking_used'] = bool(metadata['reranking_used'])
                
                if retrieval_metadata:  # Only add if not empty
                    content['retrieval_metadata'] = retrieval_metadata
            
            if 'processing_time' in api_response:
                content['processing_time'] = float(api_response['processing_time'])
            
            if 'status' in api_response:
                content['status'] = str(api_response['status'])
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting structured content from API response: {e}")
            # Return minimal content with error message
            return {
                "text": f"Error processing response: {str(e)}",
                "status": "error"
            }


class RAGAPIError(Exception):
    def __init__(self, status_code: int, detail: Union[dict, str], retriable: bool = True):
        super().__init__(f"RAG API returned status {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail
        self.retriable = retriable


# Global instance
rag_api_client = RAGAPIClient()