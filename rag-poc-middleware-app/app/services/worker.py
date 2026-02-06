import json
import asyncio
import logging
import threading
from typing import Dict, Any
import pika
import httpx
from datetime import datetime

from app.core.config import settings
from app.services.rabbitmq import rabbitmq_service
from app.services.rag_api import rag_api_client, RAGAPIError

from app.models.chat import Chats
from app.models.user import User

logger = logging.getLogger(__name__)


class ChatWorker:
    """Worker that processes chat messages from RabbitMQ queue"""
    
    def __init__(self):
        self.is_running = False
        self.worker_thread = None

    @staticmethod
    def _format_rag_error(detail: Any) -> str:
        """Convert RAG API error detail into human-readable text."""
        try:
            if isinstance(detail, dict):
                if "detail" in detail and isinstance(detail["detail"], list) and detail["detail"]:
                    first = detail["detail"][0]
                    if isinstance(first, dict):
                        msg = first.get("msg") or first.get("message")
                        if msg:
                            return str(msg)
                if "message" in detail:
                    return str(detail["message"])
                return json.dumps(detail)
            if isinstance(detail, list) and detail:
                first = detail[0]
                if isinstance(first, dict) and "msg" in first:
                    return str(first["msg"])
                return json.dumps(detail)
            return str(detail)
        except Exception:
            return str(detail)
        
    def start(self):
        """Start the worker in a separate thread"""
        if self.is_running:
            logger.warning("Worker is already running")
            return
            
        if not settings.WORKER_ENABLED:
            logger.info("Worker is disabled in configuration")
            return
            
        if not settings.SUPERUSER_TOKEN:
            logger.error("SUPERUSER_TOKEN not configured, worker cannot start")
            return
            
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self.worker_thread.start()
        logger.info("Chat worker started")
    
    def stop(self):
        """Stop the worker"""
        self.is_running = False
        rabbitmq_service.stop_consuming()
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        logger.info("Chat worker stopped")
    
    def _run_worker(self):
        """Main worker loop"""
        try:
            # Setup RabbitMQ consumer
            if not rabbitmq_service.setup_consumer(self._process_message):
                logger.error("Failed to setup RabbitMQ consumer")
                return
            
            # Start consuming messages (this blocks)
            rabbitmq_service.start_consuming()
            
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            rabbitmq_service.disconnect()
    
    def _process_message(self, ch, method, properties, body):
        """Process a single message from the queue"""
        try:
            # Parse the message
            message_data = json.loads(body)
            logger.info(f"Processing message: {message_data.get('id', 'unknown')}")
            logger.debug(f"Message data structure: {list(message_data.keys())}")
            
            # Process the message asynchronously
            result = asyncio.run(self._handle_chat_message(message_data))
            
            if result:
                # Acknowledge the message
                ch.basic_ack(delivery_tag=method.delivery_tag)
                logger.info(f"Message processed successfully: {message_data.get('id', 'unknown')}")
            else:
                # Reject and requeue the message
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                logger.error(f"Message processing failed, requeued: {message_data.get('id', 'unknown')}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message JSON: {e}")
            logger.error(f"Raw message body: {body}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(f"Message body: {body}")
            # Reject without requeue to avoid infinite loop
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    async def _handle_chat_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Handle a chat message by:
        1. Querying the RAG API with the message text
        2. Creating a response message from bot to user
        
        Args:
            message_data: The chat message data from the queue
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract message details
            original_sender_id = message_data.get('sender_id')
            original_receiver_id = message_data.get('receiver_id')
            message_content = message_data.get('message', {})
            
            # Extract text from message content (handle both old and new formats)
            if isinstance(message_content, str):
                # Old format - direct string
                query_text = message_content
            elif isinstance(message_content, dict) and 'text' in message_content:
                # New format - structured message content
                query_text = message_content['text']
            else:
                logger.error(f"Invalid message format: {type(message_content)}")
                return False
            
            if not all([original_sender_id, original_receiver_id, query_text]):
                logger.error("Missing required fields in message data")
                return False
            
            # Determine which user is the bot and get their document filter
            document_filter = None
            try:
                sender = User.objects.get(id=original_sender_id)
                receiver = User.objects.get(id=original_receiver_id)
                
                # Determine which user is the bot
                if sender.user_type == 'bot':
                    bot_user = sender
                elif receiver.user_type == 'bot':
                    bot_user = receiver
                else:
                    logger.error("Neither sender nor receiver is a bot user")
                    return False
                
                # Get the bot's fixed document filter
                document_filter = bot_user.document_filter
                logger.info(f"Bot '{bot_user.name}' using document filter: {document_filter}")
                
            except Exception as e:
                logger.error(f"Error fetching bot information: {e}")
                return False
            
            # Query the RAG API with document filter
            logger.info(f"Querying RAG API with: {query_text[:100]}... (filter: {document_filter})")
            try:
                api_response = await rag_api_client.query(query_text, document_filter)
            except RAGAPIError as e:
                logger.error(f"RAG API error ({e.status_code}): {e.detail}")
                if not getattr(e, "retriable", True):
                    error_text = self._format_rag_error(e.detail)
                    logger.info("Non-retriable RAG API error, returning message to user without retry")
                    await self._create_bot_response(
                        sender_id=original_receiver_id,
                        receiver_id=original_sender_id,
                        message_content={
                            "text": f"Unable to process your query: {error_text}",
                            "status": "error"
                        }
                    )
                    return True
                return False

            if not api_response:
                logger.error("RAG API returned empty response")
                return False
            
            # Extract structured content from API response
            structured_content = rag_api_client.extract_structured_content(api_response)
            
            # Create the bot response message with rich content
            # Bot responds as the original receiver, sending to original sender
            await self._create_bot_response(
                sender_id=original_receiver_id,  # Bot sends as itself
                receiver_id=original_sender_id,  # To the original human sender
                message_content=structured_content
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
            logger.error(f"Message data: {message_data}")
            return False
    
    async def _create_bot_response(self, sender_id: str, receiver_id: str, message_content: Dict[str, Any]) -> bool:
        """
        Create a chat response using the API endpoint with structured content
        
        Args:
            sender_id: ID of the bot sending the response
            receiver_id: ID of the user receiving the response
            message_content: The structured message content dict
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare the chat creation payload - use simple text for API compatibility
            # The structured content will be stored via direct model creation
            chat_payload = {
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "message": message_content.get('text', 'No response text available')
            }
            
            # Create message content with structured data
            from app.models.chat import MessageContent, MessageSource, RetrievalMetadata
            
            # Build MessageContent with all the structured data
            message_obj = MessageContent(text=message_content['text'])
            
            if 'confidence_score' in message_content:
                message_obj.confidence_score = message_content['confidence_score']
            if 'status' in message_content:
                message_obj.status = message_content['status']
            if 'processing_time' in message_content:
                message_obj.processing_time = message_content['processing_time']
            
            # Add sources if present
            if 'sources' in message_content and message_content['sources']:
                sources = []
                for source_data in message_content['sources']:
                    source = MessageSource(
                        document_name=source_data['document_name']
                    )
                    if 'page_number' in source_data:
                        source.page_number = source_data['page_number']
                    if 'relevance_score' in source_data:
                        source.relevance_score = source_data['relevance_score']
                    sources.append(source)
                message_obj.sources = sources
            
            # Add retrieval metadata if present
            if 'retrieval_metadata' in message_content:
                metadata = message_content['retrieval_metadata']
                retrieval_meta = RetrievalMetadata()
                if 'total_documents_searched' in metadata:
                    retrieval_meta.total_documents_searched = metadata['total_documents_searched']
                if 'query_expansion_used' in metadata:
                    retrieval_meta.query_expansion_used = metadata['query_expansion_used']
                if 'hybrid_search_used' in metadata:
                    retrieval_meta.hybrid_search_used = metadata['hybrid_search_used']
                if 'reranking_used' in metadata:
                    retrieval_meta.reranking_used = metadata['reranking_used']
                message_obj.retrieval_metadata = retrieval_meta
            
            # Create the chat message directly in the database
            chat = Chats(
                sender_id=sender_id,
                receiver_id=receiver_id,
                message=message_obj
            )
            chat.save()
            
            logger.info(f"Bot response created successfully with rich content")
            return True
                    
        except Exception as e:
            logger.error(f"Error creating bot response: {e}")
            return False


# Global worker instance
chat_worker = ChatWorker()