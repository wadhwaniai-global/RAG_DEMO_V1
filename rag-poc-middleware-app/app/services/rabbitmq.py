import pika
import json
import logging
import time
from typing import Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class RabbitMQService:
    """Improved service for handling RabbitMQ connections and operations"""
    
    def __init__(self):
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None
        self._connection_params = None
        self._consumer_setup = False  # Track if consumer is set up
        self._callback_function = None  # Store callback for reconnection
        
    def _get_connection_parameters(self):
        """Get connection parameters with proper configuration"""
        if self._connection_params is None:
            # Use individual parameters for better control
            credentials = pika.PlainCredentials(
                settings.RABBITMQ_USERNAME, 
                settings.RABBITMQ_PASSWORD
            )
            
            self._connection_params = pika.ConnectionParameters(
                host=settings.RABBITMQ_HOST,
                port=settings.RABBITMQ_PORT,
                virtual_host=settings.RABBITMQ_VIRTUAL_HOST,
                credentials=credentials,
                # Connection settings to prevent issues
                heartbeat=600,  # 10 minutes
                blocked_connection_timeout=300,  # 5 minutes
                connection_attempts=3,
                retry_delay=2,
                socket_timeout=10,
            )
        return self._connection_params
        
    def connect(self) -> bool:
        """Establish connection to RabbitMQ with retry logic"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Close existing connection if any
                self.disconnect()
                
                # Create new connection
                params = self._get_connection_parameters()
                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()
                
                # Declare the queue (creates if doesn't exist)
                self.channel.queue_declare(
                    queue=settings.CHAT_QUEUE_NAME,
                    durable=True,  # Queue survives broker restart
                    exclusive=False,
                    auto_delete=False
                )
                
                logger.info(f"Connected to RabbitMQ and declared queue: {settings.CHAT_QUEUE_NAME}")
                
                # Re-setup consumer if it was previously setup
                if self._callback_function:
                    self._setup_consumer_internal()
                
                return True
                
            except pika.exceptions.AMQPConnectionError as e:
                logger.error(f"RabbitMQ connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected error connecting to RabbitMQ (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        return False
    
    def disconnect(self):
        """Close RabbitMQ connection safely"""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.close()
                self.channel = None
        except Exception as e:
            logger.debug(f"Error closing channel: {e}")
            
        try:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
                self.connection = None
                logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.debug(f"Error closing connection: {e}")
            
        # Reset connection state but keep consumer setup info
        self.connection = None
        self.channel = None
        self._consumer_setup = False
    
    def is_connected(self) -> bool:
        """Check if connection is healthy"""
        return (
            self.connection is not None and 
            not self.connection.is_closed and
            self.channel is not None and 
            not self.channel.is_closed
        )
    
    def publish_message(self, message: Dict[str, Any]) -> bool:
        """Publish a message to the chat processing queue with connection recovery"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Ensure we have a valid connection
                if not self.is_connected():
                    if not self.connect():
                        logger.error(f"Failed to establish connection for publishing (attempt {attempt + 1})")
                        continue
                
                # Convert message to JSON
                message_body = json.dumps(message, default=str)
                
                # Publish message with persistence
                self.channel.basic_publish(
                    exchange='',
                    routing_key=settings.CHAT_QUEUE_NAME,
                    body=message_body,
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Make message persistent
                        timestamp=int(time.time())
                    )
                )
                
                logger.info(f"Published message to queue: {message.get('id', 'unknown')}")
                return True
                
            except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError) as e:
                logger.error(f"AMQP error publishing message (attempt {attempt + 1}): {e}")
                self.disconnect()  # Force reconnection on next attempt
                time.sleep(1)
            except Exception as e:
                logger.error(f"Unexpected error publishing message (attempt {attempt + 1}): {e}")
                self.disconnect()
                time.sleep(1)
        
        logger.error("Failed to publish message after all retries")
        return False
    
    def _setup_consumer_internal(self) -> bool:
        """Internal method to setup consumer"""
        try:
            if not self.is_connected():
                return False
            
            # Set QoS to process one message at a time
            self.channel.basic_qos(prefetch_count=1)
            
            # Setup consumer
            self.channel.basic_consume(
                queue=settings.CHAT_QUEUE_NAME,
                on_message_callback=self._callback_function,
                auto_ack=False  # Manual acknowledgment for reliability
            )
            
            self._consumer_setup = True
            logger.info(f"Consumer setup for queue: {settings.CHAT_QUEUE_NAME}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup consumer: {e}")
            return False
    
    def setup_consumer(self, callback_function) -> bool:
        """Setup consumer for processing messages"""
        # Store callback for reconnections
        self._callback_function = callback_function
        
        if not self.is_connected():
            if not self.connect():
                return False
        
        return self._setup_consumer_internal()
    
    def start_consuming(self):
        """Start consuming messages (blocking operation) with recovery"""
        if not self._callback_function:
            logger.error("No callback function set for consumer")
            return
            
        while True:
            try:
                if not self.is_connected():
                    logger.info("Connection lost, attempting to reconnect...")
                    if not self.connect():
                        logger.error("Failed to reconnect, retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                
                # Only log this once when actually starting to consume
                if not self._consumer_setup:
                    if not self._setup_consumer_internal():
                        logger.error("Failed to setup consumer, retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                
                logger.info("Starting message consumption...")
                self.channel.start_consuming()
                
            except pika.exceptions.AMQPConnectionError as e:
                logger.error(f"Connection error during consumption: {e}")
                self.disconnect()
                time.sleep(5)
            except KeyboardInterrupt:
                logger.info("Consumption interrupted by user")
                self.stop_consuming()
                break
            except Exception as e:
                logger.error(f"Unexpected error during consumption: {e}")
                self.disconnect()
                time.sleep(5)
    
    def stop_consuming(self):
        """Stop consuming messages"""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.stop_consuming()
                logger.info("Stopped message consumption")
        except Exception as e:
            logger.error(f"Error stopping consumption: {e}")


# Global instance
rabbitmq_service = RabbitMQService()