from mongoengine import Document, EmbeddedDocument, StringField, FloatField, IntField, BooleanField, DateTimeField, ListField, EmbeddedDocumentField, DictField, NotUniqueError
from mongoengine.base import BaseField
from datetime import datetime
from typing import List, Optional, Dict, Any


class MessageSource(EmbeddedDocument):
    """Embedded document for message sources from RAG API"""
    document_name = StringField(required=True)
    page_number = IntField()
    relevance_score = FloatField()


class RetrievalMetadata(EmbeddedDocument):
    """Embedded document for RAG API retrieval metadata"""
    total_documents_searched = IntField()
    query_expansion_used = BooleanField()
    hybrid_search_used = BooleanField()
    reranking_used = BooleanField()


class MessageContent(EmbeddedDocument):
    """Embedded document for message content - supports both simple text and rich bot responses"""
    text = StringField(required=True, max_length=10000)  # Main message text (maps to 'answer' for bots)
    
    # Optional fields for bot messages from RAG API
    confidence_score = FloatField(min_value=0.0, max_value=1.0)
    sources = ListField(EmbeddedDocumentField(MessageSource))
    retrieval_metadata = EmbeddedDocumentField(RetrievalMetadata)
    processing_time = FloatField(min_value=0.0)
    status = StringField(max_length=50)  # API status like 'success'
    
    def is_bot_message(self) -> bool:
        """Check if this is a rich bot message with additional metadata"""
        return any([
            self.confidence_score is not None,
            self.sources,
            self.retrieval_metadata is not None,
            self.processing_time is not None,
            self.status is not None
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {'text': self.text}
        
        if self.confidence_score is not None:
            result['confidence_score'] = self.confidence_score
        if self.sources:
            result['sources'] = [{
                'document_name': source.document_name,
                'page_number': source.page_number,
                'relevance_score': source.relevance_score
            } for source in self.sources]
        if self.retrieval_metadata:
            result['retrieval_metadata'] = {
                'total_documents_searched': self.retrieval_metadata.total_documents_searched,
                'query_expansion_used': self.retrieval_metadata.query_expansion_used,
                'hybrid_search_used': self.retrieval_metadata.hybrid_search_used,
                'reranking_used': self.retrieval_metadata.reranking_used
            }
        if self.processing_time is not None:
            result['processing_time'] = self.processing_time
        if self.status:
            result['status'] = self.status
            
        return result


class MixedMessageField(EmbeddedDocumentField):
    """Custom field that handles both string (legacy) and embedded document (new) message formats"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(MessageContent, *args, **kwargs)
    
    def to_mongo(self, value):
        """Convert value to MongoDB format"""
        if isinstance(value, str):
            # Convert string to MessageContent for storage
            message_content = MessageContent(text=value)
            return super().to_mongo(message_content)
        elif isinstance(value, MessageContent):
            return super().to_mongo(value)
        elif isinstance(value, dict):
            # Handle dict input (from API)
            if 'text' in value:
                message_content = MessageContent(**value)
                return super().to_mongo(message_content)
            else:
                # Assume it's a legacy string in dict format
                message_content = MessageContent(text=str(value.get('message', '')))
                return super().to_mongo(message_content)
        return super().to_mongo(value)
    
    def to_python(self, value):
        """Convert from MongoDB to Python object"""
        if isinstance(value, str):
            # Handle legacy string messages - convert to MessageContent
            return MessageContent(text=value)
        elif isinstance(value, dict):
            # Handle embedded document from database
            return super().to_python(value)
        elif isinstance(value, MessageContent):
            return value
        return super().to_python(value)


class ChatCounter(Document):
    """Counter for each conversation thread between two participants"""
    participant_1 = StringField(required=True, max_length=200)
    participant_2 = StringField(required=True, max_length=200)
    value = IntField(default=0)
    
    meta = {
        'collection': 'chat_counters',
        'indexes': [
            {'fields': ['participant_1', 'participant_2'], 'unique': True}  # Composite unique key
        ]
    }
    
    @staticmethod
    def normalize_participants(sender_id: str, receiver_id: str) -> tuple:
        """Normalize participant IDs so order doesn't matter"""
        participants = sorted([sender_id, receiver_id])
        return participants[0], participants[1]


class Chats(Document):
    """Chats model using MongoEngine"""
    sender_id = StringField(required=True, max_length=200)
    receiver_id = StringField(required=True, max_length=200)
    message = MixedMessageField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    offset = IntField(required=True)
    is_read = BooleanField(default=False)
    is_delivered = BooleanField(default=False)
    is_seen = BooleanField(default=False)
    is_deleted = BooleanField(default=False)
    is_archived = BooleanField(default=False)
    is_pinned = BooleanField(default=False)
    
    # Meta configuration
    meta = {
        'collection': 'chats',
        'indexes': [
            'sender_id',
            'receiver_id',
            ('sender_id', 'receiver_id', 'offset'),  # Composite index for conversation + offset
            [('offset', -1)]  # Index on offset field for faster queries
        ]
    }
    
    def save(self, *args, **kwargs):
        """Override save to update timestamps and set atomic incremental offset per conversation"""
        if not self.id:  # Only for new documents
            self.created_at = datetime.utcnow()
            
            # Normalize participants so order doesn't matter
            participant_1, participant_2 = ChatCounter.normalize_participants(
                str(self.sender_id), str(self.receiver_id)
            )
            
            # Atomically increment counter for this specific conversation
            from mongoengine.connection import get_db
            db = get_db()
            
            result = db.chat_counters.find_one_and_update(
                {
                    'participant_1': participant_1,
                    'participant_2': participant_2
                },
                {'$inc': {'value': 1}},
                upsert=True,  # Create counter if doesn't exist for this conversation
                return_document=True  # Return updated document
            )
            
            self.offset = result['value']
        
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
    
    def __str__(self):
        if self.message:
            if hasattr(self.message, 'text'):
                message_text = self.message.text
            elif isinstance(self.message, str):
                message_text = self.message
            else:
                message_text = str(self.message)
        else:
            message_text = "No message"
        return f"Chat(msg={message_text}, sender={self.sender_id}), receiver={self.receiver_id}"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        # Handle message field safely
        message_dict = None
        if self.message:
            if hasattr(self.message, 'to_dict'):
                message_dict = self.message.to_dict()
            elif isinstance(self.message, str):
                # Handle legacy string format
                message_dict = {'text': self.message}
            else:
                # Fallback for unexpected formats
                message_dict = {'text': str(self.message)}
        
        return {
            'id': str(self.id),
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message': message_dict,
            'offset': self.offset,
            'is_read': self.is_read,
            'is_delivered': self.is_delivered,
            'is_seen': self.is_seen,
            'is_deleted': self.is_deleted,
            'is_archived': self.is_archived,
            'is_pinned': self.is_pinned,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }