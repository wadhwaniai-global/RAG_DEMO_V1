from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime


class MessageSource(BaseModel):
    """Schema for message source from RAG API"""
    document_name: str = Field(..., description="Name of the source document")
    page_number: Optional[int] = Field(None, description="Page number in the document")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")


class RetrievalMetadata(BaseModel):
    """Schema for RAG API retrieval metadata"""
    total_documents_searched: Optional[int] = Field(None, ge=0, description="Total documents searched")
    query_expansion_used: Optional[bool] = Field(None, description="Whether query expansion was used")
    hybrid_search_used: Optional[bool] = Field(None, description="Whether hybrid search was used")
    reranking_used: Optional[bool] = Field(None, description="Whether reranking was used")


class MessageContent(BaseModel):
    """Schema for message content - supports both simple text and rich bot responses"""
    text: str = Field(..., max_length=10000, description="Main message text")
    
    # Optional fields for bot messages from RAG API
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    sources: Optional[List[MessageSource]] = Field(None, description="Source documents")
    retrieval_metadata: Optional[RetrievalMetadata] = Field(None, description="Retrieval metadata")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Processing time in seconds")
    status: Optional[str] = Field(None, max_length=50, description="API status")
    
    def is_bot_message(self) -> bool:
        """Check if this is a rich bot message with additional metadata"""
        return any([
            self.confidence_score is not None,
            self.sources is not None and len(self.sources) > 0,
            self.retrieval_metadata is not None,
            self.processing_time is not None,
            self.status is not None
        ])


class ChatMessage(BaseModel):
    """Schema for a single chat message"""
    sender_id: str = Field(..., max_length=200, description="ID of the message sender")
    receiver_id: str = Field(..., max_length=200, description="ID of the message receiver")
    message: Union[str, MessageContent] = Field(..., description="The chat message content")
    
    @validator('message', pre=True)
    def validate_message(cls, v):
        """Convert string message to MessageContent for backward compatibility"""
        if isinstance(v, str):
            return MessageContent(text=v)
        elif isinstance(v, dict):
            return MessageContent(**v)
        return v


class ChatCreate(BaseModel):
    """Schema for creating a new chat message - simplified for API input"""
    sender_id: str = Field(..., max_length=200, description="ID of the message sender")
    receiver_id: str = Field(..., max_length=200, description="ID of the message receiver")
    message: str = Field(..., max_length=10000, description="The chat message text")


class ChatInDB(ChatMessage):
    """Schema for chat message as stored in database"""
    id: str
    offset: int = Field(..., description="Message position/order within conversation")
    is_read: bool = Field(default=False)
    is_delivered: bool = Field(default=False)
    is_seen: bool = Field(default=False)
    is_deleted: bool = Field(default=False)
    is_archived: bool = Field(default=False)
    is_pinned: bool = Field(default=False)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatResponse(ChatMessage):
    """Schema for chat message response"""
    id: str
    offset: int = Field(..., description="Message position/order for frontend sorting")
    is_read: bool = Field(default=False)
    is_delivered: bool = Field(default=False)
    is_seen: bool = Field(default=False)
    is_deleted: bool = Field(default=False)
    is_archived: bool = Field(default=False)
    is_pinned: bool = Field(default=False)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatQuery(BaseModel):
    """Schema for querying chat messages with pagination"""
    sender_id: Optional[str] = Field(None, max_length=200, description="Filter by sender ID")
    receiver_id: Optional[str] = Field(None, max_length=200, description="Filter by receiver ID")
    offset: int = Field(0, ge=0, description="Reference message offset for pagination")
    limit: int = Field(10, ge=1, le=100, description="Number of messages to fetch (max 100)")
    before: bool = Field(True, description="If True, fetch messages before offset, else after")


class ChatListResponse(BaseModel):
    """Schema for paginated chat messages response"""
    messages: List[ChatResponse]
    total_count: int = Field(..., description="Total number of messages available")
    offset: int = Field(..., description="Reference offset used for pagination")
    limit: int = Field(..., description="Number of messages returned")
    has_more: bool = Field(..., description="Whether there are more messages available")


class ConversationParticipant(BaseModel):
    """Schema for conversation participant details"""
    id: str
    name: str
    email: Optional[str] = None
    description: Optional[str] = None
    user_type: str
    is_active: bool


class ConversationDetails(BaseModel):
    """Schema for conversation details"""
    participant: ConversationParticipant
    message_count: int = Field(..., description="Total number of messages in this conversation")
    last_message: Optional[ChatResponse] = Field(None, description="Last message in the conversation")


class ConversationsListResponse(BaseModel):
    """Schema for user's conversations list response"""
    conversations: List[ConversationDetails]
    total_conversations: int = Field(..., description="Total number of active conversations")

