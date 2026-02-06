from fastapi import APIRouter, HTTPException, status, Query, Depends
from typing import List, Optional
from mongoengine.errors import DoesNotExist
from bson import ObjectId
import logging
from datetime import datetime
from app.models.chat import Chats, MessageContent, ChatCounter
from app.models.user import User
from app.schemas.chat import ChatCreate, ChatResponse, ChatListResponse, ConversationsListResponse, ConversationDetails, ConversationParticipant
from app.core.security import get_current_user_from_token
from app.services.rabbitmq import rabbitmq_service
from mongoengine import Q
router = APIRouter()
logger = logging.getLogger(__name__)


def validate_sender_permission(current_user: User, sender_id: str) -> bool:
    """Validate that the current user can send as the specified sender"""
    # Users can only send as themselves (by user ID)
    return sender_id == str(current_user.id)


def validate_receiver_permission(current_user: User, receiver_id: str) -> bool:
    """Validate that the current user can send to the specified receiver"""
    try:
        receiver_user = User.objects.get(id=receiver_id)
        if current_user.user_type == 'human':
            # Human users can only send to bot users
            return receiver_user.user_type == 'bot'
        elif current_user.user_type == 'bot':
            # Bot users can send to any user (human or bot)
            return True
    except DoesNotExist:
        return False
    return False


def validate_read_permission(current_user: User, sender_id: str, receiver_id: str) -> bool:
    """Validate that the current user can read messages in this conversation"""
    current_user_id = str(current_user.id)
    
    # Users can read conversations where they are either sender or receiver
    return current_user_id == sender_id or current_user_id == receiver_id


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def create_message(
    chat_data: ChatCreate,
    current_user: User = Depends(get_current_user_from_token)
):
    """Create a new chat message"""
    # Ensure user is authenticated
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # Validate sender permission
    if not validate_sender_permission(current_user, chat_data.sender_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only send messages as yourself"
        )
    
    # Validate receiver permission (no human-to-human messaging)
    if not validate_receiver_permission(current_user, chat_data.receiver_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Human users cannot send messages to other human users"
        )
    
    # Create the chat message with embedded MessageContent
    message_content = MessageContent(text=chat_data.message)
    chat = Chats(
        sender_id=chat_data.sender_id,
        receiver_id=chat_data.receiver_id,
        message=message_content
    )
    chat.save()
    
    # If sender is a human user, publish message to RabbitMQ for processing
    if current_user.user_type == 'human':
        try:
            # Prepare message data for queue
            message_data = chat.to_dict()
            
            # Publish to RabbitMQ queue
            success = rabbitmq_service.publish_message(message_data)
            if success:
                logger.info(f"Published human message to queue: {chat.id}")
            else:
                logger.error(f"Failed to publish message to queue: {chat.id}")
        except Exception as e:
            logger.error(f"Error publishing message to queue: {e}")
            # Don't fail the API call if queue publishing fails
    
    # Safely handle message content for response
    message_dict = {"text": ""}
    if chat.message:
        if hasattr(chat.message, 'to_dict'):
            message_dict = chat.message.to_dict()
        elif isinstance(chat.message, str):
            message_dict = {"text": chat.message}
        else:
            message_dict = {"text": str(chat.message)}
    
    return ChatResponse(
        id=str(chat.id),
        sender_id=chat.sender_id,
        receiver_id=chat.receiver_id,
        message=message_dict,
        offset=chat.offset,
        is_read=chat.is_read,
        is_delivered=chat.is_delivered,
        is_seen=chat.is_seen,
        is_deleted=chat.is_deleted,
        is_archived=chat.is_archived,
        is_pinned=chat.is_pinned,
        created_at=chat.created_at,
        updated_at=chat.updated_at
    )





@router.get("/messages/", response_model=ChatListResponse)
async def get_messages(
    sender_id: str = Query(..., description="Sender ID for the conversation"),
    receiver_id: str = Query(..., description="Receiver ID for the conversation"),
    offset: int = Query(..., description="Reference message offset"),
    limit: int = Query(10, ge=1, le=100, description="Number of messages to fetch"),
    before: bool = Query(True, description="If True, fetch messages before offset, else after"),
    current_user: User = Depends(get_current_user_from_token)
):
    """Get chat messages before or after a specific offset for a conversation"""
    # Ensure user is authenticated
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # Validate read permission - user must be part of this conversation
    if not validate_read_permission(current_user, sender_id, receiver_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only read messages from conversations you are part of"
        )
    
    # Query for messages in this conversation (both directions)
    base_query = Chats.objects.filter(
        (Q(sender_id=sender_id) & Q(receiver_id=receiver_id)) |
        (Q(sender_id=receiver_id) & Q(receiver_id=sender_id))
    )
    
    # Query based on before/after offset
    if before:
        messages = base_query.filter(offset__lt=offset).order_by('-offset').limit(limit)
        total_remaining = base_query.filter(offset__lt=offset).count()
    else:
        messages = base_query.filter(offset__gt=offset).order_by('offset').limit(limit)
        total_remaining = base_query.filter(offset__gt=offset).count()

    # Convert messages to response format
    message_responses = []
    for msg in messages:
        # Safely handle message content
        message_dict = {"text": ""}
        if msg.message:
            if hasattr(msg.message, 'to_dict'):
                message_dict = msg.message.to_dict()
            elif isinstance(msg.message, str):
                message_dict = {"text": msg.message}
            else:
                message_dict = {"text": str(msg.message)}
        
        message_responses.append(ChatResponse(
            id=str(msg.id),
            sender_id=msg.sender_id,
            receiver_id=msg.receiver_id,
            message=message_dict,
            offset=msg.offset,
            is_read=msg.is_read,
            is_delivered=msg.is_delivered,
            is_seen=msg.is_seen,
            is_deleted=msg.is_deleted,
            is_archived=msg.is_archived,
            is_pinned=msg.is_pinned,
            created_at=msg.created_at,
            updated_at=msg.updated_at
        ))

    # Determine if there are more messages
    has_more = total_remaining > len(message_responses)

    return ChatListResponse(
        messages=message_responses,
        total_count=total_remaining,
        offset=offset,
        limit=limit,
        has_more=has_more
    )


@router.get("/conversations/", response_model=ConversationsListResponse)
async def get_user_conversations(
    current_user: User = Depends(get_current_user_from_token)
):
    """Get all active conversations for the current user"""
    # Ensure user is authenticated
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    current_user_id = str(current_user.id)
    
    # Find all conversation counters where the user is either participant_1 or participant_2
    conversation_counters = ChatCounter.objects.filter(
        Q(participant_1=current_user_id) | Q(participant_2=current_user_id)
    )
    
    conversations = []
    
    for counter in conversation_counters:
        # Determine the other participant
        other_participant_id = (
            counter.participant_2 if counter.participant_1 == current_user_id 
            else counter.participant_1
        )
        
        try:
            # Get other participant's details
            other_user = User.objects.get(id=other_participant_id)
            
            # Skip if the other user is not active
            if not other_user.is_active:
                continue
            
            # Get the last message in this conversation
            last_message_query = Chats.objects.filter(
                (Q(sender_id=current_user_id) & Q(receiver_id=other_participant_id)) |
                (Q(sender_id=other_participant_id) & Q(receiver_id=current_user_id))
            ).order_by('-offset').first()
            
            last_message_response = None
            if last_message_query:
                # Safely handle message content for the last message
                message_dict = {"text": ""}
                if last_message_query.message:
                    if hasattr(last_message_query.message, 'to_dict'):
                        message_dict = last_message_query.message.to_dict()
                    elif isinstance(last_message_query.message, str):
                        message_dict = {"text": last_message_query.message}
                    else:
                        message_dict = {"text": str(last_message_query.message)}
                
                last_message_response = ChatResponse(
                    id=str(last_message_query.id),
                    sender_id=last_message_query.sender_id,
                    receiver_id=last_message_query.receiver_id,
                    message=message_dict,
                    offset=last_message_query.offset,
                    is_read=last_message_query.is_read,
                    is_delivered=last_message_query.is_delivered,
                    is_seen=last_message_query.is_seen,
                    is_deleted=last_message_query.is_deleted,
                    is_archived=last_message_query.is_archived,
                    is_pinned=last_message_query.is_pinned,
                    created_at=last_message_query.created_at,
                    updated_at=last_message_query.updated_at
                )
            
            # Create participant details
            participant = ConversationParticipant(
                id=str(other_user.id),
                name=other_user.name,
                email=other_user.email,
                description=other_user.description,
                user_type=other_user.user_type,
                is_active=other_user.is_active
            )
            
            # Create conversation details
            conversation_detail = ConversationDetails(
                participant=participant,
                message_count=counter.value,
                last_message=last_message_response
            )
            
            conversations.append(conversation_detail)
            
        except DoesNotExist:
            # Skip if other participant user doesn't exist
            logger.warning(f"User {other_participant_id} not found in conversation counter")
            continue
        except Exception as e:
            # Log error but continue processing other conversations
            logger.error(f"Error processing conversation with user {other_participant_id}: {e}")
            continue
    
    # Sort conversations by last message timestamp (most recent first)
    conversations.sort(
        key=lambda x: x.last_message.updated_at if x.last_message else datetime.min,
        reverse=True
    )
    
    return ConversationsListResponse(
        conversations=conversations,
        total_conversations=len(conversations)
    )