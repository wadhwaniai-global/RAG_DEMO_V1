from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Dict
from mongoengine.errors import NotUniqueError, DoesNotExist, ValidationError

from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate, UserResponse, TokenRequest, TokenResponse, DocumentFilterUpdate, BulkDocumentFilterUpdate
from app.core.security import (
    get_password_hash, 
    authenticate_user, 
    create_access_token, 
    get_current_user_from_token,
    require_superuser
)

router = APIRouter()


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user_data: UserCreate):
    """Create a new user (human or bot)"""
    try:
        # Hash the password only for human users
        hashed_password = None
        if user_data.password and user_data.user_type == 'human':
            hashed_password = get_password_hash(user_data.password)
        
        # Create user document
        user = User(
            name=user_data.name,
            email=user_data.email,
            description=user_data.description,
            user_type=user_data.user_type,
            hashed_password=hashed_password,
            is_active=user_data.is_active,
            is_superuser=user_data.is_superuser
        )
        user.save()
        
        # Convert to response format
        return UserResponse(**user.to_dict())
        
    except NotUniqueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name or email already exists"
        )
    except (ValidationError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/", response_model=List[UserResponse])
async def get_users(skip: int = 0, limit: int = 100, user_type: str = None):
    """Get all users with pagination"""
    query = User.objects
    if user_type:
        query = query.filter(user_type=user_type)
    users = query.skip(skip).limit(limit)
    return [UserResponse(**user.to_dict()) for user in users]


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Get a specific user by ID"""
    try:
        user = User.objects.get(id=user_id)
        return UserResponse(**user.to_dict())
    except DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )


@router.get("/bots/all", response_model=List[UserResponse])
async def get_all_bot_users(
    current_user: User = Depends(get_current_user_from_token)
):
    """
    Get all bot users from the collection
    Requires authentication - any valid user can access this endpoint
    """
    # Check if user is authenticated
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # Check if user is active
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is not active"
        )
    
    try:
        # Get all bot users
        bot_users = User.objects.filter(user_type='bot', is_active=True)
        return [UserResponse(**user.to_dict()) for user in bot_users]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving bot users: {str(e)}"
        )





@router.put("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, user_data: UserUpdate):
    """Update a user"""
    try:
        user = User.objects.get(id=user_id)
        
        # Update fields if provided
        update_data = user_data.model_dump(exclude_unset=True)
        
        # Handle password hashing if password is being updated
        if "password" in update_data:
            update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
        
        # Update user
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.save()
        return UserResponse(**user.to_dict())
        
    except DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str):
    """Delete a user"""
    try:
        user = User.objects.get(id=user_id)
        user.delete()
    except DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )


@router.put("/{user_id}/document-filter", response_model=UserResponse)
async def update_bot_document_filter(
    user_id: str,
    request: DocumentFilterUpdate,
    current_user: User = Depends(get_current_user_from_token)
):
    """
    Update document filter for a bot user
    Only superusers can update document filters
    """
    try:
        # Check if current user is superuser
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only superusers can update document filters"
            )
        
        # Get the target user
        target_user = User.objects.get(id=user_id)
        
        # Check if target user is a bot
        if target_user.user_type != 'bot':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document filter can only be set for bot users"
            )
        
        # Update the document filter
        target_user.document_filter = request.document_filter
        target_user.save()
        
        return UserResponse(**target_user.to_dict())
        
    except DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/bulk-update-document-filters")
async def bulk_update_bot_document_filters(
    request: BulkDocumentFilterUpdate,
    current_user: User = Depends(get_current_user_from_token)
):
    """
    Bulk update document filters for multiple bot users
    Only superusers can perform bulk updates
    """
    try:
        # Check if current user is superuser
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only superusers can perform bulk updates"
            )
        
        updated_users = []
        errors = []
        
        for update in request.updates:
            user_id = update.get('user_id')
            document_filter = update.get('document_filter')
            
            if not user_id or document_filter is None:
                errors.append(f"Missing user_id or document_filter in update: {update}")
                continue
            
            try:
                target_user = User.objects.get(id=user_id)
                
                # Check if target user is a bot
                if target_user.user_type != 'bot':
                    errors.append(f"User {user_id} is not a bot user")
                    continue
                
                # Update the document filter
                target_user.document_filter = document_filter
                target_user.save()
                updated_users.append(UserResponse(**target_user.to_dict()))
                
            except DoesNotExist:
                errors.append(f"User {user_id} not found")
            except Exception as e:
                errors.append(f"Error updating user {user_id}: {str(e)}")
        
        return {
            "updated_users": updated_users,
            "errors": errors,
            "success_count": len(updated_users),
            "error_count": len(errors)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/generate-token", response_model=TokenResponse)
async def generate_token(
    token_request: TokenRequest,
    current_user: User = Depends(get_current_user_from_token)
):
    """
    Generate JWT token - Public API with two authentication methods:
    1. Name/password authentication (for human users)
    2. Superuser token in Authorization header to generate token for any user
    """
    target_user = None
    
    # Method 1: Name/password authentication
    if token_request.name and token_request.password:
        target_user = authenticate_user(token_request.name, token_request.password)
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid name or password"
            )
    
    # Method 2: Superuser generates token for another user
    elif current_user and current_user.is_superuser:
        if token_request.target_user_id:
            try:
                target_user = User.objects.get(id=token_request.target_user_id)
            except DoesNotExist:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Target user not found"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="target_user_id required for superuser token generation"
            )
    
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Either provide name/password or valid superuser authorization"
        )
    
    if not target_user or not target_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is not active"
        )
    
    # Create long-lived token (30+ days)
    token_data = {
        "user_id": str(target_user.id),
        "name": target_user.name,
        "user_type": target_user.user_type
    }
    
    access_token, expires_at = create_access_token(
        data=token_data,
        long_lived=True
    )
    
    # Calculate expires_in seconds
    from datetime import datetime
    expires_in = int((expires_at - datetime.utcnow()).total_seconds())
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
        user_id=str(target_user.id),
        user_type=target_user.user_type
    )