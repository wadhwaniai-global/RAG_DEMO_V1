from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, Literal, List, Dict
from datetime import datetime


class UserBase(BaseModel):
    """Base user schema with common fields"""
    name: str = Field(..., min_length=1, max_length=50)
    email: Optional[EmailStr] = None
    description: Optional[str] = Field(None, max_length=500)
    user_type: Literal['human', 'bot'] = 'human'
    document_filter: Optional[List[str]] = Field(None, description="List of document filters for bot users")
    is_active: bool = True


class UserCreate(UserBase):
    """Schema for creating a new user"""
    password: Optional[str] = Field(None, min_length=8, max_length=100)
    is_superuser: bool = False
    
    @validator('password')
    def validate_password_for_human(cls, v, values):
        """Validate that human users have a password and bot users don't"""
        user_type = values.get('user_type')
        if user_type == 'human' and not v:
            raise ValueError('Password is required for human users')
        if user_type == 'bot' and v:
            raise ValueError('Bot users should not have a password')
        return v
    
    @validator('email')
    def validate_email_for_human(cls, v, values):
        """Validate that human users have an email"""
        if values.get('user_type') == 'human' and not v:
            raise ValueError('Email is required for human users')
        return v
    
    @validator('is_superuser')
    def validate_superuser_requirements(cls, v, values):
        """Validate superuser requirements"""
        if v and values.get('user_type') == 'bot':
            raise ValueError('Bot users cannot be superusers')
        return v
    
    @validator('document_filter')
    def validate_document_filter(cls, v, values):
        """Validate document filter is only for bot users"""
        user_type = values.get('user_type')
        if v and user_type == 'human':
            raise ValueError('Document filter can only be set for bot users')
        # Ensure it's a list if provided
        if v and not isinstance(v, list):
            # For backward compatibility, convert single string to list
            return [v] if isinstance(v, str) else v
        return v


class UserUpdate(BaseModel):
    """Schema for updating a user"""
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    email: Optional[EmailStr] = None
    description: Optional[str] = Field(None, max_length=500)
    document_filter: Optional[List[str]] = Field(None, description="List of document filters for bot users")
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=8, max_length=100)
    # Note: user_type should not be updatable after creation for security


class UserInDB(UserBase):
    """Schema for user as stored in database"""
    id: str
    hashed_password: Optional[str] = None
    is_superuser: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserResponse(UserBase):
    """Schema for user response (without sensitive data)"""
    id: str
    is_superuser: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Schema for authentication token"""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Schema for token data"""
    name: Optional[str] = None
    user_id: Optional[str] = None
    user_type: Optional[str] = None


class TokenRequest(BaseModel):
    """Schema for token generation request"""
    name: Optional[str] = None
    password: Optional[str] = None
    target_user_id: Optional[str] = None  # For superuser token generation


class TokenResponse(BaseModel):
    """Schema for token response"""
    access_token: str
    token_type: str
    expires_in: int  # seconds
    user_id: str
    user_type: str


class DocumentFilterUpdate(BaseModel):
    """Schema for updating bot document filter"""
    document_filter: List[str] = Field(..., description="List of document filters", min_items=1)


class BulkDocumentFilterUpdate(BaseModel):
    """Schema for bulk document filter updates"""
    updates: List[Dict[str, List[str]]] = Field(..., description="List of {'user_id': ['filter1', 'filter2']} objects")