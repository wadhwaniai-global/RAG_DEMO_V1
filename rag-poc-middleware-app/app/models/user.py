from mongoengine import Document, StringField, EmailField, BooleanField, DateTimeField, ListField
from datetime import datetime
from typing import Optional


class User(Document):
    """User model using MongoEngine"""

    # Fields
    name = StringField(required=True, unique=True, max_length=50)
    email = EmailField(unique=True)  # Optional for bot users
    description = StringField(max_length=500)  # Description for all users
    hashed_password = StringField()  # Mandatory for human users, not set for bot users
    user_type = StringField(required=True, choices=['human', 'bot'], default='human')
    document_filter = ListField(StringField(max_length=100))  # Document filters for bot users (list)
    is_active = BooleanField(default=True)
    is_superuser = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    
    # Meta configuration
    meta = {
        'collection': 'users',
        'indexes': [
            'name',
            'email',
            'user_type',
            'created_at'
        ]
    }
    
    def clean(self):
        """Validate user data before saving"""
        if self.user_type == 'human':
            if not self.name:
                raise ValueError("Human users must have a name")
            if not self.hashed_password:
                raise ValueError("Human users must have a password")
            if not self.email:
                raise ValueError("Human users must have an email")
        elif self.user_type == 'bot':
            if not self.name:
                raise ValueError("Bot users must have a name")
            if self.hashed_password:
                raise ValueError("Bot users should not have a password")
    
    def save(self, *args, **kwargs):
        """Override save to update updated_at timestamp"""
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
    
    def __str__(self):
        return f"User(name={self.name}, email={self.email})"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'id': str(self.id),
            'name': self.name,
            'email': self.email,
            'description': self.description,
            'user_type': self.user_type,
            'document_filter': self.document_filter,
            'is_active': self.is_active,
            'is_superuser': self.is_superuser,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }