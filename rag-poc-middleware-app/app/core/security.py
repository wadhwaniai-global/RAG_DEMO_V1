from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings

# More lenient bcrypt configuration to match your old setup
pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__default_rounds=12,  # Try different rounds
    bcrypt__min_rounds=4,
    bcrypt__max_rounds=31
)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"Password verification error: {e}")
        # Try with truncated password as fallback
        if len(plain_password.encode('utf-8')) > 72:
            try:
                return pwd_context.verify(plain_password[:72], hashed_password)
            except Exception as e2:
                print(f"Truncated password verification error: {e2}")
        return False

def get_password_hash(password: str) -> str:
    """Hash a password"""
    # Truncate password to 72 bytes if it's longer (bcrypt limit)
    if len(password.encode('utf-8')) > 72:
        password = password[:72]
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None, long_lived: bool = False):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    elif long_lived:
        expire = datetime.utcnow() + timedelta(days=settings.LONG_ACCESS_TOKEN_EXPIRE_DAYS)
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt, expire

def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None

def authenticate_user(name: str, password: str):
    """Authenticate a user with username and password"""
    from app.models.user import User
    
    try:
        user = User.objects.get(name=name, user_type='human')
        if user and verify_password(password, user.hashed_password):
            return user
    except User.DoesNotExist:
        pass
    return None

# Security schemes
security = HTTPBearer(auto_error=False)

def get_current_user_from_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    from app.models.user import User
    
    if not credentials:
        return None
    
    token_data = verify_token(credentials.credentials)
    if not token_data:
        return None
    
    try:
        user = User.objects.get(id=token_data.get("user_id"))
        return user
    except User.DoesNotExist:
        return None

def require_superuser(current_user = Depends(get_current_user_from_token)):
    """Require current user to be a superuser"""
    if not current_user or not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser access required"
        )
    return current_user