from fastapi import APIRouter

from app.api.v1.endpoints import chat, users, whisper

api_router = APIRouter()
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(chat.router, prefix="/chats", tags=["chats"])
api_router.include_router(whisper.router, prefix="/whisper", tags=["whisper"])