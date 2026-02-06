from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File
import logging
import os
from openai import OpenAI
from app.models.user import User
from app.schemas.whisper import WhisperTranscriptionResponse
from app.core.security import get_current_user_from_token

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@router.post("/transcribe", response_model=WhisperTranscriptionResponse, status_code=status.HTTP_200_OK)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    current_user: User = Depends(get_current_user_from_token)
):
    """Transcribe audio file using OpenAI Whisper API"""
    # Ensure user is authenticated
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # Validate file type (Whisper supports many formats)
    allowed_extensions = {
        '.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.flac', '.ogg'
    }
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (25MB limit for Whisper API)
    file_content = await file.read()
    if len(file_content) > 25 * 1024 * 1024:  # 25MB
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds 25MB limit"
        )
    
    try:
        # Create a temporary file-like object for OpenAI API
        file.file.seek(0)  # Reset file pointer
        
        # Call OpenAI Whisper API
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(file.filename, file_content),
            response_format="verbose_json"
        )
        
        logger.info(f"Successfully transcribed audio file: {file.filename} for user: {current_user.id}")
        
        return WhisperTranscriptionResponse(
            text=transcript.text,
            language=getattr(transcript, 'language', None),
            duration=getattr(transcript, 'duration', None)
        )
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to transcribe audio file"
        )