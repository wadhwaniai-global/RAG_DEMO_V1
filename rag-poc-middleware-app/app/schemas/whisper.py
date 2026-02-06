from pydantic import BaseModel, Field
from typing import Optional


class WhisperTranscriptionResponse(BaseModel):
    """Schema for Whisper API transcription response"""
    text: str = Field(..., description="Transcribed text from the audio file")
    language: Optional[str] = Field(None, description="Detected language of the audio")
    duration: Optional[float] = Field(None, ge=0.0, description="Duration of the audio in seconds")
    
    class Config:
        from_attributes = True