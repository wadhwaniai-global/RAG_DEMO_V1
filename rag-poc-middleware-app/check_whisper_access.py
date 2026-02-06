#!/usr/bin/env python3
"""
Script to check if OpenAI API key has access to Whisper API
"""

import os
import sys
import io
import wave
import struct
from pathlib import Path

try:
    import openai
except ImportError:
    print("❌ Error: openai package not installed. Please install it with:")
    print("   pip install openai")
    sys.exit(1)


def create_test_audio_file():
    """Create a minimal test audio file for Whisper API testing"""
    # Create a very short (1 second) mono audio file at 16kHz
    sample_rate = 16000
    duration = 1  # 1 second
    frequency = 440  # A4 note
    
    # Generate sine wave data
    samples = []
    for i in range(sample_rate * duration):
        t = float(i) / sample_rate
        wave_value = int(32767 * 0.1 * (2 * 3.14159 * frequency * t) % (2 * 3.14159))  # Low volume sine wave
        samples.append(struct.pack('<h', wave_value))
    
    # Create in-memory WAV file
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b''.join(samples))
    
    audio_buffer.seek(0)
    return audio_buffer


def check_whisper_access():
    """Test if the OpenAI API key has access to Whisper API"""
    
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No OpenAI API key found!")
        print("\nPlease set your API key as an environment variable:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr you can set it temporarily for this script:")
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            print("❌ No API key provided. Exiting.")
            return False
    
    # Initialize OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {str(e)}")
        return False
    
    # Create a test audio file
    print("🔧 Creating test audio file...")
    try:
        test_audio = create_test_audio_file()
        test_audio.name = "test_audio.wav"  # Required for the API
        print("✅ Test audio file created")
    except Exception as e:
        print(f"❌ Failed to create test audio file: {str(e)}")
        return False
    
    # Test Whisper API access
    print("🧪 Testing Whisper API access...")
    try:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=test_audio,
            response_format="text"
        )
        print("✅ Whisper API access confirmed!")
        print(f"📝 Test transcription result: '{response.strip()}'")
        print("   (Note: The result might be empty or nonsensical since we used a synthetic audio file)")
        return True
        
    except openai.AuthenticationError:
        print("❌ Authentication failed!")
        print("   • Check if your API key is correct")
        print("   • Verify the API key is not expired")
        return False
        
    except openai.PermissionDeniedError:
        print("❌ Permission denied!")
        print("   • Your API key doesn't have access to Whisper API")
        print("   • You might need to upgrade your OpenAI plan")
        print("   • Contact OpenAI support if you believe this is an error")
        return False
        
    except openai.RateLimitError:
        print("⚠️  Rate limit exceeded!")
        print("   • Your API key is valid but you've hit rate limits")
        print("   • This means you DO have access to Whisper API")
        print("   • Try again in a few minutes")
        return True
        
    except openai.APIError as e:
        print(f"❌ OpenAI API error: {str(e)}")
        print("   • This might be a temporary API issue")
        print("   • Try again in a few minutes")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False


def main():
    """Main function"""
    print("🎵 OpenAI Whisper API Access Checker")
    print("=" * 40)
    
    has_access = check_whisper_access()
    
    print("\n" + "=" * 40)
    if has_access:
        print("🎉 SUCCESS: Your OpenAI API key has access to Whisper API!")
    else:
        print("💔 FAILED: Your OpenAI API key does not have access to Whisper API")
        
    print("\n📚 Additional Information:")
    print("   • Whisper API pricing: https://openai.com/pricing")
    print("   • Whisper API documentation: https://platform.openai.com/docs/guides/speech-to-text")
    print("   • OpenAI Dashboard: https://platform.openai.com/account/usage")


if __name__ == "__main__":
    main()