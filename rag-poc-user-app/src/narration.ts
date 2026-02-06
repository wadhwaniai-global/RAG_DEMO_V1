import AWS from 'aws-sdk';

// Configure AWS SDK
const configureAWS = () => {
  const accessKeyId = process.env.REACT_APP_AWS_ACCESS_KEY_ID;
  const secretAccessKey = process.env.REACT_APP_AWS_SECRET_ACCESS_KEY;
  const region = process.env.REACT_APP_AWS_REGION || 'us-east-1';

  if (!accessKeyId || !secretAccessKey) {
    console.error('AWS credentials not found in environment variables');
    return null;
  }

  AWS.config.update({
    accessKeyId,
    secretAccessKey,
    region,
  });

  return new AWS.Polly();
};

export interface NarrationState {
  isPlaying: boolean;
  isPaused: boolean;
  currentTime: number;
  duration: number;
  audioUrl: string | null;
  error: string | null;
}

export class NarrationService {
  private polly: AWS.Polly | null = null;
  private audioElement: HTMLAudioElement | null = null;
  private currentMessageId: string | null = null;
  private audioCache: Map<string, string> = new Map();
  private currentStateCallback: ((state: Partial<NarrationState>) => void) | null = null;
  private globalStopCallback: (() => void) | null = null;

  constructor() {
    this.polly = configureAWS();
  }

  // Set a global callback that gets called when any narration stops
  setGlobalStopCallback(callback: () => void) {
    this.globalStopCallback = callback;
  }

  async synthesizeSpeech(text: string, messageId: string): Promise<string> {
    if (!this.polly) {
      console.error('AWS Polly not configured. Missing environment variables:');
      console.error('- REACT_APP_AWS_ACCESS_KEY_ID');
      console.error('- REACT_APP_AWS_SECRET_ACCESS_KEY');
      console.error('- REACT_APP_AWS_REGION');
      throw new Error('AWS Polly not configured. Please add AWS credentials to your .env file.');
    }

    // Check if audio is already cached
    const cachedAudio = this.audioCache.get(messageId);
    if (cachedAudio) {
      console.log('Using cached audio for message:', messageId);
      return cachedAudio;
    }

    const params: AWS.Polly.SynthesizeSpeechInput = {
      Text: text,
      OutputFormat: 'mp3',
      VoiceId: 'Joanna', // Default female English voice
      Engine: 'neural', // Use neural engine for better quality
    };

    console.log('Synthesizing speech with AWS Polly...');
    try {
      const result = await this.polly.synthesizeSpeech(params).promise();

      if (result.AudioStream) {
        // Convert AudioStream to blob URL
        // AWS SDK v2 returns AudioStream as Buffer or Uint8Array
        let audioBlob: Blob;

        if (result.AudioStream instanceof Uint8Array || result.AudioStream instanceof ArrayBuffer) {
          audioBlob = new Blob([result.AudioStream], { type: 'audio/mpeg' });
        } else if (Buffer.isBuffer(result.AudioStream)) {
          // Node.js Buffer (shouldn't happen in browser but handle it)
          audioBlob = new Blob([result.AudioStream], { type: 'audio/mpeg' });
        } else if (result.AudioStream instanceof Blob) {
          // Already a Blob
          audioBlob = result.AudioStream;
        } else {
          // Fallback: try to convert to Uint8Array
          const buffer = new Uint8Array(result.AudioStream as any);
          audioBlob = new Blob([buffer], { type: 'audio/mpeg' });
        }

        const audioUrl = URL.createObjectURL(audioBlob);

        // Cache the audio URL
        this.audioCache.set(messageId, audioUrl);
        console.log('Speech synthesis successful, audio cached. Blob size:', audioBlob.size, 'bytes');

        return audioUrl;
      } else {
        throw new Error('No audio stream received from Polly');
      }
    } catch (error) {
      console.error('Error synthesizing speech with AWS Polly:', error);
      if (error instanceof Error) {
        throw new Error(`AWS Polly error: ${error.message}. Please verify your AWS credentials.`);
      }
      throw error;
    }
  }

  async playNarration(
    text: string,
    messageId: string,
    onStateChange: (state: Partial<NarrationState>) => void
  ): Promise<void> {
    console.log('Starting narration for message:', messageId);
    try {
      // Stop current narration if playing (this will trigger global stop callback)
      if (this.currentMessageId && this.currentMessageId !== messageId) {
        console.log('Stopping previous narration:', this.currentMessageId);
        this.stopNarration();
      }

      // Store the state callback for this narration
      this.currentStateCallback = onStateChange;

      onStateChange({ isPlaying: false, isPaused: false, error: null });

      // Get or synthesize audio
      console.log('Synthesizing speech for text length:', text.length);
      const audioUrl = await this.synthesizeSpeech(text, messageId);

      // Create audio element
      console.log('Creating audio element with URL:', audioUrl.substring(0, 50) + '...');
      this.audioElement = new Audio(audioUrl);
      this.currentMessageId = messageId;

      // Wait for audio to be ready to play
      await new Promise<void>((resolve, reject) => {
        const audio = this.audioElement!;

        // Set up event listeners
        const onCanPlay = () => {
          console.log('Audio can play - ready state:', audio.readyState);
          resolve();
        };

        const onError = (e: Event) => {
          const target = e.target as HTMLAudioElement;
          let errorMessage = 'Audio playback error';

          if (target.error) {
            switch (target.error.code) {
              case target.error.MEDIA_ERR_ABORTED:
                errorMessage = 'Audio playback aborted';
                break;
              case target.error.MEDIA_ERR_NETWORK:
                errorMessage = 'Network error while loading audio';
                break;
              case target.error.MEDIA_ERR_DECODE:
                errorMessage = 'Audio decoding error';
                break;
              case target.error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                errorMessage = 'Audio format not supported';
                break;
            }
            console.error('Audio element error:', errorMessage, target.error);
          }

          onStateChange({
            isPlaying: false,
            isPaused: false,
            error: errorMessage
          });

          reject(new Error(errorMessage));
        };

        // If already ready, resolve immediately
        if (audio.readyState >= 3) { // HAVE_FUTURE_DATA or HAVE_ENOUGH_DATA
          console.log('Audio already ready');
          resolve();
          return;
        }

        // Otherwise wait for canplay event
        audio.addEventListener('canplay', onCanPlay, { once: true });
        audio.addEventListener('error', onError, { once: true });

        // Load the audio
        audio.load();
      });

      // Now set up persistent event listeners
      this.audioElement.addEventListener('loadedmetadata', () => {
        onStateChange({
          duration: this.audioElement?.duration || 0,
          audioUrl,
        });
      });

      this.audioElement.addEventListener('timeupdate', () => {
        onStateChange({
          currentTime: this.audioElement?.currentTime || 0,
        });
      });

      this.audioElement.addEventListener('play', () => {
        onStateChange({ isPlaying: true, isPaused: false });
      });

      this.audioElement.addEventListener('pause', () => {
        onStateChange({ isPlaying: false, isPaused: true });
      });

      this.audioElement.addEventListener('ended', () => {
        onStateChange({
          isPlaying: false,
          isPaused: false,
          currentTime: 0
        });
      });

      this.audioElement.addEventListener('error', (e) => {
        const target = e.target as HTMLAudioElement;
        let errorMessage = 'Audio playback error';

        if (target.error) {
          switch (target.error.code) {
            case target.error.MEDIA_ERR_ABORTED:
              errorMessage = 'Audio playback aborted';
              break;
            case target.error.MEDIA_ERR_NETWORK:
              errorMessage = 'Network error while loading audio';
              break;
            case target.error.MEDIA_ERR_DECODE:
              errorMessage = 'Audio decoding error';
              break;
            case target.error.MEDIA_ERR_SRC_NOT_SUPPORTED:
              errorMessage = 'Audio format not supported';
              break;
          }
          console.error('Audio element error:', errorMessage, target.error);
        }

        onStateChange({
          isPlaying: false,
          isPaused: false,
          error: errorMessage
        });
      });

      // Start playing
      console.log('Starting audio playback...');
      console.log('Audio element ready state:', this.audioElement.readyState);
      console.log('Audio element can play type (audio/mpeg):', this.audioElement.canPlayType('audio/mpeg'));

      try {
        await this.audioElement.play();
        console.log('Audio playback started successfully and should be audible now!');
      } catch (playError) {
        console.error('Failed to play audio:', playError);
        throw playError;
      }

    } catch (error) {
      console.error('Error playing narration:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred while playing narration';
      onStateChange({
        isPlaying: false,
        isPaused: false,
        error: errorMessage
      });
    }
  }

  pauseNarration(): void {
    if (this.audioElement && !this.audioElement.paused) {
      this.audioElement.pause();
    }
  }

  resumeNarration(): void {
    if (this.audioElement && this.audioElement.paused) {
      this.audioElement.play();
    }
  }

  stopNarration(): void {
    if (this.audioElement) {
      this.audioElement.pause();
      this.audioElement.currentTime = 0;
      this.audioElement = null;
    }

    // Notify about state change before clearing references
    if (this.currentStateCallback) {
      this.currentStateCallback({
        isPlaying: false,
        isPaused: false,
        currentTime: 0,
      });
    }

    // Trigger global stop callback to update all message states
    if (this.globalStopCallback) {
      this.globalStopCallback();
    }

    this.currentMessageId = null;
    this.currentStateCallback = null;
  }

  restartNarration(): void {
    if (this.audioElement) {
      this.audioElement.currentTime = 0;
      if (this.audioElement.paused) {
        this.audioElement.play();
      }
      
      // Notify about state change
      if (this.currentStateCallback) {
        this.currentStateCallback({
          currentTime: 0,
          isPlaying: !this.audioElement.paused,
          isPaused: this.audioElement.paused,
        });
      }
    }
  }

  getCurrentMessageId(): string | null {
    return this.currentMessageId;
  }

  isCurrentlyPlaying(messageId: string): boolean {
    return !!(this.currentMessageId === messageId && 
              this.audioElement && 
              !this.audioElement.paused);
  }

  isCurrentlyPaused(messageId: string): boolean {
    return !!(this.currentMessageId === messageId && 
              this.audioElement && 
              this.audioElement.paused);
  }

  // Clean up cached audio URLs to prevent memory leaks
  clearCache(): void {
    this.audioCache.forEach(url => URL.revokeObjectURL(url));
    this.audioCache.clear();
  }
}

// Export singleton instance
export const narrationService = new NarrationService();