import { SubmitQueryRequest, SubmitQueryResponse, GetMessagesRequest, GetMessagesResponse, LoginRequest, LoginResponse, JWTPayload, GetConversationsResponse } from './types';

// API Configuration
const API_BASE_URL = 'https://healthcare-agents.wadhwaniaiglobal.com/api/v1';


// Pagination size for message fetches (can be adjusted)
export const PAGE_SIZE = 10;

// Dynamic auth values - set after login
let AUTH_TOKEN: string | null = null;
let SENDER_ID: string | null = null;

// JWT Utility Functions
export const decodeJWT = (token: string): JWTPayload => {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    return JSON.parse(jsonPayload);
  } catch (error) {
    console.error('Error decoding JWT:', error);
    throw new Error('Invalid JWT token');
  }
};

// Auth management functions
export const setAuthToken = (token: string): void => {
  AUTH_TOKEN = token;
  const payload = decodeJWT(token);
  SENDER_ID = payload.user_id;

  // Store in localStorage for persistence
  localStorage.setItem('auth_token', token);
  localStorage.setItem('sender_id', payload.user_id);
};

export const getAuthToken = (): string | null => {
  if (!AUTH_TOKEN) {
    AUTH_TOKEN = localStorage.getItem('auth_token');
  }
  return AUTH_TOKEN;
};

export const getSenderId = (): string | null => {
  if (!SENDER_ID) {
    SENDER_ID = localStorage.getItem('sender_id');
  }
  return SENDER_ID;
};

export const clearAuth = (): void => {
  AUTH_TOKEN = null;
  SENDER_ID = null;
  localStorage.removeItem('auth_token');
  localStorage.removeItem('sender_id');
};

export const isAuthenticated = (): boolean => {
  const token = getAuthToken();
  if (!token) return false;

  try {
    const payload = decodeJWT(token);
    const currentTime = Math.floor(Date.now() / 1000);
    return payload.exp > currentTime;
  } catch {
    return false;
  }
};

// Login API
export const login = async (request: LoginRequest): Promise<LoginResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/users/generate-token`, {
      method: 'POST',
      headers: {
        'accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data: LoginResponse = await response.json();

    // Set the auth token and extract sender ID
    setAuthToken(data.access_token);

    return data;
  } catch (error) {
    console.error('Error during login:', error);
    throw error;
  }
};

export const submitQuery = async (
  request: SubmitQueryRequest,
  receiverId: string,
  options?: { signal?: AbortSignal }
): Promise<SubmitQueryResponse> => {
  try {
    const authToken = getAuthToken();
    const senderId = getSenderId();

    if (!authToken || !senderId) {
      throw new Error('Not authenticated. Please login first.');
    }

    const response = await fetch(`${API_BASE_URL}/chats/`, {
      method: 'POST',
      headers: {
        'accept': 'application/json',
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sender_id: senderId,
        receiver_id: receiverId,
        message: request.query,
      }),
      signal: options?.signal, // Support for AbortController
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data: SubmitQueryResponse = await response.json();
    return data;
  } catch (error) {
    // Check if error is due to abort
    if (error instanceof Error && error.name === 'AbortError') {
      console.log('Request was cancelled by user');
      throw error;
    }
    console.error('Error submitting query:', error);
    throw error;
  }
};

export const getMessages = async (request: GetMessagesRequest, receiverId: string): Promise<GetMessagesResponse> => {
  try {
    const authToken = getAuthToken();
    const senderId = getSenderId();

    if (!authToken || !senderId) {
      throw new Error('Not authenticated. Please login first.');
    }

    // For initial load (afterOffset is undefined) we want the latest messages.
    // The backend supports fetching messages before an offset. To get the latest N
    // messages we pass a very large offset and before=true. For subsequent
    // polls we fetch messages after the last known offset (before=false).
    const url = new URL(`${API_BASE_URL}/chats/messages/`);
    const isInitialLoad = typeof request.afterOffset === 'undefined' || request.afterOffset === null;
    const offset = isInitialLoad ? Number.MAX_SAFE_INTEGER : request.afterOffset ?? 0;
    const before = isInitialLoad ? true : (request.before ?? false);
    url.searchParams.append('sender_id', senderId);
    url.searchParams.append('receiver_id', receiverId);
    url.searchParams.append('offset', offset.toString());
    url.searchParams.append('limit', String(PAGE_SIZE));
    url.searchParams.append('before', String(before));

    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: {
        'accept': 'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data: GetMessagesResponse = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching messages:', error);
    throw error;
  }
};

export const getConversations = async (): Promise<GetConversationsResponse> => {
  try {
    const authToken = getAuthToken();

    if (!authToken) {
      throw new Error('Not authenticated. Please login first.');
    }

    const response = await fetch(`${API_BASE_URL}/chats/conversations/`, {
      method: 'GET',
      headers: {
        'accept': 'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data: GetConversationsResponse = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching conversations:', error);
    throw error;
  }
};

export const transcribeAudio = async (audioFile: File): Promise<{ text: string }> => {
  try {
    const authToken = getAuthToken();

    if (!authToken) {
      throw new Error('Not authenticated. Please login first.');
    }

    const formData = new FormData();
    formData.append('file', audioFile);

    const response = await fetch(`${API_BASE_URL}/whisper/transcribe`, {
      method: 'POST',
      headers: {
        'accept': 'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error transcribing audio:', error);
    throw error;
  }
};

export const getAvailableBots = async (): Promise<any[]> => {
  try {
    const authToken = getAuthToken();

    if (!authToken) {
      throw new Error('Not authenticated. Please login first.');
    }

    const response = await fetch(`${API_BASE_URL}/users/bots/all`, {
      method: 'GET',
      headers: {
        'accept': 'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching available bots:', error);
    throw error;
  }
}; 
