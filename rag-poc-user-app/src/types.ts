export interface Source {
  document_name: string;
  page_number: number;
  relevance_score: number;
}

export interface RetrievalMetadata {
  total_documents_searched: number;
  query_expansion_used: boolean;
  hybrid_search_used: boolean;
  reranking_used: boolean;
}

export interface MessageContent {
  text: string;
  confidence_score: number | null;
  sources: Source[] | null;
  retrieval_metadata: RetrievalMetadata | null;
  processing_time: number | null;
  status: string | null;
}

export interface Message {
  sender_id: string;
  receiver_id: string;
  message: MessageContent | string; // Support both new and old format for backwards compatibility
  id: string;
  offset: number;
  is_read: boolean;
  is_delivered: boolean;
  is_seen: boolean;
  is_deleted: boolean;
  is_archived: boolean;
  is_pinned: boolean;
  created_at: string;
  updated_at: string;
}

export interface DefaultChips extends Message {
  intro: string | null;
  chips: string[] | null;
  defaultChips: boolean
}

export interface QuestionChips extends Message {
  category: string;
  questions: string[];
  questionChips: boolean;
}

export type MessageOrDefaultChips = Message | DefaultChips | QuestionChips;

export interface SubmitQueryRequest {
  query: string;
}

export interface SubmitQueryResponse {
  sender_id: string;
  receiver_id: string;
  message: MessageContent | string; // Support both new and old format for backwards compatibility
  id: string;
  offset: number;
  is_read: boolean;
  is_delivered: boolean;
  is_seen: boolean;
  is_deleted: boolean;
  is_archived: boolean;
  is_pinned: boolean;
  created_at: string;
  updated_at: string;
}

export interface GetMessagesRequest {
  afterOffset?: number;
  // When true, fetch messages before the provided offset (for loading older messages).
  // When false (default) the API fetches messages after the provided offset (newer messages).
  before?: boolean;
}

export interface GetMessagesResponse {
  messages: Message[];
  total_count: number;
  offset: number;
  limit: number;
  has_more: boolean;
}

export interface LoginRequest {
  name: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

export interface JWTPayload {
  user_id: string;
  name: string;
  user_type: string;
  exp: number;
}

// Conversation types
export interface Participant {
  id: string;
  name: string;
  email: string;
  description: string;
  user_type: string;
  is_active: boolean;
}

export interface Conversation {
  participant: Participant;
  message_count: number;
  last_message: Message;
}

export interface GetConversationsResponse {
  conversations: Conversation[];
  total_conversations: number;
} 