import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { getBotConfiguration } from '../constants/constants';

import {
  Box,
  Typography,
  ListItem,
  Avatar,
  Chip,
  Collapse,
  Card,
  CardContent,
  Grid,
  Paper,
  IconButton,
} from '@mui/material';
import {
  Person as PersonIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Info as InfoIcon,
  Schedule as ScheduleIcon,
  Source as SourceIcon,
  DataObject as DataObjectIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  Replay as ReplayIcon,
  VolumeUp as VolumeUpIcon,
  VolumeUpOutlined as VolumeUpOutlinedIcon,
} from '@mui/icons-material';
import { Message, MessageContent, DefaultChips, QuestionChips, MessageOrDefaultChips } from '../types';
import { NarrationState } from '../narration';

/**
 * A complete and robust parser for message content. It handles various
 * data formats, intelligently extracts key information, and provides safe
 * fallbacks to prevent UI errors or the display of raw JSON.
 */
export const parseMessageContent = (message: MessageContent | string): MessageContent => {
  // Rule 1: If it's already a valid, parsed object, do nothing.
  if (typeof message === 'object' && message !== null) {
    return message;
  }

  // Rule 2: If it's not a string or object, it's an unsupported format.
  if (typeof message !== 'string') {
    console.warn('Received unsupported message content type:', typeof message);
    return {
      text: '[Unsupported message format]',
      confidence_score: null,
      sources: null,
      retrieval_metadata: null,
      processing_time: null,
      status: 'error',
    };
  }

  // Rule 3: Attempt to parse the string as JSON.
  try {
    const parsed = JSON.parse(message);

    // Rule 4: Handle cases where JSON is valid but not an object (e.g., "123").
    if (typeof parsed !== 'object' || parsed === null) {
      return { text: message, confidence_score: null, sources: null, retrieval_metadata: null, processing_time: null, status: 'plain_text' };
    }

    // Rule 5: Intelligently find the main text, checking for common key names.
    const mainText = parsed.text || parsed.answer;

    if (typeof mainText === 'string') {
      // It's a valid RAG response. Normalize the structure for consistency.
      return {
        text: mainText,
        confidence_score: parsed.confidence_score || null,
        sources: parsed.sources || null,
        retrieval_metadata: parsed.retrieval_metadata || null,
        processing_time: parsed.processing_time || null,
        status: parsed.status || null,
      };
    }

    // Rule 6: Handle unrecognized but valid JSON objects gracefully.
    // This prevents showing raw JSON in the chat bubble.
    return {
      text: "[Received structured data without a displayable message]",
      confidence_score: parsed.confidence_score || null,
      sources: parsed.sources || [{ document_name: "Raw Data", page_number: 1, relevance_score: 1, content: JSON.stringify(parsed, null, 2) }],
      retrieval_metadata: parsed,
      processing_time: parsed.processing_time || null,
      status: 'unknown_format',
    };

  } catch (error) {
    // Rule 7: If JSON.parse fails, the string is treated as plain text.
    return { text: message, confidence_score: null, sources: null, retrieval_metadata: null, processing_time: null, status: 'plain_text' };
  }
};

// Helper function to extract text from message (backwards compatibility)
const getMessageText = (message: MessageContent | string): string => {
  const parsedMessage = parseMessageContent(message);
  return parsedMessage.text;
};

// Helper function to check if message has metadata
const hasMetadata = (message: MessageContent | string): boolean => {
  const parsedMessage = parseMessageContent(message);
  return !!(
    parsedMessage.confidence_score !== null ||
    parsedMessage.sources !== null ||
    parsedMessage.retrieval_metadata !== null ||
    parsedMessage.processing_time !== null ||
    parsedMessage.status !== null
  );
};

// Helper function to format time
const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

// Metadata dropdown component
const MessageMetadata: React.FC<{ messageContent: MessageContent }> = ({ messageContent }) => {
  const [expanded, setExpanded] = useState(false);

  const toggleExpanded = () => {
    setExpanded(!expanded);
  };

  return (
    <Box sx={{ mt: 1 }}>
      <IconButton
        onClick={toggleExpanded}
        size="small"
        sx={{
          padding: 0.5,
          color: 'text.secondary',
          '&:hover': {
            backgroundColor: 'rgba(0, 0, 0, 0.04)',
          },
        }}
      >
        <InfoIcon fontSize="small" sx={{ mr: 0.5 }} />
        <Typography variant="caption" sx={{ mr: 0.5 }}>
          Details
        </Typography>
        {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
      </IconButton>

      <Collapse in={expanded}>
        <Card
          elevation={0}
          sx={{
            mt: 1,
            backgroundColor: 'rgba(0, 0, 0, 0.02)',
            border: '1px solid rgba(0, 0, 0, 0.08)',
          }}
        >
          <CardContent sx={{ padding: 1.5, '&:last-child': { paddingBottom: 1.5 } }}>
            <Grid container spacing={2}>
              {/* Confidence Score */}
              {messageContent.confidence_score !== null && (
                <Grid item xs={12} sm={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <InfoIcon fontSize="small" sx={{ mr: 1, color: '#2c2c2c' }} />
                    <Typography variant="subtitle2" fontWeight="bold">
                      Confidence
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {(messageContent.confidence_score * 100).toFixed(1)}%
                  </Typography>
                </Grid>
              )}

              {/* Processing Time */}
              {messageContent.processing_time !== null && (
                <Grid item xs={12} sm={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <ScheduleIcon fontSize="small" sx={{ mr: 1, color: '#2c2c2c' }} />
                    <Typography variant="subtitle2" fontWeight="bold">
                      Processing Time
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {messageContent.processing_time.toFixed(2)}s
                  </Typography>
                </Grid>
              )}

              {/* Status */}
              {messageContent.status !== null && (
                <Grid item xs={12} sm={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <DataObjectIcon fontSize="small" sx={{ mr: 1, color: '#2c2c2c' }} />
                    <Typography variant="subtitle2" fontWeight="bold">
                      Status
                    </Typography>
                  </Box>
                  <Chip
                    label={messageContent.status}
                    size="small"
                    color={messageContent.status === 'success' ? 'success' : 'default'}
                    variant="outlined"
                  />
                </Grid>
              )}

              {/* Sources */}
              {messageContent.sources && messageContent.sources.length > 0 && (
                <Grid item xs={12}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <SourceIcon fontSize="small" sx={{ mr: 1, color: '#2c2c2c' }} />
                    <Typography variant="subtitle2" fontWeight="bold">
                      Sources ({messageContent.sources.length})
                    </Typography>
                  </Box>
                  {messageContent.sources.map((source, index) => (
                    <Paper
                      key={index}
                      elevation={0}
                      sx={{
                        p: 1,
                        mb: 1,
                        backgroundColor: 'background.paper',
                        border: '1px solid rgba(0, 0, 0, 0.08)',
                      }}
                    >
                      <Typography variant="caption" fontWeight="bold" display="block">
                        {source.document_name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Page {source.page_number} • Relevance: {(source.relevance_score * 100).toFixed(1)}%
                      </Typography>
                    </Paper>
                  ))}
                </Grid>
              )}

              {/* Retrieval Metadata */}
              {messageContent.retrieval_metadata && (
                <Grid item xs={12}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <DataObjectIcon fontSize="small" sx={{ mr: 1, color: '#2c2c2c' }} />
                    <Typography variant="subtitle2" fontWeight="bold">
                      Retrieval Details
                    </Typography>
                  </Box>
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Typography variant="caption" display="block">
                        Documents Searched
                      </Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {messageContent.retrieval_metadata.total_documents_searched}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" display="block">
                        Hybrid Search
                      </Typography>
                      <Chip
                        label={messageContent.retrieval_metadata.hybrid_search_used ? 'Yes' : 'No'}
                        size="small"
                        color={messageContent.retrieval_metadata.hybrid_search_used ? 'success' : 'default'}
                        variant="outlined"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" display="block">
                        Query Expansion
                      </Typography>
                      <Chip
                        label={messageContent.retrieval_metadata.query_expansion_used ? 'Yes' : 'No'}
                        size="small"
                        color={messageContent.retrieval_metadata.query_expansion_used ? 'success' : 'default'}
                        variant="outlined"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" display="block">
                        Reranking
                      </Typography>
                      <Chip
                        label={messageContent.retrieval_metadata.reranking_used ? 'Yes' : 'No'}
                        size="small"
                        color={messageContent.retrieval_metadata.reranking_used ? 'success' : 'default'}
                        variant="outlined"
                      />
                    </Grid>
                  </Grid>
                </Grid>
              )}
            </Grid>
          </CardContent>
        </Card>
      </Collapse>
    </Box>
  );
};

interface NarrationControlsProps {
  messageText: string;
  messageId: string;
  narrationState: NarrationState | undefined;
  onStartNarration: (messageText: string, messageId: string) => void;
  onPauseNarration: () => void;
  onResumeNarration: () => void;
  onStopNarration: () => void;
  onRestartNarration: () => void;
}

const NarrationControls: React.FC<NarrationControlsProps> = ({
  messageText,
  messageId,
  narrationState,
  onStartNarration,
  onPauseNarration,
  onResumeNarration,
  onStopNarration,
  onRestartNarration,
}) => {
  const isPlaying = narrationState?.isPlaying || false;
  const isPaused = narrationState?.isPaused || false;
  const currentTime = narrationState?.currentTime || 0;
  const duration = narrationState?.duration || 0;
  const error = narrationState?.error;

  if (error) {
    return (
      <Typography
        variant="caption"
        sx={{
          color: 'error.main',
          fontSize: '0.7rem',
          opacity: 0.8,
        }}
      >
        Narration unavailable
      </Typography>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 0.5,
        padding: 0.5,
        borderRadius: 1,
        backgroundColor: isPlaying ? 'rgba(8, 145, 178, 0.08)' : 'transparent',
        transition: 'background-color 0.3s ease',
      }}
    >
      {/* Play/Pause/Resume Button */}
      <IconButton
        size="small"
        onClick={() => {
          if (isPlaying) {
            onPauseNarration();
          } else if (isPaused) {
            onResumeNarration();
          } else {
            onStartNarration(messageText, messageId);
          }
        }}
        sx={{
          width: 28,
          height: 28,
          color: isPlaying ? 'primary.main' : 'text.secondary',
          '&:hover': {
            backgroundColor: 'rgba(44, 44, 44, 0.08)',
          },
        }}
      >
        {isPlaying ? (
          <PauseIcon fontSize="small" />
        ) : (
          <VolumeUpOutlinedIcon fontSize="small" />
        )}
      </IconButton>

      {/* Stop Button */}
      {(isPlaying || isPaused) && (
        <IconButton
          size="small"
          onClick={onStopNarration}
          sx={{
            width: 28,
            height: 28,
            color: 'error.main',
            '&:hover': {
              backgroundColor: 'rgba(244, 67, 54, 0.08)',
            },
          }}
        >
          <StopIcon fontSize="small" />
        </IconButton>
      )}

      {/* Restart Button */}
      {(isPlaying || isPaused) && (
        <IconButton
          size="small"
          onClick={onRestartNarration}
          sx={{
            width: 28,
            height: 28,
            color: 'text.secondary',
            '&:hover': {
              backgroundColor: 'rgba(0, 0, 0, 0.04)',
            },
          }}
        >
          <ReplayIcon fontSize="small" />
        </IconButton>
      )}

      {/* Progress Indicator */}
      {(isPlaying || isPaused) && duration > 0 && (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            ml: 1,
          }}
        >
          <VolumeUpIcon
            fontSize="small"
            sx={{
              color: isPlaying ? 'primary.main' : 'text.secondary',
              opacity: isPlaying ? 1 : 0.6,
              animation: isPlaying ? 'pulse 1.5s ease-in-out infinite' : 'none',
            }}
          />
          <Typography
            variant="caption"
            sx={{
              color: 'text.secondary',
              fontSize: '0.7rem',
              fontFamily: 'monospace',
              minWidth: '40px',
            }}
          >
            {formatTime(Math.floor(currentTime))}/{formatTime(Math.floor(duration))}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

interface BotMessageProps {
  message: Message;
  index: number;
  narrationState: NarrationState | undefined;
  onStartNarration: (messageText: string, messageId: string) => void;
  onPauseNarration: () => void;
  onResumeNarration: () => void;
  onStopNarration: () => void;
  onRestartNarration: () => void;
}

export const BotMessage: React.FC<BotMessageProps> = ({
  message,
  index,
  narrationState,
  onStartNarration,
  onPauseNarration,
  onResumeNarration,
  onStopNarration,
  onRestartNarration,
}) => {
  const messageText = getMessageText(message.message);
  const showMetadata = hasMetadata(message.message);

  return (
    <ListItem
      key={`${message.id}-${index}`}
      sx={{
        display: 'flex',
        justifyContent: 'flex-start',
        paddingX: { xs: 1, sm: 2 },
        paddingY: { xs: 0.5, sm: 1 },
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'flex-start',
          maxWidth: { xs: '85%', sm: '75%', md: '70%' },
          flexDirection: 'row',
        }}
      >
        <Avatar
          sx={{
            bgcolor: '#0891b2',
            marginX: 1,
            width: { xs: 32, sm: 36 },
            height: { xs: 32, sm: 36 },
            boxShadow: '0 3px 6px rgba(8, 145, 178, 0.3)',
            border: '2px solid rgba(8, 145, 178, 0.1)',
            fontSize: '14px',
            fontWeight: 'bold',
          }}
        >
          AI
        </Avatar>
        
        <Box sx={{ width: '100%' }}>
          <Paper
            elevation={1}
            sx={{
              padding: { xs: 1.5, sm: 2 },
              backgroundColor: '#e0f2fe',
              borderRadius: '18px 18px 18px 4px',
              maxWidth: '100%',
              boxShadow: '0 2px 8px rgba(8, 145, 178, 0.15)',
              border: '2px solid rgba(8, 145, 178, 0.25)',
              position: 'relative',
            }}
          >
            <Box
              sx={{
                color: '#1e293b',
                '& p': {
                  margin: '0.5em 0',
                  lineHeight: 1.6,
                },
                '& p:first-of-type': {
                  marginTop: 0,
                },
                '& p:last-of-type': {
                  marginBottom: 0,
                },
                '& strong': {
                  fontWeight: 700,
                  color: '#0f172a',
                },
                '& em': {
                  fontStyle: 'italic',
                  color: '#334155',
                },
                '& ul, & ol': {
                  marginLeft: '1.5em',
                  marginTop: '0.5em',
                  marginBottom: '0.5em',
                  paddingLeft: '0.5em',
                },
                '& li': {
                  marginBottom: '0.25em',
                  lineHeight: 1.6,
                },
                '& code': {
                  backgroundColor: 'rgba(15, 23, 42, 0.1)',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  fontSize: '0.9em',
                  fontFamily: 'monospace',
                  color: '#0891b2',
                  fontWeight: 600,
                },
                '& pre': {
                  backgroundColor: 'rgba(15, 23, 42, 0.05)',
                  padding: '12px',
                  borderRadius: '8px',
                  overflow: 'auto',
                  marginTop: '0.5em',
                  marginBottom: '0.5em',
                },
                '& pre code': {
                  backgroundColor: 'transparent',
                  padding: 0,
                  color: '#1e293b',
                  fontWeight: 400,
                },
                '& blockquote': {
                  borderLeft: '4px solid #0891b2',
                  paddingLeft: '12px',
                  marginLeft: 0,
                  marginTop: '0.5em',
                  marginBottom: '0.5em',
                  color: '#475569',
                  fontStyle: 'italic',
                  backgroundColor: 'rgba(8, 145, 178, 0.05)',
                  paddingTop: '8px',
                  paddingBottom: '8px',
                  borderRadius: '4px',
                },
                '& h1, & h2, & h3, & h4, & h5, & h6': {
                  marginTop: '1em',
                  marginBottom: '0.5em',
                  fontWeight: 700,
                  color: '#0f172a',
                },
                '& h1': { fontSize: '1.5em' },
                '& h2': { fontSize: '1.3em' },
                '& h3': { fontSize: '1.1em' },
                '& hr': {
                  border: 'none',
                  borderTop: '2px solid rgba(8, 145, 178, 0.2)',
                  marginTop: '1em',
                  marginBottom: '1em',
                },
                '& a': {
                  color: '#0891b2',
                  textDecoration: 'underline',
                  '&:hover': {
                    color: '#0e7490',
                  },
                },
                '& table': {
                  borderCollapse: 'collapse',
                  width: '100%',
                  marginTop: '0.5em',
                  marginBottom: '0.5em',
                },
                '& th, & td': {
                  border: '1px solid rgba(8, 145, 178, 0.2)',
                  padding: '8px',
                  textAlign: 'left',
                },
                '& th': {
                  backgroundColor: 'rgba(8, 145, 178, 0.1)',
                  fontWeight: 600,
                },
              }}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {messageText}
              </ReactMarkdown>
            </Box>

            <Chip
              label={`#${message.offset}`}
              size="small"
              variant="outlined"
              sx={{
                marginTop: 1,
                height: 20,
                fontSize: '0.7rem',
                opacity: 0.7,
              }}
            />
          </Paper>

          {/* Narration controls */}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              mt: 1,
              ml: 1,
            }}
          >
            <NarrationControls
              messageText={messageText}
              messageId={message.id}
              narrationState={narrationState}
              onStartNarration={onStartNarration}
              onPauseNarration={onPauseNarration}
              onResumeNarration={onResumeNarration}
              onStopNarration={onStopNarration}
              onRestartNarration={onRestartNarration}
            />
          </Box>
          
          {/* Show metadata dropdown for bot messages */}
          {showMetadata && (
            <MessageMetadata messageContent={parseMessageContent(message.message)} />
          )}
        </Box>
      </Box>
    </ListItem>
  );
};

interface UserMessageProps {
  message: Message;
  index: number;
}

export const UserMessage: React.FC<UserMessageProps> = ({ message, index }) => {
  const messageText = getMessageText(message.message);

  return (
    <ListItem
      key={`${message.id}-${index}`}
      sx={{
        display: 'flex',
        justifyContent: 'flex-end',
        paddingX: { xs: 1, sm: 2 },
        paddingY: { xs: 0.5, sm: 1 },
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'flex-start',
          maxWidth: { xs: '85%', sm: '75%', md: '70%' },
          flexDirection: 'row-reverse',
        }}
      >
        <Avatar
          sx={{
            bgcolor: '#6366f1',
            marginX: 1,
            width: { xs: 32, sm: 36 },
            height: { xs: 32, sm: 36 },
            boxShadow: '0 3px 6px rgba(99, 102, 241, 0.3)',
            border: '2px solid rgba(99, 102, 241, 0.2)',
            fontSize: '14px',
            fontWeight: 'bold',
          }}
        >
          <PersonIcon fontSize="small" />
        </Avatar>
        
        <Box sx={{ width: '100%' }}>
          <Paper
            elevation={1}
            sx={{
              padding: { xs: 1.5, sm: 2 },
              backgroundColor: '#f8fafc',
              borderRadius: '18px 18px 4px 18px',
              maxWidth: '100%',
              boxShadow: '0 2px 8px rgba(99, 102, 241, 0.1)',
              border: '1px solid rgba(99, 102, 241, 0.15)',
              position: 'relative',
            }}
          >
            <Typography
              variant="body1"
              sx={{
                wordBreak: 'break-word',
                color: '#374151',
                whiteSpace: 'pre-wrap',
                fontWeight: '400',
              }}
            >
              {messageText}
            </Typography>
            
            <Chip
              label={`#${message.offset}`}
              size="small"
              variant="outlined"
              sx={{
                marginTop: 1,
                height: 20,
                fontSize: '0.7rem',
                opacity: 0.7,
              }}
            />
          </Paper>
        </Box>
      </Box>
    </ListItem>
  );
};

interface DefaultChipsProps {
  message: DefaultChips;
  index: number;
  onChipClick: (chipText: string, isQuestionChip?: boolean) => void;
}

export const DefaultChipsComponent: React.FC<DefaultChipsProps> = ({ message, index, onChipClick }) => {
  const [hoveredChip, setHoveredChip] = useState<number | null>(null);
  const [clickedChip, setClickedChip] = useState<number | null>(null);

  // Get bot configuration to access chip metadata
  const botId = message.sender_id;
  const botConfig = getBotConfiguration(botId);
  const chipMetadata = botConfig.chipMetadata || {};

  // Determine number of columns based on screen size and number of chips
  const getColumns = (chipCount: number) => {
    if (chipCount <= 2) return 2;
    if (chipCount <= 6) return 3;
    return 4;
  };

  const chipCount = message.chips?.length || 0;
  const columns = getColumns(chipCount);

  const handleChipClick = (chip: string, chipIndex: number) => {
    setClickedChip(chipIndex);
    setTimeout(() => setClickedChip(null), 200);
    onChipClick(chip, false); // Pass false to indicate this is NOT a question chip
  };

  return (
    <>
      {/* CSS Keyframes for animations */}
      <style>
        {`
          @keyframes fadeInUp {
            from {
              opacity: 0;
              transform: translateY(20px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }

          @keyframes slideIn {
            0% {
              transform: translateX(-8px);
              opacity: 0;
            }
            100% {
              transform: translateX(0);
              opacity: 1;
            }
          }

          .chip-card {
            animation: fadeInUp 0.5s ease-out;
            animation-fill-mode: both;
          }

          .chip-card:nth-child(1) { animation-delay: 0.05s; }
          .chip-card:nth-child(2) { animation-delay: 0.1s; }
          .chip-card:nth-child(3) { animation-delay: 0.15s; }
          .chip-card:nth-child(4) { animation-delay: 0.2s; }
          .chip-card:nth-child(5) { animation-delay: 0.25s; }
          .chip-card:nth-child(6) { animation-delay: 0.3s; }
        `}
      </style>

      <ListItem
        key={`${message.id}-${index}`}
        sx={{
          display: 'flex',
          justifyContent: 'center',
          paddingX: { xs: 1, sm: 2 },
          paddingY: { xs: 2, sm: 3 },
        }}
      >
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            maxWidth: { xs: '95%', sm: '90%', md: '85%' },
            width: '100%',
          }}
        >
          {message.intro && (
            <Box
              sx={{
                textAlign: 'center',
                mb: 4,
                animation: 'fadeInUp 0.4s ease-out',
              }}
            >
              <Typography
                variant="h5"
                sx={{
                  color: '#0f172a',
                  fontWeight: '600',
                  fontSize: { xs: '1.25rem', sm: '1.4rem', md: '1.5rem' },
                  mb: 1,
                  letterSpacing: '-0.01em',
                }}
              >
                {message.intro}
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  color: '#64748b',
                  fontSize: { xs: '0.875rem', sm: '0.9rem' },
                  fontWeight: '400',
                  letterSpacing: '0.01em',
                }}
              >
              Your trusted companion for health guidance and support. Select a topic to begin.
              </Typography>
            </Box>
          )}

          {message.chips && message.chips.length > 0 && (
            <Grid
              container
              spacing={{ xs: 2, sm: 2.5, md: 3 }}
              sx={{
                width: '100%',
                justifyContent: 'center',
                mb: 2,
              }}
            >
              {message.chips.map((chip, chipIndex) => {
                const metadata = chipMetadata[chip];
                const isHovered = hoveredChip === chipIndex;
                const isClicked = clickedChip === chipIndex;

                return (
                  <Grid
                    item
                    xs={12 / Math.min(columns, 2)}
                    sm={12 / Math.min(columns, 3)}
                    md={12 / columns}
                    key={chipIndex}
                    sx={{
                      display: 'flex',
                      justifyContent: 'center',
                    }}
                    className="chip-card"
                  >
                    <Card
                      onClick={() => handleChipClick(chip, chipIndex)}
                      onMouseEnter={() => setHoveredChip(chipIndex)}
                      onMouseLeave={() => setHoveredChip(null)}
                      sx={{
                        width: '100%',
                        minHeight: { xs: '140px', sm: '150px' },
                        cursor: 'pointer',
                        borderRadius: '16px',
                        position: 'relative',
                        overflow: 'hidden',
                        background: '#ffffff',
                        border: `1px solid ${metadata?.secondaryColor ? `${metadata.secondaryColor}20` : '#e2e8f0'}`,
                        boxShadow: isHovered
                          ? `0 8px 16px ${metadata?.secondaryColor ? `${metadata.secondaryColor}30` : 'rgba(0, 0, 0, 0.12)'}, 0 2px 4px ${metadata?.secondaryColor ? `${metadata.secondaryColor}20` : 'rgba(0, 0, 0, 0.08)'}`
                          : '0 2px 4px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04)',
                        transform: isClicked
                          ? 'scale(0.98)'
                          : isHovered
                            ? 'translateY(-4px)'
                            : 'translateY(0)',
                        transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          height: '4px',
                          background: metadata?.secondaryColor || '#0891b2',
                          opacity: isHovered ? 1 : 0.7,
                          transition: 'opacity 0.25s ease',
                        },
                      }}
                    >
                      <CardContent
                        sx={{
                          height: '100%',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'flex-start',
                          justifyContent: 'space-between',
                          padding: { xs: 2.5, sm: 3 },
                          paddingTop: { xs: 3, sm: 3.5 },
                          position: 'relative',
                          zIndex: 1,
                          '&:last-child': {
                            paddingBottom: { xs: 2.5, sm: 3 },
                          },
                        }}
                      >
                        {/* Icon Badge */}
                        {metadata?.iconText && (
                          <Box
                            sx={{
                              width: { xs: '48px', sm: '52px' },
                              height: { xs: '48px', sm: '52px' },
                              borderRadius: '12px',
                              background: `linear-gradient(135deg, ${metadata.primaryColor || '#1e3a8a'} 0%, ${metadata.secondaryColor || '#3b82f6'} 100%)`,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              mb: 2,
                              boxShadow: `0 4px 12px ${metadata.secondaryColor ? `${metadata.secondaryColor}30` : 'rgba(0, 0, 0, 0.15)'}`,
                              transform: isHovered ? 'scale(1.05)' : 'scale(1)',
                              transition: 'transform 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                            }}
                          >
                            <Typography
                              sx={{
                                color: '#ffffff',
                                fontSize: { xs: '0.9rem', sm: '1rem' },
                                fontWeight: '700',
                                letterSpacing: '0.05em',
                                textTransform: 'uppercase',
                              }}
                            >
                              {metadata.iconText}
                            </Typography>
                          </Box>
                        )}

                        {/* Chip Text */}
                        <Box sx={{ width: '100%' }}>
                          <Typography
                            variant="h6"
                            sx={{
                              color: '#0f172a',
                              fontWeight: '600',
                              fontSize: { xs: '0.95rem', sm: '1.05rem' },
                              lineHeight: 1.4,
                              mb: 0.5,
                              letterSpacing: '-0.01em',
                            }}
                          >
                            {chip}
                          </Typography>

                          {/* Description */}
                          {metadata?.description && (
                            <Typography
                              variant="body2"
                              sx={{
                                color: '#64748b',
                                fontSize: { xs: '0.8rem', sm: '0.85rem' },
                                lineHeight: 1.5,
                                letterSpacing: '0.01em',
                                opacity: isHovered ? 1 : 0.8,
                                transition: 'opacity 0.25s ease',
                              }}
                            >
                              {metadata.description}
                            </Typography>
                          )}
                        </Box>

                        {/* Arrow indicator */}
                        <Box
                          sx={{
                            position: 'absolute',
                            bottom: { xs: 16, sm: 20 },
                            right: { xs: 16, sm: 20 },
                            width: '28px',
                            height: '28px',
                            borderRadius: '50%',
                            background: metadata?.secondaryColor ? `${metadata.secondaryColor}15` : '#e0f2fe',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            transform: isHovered ? 'translateX(4px)' : 'translateX(0)',
                            opacity: isHovered ? 1 : 0.6,
                            transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                          }}
                        >
                          <Typography
                            sx={{
                              color: metadata?.primaryColor || '#0891b2',
                              fontSize: '1.1rem',
                              fontWeight: '600',
                            }}
                          >
                            →
                          </Typography>
                        </Box>
                      </CardContent>

                      {/* Subtle gradient overlay on hover */}
                      <Box
                        sx={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          bottom: 0,
                          background: metadata?.secondaryColor
                            ? `linear-gradient(135deg, ${metadata.secondaryColor}08 0%, transparent 100%)`
                            : 'linear-gradient(135deg, rgba(8, 145, 178, 0.03) 0%, transparent 100%)',
                          opacity: isHovered ? 1 : 0,
                          transition: 'opacity 0.25s ease-in-out',
                          pointerEvents: 'none',
                        }}
                      />
                    </Card>
                  </Grid>
                );
              })}
            </Grid>
          )}
        </Box>
      </ListItem>
    </>
  );
};

interface QuestionChipsProps {
  message: QuestionChips;
  index: number;
  onChipClick: (chipText: string, isQuestionChip: boolean) => void;
}

export const QuestionChipsComponent: React.FC<QuestionChipsProps> = ({ message, index, onChipClick }) => {
  const [hoveredQuestion, setHoveredQuestion] = useState<number | null>(null);
  const [clickedQuestion, setClickedQuestion] = useState<number | null>(null);

  const handleQuestionClick = (question: string, questionIndex: number) => {
    setClickedQuestion(questionIndex);
    setTimeout(() => setClickedQuestion(null), 200);
    onChipClick(question, true); // Pass true to indicate this is a question chip
  };

  return (
    <ListItem
      key={`${message.id}-${index}`}
      sx={{
        display: 'flex',
        justifyContent: 'center',
        paddingX: { xs: 1, sm: 2 },
        paddingY: { xs: 2, sm: 2.5 },
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          maxWidth: { xs: '95%', sm: '90%', md: '85%' },
          width: '100%',
        }}
      >
        {/* Category header */}
        <Box
          sx={{
            textAlign: 'center',
            mb: 3,
            animation: 'fadeInUp 0.3s ease-out',
          }}
        >
          <Typography
            variant="h6"
            sx={{
              color: '#0f172a',
              fontWeight: '600',
              fontSize: { xs: '1.1rem', sm: '1.2rem' },
              mb: 0.5,
            }}
          >
            Questions about {message.category}
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: '#64748b',
              fontSize: { xs: '0.8rem', sm: '0.85rem' },
            }}
          >
            Select a question to continue
          </Typography>
        </Box>

        {/* Question chips */}
        <Grid
          container
          spacing={{ xs: 1.5, sm: 2 }}
          sx={{
            width: '100%',
            justifyContent: 'center',
          }}
        >
          {message.questions.map((question, questionIndex) => {
            const isHovered = hoveredQuestion === questionIndex;
            const isClicked = clickedQuestion === questionIndex;

            return (
              <Grid
                item
                xs={12}
                key={questionIndex}
                sx={{
                  display: 'flex',
                  justifyContent: 'center',
                  animation: `slideIn 0.3s ease-out ${questionIndex * 0.1}s both`,
                }}
              >
                <Card
                  onClick={() => handleQuestionClick(question, questionIndex)}
                  onMouseEnter={() => setHoveredQuestion(questionIndex)}
                  onMouseLeave={() => setHoveredQuestion(null)}
                  sx={{
                    width: '100%',
                    cursor: 'pointer',
                    borderRadius: '12px',
                    background: '#ffffff',
                    border: '1px solid #e0f2fe',
                    boxShadow: isHovered
                      ? '0 6px 12px rgba(8, 145, 178, 0.2)'
                      : '0 2px 4px rgba(0, 0, 0, 0.06)',
                    transform: isClicked
                      ? 'scale(0.98)'
                      : isHovered
                        ? 'translateY(-2px)'
                        : 'translateY(0)',
                    transition: 'all 0.2s ease',
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: '4px',
                      background: '#0891b2',
                      opacity: isHovered ? 1 : 0.5,
                      transition: 'opacity 0.2s ease',
                    },
                  }}
                >
                  <CardContent
                    sx={{
                      padding: { xs: 2, sm: 2.5 },
                      paddingLeft: { xs: 3, sm: 3.5 },
                      '&:last-child': {
                        paddingBottom: { xs: 2, sm: 2.5 },
                      },
                    }}
                  >
                    <Typography
                      variant="body1"
                      sx={{
                        color: '#1e293b',
                        fontSize: { xs: '0.9rem', sm: '0.95rem' },
                        lineHeight: 1.5,
                        fontWeight: isHovered ? '500' : '400',
                        transition: 'font-weight 0.2s ease',
                      }}
                    >
                      {question}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      </Box>
    </ListItem>
  );
};

interface MessageRendererProps {
  message: MessageOrDefaultChips;
  index: number;
  narrationStates: Map<string, NarrationState>;
  onStartNarration: (messageText: string, messageId: string) => void;
  onPauseNarration: () => void;
  onResumeNarration: () => void;
  onStopNarration: () => void;
  onRestartNarration: () => void;
  onChipClick: (chipText: string, isQuestionChip?: boolean) => void;
  agent_id: string;
}

export const MessageRenderer: React.FC<MessageRendererProps> = ({
  message,
  index,
  narrationStates,
  onStartNarration,
  onPauseNarration,
  onResumeNarration,
  onStopNarration,
  onRestartNarration,
  onChipClick,
  agent_id,
}) => {
  const isDefaultChips = 'defaultChips' in message && message.defaultChips === true;
  const isQuestionChips = 'questionChips' in message && message.questionChips === true;

  if (isDefaultChips) {
    return <DefaultChipsComponent message={message} index={index} onChipClick={onChipClick} />;
  }

  if (isQuestionChips) {
    return <QuestionChipsComponent message={message} index={index} onChipClick={onChipClick} />;
  }

  const isBot = message.sender_id === agent_id;
  const narrationState = narrationStates.get(message.id);

  if (isBot) {
    return (
      <BotMessage
        message={message}
        index={index}
        narrationState={narrationState}
        onStartNarration={onStartNarration}
        onPauseNarration={onPauseNarration}
        onResumeNarration={onResumeNarration}
        onStopNarration={onStopNarration}
        onRestartNarration={onRestartNarration}
      />
    );
  }

  return <UserMessage message={message} index={index} />;
};
