import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Container,
  Paper,
  Box,
  TextField,
  IconButton,
  List,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  Button,
  Snackbar,
  Alert,
  LinearProgress,
  Typography,
} from '@mui/material';
import {
  Send as SendIcon,
  Mic as MicIcon,
  MicOff as MicOffIcon,
  Close as CloseIcon,
  Stop as StopIcon,
  MoreHoriz as MoreHorizIcon,
  ArrowBack as ArrowBackIcon,
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  Delete as DeleteIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';
import { Message, DefaultChips, QuestionChips, SubmitQueryResponse } from '../../types';
import { NarrationState } from '../../narration';
import { submitQuery, getMessages, transcribeAudio } from '../../api';
import { narrationService } from '../../narration';
import { MessageRenderer } from '../../components/messages';
import { createDefaultSelections, getBotConfiguration } from '../../constants/constants';

interface ChatBoxProps {
  agent_id: string;
  onMobileBack?: () => void; // For mobile navigation
}

const ChatBox: React.FC<ChatBoxProps> = ({ agent_id, onMobileBack }) => {
  // State declarations
  const [messages, setMessages] = useState<(Message | DefaultChips | QuestionChips)[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingOlder, setIsLoadingOlder] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement | null>(null);
  const skipAutoScrollRef = useRef(false);
  const scrollDebounceRef = useRef<NodeJS.Timeout | null>(null);
  const loaderStartRef = useRef<number | null>(null);
  const MIN_LOADER_MS = 1000; // minimum spinner display time in ms (between 1-2s)
  const ACTIVE_POLL_DELAY_MS = 1500;
  const IDLE_POLL_DELAY_MS = 10000;
  const latestFetchIdRef = useRef(0);
  const lastOffsetRef = useRef<number | undefined>(undefined);
  const hasInitialLoadRef = useRef(false);
  const pollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const awaitingBotReplyRef = useRef(false);
  const scheduleNextPollRef = useRef<(delay: number) => void>(() => {});
  const suggestionsDismissedRef = useRef(false);

  // Query cancellation state
  const [pendingRequest, setPendingRequest] = useState<AbortController | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const currentRequestRef = useRef<AbortController | null>(null);
  // Synchronous lock to prevent multiple rapid submissions (covers React state update delays)
  const sendLockRef = useRef(false);

  // Hierarchical chip flow state
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  // Bot switching state
  const [isSwitchingBot, setIsSwitchingBot] = useState(false);

  // Speech-to-text related state
  const [isRecordingModalOpen, setIsRecordingModalOpen] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioPermissionDenied, setAudioPermissionDenied] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error' | 'info'>('info');
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [recordedAudioUrl, setRecordedAudioUrl] = useState<string | null>(null);
  const [isPlayingRecording, setIsPlayingRecording] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

  // Narration state
  const [narrationStates, setNarrationStates] = useState<Map<string, NarrationState>>(new Map());

  // Type guards
  const isDefaultChips = (msg: Message | DefaultChips | QuestionChips): msg is DefaultChips => {
    return 'defaultChips' in msg && msg.defaultChips === true;
  };

  const isQuestionChips = (msg: Message | DefaultChips | QuestionChips): msg is QuestionChips => {
    return 'questionChips' in msg && msg.questionChips === true;
  };

  const isRegularMessage = (msg: Message | DefaultChips | QuestionChips): msg is Message => {
    return !isDefaultChips(msg) && !isQuestionChips(msg);
  };

  const partitionMessages = useCallback((msgs: (Message | DefaultChips | QuestionChips)[]) => {
    const regular: Message[] = [];
    const question: QuestionChips[] = [];
    const defaults: DefaultChips[] = [];

    msgs.forEach((msg) => {
      if (isDefaultChips(msg)) {
        defaults.push(msg);
        return;
      }
      if (isQuestionChips(msg)) {
        question.push(msg);
      } else {
        regular.push(msg);
      }
    });

    return { regular, question, defaults };
  }, []);

  const getDefaultChipsForRender = useCallback(
    (defaults: DefaultChips[] = []): DefaultChips[] => {
      if (defaults.length > 0) {
        const unique = new Map<string, DefaultChips>();
        defaults.forEach(chip => {
          unique.set(chip.id, chip);
        });
        return Array.from(unique.values());
      }
      return [createDefaultSelections(agent_id)];
    },
    [agent_id]
  );

  const stripSuggestionMessages = (msgs: (Message | DefaultChips | QuestionChips)[]) => {
    return msgs.filter(isRegularMessage);
  };

  const updateSuggestionsDismissed = (value: boolean) => {
    suggestionsDismissedRef.current = value;
  };

  // Helper function to create question chips message
  const createQuestionChips = (category: string, questions: string[]): QuestionChips => {
    return {
      sender_id: agent_id,
      receiver_id: 'user',
      message: {
        text: `Select a question about ${category}:`,
        confidence_score: null,
        sources: null,
        retrieval_metadata: null,
        processing_time: null,
        status: null
      },
      id: `question-chips-${category}-${Date.now()}`,
      offset: messages.length > 0 ? Math.max(...messages.map(m => m.offset)) + 1 : 1,
      is_read: false,
      is_delivered: false,
      is_seen: false,
      is_deleted: false,
      is_archived: false,
      is_pinned: false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      category: category,
      questions: questions,
      questionChips: true
    };
  };

  const addOutgoingMessage = (payload: SubmitQueryResponse, fallbackText: string) => {
    const normalized: Message = {
      ...payload,
      message: payload.message || {
        text: fallbackText,
        confidence_score: null,
        sources: null,
        retrieval_metadata: null,
        processing_time: null,
        status: null,
      },
    };

    updateSuggestionsDismissed(true);

    setMessages(prev => {
      const { regular } = partitionMessages(prev);
      const withoutDuplicate = regular.filter(msg => msg.id !== normalized.id);
      const updated = [...withoutDuplicate, normalized].sort((a, b) => a.offset - b.offset);
      return updated;
    });

    if (lastOffsetRef.current === undefined || normalized.offset > lastOffsetRef.current) {
      lastOffsetRef.current = normalized.offset;
    }
  };

  // Set up global stop callback for narration service
  useEffect(() => {
    narrationService.setGlobalStopCallback(() => {
      // Clear all playing/paused states when any narration stops
      setNarrationStates(prev => {
        const updated = new Map(prev);
        let hasChanges = false;
        updated.forEach((state, id) => {
          if (state.isPlaying || state.isPaused) {
            updated.set(id, {
              ...state,
              isPlaying: false,
              isPaused: false,
              currentTime: 0,
            });
            hasChanges = true;
          }
        });
        return hasChanges ? updated : prev;
      });
    });

    return () => {
      narrationService.setGlobalStopCallback(() => {});
    };
  }, []);

  // Scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    // If we recently loaded older messages, skip auto-scrolling to bottom
    if (skipAutoScrollRef.current) {
      skipAutoScrollRef.current = false;
      return;
    }
    // Don't scroll if messages are empty (during bot switching)
    if (messages.length === 0) {
      return;
    }
    scrollToBottom();
  }, [messages]);

  // Handler to load older messages when user scrolls to top
  // Debounced scroll handler that checks the container directly to avoid synthetic event reuse
  const handleScroll = async () => {
    const container = messagesContainerRef.current;
    if (!container) return;
    if (container.scrollTop > 50) return;
    if (isLoadingOlder) return;
    if (messages.length === 0) return;

    // Preserve scroll position: measure before update
    const prevScrollHeight = container.scrollHeight;
    const prevScrollTop = container.scrollTop;

  // Mark to skip auto-scroll-to-bottom effect
  skipAutoScrollRef.current = true;
  loaderStartRef.current = Date.now();
  setIsLoadingOlder(true);
    try {
      const earliestOffset = Math.min(...messages.map(m => m.offset));
      const response = await getMessages({ afterOffset: earliestOffset, before: true }, agent_id);
      if (response.messages && response.messages.length > 0) {
        setMessages(prev => {
          const combined = [...response.messages, ...prev];
          return combined.sort((a, b) => a.offset - b.offset);
        });

        // After DOM updates, restore scroll position so user stays at same message
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            const newContainer = messagesContainerRef.current;
            if (newContainer) {
              const newScrollHeight = newContainer.scrollHeight;
              const target = newScrollHeight - prevScrollHeight + prevScrollTop;
              // Immediately restore the scroll position to avoid flicker
              newContainer.scrollTop = target;
            }
          });
        });
      }
    } catch (error) {
      console.error('Error loading older messages:', error);
    } finally {
      const elapsed = loaderStartRef.current ? Date.now() - loaderStartRef.current : 0;
      const remaining = Math.max(0, MIN_LOADER_MS - elapsed);
      if (remaining > 0) {
        setTimeout(() => setIsLoadingOlder(false), remaining);
      } else {
        setIsLoadingOlder(false);
      }
      loaderStartRef.current = null;
    }
  };

  const onScrollDebounced = () => {
    if (scrollDebounceRef.current) clearTimeout(scrollDebounceRef.current);
    scrollDebounceRef.current = setTimeout(() => {
      handleScroll();
    }, 100);
  };

  // Animate scrollTop from current to target over durationMs for smoother visual
  const animateScrollTop = (el: HTMLDivElement, targetScrollTop: number, durationMs = 200) => {
    const start = el.scrollTop;
    const delta = targetScrollTop - start;
    if (Math.abs(delta) < 1) {
      el.scrollTop = targetScrollTop;
      return;
    }
    const startTime = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - startTime) / durationMs);
      // easeOutQuad
      const eased = 1 - (1 - t) * (1 - t);
      el.scrollTop = start + delta * eased;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  // Cleanup narration cache on unmount
  useEffect(() => {
    return () => {
      narrationService.clearCache();
    };
  }, []);

  // Initial load and polling for new messages
  useEffect(() => {
    if (!agent_id) return;

    latestFetchIdRef.current++;
    setIsSwitchingBot(true);
    lastOffsetRef.current = undefined;
    hasInitialLoadRef.current = false;
    awaitingBotReplyRef.current = false;
    skipAutoScrollRef.current = true;
  updateSuggestionsDismissed(false);

    const clearPollTimeout = () => {
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
        pollTimeoutRef.current = null;
      }
    };

    const loadMessages = async () => {
      if (!hasInitialLoadRef.current) return;

      const fetchId = ++latestFetchIdRef.current;
      try {
        const response = await getMessages({ afterOffset: lastOffsetRef.current ?? 0 }, agent_id);
        if (fetchId !== latestFetchIdRef.current) return;

        if (response.messages.length > 0) {
          let detectedBotReply = false;
          setMessages(prev => {
            const { regular, question, defaults } = partitionMessages(prev);
            const existingIds = new Set(regular.map(msg => msg.id));

            const getTextForComparison = (msg: Message | DefaultChips | QuestionChips): string => {
              if ('defaultChips' in msg && msg.defaultChips === true) return '';
              if (typeof msg.message === 'string') return msg.message;
              if (msg.message && typeof msg.message === 'object' && 'text' in msg.message) {
                return msg.message.text || '';
              }
              return '';
            };

            const newMessages = response.messages.filter(msg => {
              if (existingIds.has(msg.id)) return false;

              if (msg.sender_id === agent_id) {
                detectedBotReply = true;
              }

              if (regular.length > 0) {
                const lastMsg = regular[regular.length - 1];
                if (!isDefaultChips(lastMsg)) {
                  const lastText = getTextForComparison(lastMsg).trim();
                  const newText = getTextForComparison(msg).trim();

                  if (lastText && newText && lastText === newText && lastMsg.sender_id === msg.sender_id) {
                    console.warn('Duplicate consecutive message detected and filtered:', msg.id);
                    return false;
                  }
                }
              }

              return true;
            });

            const includeQuestions = !suggestionsDismissedRef.current && question.length > 0;
            const includeDefaultChips = !suggestionsDismissedRef.current && question.length === 0;
            const defaultChipsForRender = includeDefaultChips ? getDefaultChipsForRender(defaults) : [];

            if (newMessages.length === 0) {
              const base = includeQuestions ? [...regular, ...question] : regular;
              return includeDefaultChips ? [...base, ...defaultChipsForRender] : base;
            }

            const combined = [...regular, ...newMessages];
            const sorted = combined.sort((a, b) => a.offset - b.offset);
            const base = includeQuestions ? [...sorted, ...question] : sorted;
            return includeDefaultChips ? [...base, ...defaultChipsForRender] : base;
          });

          const maxOffset = Math.max(...response.messages.map(m => m.offset));
          if (lastOffsetRef.current === undefined || maxOffset > lastOffsetRef.current) {
            lastOffsetRef.current = maxOffset;
          }

          if (detectedBotReply) {
            awaitingBotReplyRef.current = false;
            // Bot reply detected - release synchronous send lock so user can send next question
            sendLockRef.current = false;
          }
        }
      } catch (error) {
        console.error('Error loading messages:', error);
      } finally {
        if (hasInitialLoadRef.current) {
          const delay = awaitingBotReplyRef.current ? ACTIVE_POLL_DELAY_MS : IDLE_POLL_DELAY_MS;
          scheduleNextPoll(delay);
        }
      }
    };

    const scheduleNextPoll = (delay: number) => {
      if (!hasInitialLoadRef.current) return;
      clearPollTimeout();
      pollTimeoutRef.current = setTimeout(() => {
        loadMessages();
      }, delay);
    };

    scheduleNextPollRef.current = (delay: number) => {
      scheduleNextPoll(delay);
    };

    const initialFetch = async () => {
      const fetchId = ++latestFetchIdRef.current;
      try {
        const response = await getMessages({ afterOffset: Number.MAX_SAFE_INTEGER, before: true }, agent_id);
        if (fetchId !== latestFetchIdRef.current) return;

        setTimeout(() => {
          if (fetchId !== latestFetchIdRef.current) return;

          if (response.messages && response.messages.length > 0) {
            const sorted = response.messages.sort((a, b) => a.offset - b.offset);
            const defaults = getDefaultChipsForRender();
            setMessages([...sorted, ...defaults]);
            const maxOffset = Math.max(...response.messages.map(m => m.offset));
            lastOffsetRef.current = maxOffset;
            updateSuggestionsDismissed(false);
          } else {
            setMessages([createDefaultSelections(agent_id)]);
            lastOffsetRef.current = undefined;
            updateSuggestionsDismissed(false);
          }

          hasInitialLoadRef.current = true;
          setIsSwitchingBot(false);
          skipAutoScrollRef.current = false;

          setTimeout(() => {
            if (messagesEndRef.current) {
              messagesEndRef.current.scrollIntoView({ behavior: 'auto' });
            }
          }, 0);
          scheduleNextPoll(awaitingBotReplyRef.current ? ACTIVE_POLL_DELAY_MS : IDLE_POLL_DELAY_MS);
        }, 0);
      } catch (error) {
        console.error('Error in initial fetch:', error);
        setMessages([createDefaultSelections(agent_id)]);
        lastOffsetRef.current = undefined;
        setIsSwitchingBot(false);
        skipAutoScrollRef.current = false;
        hasInitialLoadRef.current = true;
        updateSuggestionsDismissed(false);
        scheduleNextPoll(awaitingBotReplyRef.current ? ACTIVE_POLL_DELAY_MS : IDLE_POLL_DELAY_MS);
      }
    };

    initialFetch();

    return () => {
      latestFetchIdRef.current++;
      hasInitialLoadRef.current = false;
      awaitingBotReplyRef.current = false;
      clearPollTimeout();
      scheduleNextPollRef.current = () => {};
    };
  }, [agent_id]);

  // Note: Old auto-sequential question flow has been removed
  // Now using hierarchical flow where users manually select questions from QuestionChips

  // Utility functions
  const showSnackbar = (message: string, severity: 'success' | 'error' | 'info') => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Waveform visualization
  const drawWaveform = () => {
    if (!canvasRef.current || !analyserRef.current) return;

    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext('2d');
    if (!canvasCtx) return;

    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!isRecording) return;

      animationFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      canvasCtx.fillStyle = 'rgba(44, 44, 44, 0.9)';
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

      canvasCtx.lineWidth = 2;
      canvasCtx.strokeStyle = '#0891b2';
      canvasCtx.beginPath();

      const sliceWidth = (canvas.width * 1.0) / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          canvasCtx.moveTo(x, y);
        } else {
          canvasCtx.lineTo(x, y);
        }

        x += sliceWidth;
      }

      canvasCtx.lineTo(canvas.width, canvas.height / 2);
      canvasCtx.stroke();
    };

    draw();
  };

  // Cleanup audio context and waveform on unmount or when recording stops
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      if (recordedAudioUrl) {
        URL.revokeObjectURL(recordedAudioUrl);
      }
    };
  }, [recordedAudioUrl]);

  // Narration helper functions
  const updateNarrationState = (messageId: string, newState: Partial<NarrationState>) => {
    setNarrationStates(prev => {
      const updated = new Map(prev);
      const currentState = updated.get(messageId) || {
        isPlaying: false,
        isPaused: false,
        currentTime: 0,
        duration: 0,
        audioUrl: null,
        error: null,
      };
      updated.set(messageId, { ...currentState, ...newState });
      return updated;
    });
  };

  const startNarration = async (messageText: string, messageId: string) => {
    console.log('ChatBox: startNarration called for message:', messageId);
    try {
      // Check if this message is already playing
      const currentState = narrationStates.get(messageId);
      const currentlyPlayingId = narrationService.getCurrentMessageId();

      // If another message is playing, stop it
      if (currentlyPlayingId && currentlyPlayingId !== messageId) {
        console.log('Stopping currently playing narration:', currentlyPlayingId);
        narrationService.stopNarration();
      }

      // If this message is already playing, don't restart it
      if (currentState?.isPlaying && currentlyPlayingId === messageId) {
        console.log('Message already playing, ignoring start request');
        return;
      }

      // Start the new narration
      await narrationService.playNarration(
        messageText,
        messageId,
        (state) => updateNarrationState(messageId, state)
      );
    } catch (error) {
      console.error('ChatBox: Failed to start narration:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to start narration';
      showSnackbar(errorMessage, 'error');
    }
  };

  const pauseNarration = () => {
    narrationService.pauseNarration();
  };

  const resumeNarration = () => {
    narrationService.resumeNarration();
  };

  const stopNarration = () => {
    narrationService.stopNarration();
  };

  const restartNarration = () => {
    narrationService.restartNarration();
  };

  // Stop ongoing generation
  const handleStopGeneration = () => {
    if (currentRequestRef.current) {
      currentRequestRef.current.abort();
      currentRequestRef.current = null;
      setPendingRequest(null);
      setIsGenerating(false);
      setIsLoading(false);
      showSnackbar('Generation stopped', 'info');
      // Allow new submissions when the user stops generation
      sendLockRef.current = false;
    }
  };

  // Event handlers
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!inputValue.trim() || !agent_id) return;

    // If a send lock is active, block immediate submissions (covers rapid clicks before state updates)
    if (sendLockRef.current) {
      showSnackbar('Please wait for the current response or click Stop', 'info');
      return;
    }

    // If currently generating (React state) also block
    if (isGenerating && currentRequestRef.current) {
      showSnackbar('Please wait for the current response or click Stop', 'info');
      return;
    }

    const query = inputValue.trim();
    setInputValue('');
    setIsLoading(true);
    setIsGenerating(true);

  // Acquire synchronous send lock immediately to prevent other submits
  sendLockRef.current = true;

  // Create new AbortController for this request
  const controller = new AbortController();
  currentRequestRef.current = controller;
  setPendingRequest(controller);

    try {
      const response = await submitQuery({ query }, agent_id, { signal: controller.signal });
      addOutgoingMessage(response, query);
      setSelectedCategory(null);
      // Mark that we're awaiting a bot reply; keep the send lock until the bot reply is observed
      awaitingBotReplyRef.current = true;
      scheduleNextPollRef.current(ACTIVE_POLL_DELAY_MS);

      if (currentRequestRef.current === controller) {
        setIsLoading(false);
        setIsGenerating(false);
        setPendingRequest(null);
        currentRequestRef.current = null;
      }
      // Do not clear sendLockRef here - we want to keep it locked until a bot reply arrives
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        // Request was cancelled, don't show error or reset states
        console.log('Query submission cancelled');
        // If user cancelled the request, allow new submissions
        sendLockRef.current = false;
      } else {
        // Only show error and reset if this is still the current request
        if (currentRequestRef.current === controller) {
          console.error('Error submitting query:', error);
          showSnackbar('Failed to send message', 'error');
          setIsLoading(false);
          setIsGenerating(false);
          setPendingRequest(null);
          currentRequestRef.current = null;
        }
        awaitingBotReplyRef.current = false;
        // Clear the synchronous send lock on error so user can retry
        sendLockRef.current = false;
      }
    }
  };

  const handleChipClick = async (chipText: string, isQuestionChip: boolean = false) => {
    if (!agent_id) return;

    // Check if this chip is a category chip
    const botConfig = getBotConfiguration(agent_id);
    const isCategory = botConfig.questionCategories && chipText in botConfig.questionCategories;

    if (isCategory && !isQuestionChip) {
      // NEW HIERARCHICAL FLOW: Show questions instead of auto-sending
      const questions = botConfig.questionCategories![chipText];
      setSelectedCategory(chipText);
      updateSuggestionsDismissed(false);

      const questionChipsMessage = createQuestionChips(chipText, questions);
      setMessages((prev) => {
        const retained = stripSuggestionMessages(prev);
        return [...retained, questionChipsMessage];
      });

      showSnackbar(`Select a question about ${chipText}`, 'info');
    } else {
      // If a send lock is active, block immediate submissions (covers rapid clicks before state updates)
      if (sendLockRef.current) {
        showSnackbar('Please wait for the current response or click Stop', 'info');
        return;
      }

      // If currently generating (React state) also block
      if (isGenerating && currentRequestRef.current) {
        showSnackbar('Please wait for the current response or click Stop', 'info');
        return;
      }

      // Acquire synchronous send lock immediately
      sendLockRef.current = true;

      // Regular chip click OR question chip clicked - send the query
      setIsLoading(true);
      setIsGenerating(true);

      const controller = new AbortController();
      currentRequestRef.current = controller;
      setPendingRequest(controller);

      try {
        const response = await submitQuery({ query: chipText }, agent_id, { signal: controller.signal });
        addOutgoingMessage(response, chipText);

        if (isQuestionChip) {
          setSelectedCategory(null);
        }

        // Mark awaiting reply and keep lock until reply observed
        awaitingBotReplyRef.current = true;
        scheduleNextPollRef.current(ACTIVE_POLL_DELAY_MS);

        if (currentRequestRef.current === controller) {
          setIsLoading(false);
          setIsGenerating(false);
          setPendingRequest(null);
          currentRequestRef.current = null;
        }
        // Do not clear sendLockRef here - it will be cleared when bot reply is observed
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          console.log('Query cancelled');
          // User cancelled - allow new submissions
          sendLockRef.current = false;
        } else {
          // Only show error and reset if this is still the current request
          if (currentRequestRef.current === controller) {
            console.error('Error submitting query:', error);
            showSnackbar('Failed to send message', 'error');
            setIsLoading(false);
            setIsGenerating(false);
            setPendingRequest(null);
            currentRequestRef.current = null;
          }
          awaitingBotReplyRef.current = false;
          // Clear synchronous lock so user can retry
          sendLockRef.current = false;
        }
      }
    }
  };

  const requestMicrophonePermission = async (): Promise<boolean> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
      return true;
    } catch (error) {
      console.error('Microphone permission denied:', error);
      setAudioPermissionDenied(true);
      showSnackbar('Microphone permission is required for voice recording', 'error');
      return false;
    }
  };

  const startRecording = async () => {
    try {
      const hasPermission = await requestMicrophonePermission();
      if (!hasPermission) return;

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        }
      });

      // Set up audio context and analyser for waveform
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm',
      });

      audioChunksRef.current = [];
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop());

        // Stop waveform animation
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }

        // Create audio URL for playback
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(audioBlob);
        setRecordedAudioUrl(url);

        // Close audio context only if it's not already closed
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close();
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      setRecordedAudioUrl(null);

      // Start timer
      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

      // Start waveform visualization
      drawWaveform();

      showSnackbar('Recording started', 'info');
    } catch (error) {
      console.error('Error starting recording:', error);
      showSnackbar('Failed to start recording. Please check your microphone.', 'error');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
    }
  };

  const handlePlayRecording = () => {
    if (!recordedAudioUrl) return;

    if (audioPlayerRef.current) {
      if (isPlayingRecording) {
        audioPlayerRef.current.pause();
        setIsPlayingRecording(false);
      } else {
        audioPlayerRef.current.play();
        setIsPlayingRecording(true);
      }
    }
  };

  const handleTranscribeRecording = async () => {
    if (audioChunksRef.current.length === 0) {
      showSnackbar('No audio data recorded', 'error');
      return;
    }

    setIsTranscribing(true);
    showSnackbar('Transcribing audio...', 'info');

    try {
      const audioBlob = audioChunksRef.current[0];

      // Determine the mime type and filename based on the blob type
      const mimeType = audioBlob.type || 'audio/webm';
      const extension = mimeType.includes('m4a') ? 'm4a' : 'webm';
      const audioFile = new File([audioBlob], `recording.${extension}`, { type: mimeType });

      const transcriptionResult = await transcribeAudio(audioFile);

      if (transcriptionResult.text && transcriptionResult.text.trim()) {
        // Append transcription to existing input value
        const newText = inputValue ? `${inputValue} ${transcriptionResult.text}` : transcriptionResult.text;
        setInputValue(newText);
        showSnackbar('Audio transcribed successfully!', 'success');
      } else {
        showSnackbar('No speech detected in the recording', 'info');
      }
    } catch (error) {
      console.error('Error transcribing audio:', error);
      showSnackbar('Failed to transcribe audio. Please try again.', 'error');
    } finally {
      setIsTranscribing(false);
      setIsRecordingModalOpen(false);
      setRecordingTime(0);
      setRecordedAudioUrl(null);
    }
  };

  const handleDiscardRecording = () => {
    if (recordedAudioUrl) {
      URL.revokeObjectURL(recordedAudioUrl);
    }
    setRecordedAudioUrl(null);
    setRecordingTime(0);
    audioChunksRef.current = [];
  };

  const handleMicButtonClick = () => {
    if (audioPermissionDenied) {
      showSnackbar('Please enable microphone permission in your browser settings', 'error');
      return;
    }
    setIsRecordingModalOpen(true);
  };

  const handleCloseRecordingModal = () => {
    if (isRecording) {
      stopRecording();
    }
    if (recordedAudioUrl) {
      URL.revokeObjectURL(recordedAudioUrl);
    }
    setIsRecordingModalOpen(false);
    setRecordingTime(0);
    setRecordedAudioUrl(null);
    setIsPlayingRecording(false);
    audioChunksRef.current = [];
  };

  const handleAudioFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('audio/')) {
      showSnackbar('Please select a valid audio file', 'error');
      return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      showSnackbar('File size must be less than 10MB', 'error');
      return;
    }

    setIsTranscribing(true);
    showSnackbar('Transcribing uploaded audio...', 'info');

    try {
      const transcriptionResult = await transcribeAudio(file);

      if (transcriptionResult.text && transcriptionResult.text.trim()) {
        // Append transcription to existing input value
        const newText = inputValue ? `${inputValue} ${transcriptionResult.text}` : transcriptionResult.text;
        setInputValue(newText);
        showSnackbar('Audio transcribed successfully!', 'success');
      } else {
        showSnackbar('No speech detected in the audio file', 'info');
      }
    } catch (error) {
      console.error('Error transcribing uploaded audio:', error);
      showSnackbar('Failed to transcribe audio. Please try again.', 'error');
    } finally {
      setIsTranscribing(false);
      setIsRecordingModalOpen(false);
      // Reset the file input
      event.target.value = '';
    }
  };

  const handleLoadDemoAudio = async () => {
    try {
      showSnackbar('Loading demo audio...', 'info');

      // Fetch the demo audio file from public folder
      const response = await fetch('/demo_audio.m4a');
      if (!response.ok) {
        throw new Error('Failed to load demo audio');
      }

      const audioBlob = await response.blob();

      // Store the blob in audioChunksRef for later transcription
      audioChunksRef.current = [audioBlob];

      // Create URL for playback
      const url = URL.createObjectURL(audioBlob);
      setRecordedAudioUrl(url);

      // Set a dummy recording time (you can calculate actual duration if needed)
      setRecordingTime(15); // Default 15 seconds for demo

      showSnackbar('Demo audio loaded! Click play to listen or transcribe.', 'success');
    } catch (error) {
      console.error('Error loading demo audio:', error);
      showSnackbar('Failed to load demo audio. Please try again.', 'error');
    }
  };

  const handleToggleDefaultChips = () => {
    console.log('3-dot button clicked, current messages:', messages.length);

    updateSuggestionsDismissed(false);

    // Remove existing chips and add fresh ones at the end
    setMessages((prev) => {
      // Filter out any existing default chips
      const withoutChips = prev.filter(m => !isDefaultChips(m));
      // Add new chips at the end
      const newChips = getDefaultChipsForRender([createDefaultSelections(agent_id)]);
      console.log('Adding fresh chips at end');
      return [...withoutChips, ...newChips];
    });

    // Scroll to bottom to show the chips
    setTimeout(() => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
    }, 150);
  };
  return (
    <Container
      maxWidth={false}
      disableGutters
      sx={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        padding: { xs: 1, sm: 2, md: 3 },
        backgroundColor: 'background.default',
        overflow: 'hidden',
      }}
    >
      {/* Chat Messages Container */}
      <Paper
        elevation={1}
        sx={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          borderRadius: { xs: 1, sm: 2 },
          maxWidth: '1400px',
          width: '100%',
          margin: '0 auto',
          background: `
            linear-gradient(45deg, transparent 25%, rgba(0,0,0,0.01) 25%),
            linear-gradient(-45deg, transparent 25%, rgba(0,0,0,0.01) 25%),
            linear-gradient(45deg, rgba(0,0,0,0.01) 75%, transparent 75%),
            linear-gradient(-45deg, rgba(0,0,0,0.01) 75%, transparent 75%)
          `,
          backgroundSize: '20px 20px',
          backgroundPosition: '0 0, 0 10px, 10px -10px, -10px 0px',
          backgroundColor: '#fafafa',
        }}
      >
        {/* Mobile Back Button */}
        {onMobileBack && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              padding: 1,
              borderBottom: '1px solid #e0e0e0',
              backgroundColor: '#ffffff',
            }}
          >
            <IconButton
              onClick={onMobileBack}
              sx={{
                marginRight: 1,
                color: '#0891b2',
                '&:hover': {
                  backgroundColor: 'rgba(8, 145, 178, 0.1)',
                },
              }}
            >
              <ArrowBackIcon />
            </IconButton>
            <Typography variant="h6" sx={{ color: '#2c2c2c', fontWeight: 'bold' }}>
              Chat
            </Typography>
          </Box>
        )}

        {/* Messages List */}
        <Box
          ref={el => (messagesContainerRef.current = el as HTMLDivElement | null)}
          onScroll={onScrollDebounced}
          sx={{
            flex: 1,
            overflow: 'auto',
            backgroundColor: 'background.paper',
            position: 'relative', // allow absolute loader overlay
          }}
        >
          {/* Top loader overlay shown when older messages are being fetched - always mounted and faded */}
          <Box
            sx={{
              position: 'absolute',
              top: 8,
              left: 0,
              right: 0,
              display: 'flex',
              justifyContent: 'center',
              pointerEvents: 'none',
              zIndex: 10,
              opacity: isLoadingOlder ? 1 : 0,
              transition: 'opacity 280ms ease',
            }}
          >
            <CircularProgress size={28} />
            <Box sx={{ ml: 1, pointerEvents: 'none' }}>
              <Typography variant="caption" sx={{ ml: 1 }}>Loading older messages...</Typography>
            </Box>
          </Box>

          {/* Bot switching overlay - shows subtle banner when switching bots */}
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              pointerEvents: 'none',
              zIndex: 9,
              opacity: isSwitchingBot ? 1 : 0,
              transition: 'opacity 300ms ease',
              backgroundColor: 'rgba(8, 145, 178, 0.95)',
              padding: '12px',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
            }}
          >
            <CircularProgress size={20} sx={{ color: 'white' }} />
            <Typography variant="body2" sx={{ ml: 1.5, color: 'white', fontWeight: 500 }}>
              Loading conversation...
            </Typography>
          </Box>

          {/* Fade overlay for old messages during bot switch */}
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(255, 255, 255, 0.7)',
              pointerEvents: 'none',
              zIndex: 8,
              opacity: isSwitchingBot ? 1 : 0,
              transition: 'opacity 300ms ease',
            }}
          />

          <List sx={{ paddingY: 1 }}>
            {messages.map((message, index) => (
              <MessageRenderer
                key={`${message.id}-${index}`}
                message={message}
                index={index}
                narrationStates={narrationStates}
                onStartNarration={startNarration}
                onPauseNarration={pauseNarration}
                onResumeNarration={resumeNarration}
                onStopNarration={stopNarration}
                onRestartNarration={restartNarration}
                onChipClick={handleChipClick}
                agent_id={agent_id}
              />
            ))}
          </List>
          <div ref={messagesEndRef} />
        </Box>

        {/* Input Form */}
        <Box
          component="form"
          onSubmit={handleSubmit}
          sx={{
            display: 'flex',
            alignItems: 'center',
            padding: { xs: 1.5, sm: 2 },
            borderTop: '1px solid',
            borderTopColor: '#d0d0d0',
            backgroundColor: '#ffffff',
            boxShadow: '0 -2px 8px rgba(0,0,0,0.1)',
          }}
        >
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Type your message..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            disabled={isLoading}
            sx={{
              marginRight: 1,
              '& .MuiOutlinedInput-root': {
                borderRadius: '24px',
                backgroundColor: '#ffffff',
                boxShadow: '0 2px 8px rgba(8, 145, 178, 0.1)',
                border: '2px solid rgba(8, 145, 178, 0.2)',
                '&:hover': {
                  boxShadow: '0 4px 12px rgba(8, 145, 178, 0.2)',
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: '#0891b2',
                  },
                },
                '&.Mui-focused': {
                  boxShadow: '0 4px 16px rgba(8, 145, 178, 0.3)',
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: '#0891b2',
                    borderWidth: '2px',
                  },
                },
                '& .MuiOutlinedInput-notchedOutline': {
                  border: 'none',
                },
              },
              '& .MuiInputBase-input': {
                color: '#2c2c2c',
              },
              '& .MuiInputBase-input::placeholder': {
                color: '#6a6a6a',
                opacity: 1,
              },
            }}
          />
          <IconButton
            onClick={handleToggleDefaultChips}
            disabled={isLoading}
            sx={{
              padding: { xs: 1.2, sm: 1.5 },
              backgroundColor: '#ffffff',
              color: '#0891b2',
              borderRadius: '50%',
              marginRight: 1,
              boxShadow: '0 3px 8px rgba(8, 145, 178, 0.2)',
              border: '2px solid rgba(8, 145, 178, 0.3)',
              '&:hover': {
                backgroundColor: '#f0f9ff',
                transform: 'scale(1.05)',
                boxShadow: '0 4px 12px rgba(8, 145, 178, 0.3)',
              },
              '&:disabled': {
                backgroundColor: '#f5f5f5',
                color: '#bdc3c7',
                boxShadow: 'none',
                border: '2px solid #e0e0e0',
              },
              transition: 'all 0.2s ease-in-out',
            }}
          >
            <MoreHorizIcon />
          </IconButton>
          {/* Microphone Button */}
          <IconButton
            onClick={handleMicButtonClick}
            disabled={isLoading}
            sx={{
              padding: { xs: 1.2, sm: 1.5 },
              backgroundColor: audioPermissionDenied ? '#f44336' : '#ffffff',
              color: audioPermissionDenied ? 'white' : '#0891b2',
              borderRadius: '50%',
              marginRight: 1,
              boxShadow: '0 3px 8px rgba(8, 145, 178, 0.2)',
              border: '2px solid rgba(8, 145, 178, 0.3)',
              '&:hover': {
                backgroundColor: audioPermissionDenied ? '#d32f2f' : '#f0f9ff',
                transform: 'scale(1.05)',
                boxShadow: '0 4px 12px rgba(8, 145, 178, 0.3)',
              },
              '&:disabled': {
                backgroundColor: '#f5f5f5',
                color: '#bdc3c7',
                boxShadow: 'none',
                border: '2px solid #e0e0e0',
              },
              transition: 'all 0.2s ease-in-out',
            }}
          >
            {audioPermissionDenied ? <MicOffIcon /> : <MicIcon />}
          </IconButton>
          
          {/* Send/Stop Button - shows Stop when generating */}
          {isGenerating ? (
            <IconButton
              onClick={handleStopGeneration}
              sx={{
                padding: { xs: 1.2, sm: 1.5 },
                backgroundColor: '#ef4444',
                color: 'white',
                borderRadius: '50%',
                boxShadow: '0 3px 8px rgba(239, 68, 68, 0.4)',
                border: '2px solid rgba(239, 68, 68, 0.2)',
                '&:hover': {
                  backgroundColor: '#dc2626',
                  transform: 'scale(1.05)',
                  boxShadow: '0 4px 12px rgba(239, 68, 68, 0.5)',
                },
                transition: 'all 0.2s ease-in-out',
              }}
            >
              <StopIcon />
            </IconButton>
          ) : (
            <IconButton
              type="submit"
              color="primary"
              disabled={!inputValue.trim() || isLoading}
              sx={{
                padding: { xs: 1.2, sm: 1.5 },
                backgroundColor: '#0891b2',
                color: 'white',
                borderRadius: '50%',
                boxShadow: '0 3px 8px rgba(8, 145, 178, 0.4)',
                border: '2px solid rgba(8, 145, 178, 0.2)',
                '&:hover': {
                  backgroundColor: '#0e7490',
                  transform: 'scale(1.05)',
                  boxShadow: '0 4px 12px rgba(8, 145, 178, 0.5)',
                },
                '&:disabled': {
                  backgroundColor: '#f5f5f5',
                  color: '#bdc3c7',
                  boxShadow: 'none',
                  border: '2px solid #e0e0e0',
                },
                transition: 'all 0.2s ease-in-out',
              }}
            >
              {isLoading ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                <SendIcon />
              )}
            </IconButton>
          )}
        </Box>
      </Paper>

      {/* Recording Modal */}
      <Dialog
        open={isRecordingModalOpen}
        onClose={handleCloseRecordingModal}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: 3,
            padding: 2,
            background: 'linear-gradient(135deg, #2c2c2c 0%, #4a4a4a 100%)',
            color: 'white',
          },
        }}
      >
        <DialogTitle sx={{ textAlign: 'center', color: 'white', paddingBottom: 1 }}>
          <Typography variant="h6" fontWeight="bold">
            Voice Recording
          </Typography>
          <IconButton
            onClick={handleCloseRecordingModal}
            sx={{
              position: 'absolute',
              right: 16,
              top: 16,
              color: 'white',
            }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        
        <DialogContent sx={{ textAlign: 'center', paddingTop: 2 }}>
          {/* Recording Status */}
          {isTranscribing ? (
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Transcribing Audio...
              </Typography>
              <LinearProgress
                sx={{
                  mb: 2,
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: 'white',
                  },
                }}
              />
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Please wait while we convert your speech to text
              </Typography>
            </Box>
          ) : (
            <Box sx={{ mb: 3 }}>
              {/* Waveform Canvas - shown during recording */}
              {isRecording && (
                <Box sx={{ mb: 3 }}>
                  <canvas
                    ref={canvasRef}
                    width={400}
                    height={100}
                    style={{
                      width: '100%',
                      maxWidth: '400px',
                      height: 'auto',
                      borderRadius: '8px',
                      border: '2px solid rgba(8, 145, 178, 0.5)',
                    }}
                  />
                </Box>
              )}

              {/* Recording Indicator */}
              {!recordedAudioUrl && (
                <Box
                  sx={{
                    width: 120,
                    height: 120,
                    borderRadius: '50%',
                    backgroundColor: isRecording ? '#ff1744' : 'rgba(255, 255, 255, 0.2)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto 20px',
                    animation: isRecording ? 'pulse 1.5s infinite' : 'none',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'scale(1.05)',
                    },
                  }}
                  onClick={isRecording ? stopRecording : startRecording}
                >
                  {isRecording ? (
                    <StopIcon sx={{ fontSize: 60, color: 'white' }} />
                  ) : (
                    <MicIcon sx={{ fontSize: 60, color: 'white' }} />
                  )}
                </Box>
              )}

              {/* Recording Time */}
              {(isRecording || recordedAudioUrl) && (
                <Typography variant="h4" sx={{ mb: 2, fontFamily: 'monospace' }}>
                  {formatTime(recordingTime)}
                </Typography>
              )}

              {/* Audio Playback Section */}
              {recordedAudioUrl && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h6" sx={{ mb: 2, color: '#4ade80' }}>
                    Recording Complete!
                  </Typography>

                  {/* Hidden audio element */}
                  <audio
                    ref={audioPlayerRef}
                    src={recordedAudioUrl}
                    onEnded={() => setIsPlayingRecording(false)}
                    style={{ display: 'none' }}
                  />

                  {/* Playback Controls */}
                  <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mb: 3 }}>
                    <IconButton
                      onClick={handlePlayRecording}
                      sx={{
                        backgroundColor: 'rgba(8, 145, 178, 0.2)',
                        color: '#0891b2',
                        border: '2px solid #0891b2',
                        '&:hover': {
                          backgroundColor: 'rgba(8, 145, 178, 0.3)',
                        },
                      }}
                    >
                      {isPlayingRecording ? <PauseIcon /> : <PlayArrowIcon />}
                    </IconButton>

                    <IconButton
                      onClick={handleDiscardRecording}
                      sx={{
                        backgroundColor: 'rgba(239, 68, 68, 0.2)',
                        color: '#ef4444',
                        border: '2px solid #ef4444',
                        '&:hover': {
                          backgroundColor: 'rgba(239, 68, 68, 0.3)',
                        },
                      }}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>

                  {/* Transcribe Button */}
                  <Button
                    variant="contained"
                    startIcon={<CheckCircleIcon />}
                    onClick={handleTranscribeRecording}
                    sx={{
                      backgroundColor: '#4ade80',
                      color: '#1a1a1a',
                      borderRadius: 25,
                      padding: '12px 30px',
                      fontWeight: 'bold',
                      '&:hover': {
                        backgroundColor: '#22c55e',
                      },
                    }}
                  >
                    Transcribe Audio
                  </Button>
                </Box>
              )}

              {/* Instructions */}
              {!recordedAudioUrl && (
                <Typography variant="body1" sx={{ mb: 2, opacity: 0.9 }}>
                  {isRecording
                    ? 'Recording... Click the stop button when finished'
                    : 'Click the microphone to start recording'
                  }
                </Typography>
              )}

              {/* Action Buttons */}
              {!recordedAudioUrl && (
                <Box sx={{ mt: 3, display: 'flex', flexDirection: 'column', gap: 2, alignItems: 'center' }}>
                  {!isRecording ? (
                    <>
                      <Button
                        variant="contained"
                        startIcon={<MicIcon />}
                        onClick={startRecording}
                        sx={{
                          backgroundColor: 'rgba(255, 255, 255, 0.2)',
                          color: 'white',
                          border: '2px solid white',
                          borderRadius: 25,
                          padding: '12px 30px',
                          fontWeight: 'bold',
                          '&:hover': {
                            backgroundColor: 'rgba(255, 255, 255, 0.3)',
                          },
                        }}
                      >
                        Start Recording
                      </Button>

                      <Typography variant="body2" sx={{ opacity: 0.7, my: 1 }}>
                        OR
                      </Typography>

                      <Button
                        variant="outlined"
                        onClick={handleLoadDemoAudio}
                        sx={{
                          color: 'white',
                          borderColor: 'white',
                          borderRadius: 25,
                          padding: '12px 30px',
                          fontWeight: 'bold',
                          '&:hover': {
                            borderColor: 'white',
                            backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          },
                        }}
                      >
                        Try Demo Audio
                      </Button>
                    </>
                  ) : (
                    <Button
                      variant="contained"
                      startIcon={<StopIcon />}
                      onClick={stopRecording}
                      sx={{
                        backgroundColor: '#ff1744',
                        color: 'white',
                        borderRadius: 25,
                        padding: '12px 30px',
                        fontWeight: 'bold',
                        '&:hover': {
                          backgroundColor: '#d50000',
                        },
                      }}
                    >
                      Stop Recording
                    </Button>
                  )}
                </Box>
              )}
            </Box>
          )}

          {/* Tips */}
          {!isRecording && !isTranscribing && !recordedAudioUrl && (
            <Paper
              elevation={0}
              sx={{
                padding: 2,
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                borderRadius: 2,
                mt: 2,
              }}
            >
              <Typography variant="caption" sx={{ color: 'white', opacity: 0.8 }}>
                💡 <strong>Tips:</strong> Speak clearly and ensure you're in a quiet environment for best results
              </Typography>
            </Paper>
          )}
        </DialogContent>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={4000}
        onClose={() => setSnackbarOpen(false)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setSnackbarOpen(false)}
          severity={snackbarSeverity}
          sx={{ width: '100%' }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>

      {/* CSS for pulse animation */}
      <style>
        {`
          @keyframes pulse {
            0% {
              box-shadow: 0 0 0 0 rgba(255, 23, 68, 0.7);
            }
            70% {
              box-shadow: 0 0 0 10px rgba(255, 23, 68, 0);
            }
            100% {
              box-shadow: 0 0 0 0 rgba(255, 23, 68, 0);
            }
          }

          @keyframes pulse {
            0%, 100% {
              opacity: 1;
              transform: scale(1);
            }
            50% {
              opacity: 0.7;
              transform: scale(1.1);
            }
          }
        `}
      </style>
    </Container>
  );
};

export default ChatBox;
