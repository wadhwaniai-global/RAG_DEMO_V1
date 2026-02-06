import React, { useState, useEffect, useRef } from 'react';
import { Box } from '@mui/material';
import { Conversation } from '../types';
import { getConversations, getAvailableBots } from '../api';
import ChatBox from './Chatscreen/ChatBox';
import Conversations from './Chatscreen/Conversations';
import ResizableLayout from '../components/ResizableLayout';

function ChatScreen({ onMobileViewChange }: { onMobileViewChange?: (view: 'conversations' | 'chatbox') => void }) {
  // Conversations state
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [isLoadingConversations, setIsLoadingConversations] = useState(false);
  const hasAutoSelected = useRef(false);

  // Available bots state
  const [availableBots, setAvailableBots] = useState<any[]>([]);
  const [isBotsModalOpen, setIsBotsModalOpen] = useState(false);
  const [isLoadingBots, setIsLoadingBots] = useState(false);
  const [selectedBot, setSelectedBot] = useState<any>(null);

  // Load conversations on component mount and every 5 seconds
  useEffect(() => {
    const loadConversations = async (showSpinner = false) => {
      try {
        if (showSpinner) {
          setIsLoadingConversations(true);
        }
        const response = await getConversations();
        // Only update state if conversations actually changed to avoid re-renders/flicker
        setConversations(prev => {
          const prevIds = prev.map(c => c.participant.id);
          const nextIds = response.conversations.map(c => c.participant.id);
          const identical = prevIds.length === nextIds.length && prevIds.every((id, i) => id === nextIds[i]);
          return identical ? prev : response.conversations;
        });
        
        // Select the first conversation by default ONLY on initial load
        if (response.conversations.length > 0 && !hasAutoSelected.current) {
          const firstConv = response.conversations[0];
          setSelectedConversation(firstConv);
          setSelectedBot({
            id: firstConv.participant.id,
            name: firstConv.participant.name,
            email: firstConv.participant.email || '',
            description: `Conversation with ${firstConv.participant.name}`,
            user_type: 'bot',
            document_filter: '',
            is_active: true,
          });
          hasAutoSelected.current = true;
        }
      } catch (error) {
        console.error('Error loading conversations:', error);
      } finally {
        if (showSpinner) {
          setIsLoadingConversations(false);
        }
      }
    };

    // Load conversations immediately
    loadConversations(true);

    const handleVisibilityChange = () => {
      if (!document.hidden) {
        loadConversations();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Set up interval to load conversations every 15 seconds while tab is visible
    const interval = setInterval(() => {
      if (!document.hidden) {
        loadConversations();
      }
    }, 15000);

    // Cleanup interval on component unmount
    return () => {
      clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // Handle conversation selection
  const handleConversationSelect = (conversation: Conversation) => {
    setSelectedConversation(conversation);
    
    // Set the selected bot based on the conversation's participant (agent)
    setSelectedBot({
      id: conversation.participant.id,
      name: conversation.participant.name,
      email: conversation.participant.email || '',
      description: `Conversation with ${conversation.participant.name}`,
      user_type: 'bot',
      document_filter: '',
      is_active: true,
    });
  };

  // Helper function to truncate text
  const truncateText = (text: string, maxLength: number = 50) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  // Helper function to extract text from message content
  const getMessageText = (message: any): string => {
    if (typeof message === 'string') {
      return message;
    }
    if (message && typeof message === 'object' && message.text) {
      return message.text;
    }
    return 'No message content';
  };

  // Handle opening bots modal
  const handleOpenBotsModal = async () => {
    // If we already have bots cached, just open the modal without re-fetching
    if (availableBots && availableBots.length > 0) {
      setIsBotsModalOpen(true);
      return;
    }

    setIsBotsModalOpen(true);
    setIsLoadingBots(true);
    try {
      const bots = await getAvailableBots();
      setAvailableBots(bots);
    } catch (error) {
      console.error('Error fetching available bots:', error);
    } finally {
      setIsLoadingBots(false);
    }
  };

  // Handle closing bots modal
  const handleCloseBotsModal = () => {
    console.log('Closing modal...');
    // Keep the availableBots cached to avoid repeated refetch and UI jitter
    setIsBotsModalOpen(false);
  };

  // Handle bot selection
  const handleBotSelect = (bot: any) => {
    setSelectedBot(bot);
    setIsBotsModalOpen(false);
    // keep availableBots cached
    setSelectedConversation(null);
  };

  return (
    <Box sx={{ height: '100%', width: '100%' }}>
      <ResizableLayout
        leftPanel={
          <Conversations
            conversations={conversations}
            selectedConversation={selectedConversation}
            selectedBot={selectedBot}
            isLoadingConversations={isLoadingConversations}
            isLoadingBots={isLoadingBots}
            availableBots={availableBots}
            isBotsModalOpen={isBotsModalOpen}
            onConversationSelect={handleConversationSelect}
            onOpenBotsModal={handleOpenBotsModal}
            onCloseBotsModal={handleCloseBotsModal}
            onBotSelect={handleBotSelect}
            onTruncateText={truncateText}
            onGetMessageText={getMessageText}
          />
        }
        rightPanel={
          <ChatBox
            agent_id={selectedBot?.id || ''}
          />
        }
        initialLeftWidth={50}
        minLeftWidth={25}
        maxLeftWidth={75}
        onMobileViewChange={onMobileViewChange}
      />
    </Box>
  );
}

export default ChatScreen;