import React from 'react';
import {
  Paper,
  Box,
  IconButton,
  List,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  Typography,
  Avatar,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Add as AddIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { Conversation } from '../../types';

interface ConversationsProps {
  conversations: Conversation[];
  selectedConversation: Conversation | null;
  selectedBot: any;
  isLoadingConversations: boolean;
  isLoadingBots: boolean;
  availableBots: any[];
  isBotsModalOpen: boolean;
  onConversationSelect: (conversation: Conversation) => void;
  onOpenBotsModal: () => void;
  onCloseBotsModal: () => void;
  onBotSelect: (bot: any) => void;
  onTruncateText: (text: string, maxLength?: number) => string;
  onGetMessageText: (message: any) => string;
  onMobileChatSelect?: () => void; // For mobile navigation
}

const ConversationsComponent: React.FC<ConversationsProps> = ({
  conversations,
  selectedConversation,
  selectedBot,
  isLoadingConversations,
  isLoadingBots,
  availableBots,
  isBotsModalOpen,
  onConversationSelect,
  onOpenBotsModal,
  onCloseBotsModal,
  onBotSelect,
  onTruncateText,
  onGetMessageText,
  onMobileChatSelect,
}) => {
  return (
    <>
      {/* Left Side - Conversations List */}
      <Paper
        elevation={2}
        sx={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          borderRadius: 0,
          backgroundColor: '#fafafa',
        }}
      >
        {/* Conversations Header */}
        <Box
          sx={{
            padding: 2,
            borderBottom: '1px solid #e0e0e0',
            backgroundColor: '#ffffff',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <Box>
            <Typography variant="h6" sx={{ color: '#2c2c2c', fontWeight: 'bold' }}>
              Conversations
            </Typography>
          </Box>
          <IconButton
            onClick={onOpenBotsModal}
            sx={{
              padding: 1,
              backgroundColor: '#0891b2',
              color: 'white',
              borderRadius: '50%',
              boxShadow: '0 2px 8px rgba(8, 145, 178, 0.3)',
              '&:hover': {
                backgroundColor: '#0e7490',
                transform: 'scale(1.05)',
                boxShadow: '0 4px 12px rgba(8, 145, 178, 0.4)',
              },
              transition: 'all 0.2s ease-in-out',
            }}
          >
            <AddIcon />
          </IconButton>
        </Box>

        {/* Conversations List */}
        <Box
          sx={{
            flex: 1,
            overflow: 'auto',
            backgroundColor: '#ffffff',
          }}
        >
          {isLoadingConversations ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', padding: 4 }}>
              <CircularProgress size={40} sx={{ color: '#0891b2' }} />
            </Box>
          ) : conversations.length === 0 ? (
            <Box sx={{ padding: 4, textAlign: 'center' }}>
              <Typography variant="body2" sx={{ color: '#6a6a6a' }}>
                No conversations found
              </Typography>
            </Box>
          ) : (
            <List sx={{ padding: 0 }}>
              {conversations.map((conversation, index) => (
                <React.Fragment key={conversation.participant.id}>
                  <ListItem
                    button
                    onClick={() => {
                      onConversationSelect(conversation);
                      // Trigger mobile navigation if callback is provided
                      if (onMobileChatSelect) {
                        onMobileChatSelect();
                      }
                    }}
                    sx={{
                      padding: 2,
                      backgroundColor: selectedConversation?.participant.id === conversation.participant.id 
                        ? 'rgba(8, 145, 178, 0.1)' 
                        : 'transparent',
                      borderLeft: selectedConversation?.participant.id === conversation.participant.id 
                        ? '4px solid #0891b2' 
                        : '4px solid transparent',
                      '&:hover': {
                        backgroundColor: 'rgba(8, 145, 178, 0.05)',
                      },
                      transition: 'all 0.2s ease',
                    }}
                  >
                    <ListItemAvatar>
                      <Avatar
                        sx={{
                          backgroundColor: '#0891b2',
                          color: 'white',
                          fontWeight: 'bold',
                        }}
                      >
                        {conversation.participant.name.charAt(0).toUpperCase()}
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={
                        <Typography
                          variant="subtitle2"
                          sx={{
                            fontWeight: 'bold',
                            color: '#2c2c2c',
                            marginBottom: 0.5,
                          }}
                        >
                          {conversation.participant.name}
                        </Typography>
                      }
                      secondary={
                        <Typography
                          variant="body2"
                          sx={{
                            color: '#6a6a6a',
                            lineHeight: 1.3,
                          }}
                        >
                          {onTruncateText(onGetMessageText(conversation.last_message.message))}
                        </Typography>
                      }
                    />
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', marginLeft: 1 }}>
                      <Typography variant="caption" sx={{ color: '#9e9e9e', marginBottom: 0.5 }}>
                        {conversation.message_count} message{conversation.message_count !== 1 ? 's' : ''}
                      </Typography>
                    </Box>
                  </ListItem>
                  {index < conversations.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          )}
        </Box>
      </Paper>

      {/* Available Agents Modal */}
      <Dialog
        open={isBotsModalOpen}
        onClose={onCloseBotsModal}
        maxWidth="md"
        fullWidth
        disableEscapeKeyDown={false}
        PaperProps={{
          sx: {
            borderRadius: 3,
            padding: 2,
            backgroundColor: '#ffffff',
            boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
          },
        }}
      >
        <DialogTitle sx={{
          textAlign: 'center',
          color: '#2c2c2c',
          paddingBottom: 1,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <Typography variant="h5" fontWeight="bold">
            Available Agents
          </Typography>
          <IconButton
            aria-label="close"
            onClick={() => {
              console.log('Close button clicked!');
              onCloseBotsModal();
            }}
            disableRipple={false}
            sx={{
              color: '#6a6a6a',
              padding: '8px',
              minWidth: '40px',
              minHeight: '40px',
              '&:hover': {
                backgroundColor: 'rgba(0,0,0,0.08)',
              },
            }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        
        <DialogContent sx={{ paddingTop: 2 }}>
          {isLoadingBots ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', padding: 4 }}>
              <CircularProgress size={40} sx={{ color: '#0891b2' }} />
            </Box>
          ) : availableBots.length === 0 ? (
            <Box sx={{ padding: 4, textAlign: 'center' }}>
              <Typography variant="body2" sx={{ color: '#6a6a6a' }}>
                No agents available
              </Typography>
            </Box>
          ) : (
            <List sx={{ padding: 0 }}>
              {availableBots.map((bot, index) => (
                <React.Fragment key={bot.id}>
                  <ListItem
                    button
                    onClick={() => onBotSelect(bot)}
                    sx={{
                      padding: 2,
                      borderRadius: 2,
                      marginBottom: 1,
                      backgroundColor: selectedBot?.id === bot.id ? 'rgba(8, 145, 178, 0.1)' : '#fafafa',
                      border: selectedBot?.id === bot.id ? '2px solid #0891b2' : '1px solid #e0e0e0',
                      cursor: 'pointer',
                      '&:hover': {
                        backgroundColor: 'rgba(8, 145, 178, 0.05)',
                        borderColor: '#0891b2',
                      },
                      transition: 'all 0.2s ease',
                    }}
                  >
                    <ListItemAvatar>
                      <Avatar
                        sx={{
                          backgroundColor: '#0891b2',
                          color: 'white',
                          fontWeight: 'bold',
                        }}
                      >
                        {bot.name.charAt(0).toUpperCase()}
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={
                        <Typography
                          variant="subtitle1"
                          sx={{
                            fontWeight: 'bold',
                            color: '#2c2c2c',
                            marginBottom: 0.5,
                          }}
                        >
                          {bot.name}
                        </Typography>
                      }
                      secondary={
                        <Box>
                          <Typography
                            variant="body2"
                            sx={{
                              color: '#6a6a6a',
                              lineHeight: 1.3,
                              marginBottom: 0.5,
                            }}
                          >
                            {bot.description}
                          </Typography>
                          <Typography
                            variant="caption"
                            sx={{
                              color: '#9e9e9e',
                              display: 'block',
                            }}
                          >
                            Email: {bot.email}
                          </Typography>
                          <Typography
                            variant="caption"
                            sx={{
                              color: '#9e9e9e',
                              display: 'block',
                            }}
                          >
                            Document Filter: {bot.document_filter}
                          </Typography>
                          <Typography
                            variant="caption"
                            sx={{
                              color: bot.is_active ? '#4caf50' : '#f44336',
                              display: 'block',
                              fontWeight: 'bold',
                            }}
                          >
                            Status: {bot.is_active ? 'Active' : 'Inactive'}
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItem>
                  {index < availableBots.length - 1 && <Divider sx={{ marginY: 1 }} />}
                </React.Fragment>
              ))}
            </List>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
};

// Simple shallow prop comparison for memoization (only checks conversations length and selected ids)
const areEqual = (prev: ConversationsProps, next: ConversationsProps) => {
  if (prev.selectedConversation?.participant.id !== next.selectedConversation?.participant.id) return false;
  if (prev.selectedBot?.id !== next.selectedBot?.id) return false;
  if (prev.availableBots?.length !== next.availableBots?.length) return false;
  if (prev.conversations?.length !== next.conversations?.length) return false;
  if (prev.isBotsModalOpen !== next.isBotsModalOpen) return false;
  if (prev.isLoadingBots !== next.isLoadingBots) return false;
  // Further deep checks could be added if needed
  return true;
};

const Conversations = React.memo(ConversationsComponent, areEqual);

export default Conversations;
