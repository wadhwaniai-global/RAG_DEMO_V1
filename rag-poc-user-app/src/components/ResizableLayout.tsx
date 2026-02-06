import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Box, useMediaQuery, useTheme, IconButton } from '@mui/material';
import { ChevronLeft, ChevronRight } from '@mui/icons-material';

interface ResizableLayoutProps {
  leftPanel: React.ReactNode;
  rightPanel: React.ReactNode;
  initialLeftWidth?: number; // Percentage (0-100)
  minLeftWidth?: number; // Percentage (0-100)
  maxLeftWidth?: number; // Percentage (0-100)
  onMobileViewChange?: (view: 'conversations' | 'chatbox') => void;
}

const ResizableLayout: React.FC<ResizableLayoutProps> = ({
  leftPanel,
  rightPanel,
  initialLeftWidth = 50,
  minLeftWidth = 20,
  maxLeftWidth = 80,
  onMobileViewChange,
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md')); // Below 900px
  const [leftWidth, setLeftWidth] = useState(initialLeftWidth);
  const [isDragging, setIsDragging] = useState(false);
  const [isLeftPanelVisible, setIsLeftPanelVisible] = useState(true);
  const [mobileView, setMobileView] = useState<'conversations' | 'chatbox'>('conversations');
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<HTMLDivElement>(null);

  // Clamp the width between min and max values
  const clampWidth = useCallback((width: number) => {
    return Math.min(Math.max(width, minLeftWidth), maxLeftWidth);
  }, [minLeftWidth, maxLeftWidth]);

  // Handle mouse move during drag
  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!containerRef.current) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const newLeftWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100;
    const clampedWidth = clampWidth(newLeftWidth);
    
    setLeftWidth(clampedWidth);
  }, [clampWidth]);

  // Handle mouse up to stop dragging
  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    
    // Remove global event listeners
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
    
    // Restore normal cursor and text selection
    document.body.style.userSelect = '';
    document.body.style.cursor = '';
  }, [handleMouseMove]);

  // Handle mouse down on drag handle
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    
    // Add global mouse event listeners
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    
    // Prevent text selection during drag
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';
  }, [handleMouseMove, handleMouseUp]);

  // Mobile navigation functions
  const showChatBox = useCallback(() => {
    setMobileView('chatbox');
    onMobileViewChange?.('chatbox');
  }, [onMobileViewChange]);

  const showConversations = useCallback(() => {
    setMobileView('conversations');
    onMobileViewChange?.('conversations');
  }, [onMobileViewChange]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
  }, [handleMouseMove, handleMouseUp]);

  // Mobile view - show only one panel at a time
  if (isMobile) {
    return (
      <Box
        sx={{
          height: '100%',
          width: '100%',
          backgroundColor: 'background.default',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        {mobileView === 'conversations' ? (
          <Box sx={{ height: '100%', width: '100%' }}>
            {React.cloneElement(leftPanel as React.ReactElement, {
              onMobileChatSelect: showChatBox,
            })}
          </Box>
        ) : (
          <Box sx={{ height: '100%', width: '100%' }}>
            {React.cloneElement(rightPanel as React.ReactElement, {
              onMobileBack: showConversations,
            })}
          </Box>
        )}
      </Box>
    );
  }

  // Desktop view - show both panels with drag handle
  return (
    <Box
      ref={containerRef}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'row',
        backgroundColor: 'background.default',
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      {/* Left Panel */}
      {isLeftPanelVisible && (
        <Box
          sx={{
            width: `${leftWidth}%`,
            height: '100%',
            overflow: 'hidden',
            transition: isDragging ? 'none' : 'width 0.2s ease-in-out',
          }}
        >
          {leftPanel}
        </Box>
      )}

      {/* Drag Handle */}
      {isLeftPanelVisible && (
        <Box
          ref={dragRef}
          onMouseDown={handleMouseDown}
          sx={{
            width: '6px',
            height: '100%',
            backgroundColor: isDragging ? '#0891b2' : '#d0d0d0',
            cursor: 'col-resize',
            position: 'relative',
            transition: 'background-color 0.2s ease',
            flexShrink: 0,
            '&:hover': {
              backgroundColor: '#0891b2',
              width: '8px',
            },
            '&::before': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: '-4px',
              right: '-4px',
              bottom: 0,
              backgroundColor: 'transparent',
              cursor: 'col-resize',
            },
          }}
        />
      )}

      {/* Toggle Button - Show/Hide Left Panel */}
      <IconButton
        onClick={() => setIsLeftPanelVisible(!isLeftPanelVisible)}
        sx={{
          position: 'absolute',
          left: isLeftPanelVisible ? `${leftWidth}%` : '0',
          top: '50%',
          transform: 'translateY(-50%)',
          zIndex: 1000,
          backgroundColor: '#0891b2',
          color: 'white',
          width: '32px',
          height: '32px',
          borderRadius: '50%',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            backgroundColor: '#0e7490',
            transform: 'translateY(-50%) scale(1.1)',
          },
        }}
      >
        {isLeftPanelVisible ? <ChevronLeft /> : <ChevronRight />}
      </IconButton>

      {/* Right Panel */}
      <Box
        sx={{
          width: isLeftPanelVisible ? `${100 - leftWidth}%` : '100%',
          height: '100%',
          overflow: 'hidden',
          transition: isDragging ? 'none' : 'width 0.3s ease-in-out',
        }}
      >
        {rightPanel}
      </Box>
    </Box>
  );
};

export default ResizableLayout;
