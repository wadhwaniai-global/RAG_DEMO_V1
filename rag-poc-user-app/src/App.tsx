import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import {
  Typography,
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Box,
  useMediaQuery,
} from '@mui/material';
import ChatScreen from './pages/ChatScreen';
import Login from './pages/Login';

const theme = createTheme({
  typography: {
    fontFamily: '"Trebuchet MS", "Lucida Grande", "Lucida Sans Unicode", "Lucida Sans", Tahoma, sans-serif',
    allVariants: {
      fontFamily: '"Trebuchet MS", "Lucida Grande", "Lucida Sans Unicode", "Lucida Sans", Tahoma, sans-serif',
    },
  },
  palette: {
    primary: {
      main: '#2c2c2c', // Charcoal black
    },
    secondary: {
      main: '#0891b2', // Teal accent color for textboxes and avatars
    },
    background: {
      default: '#ffffff', // Pure white background
      paper: '#ffffff',
    },
    text: {
      primary: '#2c2c2c', // Charcoal black for primary text
      secondary: '#6a6a6a', // Medium gray for secondary text
    },
  },
});







function App() {
  const [mobileView, setMobileView] = useState<'conversations' | 'chatbox'>('conversations');
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // Determine if navbar should be hidden
  const shouldHideNavbar = isMobile && mobileView === 'chatbox';
  
  // Callback to receive mobile view updates from ChatScreen
  const handleMobileViewChange = (view: 'conversations' | 'chatbox') => {
    setMobileView(view);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        {/* Branding Header - conditionally hidden on mobile ChatBox view */}
        {!shouldHideNavbar && (
          <AppBar 
            position="fixed" 
            elevation={0} 
            sx={{ 
              backgroundColor: '#ffffff', 
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              borderBottom: '1px solid #e0e0e0',
              zIndex: (theme) => theme.zIndex.drawer + 1
            }}
          >
            <Toolbar>
              <img
                src="/static/images/logo.png"
                alt="Logo"
                style={{
                  height: '68px',
                  width: 'auto',
                  marginRight: '16px',
                  padding: '4px 8px'
                }}
              />
              <Typography variant="h6" component="div" sx={{ flexGrow: 1, color: '#2c2c2c', textAlign: 'right' }}>
                HealthCare Assistant
              </Typography>
            </Toolbar>
          </AppBar>
        )}
        
        {/* Routes Container - Dynamic height based on navbar visibility */}
        <Box
          sx={{
            height: shouldHideNavbar ? '100vh' : 'calc(100vh - 64px)', // Full height when navbar hidden
            overflow: 'hidden', // Prevent container from scrolling
            display: 'flex',
            flexDirection: 'column',
            position: 'relative',
            marginTop: shouldHideNavbar ? '0px' : '64px', // No margin when navbar hidden
          }}
        >
          <Routes>
            <Route path="/" element={<Login />} />
            <Route path="/chats" element={<ChatScreen onMobileViewChange={handleMobileViewChange} />} />
            {/* Future routes can be added here */}
          </Routes>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App; 
