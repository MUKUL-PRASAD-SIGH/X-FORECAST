import React from 'react';
import { CyberpunkThemeProvider, GlobalCyberpunkStyles, CyberpunkContainer } from './theme/ThemeProvider';
import { AuthProvider } from './contexts/AuthContext';
import { MainDashboard } from './components/MainDashboard';

const App: React.FC = () => {
  return (
    <CyberpunkThemeProvider>
      <AuthProvider>
        <GlobalCyberpunkStyles />
        <CyberpunkContainer>
          <MainDashboard />
        </CyberpunkContainer>
      </AuthProvider>
    </CyberpunkThemeProvider>
  );
};

export default App;