import React from 'react';
import { CyberpunkThemeProvider, GlobalCyberpunkStyles, CyberpunkContainer } from './theme/ThemeProvider';
import { MainDashboard } from './components/MainDashboard';

const App: React.FC = () => {
  return (
    <CyberpunkThemeProvider>
      <GlobalCyberpunkStyles />
      <CyberpunkContainer>
        <MainDashboard />
      </CyberpunkContainer>
    </CyberpunkThemeProvider>
  );
};

export default App;