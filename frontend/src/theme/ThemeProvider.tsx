import React, { createContext, useContext, ReactNode } from 'react';
import { ThemeProvider as StyledThemeProvider } from 'styled-components';
import { cyberpunkTheme, CyberpunkTheme } from './cyberpunkTheme';
import { PropFilterWrapper } from './StyleSheetManager';

// Theme context
const ThemeContext = createContext<CyberpunkTheme>(cyberpunkTheme);

// Custom hook to use theme
export const useCyberpunkTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useCyberpunkTheme must be used within a CyberpunkThemeProvider');
  }
  return context;
};

// Theme provider component
interface CyberpunkThemeProviderProps {
  children: ReactNode;
  theme?: CyberpunkTheme;
}

export const CyberpunkThemeProvider: React.FC<CyberpunkThemeProviderProps> = ({
  children,
  theme = cyberpunkTheme,
}) => {
  return (
    <PropFilterWrapper>
      <ThemeContext.Provider value={theme}>
        <StyledThemeProvider theme={theme}>
          {children}
        </StyledThemeProvider>
      </ThemeContext.Provider>
    </PropFilterWrapper>
  );
};

// Global styles component
import styled, { createGlobalStyle } from 'styled-components';
import { cyberpunkCSSVariables } from './cyberpunkTheme';

export const GlobalCyberpunkStyles = createGlobalStyle`
  ${cyberpunkCSSVariables}
  
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  
  body {
    font-family: var(--font-primary);
    background: var(--gradient-background);
    color: var(--color-primary-text);
    overflow-x: hidden;
    min-height: 100vh;
    

    
    /* Subtle grid pattern */
    &::after {
      content: '';
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-image: 
        linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
      background-size: 50px 50px;
      pointer-events: none;
      z-index: -1;
    }
  }
  
  html {
    scroll-behavior: smooth;
  }
  
  /* Custom scrollbar */
  ::-webkit-scrollbar {
    width: 8px;
  }
  
  ::-webkit-scrollbar-track {
    background: var(--color-darker-bg);
  }
  
  ::-webkit-scrollbar-thumb {
    background: var(--color-neon-blue);
    border-radius: 4px;
    box-shadow: var(--effect-soft-glow);
  }
  
  ::-webkit-scrollbar-thumb:hover {
    background: var(--color-hot-pink);
    box-shadow: var(--effect-neon-glow);
  }
  
  /* Selection styles */
  ::selection {
    background: var(--color-neon-blue);
    color: var(--color-dark-bg);
  }
  
  ::-moz-selection {
    background: var(--color-neon-blue);
    color: var(--color-dark-bg);
  }
  
  /* Focus styles */
  *:focus {
    outline: 2px solid var(--color-neon-blue);
    outline-offset: 2px;
  }
  
  /* Link styles */
  a {
    color: var(--color-accent-text);
    text-decoration: none;
    transition: all 0.3s ease;
    
    &:hover {
      color: var(--color-hot-pink);
      text-shadow: var(--effect-soft-glow);
    }
  }
  
  /* Button reset */
  button {
    border: none;
    background: none;
    cursor: pointer;
    font-family: inherit;
  }
  
  /* Input styles */
  input, textarea, select {
    font-family: inherit;
    background: var(--color-card-bg);
    border: 1px solid var(--color-neon-blue);
    color: var(--color-primary-text);
    border-radius: 4px;
    padding: 8px 12px;
    transition: all 0.3s ease;
    
    &:focus {
      border-color: var(--color-hot-pink);
      box-shadow: var(--effect-soft-glow);
    }
    
    &::placeholder {
      color: var(--color-secondary-text);
    }
  }
  
  /* Headings */
  h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-display);
    font-weight: 700;
    text-shadow: var(--effect-soft-glow);
  }
  
  h1 {
    font-size: var(--font-size-display);
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  /* Code styles */
  code, pre {
    font-family: var(--font-mono);
    background: var(--color-card-bg);
    border: 1px solid var(--color-neon-blue);
    border-radius: 4px;
    padding: 2px 4px;
  }
  
  pre {
    padding: 16px;
    overflow-x: auto;
  }
  
  /* Utility classes */
  .glow-text {
    text-shadow: var(--effect-soft-glow);
  }
  
  .intense-glow-text {
    text-shadow: var(--effect-neon-glow);
  }
  
  .glitch-effect {
    animation: glitch 0.3s ease-in-out infinite alternate;
  }
  
  .pulse-effect {
    animation: pulse 2s ease-in-out infinite;
  }
  
  .gradient-text {
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  .hologram-filter {
    filter: var(--hologram-filter);
  }
  
  /* Loading animation */
  .loading-dots {
    &::after {
      content: '';
      animation: loading-dots 1.5s infinite;
    }
  }
  
  @keyframes loading-dots {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
  }
`;

// Styled container for the entire app
export const CyberpunkContainer = styled.div`
  min-height: 100vh;
  background: ${props => props.theme.effects.backgroundGradient};
  position: relative;
  
  /* Ambient particles effect */
  &::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
      radial-gradient(circle at 20% 80%, rgba(0, 255, 255, 0.1) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(255, 20, 147, 0.1) 0%, transparent 50%),
      radial-gradient(circle at 40% 40%, rgba(57, 255, 20, 0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: -1;
  }
`;