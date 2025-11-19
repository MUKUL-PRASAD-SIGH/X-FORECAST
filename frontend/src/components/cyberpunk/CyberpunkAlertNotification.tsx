import React, { useState, useEffect } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkButton } from '../ui';

// Animation keyframes
const glitchAnimation = keyframes`
  0%, 100% { 
    transform: translate(0);
    filter: hue-rotate(0deg);
  }
  10% { 
    transform: translate(-2px, 2px);
    filter: hue-rotate(90deg);
  }
  20% { 
    transform: translate(-2px, -2px);
    filter: hue-rotate(180deg);
  }
  30% { 
    transform: translate(2px, 2px);
    filter: hue-rotate(270deg);
  }
  40% { 
    transform: translate(2px, -2px);
    filter: hue-rotate(360deg);
  }
  50% { 
    transform: translate(-1px, 1px);
    filter: hue-rotate(45deg);
  }
  60% { 
    transform: translate(-1px, -1px);
    filter: hue-rotate(135deg);
  }
  70% { 
    transform: translate(1px, 1px);
    filter: hue-rotate(225deg);
  }
  80% { 
    transform: translate(1px, -1px);
    filter: hue-rotate(315deg);
  }
  90% { 
    transform: translate(0);
    filter: hue-rotate(0deg);
  }
`;

const scanLine = keyframes`
  0% { left: -100%; }
  100% { left: 100%; }
`;

const alertPulse = keyframes`
  0%, 100% { 
    box-shadow: 0 0 20px rgba(255, 0, 64, 0.6);
    border-color: rgba(255, 0, 64, 0.8);
  }
  50% { 
    box-shadow: 0 0 40px rgba(255, 0, 64, 0.9);
    border-color: rgba(255, 0, 64, 1);
  }
`;

const warningPulse = keyframes`
  0%, 100% { 
    box-shadow: 0 0 20px rgba(255, 255, 0, 0.6);
    border-color: rgba(255, 255, 0, 0.8);
  }
  50% { 
    box-shadow: 0 0 40px rgba(255, 255, 0, 0.9);
    border-color: rgba(255, 255, 0, 1);
  }
`;

const infoPulse = keyframes`
  0%, 100% { 
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.6);
    border-color: rgba(0, 255, 255, 0.8);
  }
  50% { 
    box-shadow: 0 0 40px rgba(0, 255, 255, 0.9);
    border-color: rgba(0, 255, 255, 1);
  }
`;

// Styled components
const AlertContainer = styled(motion.div).withConfig({
  shouldForwardProp: (prop) => !['$severity', '$glitchActive'].includes(prop)
})<{ 
  $severity: 'critical' | 'warning' | 'info';
  $glitchActive: boolean;
}>`
  position: relative;
  background: rgba(0, 0, 0, 0.9);
  border: 2px solid ${props => {
    switch (props.$severity) {
      case 'critical': return props.theme.colors.error;
      case 'warning': return props.theme.colors.cyberYellow;
      case 'info': return props.theme.colors.neonBlue;
      default: return props.theme.colors.secondaryText;
    }
  }};
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  overflow: hidden;
  backdrop-filter: blur(10px);
  
  ${props => props.$severity === 'critical' && css`
    animation: ${alertPulse} 2s infinite;
  `}
  
  ${props => props.$severity === 'warning' && css`
    animation: ${warningPulse} 3s infinite;
  `}
  
  ${props => props.$severity === 'info' && css`
    animation: ${infoPulse} 4s infinite;
  `}
  
  ${props => props.$glitchActive && css`
    animation: ${glitchAnimation} 0.5s infinite;
  `}
  
  /* Cyberpunk corner accents */
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 20px;
    height: 20px;
    border-top: 3px solid ${props => props.theme.colors.hotPink};
    border-left: 3px solid ${props => props.theme.colors.hotPink};
    opacity: 0.8;
  }
  
  &::after {
    content: '';
    position: absolute;
    bottom: 0;
    right: 0;
    width: 20px;
    height: 20px;
    border-bottom: 3px solid ${props => props.theme.colors.acidGreen};
    border-right: 3px solid ${props => props.theme.colors.acidGreen};
    opacity: 0.8;
  }
`;

const ScanLineEffect = styled.div.withConfig({
  shouldForwardProp: (prop) => !['$severity'].includes(prop)
})<{ $severity: 'critical' | 'warning' | 'info' }>`
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 2px;
  background: linear-gradient(90deg, 
    transparent 0%, 
    ${props => {
      switch (props.$severity) {
        case 'critical': return props.theme.colors.error;
        case 'warning': return props.theme.colors.cyberYellow;
        case 'info': return props.theme.colors.neonBlue;
        default: return props.theme.colors.secondaryText;
      }
    }} 50%, 
    transparent 100%
  );
  animation: ${scanLine} 2s infinite;
  z-index: 2;
`;

const AlertHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
  position: relative;
  z-index: 3;
`;

const AlertIcon = styled.div.withConfig({
  shouldForwardProp: (prop) => !['$severity'].includes(prop)
})<{ $severity: 'critical' | 'warning' | 'info' }>`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  font-weight: bold;
  background: ${props => {
    switch (props.$severity) {
      case 'critical': return props.theme.colors.error;
      case 'warning': return props.theme.colors.cyberYellow;
      case 'info': return props.theme.colors.neonBlue;
      default: return props.theme.colors.secondaryText;
    }
  }};
  color: ${props => props.theme.colors.darkBg};
  box-shadow: 0 0 20px ${props => {
    switch (props.$severity) {
      case 'critical': return 'rgba(255, 0, 64, 0.6)';
      case 'warning': return 'rgba(255, 255, 0, 0.6)';
      case 'info': return 'rgba(0, 255, 255, 0.6)';
      default: return 'rgba(255, 255, 255, 0.3)';
    }
  }};
  margin-right: 1rem;
  flex-shrink: 0;
`;

const AlertContent = styled.div`
  flex: 1;
  position: relative;
  z-index: 3;
`;

const AlertTitle = styled.h4.withConfig({
  shouldForwardProp: (prop) => !['$severity'].includes(prop)
})<{ $severity: 'critical' | 'warning' | 'info' }>`
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.lg};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  color: ${props => {
    switch (props.$severity) {
      case 'critical': return props.theme.colors.error;
      case 'warning': return props.theme.colors.cyberYellow;
      case 'info': return props.theme.colors.neonBlue;
      default: return props.theme.colors.primaryText;
    }
  }};
  margin: 0 0 0.5rem 0;
  text-shadow: ${props => props.theme.effects.softGlow};
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const AlertMessage = styled.p`
  font-family: ${props => props.theme.typography.fontFamily.primary};
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.primaryText};
  margin: 0 0 1rem 0;
  line-height: 1.5;
`;

const AlertDetails = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.secondaryText};
  background: rgba(0, 0, 0, 0.5);
  padding: 0.75rem;
  border-radius: 4px;
  border: 1px solid rgba(0, 255, 255, 0.3);
  margin-bottom: 1rem;
  white-space: pre-wrap;
  overflow-x: auto;
`;

const AlertActions = styled.div`
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  position: relative;
  z-index: 3;
`;

const AlertTimestamp = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.secondaryText};
  text-align: right;
  margin-left: auto;
  opacity: 0.7;
`;

const GlitchToggle = styled.button.withConfig({
  shouldForwardProp: (prop) => !['$active'].includes(prop)
})<{ $active: boolean }>`
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  width: 30px;
  height: 30px;
  border: none;
  background: ${props => props.$active ? props.theme.colors.error : 'transparent'};
  color: ${props => props.$active ? props.theme.colors.darkBg : props.theme.colors.secondaryText};
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.8rem;
  transition: all 0.3s ease;
  z-index: 4;
  
  &:hover {
    background: ${props => props.theme.colors.error};
    color: ${props => props.theme.colors.darkBg};
  }
`;

// Interfaces
export interface AlertNotificationProps {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  title: string;
  message: string;
  details?: string | object;
  timestamp: string;
  autoClose?: boolean;
  autoCloseDelay?: number;
  onClose?: (id: string) => void;
  onAction?: (id: string, action: string) => void;
  actions?: Array<{
    label: string;
    action: string;
    variant?: 'primary' | 'secondary' | 'danger';
  }>;
}

export const CyberpunkAlertNotification: React.FC<AlertNotificationProps> = ({
  id,
  severity,
  title,
  message,
  details,
  timestamp,
  autoClose = false,
  autoCloseDelay = 5000,
  onClose,
  onAction,
  actions = []
}) => {
  const [glitchActive, setGlitchActive] = useState(severity === 'critical');
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    if (autoClose && autoCloseDelay > 0) {
      const timer = setTimeout(() => {
        handleClose();
      }, autoCloseDelay);

      return () => clearTimeout(timer);
    }
  }, [autoClose, autoCloseDelay]);

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => {
      onClose?.(id);
    }, 300);
  };

  const handleAction = (action: string) => {
    onAction?.(id, action);
  };

  const getAlertIcon = () => {
    switch (severity) {
      case 'critical': return '⚠';
      case 'warning': return '⚡';
      case 'info': return 'ℹ';
      default: return '•';
    }
  };

  const formatDetails = () => {
    if (!details) return '';
    if (typeof details === 'string') return details;
    return JSON.stringify(details, null, 2);
  };

  const formatTimestamp = () => {
    return new Date(timestamp).toLocaleString();
  };

  if (!isVisible) return null;

  return (
    <AnimatePresence>
      <AlertContainer
        $severity={severity}
        $glitchActive={glitchActive}
        initial={{ opacity: 0, x: 300, scale: 0.8 }}
        animate={{ opacity: 1, x: 0, scale: 1 }}
        exit={{ opacity: 0, x: 300, scale: 0.8 }}
        transition={{ 
          type: "spring", 
          stiffness: 300, 
          damping: 30,
          duration: 0.5 
        }}
        whileHover={{ scale: 1.02 }}
      >
        <ScanLineEffect $severity={severity} />
        
        <GlitchToggle
          $active={glitchActive}
          onClick={() => setGlitchActive(!glitchActive)}
          title="Toggle glitch effect"
        >
          ⚡
        </GlitchToggle>

        <AlertHeader>
          <div style={{ display: 'flex', alignItems: 'flex-start' }}>
            <AlertIcon $severity={severity}>
              {getAlertIcon()}
            </AlertIcon>
            <AlertContent>
              <AlertTitle $severity={severity}>
                {title}
              </AlertTitle>
              <AlertMessage>
                {message}
              </AlertMessage>
            </AlertContent>
          </div>
          <AlertTimestamp>
            {formatTimestamp()}
          </AlertTimestamp>
        </AlertHeader>

        {details && (
          <AlertDetails>
            {formatDetails()}
          </AlertDetails>
        )}

        <AlertActions>
          {actions.map((action, index) => (
            <CyberpunkButton
              key={index}
              variant={action.variant || 'secondary'}
              size="sm"
              onClick={() => handleAction(action.action)}
            >
              {action.label}
            </CyberpunkButton>
          ))}
          
          <CyberpunkButton
            variant="ghost"
            size="sm"
            onClick={handleClose}
          >
            Dismiss
          </CyberpunkButton>
        </AlertActions>
      </AlertContainer>
    </AnimatePresence>
  );
};

export default CyberpunkAlertNotification;