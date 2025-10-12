import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkButton, CyberpunkInput } from '../ui';

// Types for chat interface
interface ChatMessage {
  id: string;
  type: 'user' | 'ai' | 'system';
  content: string;
  timestamp: Date;
  confidence?: number;
  sources?: string[];
  followUpQuestions?: string[];
  dataVisualization?: any;
}

interface ChatInterfaceProps {
  isOpen: boolean;
  onClose: () => void;
  onSendMessage: (message: string) => Promise<ChatMessage>;
  isLoading?: boolean;
}

// Styled components
const ChatOverlay = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(10px);
  z-index: 1000;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
`;

const ChatContainer = styled(motion.div)`
  width: 100%;
  max-width: 800px;
  height: 80vh;
  background: ${props => props.theme.colors.cardBg};
  border: 2px solid ${props => props.theme.colors.neonBlue};
  border-radius: 12px;
  box-shadow: ${props => props.theme.effects.neonGlow};
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: ${props => props.theme.effects.primaryGradient};
    animation: scan 3s linear infinite;
  }
  
  @keyframes scan {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
`;

const ChatHeader = styled.div`
  padding: 16px 20px;
  background: ${props => props.theme.colors.darkerBg};
  border-bottom: 1px solid ${props => props.theme.colors.neonBlue};
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ChatTitle = styled.h3`
  color: ${props => props.theme.colors.neonBlue};
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.lg};
  text-shadow: ${props => props.theme.effects.softGlow};
  margin: 0;
  
  &::before {
    content: '🤖 ';
    margin-right: 8px;
  }
`;

const CloseButton = styled.button`
  background: none;
  border: 1px solid ${props => props.theme.colors.hotPink};
  color: ${props => props.theme.colors.hotPink};
  width: 32px;
  height: 32px;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  
  &:hover {
    background: ${props => props.theme.colors.hotPink};
    color: ${props => props.theme.colors.darkBg};
    box-shadow: ${props => props.theme.effects.softGlow};
  }
`;

const ChatMessages = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  
  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: ${props => props.theme.colors.darkerBg};
  }
  
  &::-webkit-scrollbar-thumb {
    background: ${props => props.theme.colors.neonBlue};
    border-radius: 3px;
  }
`;

const MessageBubble = styled(motion.div)<{ messageType: 'user' | 'ai' | 'system' }>`
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 12px;
  position: relative;
  word-wrap: break-word;
  
  ${props => props.messageType === 'user' && `
    align-self: flex-end;
    background: ${props.theme.colors.neonBlue};
    color: ${props.theme.colors.darkBg};
    border-bottom-right-radius: 4px;
    
    &::after {
      content: '';
      position: absolute;
      bottom: 0;
      right: -8px;
      width: 0;
      height: 0;
      border: 8px solid transparent;
      border-left-color: ${props.theme.colors.neonBlue};
      border-bottom: none;
      border-right: none;
    }
  `}
  
  ${props => props.messageType === 'ai' && `
    align-self: flex-start;
    background: ${props.theme.colors.cardBg};
    border: 1px solid ${props.theme.colors.hotPink};
    color: ${props.theme.colors.primaryText};
    border-bottom-left-radius: 4px;
    box-shadow: ${props.theme.effects.softGlow};
    
    &::after {
      content: '';
      position: absolute;
      bottom: 0;
      left: -8px;
      width: 0;
      height: 0;
      border: 8px solid transparent;
      border-right-color: ${props.theme.colors.hotPink};
      border-bottom: none;
      border-left: none;
    }
  `}
  
  ${props => props.messageType === 'system' && `
    align-self: center;
    background: ${props.theme.colors.darkerBg};
    border: 1px solid ${props.theme.colors.acidGreen};
    color: ${props.theme.colors.acidGreen};
    font-size: ${props.theme.typography.fontSize.sm};
    font-family: ${props.theme.typography.fontFamily.mono};
    text-align: center;
  `}
`;

const MessageContent = styled.div`
  margin-bottom: 8px;
  line-height: 1.5;
`;

const MessageMeta = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: ${props => props.theme.typography.fontSize.xs};
  opacity: 0.7;
  margin-top: 8px;
`;

const ConfidenceBar = styled.div<{ confidence: number }>`
  width: 60px;
  height: 4px;
  background: ${props => props.theme.colors.darkerBg};
  border-radius: 2px;
  overflow: hidden;
  
  &::after {
    content: '';
    display: block;
    width: ${props => props.confidence * 100}%;
    height: 100%;
    background: ${props => props.confidence > 0.8 ? props.theme.colors.acidGreen : 
                      props.confidence > 0.6 ? props.theme.colors.cyberYellow : 
                      props.theme.colors.hotPink};
    transition: width 0.3s ease;
  }
`;

const FollowUpQuestions = styled.div`
  margin-top: 12px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
`;

const FollowUpButton = styled.button`
  background: none;
  border: 1px solid ${props => props.theme.colors.acidGreen};
  color: ${props => props.theme.colors.acidGreen};
  padding: 4px 8px;
  border-radius: 12px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: ${props => props.theme.colors.acidGreen};
    color: ${props => props.theme.colors.darkBg};
    box-shadow: ${props => props.theme.effects.softGlow};
  }
`;

const ChatInput = styled.div`
  padding: 20px;
  background: ${props => props.theme.colors.darkerBg};
  border-top: 1px solid ${props => props.theme.colors.neonBlue};
  display: flex;
  gap: 12px;
  align-items: flex-end;
`;

const InputContainer = styled.div`
  flex: 1;
  position: relative;
`;

const VoiceButton = styled.button`
  background: ${props => props.theme.colors.cardBg};
  border: 1px solid ${props => props.theme.colors.electricPurple};
  color: ${props => props.theme.colors.electricPurple};
  width: 40px;
  height: 40px;
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  
  &:hover {
    background: ${props => props.theme.colors.electricPurple};
    color: ${props => props.theme.colors.darkBg};
    box-shadow: ${props => props.theme.effects.softGlow};
  }
  
  &.recording {
    animation: pulse 1s infinite;
    background: ${props => props.theme.colors.error};
    border-color: ${props => props.theme.colors.error};
  }
`;

const TypingIndicator = styled(motion.div)`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: ${props => props.theme.colors.cardBg};
  border: 1px solid ${props => props.theme.colors.hotPink};
  border-radius: 12px;
  align-self: flex-start;
  
  .dot {
    width: 6px;
    height: 6px;
    background: ${props => props.theme.colors.hotPink};
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
    
    &:nth-child(2) { animation-delay: 0.2s; }
    &:nth-child(3) { animation-delay: 0.4s; }
  }
  
  @keyframes typing {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-10px); }
  }
`;

const SuggestedQuestions = styled.div`
  padding: 16px 20px;
  background: ${props => props.theme.colors.darkerBg};
  border-top: 1px solid ${props => props.theme.colors.neonBlue};
`;

const SuggestedQuestionsTitle = styled.h4`
  color: ${props => props.theme.colors.secondaryText};
  font-size: ${props => props.theme.typography.fontSize.sm};
  margin: 0 0 12px 0;
  font-family: ${props => props.theme.typography.fontFamily.mono};
`;

const SuggestedQuestionsList = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
`;

const SuggestedQuestionButton = styled.button`
  background: none;
  border: 1px solid ${props => props.theme.colors.secondaryText};
  color: ${props => props.theme.colors.secondaryText};
  padding: 6px 12px;
  border-radius: 16px;
  font-size: ${props => props.theme.typography.fontSize.xs};
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    border-color: ${props => props.theme.colors.neonBlue};
    color: ${props => props.theme.colors.neonBlue};
    box-shadow: ${props => props.theme.effects.softGlow};
  }
`;

// Main component
export const CyberpunkChatInterface: React.FC<ChatInterfaceProps> = ({
  isOpen,
  onClose,
  onSendMessage,
  isLoading = false
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'system',
      content: 'AI Assistant initialized. Ready to help with forecasting and analytics.',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const suggestedQuestions = [
    "What's the forecast for next quarter?",
    "Show me customer retention trends",
    "Which products are at risk?",
    "Explain the latest anomalies",
    "What opportunities should we focus on?",
    "How accurate are our models?"
  ];

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: content.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');

    try {
      // Get AI response
      const aiResponse = await onSendMessage(content.trim());
      setMessages(prev => [...prev, aiResponse]);
    } catch (error) {
      // Add error message
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'system',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(inputValue);
    }
  };

  const handleFollowUpClick = (question: string) => {
    handleSendMessage(question);
  };

  const handleVoiceToggle = () => {
    if (isRecording) {
      // Stop recording
      setIsRecording(false);
      // In a real implementation, you would stop speech recognition here
    } else {
      // Start recording
      setIsRecording(true);
      // In a real implementation, you would start speech recognition here
      // For demo, we'll just simulate it
      setTimeout(() => {
        setIsRecording(false);
        setInputValue("What's the forecast for next month?");
      }, 3000);
    }
  };

  if (!isOpen) return null;

  return (
    <ChatOverlay
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <ChatContainer
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
      >
        <ChatHeader>
          <ChatTitle>AI Assistant</ChatTitle>
          <CloseButton onClick={onClose}>×</CloseButton>
        </ChatHeader>

        <ChatMessages>
          <AnimatePresence>
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                messageType={message.type}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <MessageContent>{message.content}</MessageContent>
                
                {message.type === 'ai' && (
                  <MessageMeta>
                    <span>{message.timestamp.toLocaleTimeString()}</span>
                    {message.confidence && (
                      <ConfidenceBar confidence={message.confidence} />
                    )}
                  </MessageMeta>
                )}
                
                {message.followUpQuestions && message.followUpQuestions.length > 0 && (
                  <FollowUpQuestions>
                    {message.followUpQuestions.map((question, index) => (
                      <FollowUpButton
                        key={index}
                        onClick={() => handleFollowUpClick(question)}
                      >
                        {question}
                      </FollowUpButton>
                    ))}
                  </FollowUpQuestions>
                )}
              </MessageBubble>
            ))}
          </AnimatePresence>

          {isLoading && (
            <TypingIndicator
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="dot" />
              <div className="dot" />
              <div className="dot" />
              <span style={{ marginLeft: '8px', fontSize: '12px' }}>AI is thinking...</span>
            </TypingIndicator>
          )}

          <div ref={messagesEndRef} />
        </ChatMessages>

        {messages.length === 1 && (
          <SuggestedQuestions>
            <SuggestedQuestionsTitle>Try asking:</SuggestedQuestionsTitle>
            <SuggestedQuestionsList>
              {suggestedQuestions.map((question, index) => (
                <SuggestedQuestionButton
                  key={index}
                  onClick={() => handleSendMessage(question)}
                >
                  {question}
                </SuggestedQuestionButton>
              ))}
            </SuggestedQuestionsList>
          </SuggestedQuestions>
        )}

        <ChatInput>
          <VoiceButton
            onClick={handleVoiceToggle}
            className={isRecording ? 'recording' : ''}
            title={isRecording ? 'Stop recording' : 'Start voice input'}
          >
            {isRecording ? '⏹' : '🎤'}
          </VoiceButton>
          
          <InputContainer>
            <CyberpunkInput
              ref={inputRef}
              value={inputValue}
              onChange={(value) => setInputValue(value)}
              onKeyDown={handleKeyPress}
              placeholder="Ask me about forecasts, analytics, or business insights..."
              disabled={isLoading}
              variant="neon"
            />
          </InputContainer>
          
          <CyberpunkButton
            onClick={() => handleSendMessage(inputValue)}
            disabled={!inputValue.trim() || isLoading}
            variant="primary"
            glitch
          >
            Send
          </CyberpunkButton>
        </ChatInput>
      </ChatContainer>
    </ChatOverlay>
  );
};

export default CyberpunkChatInterface;