import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkButton, CyberpunkInput } from '../ui';

// Enhanced types for ensemble chat
interface EnsembleChatMessage {
  id: string;
  type: 'user' | 'ai' | 'system';
  content: string;
  timestamp: Date;
  confidence?: number;
  sources?: string[];
  followUpQuestions?: string[];
  suggestedActions?: string[];
  ensembleData?: any;
  modelPerformanceData?: any;
  forecastData?: any;
  insightsData?: any;
  technicalExplanation?: string;
  plainLanguageSummary?: string;
}

interface EnsembleChatResponse {
  response_id: string;
  response_text: string;
  confidence: number;
  sources: string[];
  timestamp: string;
  follow_up_questions: string[];
  suggested_actions: string[];
  ensemble_data?: any;
  model_performance_data?: any;
  forecast_data?: any;
  insights_data?: any;
  technical_explanation?: string;
  plain_language_summary?: string;
}

interface EnsembleChatInterfaceProps {
  isOpen: boolean;
  onClose: () => void;
  isLoading?: boolean;
  companyId?: string;
}

// Styled components (reusing from original with enhancements)
const ChatOverlay = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 20px;
`;

const ChatContainer = styled(motion.div)`
  width: 100%;
  max-width: 900px;
  height: 80vh;
  background: ${props => props.theme.colors.darkBg};
  border: 2px solid ${props => props.theme.colors.neonBlue};
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 
    0 0 30px ${props => props.theme.colors.neonBlue}40,
    inset 0 0 30px rgba(0, 255, 255, 0.1);
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, 
      transparent, 
      ${props => props.theme.colors.neonBlue}, 
      transparent
    );
    animation: scanline 2s ease-in-out infinite;
  }
  
  @keyframes scanline {
    0%, 100% { opacity: 0; }
    50% { opacity: 1; }
  }
`;

const ChatHeader = styled.div`
  padding: 16px 20px;
  background: ${props => props.theme.colors.darkerBg};
  border-bottom: 1px solid ${props => props.theme.colors.neonBlue}40;
  display: flex;
  justify-content: between;
  align-items: center;
`;

const ChatTitle = styled.h3`
  margin: 0;
  color: ${props => props.theme.colors.neonBlue};
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: 1.2rem;
  text-shadow: 0 0 10px ${props => props.theme.colors.neonBlue};
  
  &::before {
    content: '🤖 ';
    margin-right: 8px;
  }
`;

const ChatMessages = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  
  &::-webkit-scrollbar {
    width: 8px;
  }
  
  &::-webkit-scrollbar-track {
    background: ${props => props.theme.colors.darkerBg};
  }
  
  &::-webkit-scrollbar-thumb {
    background: ${props => props.theme.colors.neonBlue}60;
    border-radius: 4px;
  }
`;

const MessageBubble = styled(motion.div)<{ $isUser: boolean; $confidence?: number }>`
  max-width: 80%;
  align-self: ${props => props.$isUser ? 'flex-end' : 'flex-start'};
  background: ${props => props.$isUser 
    ? `linear-gradient(135deg, ${props.theme.colors.neonBlue}20, ${props.theme.colors.neonBlue}10)`
    : `linear-gradient(135deg, ${props.theme.colors.darkerBg}, ${props.theme.colors.darkBg})`
  };
  border: 1px solid ${props => props.$isUser 
    ? props.theme.colors.neonBlue 
    : props.theme.colors.acidGreen
  };
  border-radius: 12px;
  padding: 16px;
  position: relative;
  
  ${props => !props.$isUser && props.$confidence && `
    &::after {
      content: 'Confidence: ${Math.round(props.$confidence * 100)}%';
      position: absolute;
      top: -8px;
      right: 8px;
      background: ${props.theme.colors.acidGreen};
      color: ${props.theme.colors.darkBg};
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: bold;
    }
  `}
`;

const MessageContent = styled.div`
  color: ${props => props.theme.colors.primaryText};
  line-height: 1.6;
  white-space: pre-wrap;
  
  strong {
    color: ${props => props.theme.colors.neonBlue};
  }
  
  em {
    color: ${props => props.theme.colors.acidGreen};
  }
`;

const MessageMeta = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 12px;
  padding-top: 8px;
  border-top: 1px solid ${props => props.theme.colors.neonBlue}20;
  font-size: 0.8rem;
  color: ${props => props.theme.colors.secondaryText};
`;

const SourcesList = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 8px;
`;

const SourceTag = styled.span`
  background: ${props => props.theme.colors.acidGreen}20;
  color: ${props => props.theme.colors.acidGreen};
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.7rem;
  border: 1px solid ${props => props.theme.colors.acidGreen}40;
`;

const FollowUpQuestions = styled.div`
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const FollowUpButton = styled.button`
  background: transparent;
  border: 1px solid ${props => props.theme.colors.neonBlue}40;
  color: ${props => props.theme.colors.neonBlue};
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.3s ease;
  text-align: left;
  
  &:hover {
    background: ${props => props.theme.colors.neonBlue}20;
    border-color: ${props => props.theme.colors.neonBlue};
    box-shadow: 0 0 10px ${props => props.theme.colors.neonBlue}40;
  }
`;

const TechnicalToggle = styled.button`
  background: transparent;
  border: 1px solid ${props => props.theme.colors.acidGreen}40;
  color: ${props => props.theme.colors.acidGreen};
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.7rem;
  cursor: pointer;
  margin-top: 8px;
  
  &:hover {
    background: ${props => props.theme.colors.acidGreen}20;
  }
`;

const TechnicalDetails = styled(motion.div)`
  margin-top: 8px;
  padding: 12px;
  background: ${props => props.theme.colors.darkerBg};
  border: 1px solid ${props => props.theme.colors.acidGreen}40;
  border-radius: 6px;
  font-family: monospace;
  font-size: 0.8rem;
  color: ${props => props.theme.colors.acidGreen};
`;

const ChatInput = styled.div`
  padding: 20px;
  background: ${props => props.theme.colors.darkerBg};
  border-top: 1px solid ${props => props.theme.colors.neonBlue}40;
  display: flex;
  gap: 12px;
  align-items: flex-end;
`;

const InputContainer = styled.div`
  flex: 1;
  position: relative;
`;

const SuggestedQuestions = styled.div`
  padding: 16px 20px;
  background: ${props => props.theme.colors.darkerBg};
  border-top: 1px solid ${props => props.theme.colors.neonBlue}20;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
`;

const SuggestedButton = styled.button`
  background: transparent;
  border: 1px solid ${props => props.theme.colors.neonBlue}40;
  color: ${props => props.theme.colors.neonBlue};
  padding: 6px 12px;
  border-radius: 16px;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: ${props => props.theme.colors.neonBlue}20;
    border-color: ${props => props.theme.colors.neonBlue};
  }
`;

const CloseButton = styled.button`
  background: transparent;
  border: none;
  color: ${props => props.theme.colors.neonBlue};
  font-size: 1.5rem;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 4px;
  transition: all 0.3s ease;
  
  &:hover {
    background: ${props => props.theme.colors.neonBlue}20;
    color: ${props => props.theme.colors.neonBlue};
  }
`;

export const EnsembleChatInterface: React.FC<EnsembleChatInterfaceProps> = ({
  isOpen,
  onClose,
  isLoading = false,
  companyId
}) => {
  const [messages, setMessages] = useState<EnsembleChatMessage[]>([
    {
      id: '1',
      type: 'system',
      content: '🤖 Ensemble AI Assistant initialized. I can help you with forecasting, model performance, business insights, and more. Ask me anything about your ensemble forecasting system!',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [showTechnical, setShowTechnical] = useState<{[key: string]: boolean}>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const ensembleSuggestedQuestions = [
    "What's the ensemble forecast for next quarter?",
    "How accurate are the forecasting models?",
    "Which model is performing best right now?",
    "Show me the model weight distribution",
    "What business insights do you have?",
    "How confident is the current forecast?",
    "Compare ARIMA vs LSTM performance",
    "What patterns do you detect in the data?",
    "Give me actionable recommendations",
    "Explain the ensemble methodology"
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

  const sendEnsembleQuery = async (message: string): Promise<EnsembleChatResponse> => {
    try {
      const response = await fetch('/api/v1/ensemble-chat/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          user_id: 'user_1',
          session_id: 'default',
          context: {
            company_id: companyId,
            timestamp: new Date().toISOString()
          }
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Ensemble chat API error:', error);
      // Fallback to basic chat API
      try {
        const fallbackResponse = await fetch('/api/v1/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message,
            user_id: 'user_1'
          }),
        });

        if (fallbackResponse.ok) {
          const data = await fallbackResponse.json();
          return {
            response_id: data.response_id,
            response_text: data.response_text,
            confidence: data.confidence || 0.8,
            sources: data.sources || ['Fallback AI'],
            timestamp: new Date().toISOString(),
            follow_up_questions: [],
            suggested_actions: []
          };
        }
      } catch (fallbackError) {
        console.error('Fallback chat API error:', fallbackError);
      }

      // Final fallback
      return {
        response_id: `error_${Date.now()}`,
        response_text: 'I apologize, but I encountered an error processing your request. The ensemble chat system may be temporarily unavailable. Please try again later or contact support.',
        confidence: 0.1,
        sources: ['Error Handler'],
        timestamp: new Date().toISOString(),
        follow_up_questions: ['Try a simpler question', 'Check system status'],
        suggested_actions: ['Refresh the page', 'Contact support']
      };
    }
  };

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    // Add user message
    const userMessage: EnsembleChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: content.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');

    try {
      // Get ensemble AI response
      const aiResponse = await sendEnsembleQuery(content.trim());
      
      const aiMessage: EnsembleChatMessage = {
        id: aiResponse.response_id,
        type: 'ai',
        content: aiResponse.response_text,
        timestamp: new Date(aiResponse.timestamp),
        confidence: aiResponse.confidence,
        sources: aiResponse.sources,
        followUpQuestions: aiResponse.follow_up_questions,
        suggestedActions: aiResponse.suggested_actions,
        ensembleData: aiResponse.ensemble_data,
        modelPerformanceData: aiResponse.model_performance_data,
        forecastData: aiResponse.forecast_data,
        insightsData: aiResponse.insights_data,
        technicalExplanation: aiResponse.technical_explanation,
        plainLanguageSummary: aiResponse.plain_language_summary
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      // Add error message
      const errorMessage: EnsembleChatMessage = {
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

  const toggleTechnicalDetails = (messageId: string) => {
    setShowTechnical(prev => ({
      ...prev,
      [messageId]: !prev[messageId]
    }));
  };

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
          <ChatTitle>Ensemble AI Assistant</ChatTitle>
          <CloseButton onClick={onClose}>×</CloseButton>
        </ChatHeader>

        <ChatMessages>
          <AnimatePresence>
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                $isUser={message.type === 'user'}
                $confidence={message.confidence}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <MessageContent>{message.content}</MessageContent>
                
                {message.sources && message.sources.length > 0 && (
                  <SourcesList>
                    {message.sources.map((source, index) => (
                      <SourceTag key={index}>{source}</SourceTag>
                    ))}
                  </SourcesList>
                )}

                {message.technicalExplanation && (
                  <>
                    <TechnicalToggle onClick={() => toggleTechnicalDetails(message.id)}>
                      {showTechnical[message.id] ? 'Hide' : 'Show'} Technical Details
                    </TechnicalToggle>
                    <AnimatePresence>
                      {showTechnical[message.id] && (
                        <TechnicalDetails
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                        >
                          {message.technicalExplanation}
                        </TechnicalDetails>
                      )}
                    </AnimatePresence>
                  </>
                )}

                {message.followUpQuestions && message.followUpQuestions.length > 0 && (
                  <FollowUpQuestions>
                    {message.followUpQuestions.map((question, index) => (
                      <FollowUpButton
                        key={index}
                        onClick={() => handleSendMessage(question)}
                      >
                        💡 {question}
                      </FollowUpButton>
                    ))}
                  </FollowUpQuestions>
                )}

                <MessageMeta>
                  <span>{message.timestamp.toLocaleTimeString()}</span>
                  {message.confidence && (
                    <span>Confidence: {Math.round(message.confidence * 100)}%</span>
                  )}
                </MessageMeta>
              </MessageBubble>
            ))}
          </AnimatePresence>

          <div ref={messagesEndRef} />
        </ChatMessages>

        {messages.length === 1 && (
          <SuggestedQuestions>
            {ensembleSuggestedQuestions.slice(0, 6).map((question, index) => (
              <SuggestedButton
                key={index}
                onClick={() => handleSendMessage(question)}
              >
                {question}
              </SuggestedButton>
            ))}
          </SuggestedQuestions>
        )}

        <ChatInput>
          <InputContainer>
            <CyberpunkInput
              ref={inputRef}
              value={inputValue}
              onChange={setInputValue}
              onKeyDown={handleKeyPress}
              placeholder="Ask about forecasts, model performance, insights..."
              disabled={isLoading}
            />
          </InputContainer>
          <CyberpunkButton
            onClick={() => handleSendMessage(inputValue)}
            disabled={!inputValue.trim() || isLoading}
          >
            {isLoading ? 'Processing...' : 'Send'}
          </CyberpunkButton>
        </ChatInput>
      </ChatContainer>
    </ChatOverlay>
  );
};

export default EnsembleChatInterface;