import React from 'react';
import styled, { css } from 'styled-components';
import { motion } from 'framer-motion';

interface ConfidenceIndicatorProps {
  confidence: number; // 0-1
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  showPercentage?: boolean;
  className?: string;
}

const sizeVariants = {
  sm: css`
    width: 60px;
    height: 8px;
    font-size: ${props => props.theme.typography.fontSize.xs};
  `,
  md: css`
    width: 80px;
    height: 10px;
    font-size: ${props => props.theme.typography.fontSize.sm};
  `,
  lg: css`
    width: 120px;
    height: 12px;
    font-size: ${props => props.theme.typography.fontSize.md};
  `,
};

const IndicatorContainer = styled.div<{ size: string }>`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
  
  ${props => sizeVariants[props.size as keyof typeof sizeVariants]}
`;

const IndicatorBar = styled.div<{ size: string }>`
  position: relative;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 4px;
  overflow: hidden;
  
  ${props => sizeVariants[props.size as keyof typeof sizeVariants]}
`;

const IndicatorFill = styled(motion.div)<{ confidence: number }>`
  height: 100%;
  border-radius: 3px;
  position: relative;
  
  ${props => {
    if (props.confidence >= 0.8) {
      return css`
        background: linear-gradient(90deg, ${props.theme.colors.acidGreen}, ${props.theme.colors.neonBlue});
        box-shadow: 0 0 10px rgba(57, 255, 20, 0.5);
      `;
    } else if (props.confidence >= 0.6) {
      return css`
        background: linear-gradient(90deg, ${props.theme.colors.cyberYellow}, ${props.theme.colors.neonBlue});
        box-shadow: 0 0 8px rgba(255, 255, 0, 0.4);
      `;
    } else if (props.confidence >= 0.4) {
      return css`
        background: linear-gradient(90deg, ${props.theme.colors.warning}, ${props.theme.colors.hotPink});
        box-shadow: 0 0 6px rgba(255, 165, 0, 0.4);
      `;
    } else {
      return css`
        background: linear-gradient(90deg, ${props.theme.colors.error}, ${props.theme.colors.hotPink});
        box-shadow: 0 0 8px rgba(255, 0, 64, 0.4);
      `;
    }
  }}
  
  /* Animated scanning line effect */
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, 
      transparent 0%, 
      rgba(255, 255, 255, 0.3) 50%, 
      transparent 100%
    );
    transform: translateX(-100%);
    animation: scan 2s ease-in-out infinite;
  }
  
  @keyframes scan {
    0% { transform: translateX(-100%); }
    50% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
  }
`;

const ConfidenceLabel = styled.span`
  color: ${props => props.theme.colors.secondaryText};
  font-size: inherit;
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const ConfidenceValue = styled.span<{ confidence: number }>`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  font-size: inherit;
  
  ${props => {
    if (props.confidence >= 0.8) {
      return css`color: ${props.theme.colors.acidGreen};`;
    } else if (props.confidence >= 0.6) {
      return css`color: ${props.theme.colors.cyberYellow};`;
    } else if (props.confidence >= 0.4) {
      return css`color: ${props.theme.colors.warning};`;
    } else {
      return css`color: ${props.theme.colors.error};`;
    }
  }}
`;

const IndicatorLabels = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  gap: 0.5rem;
`;

const ConfidenceText = styled.div<{ confidence: number }>`
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  text-transform: uppercase;
  letter-spacing: 1px;
  
  ${props => {
    if (props.confidence >= 0.8) {
      return css`color: ${props.theme.colors.acidGreen};`;
    } else if (props.confidence >= 0.6) {
      return css`color: ${props.theme.colors.cyberYellow};`;
    } else if (props.confidence >= 0.4) {
      return css`color: ${props.theme.colors.warning};`;
    } else {
      return css`color: ${props.theme.colors.error};`;
    }
  }}
`;

export const ConfidenceIndicator: React.FC<ConfidenceIndicatorProps> = ({
  confidence,
  size = 'md',
  showLabel = true,
  showPercentage = true,
  className
}) => {
  const clampedConfidence = Math.max(0, Math.min(1, confidence));
  const percentage = Math.round(clampedConfidence * 100);
  
  const getConfidenceText = (conf: number): string => {
    if (conf >= 0.8) return 'High';
    if (conf >= 0.6) return 'Medium';
    if (conf >= 0.4) return 'Low';
    return 'Very Low';
  };

  return (
    <IndicatorContainer size={size} className={className}>
      <IndicatorBar size={size}>
        <IndicatorFill
          confidence={clampedConfidence}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
      </IndicatorBar>
      
      {(showLabel || showPercentage) && (
        <IndicatorLabels>
          {showLabel && (
            <ConfidenceLabel>
              <ConfidenceText confidence={clampedConfidence}>
                {getConfidenceText(clampedConfidence)}
              </ConfidenceText>
            </ConfidenceLabel>
          )}
          {showPercentage && (
            <ConfidenceValue confidence={clampedConfidence}>
              {percentage}%
            </ConfidenceValue>
          )}
        </IndicatorLabels>
      )}
    </IndicatorContainer>
  );
};