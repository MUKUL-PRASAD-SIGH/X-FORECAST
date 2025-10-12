import React from 'react';
import styled, { css, keyframes } from 'styled-components';
import { motion } from 'framer-motion';

interface CyberpunkLoaderProps {
  variant?: 'spinner' | 'matrix' | 'pulse' | 'glitch' | 'hologram';
  size?: 'sm' | 'md' | 'lg';
  color?: 'blue' | 'pink' | 'green' | 'purple';
  text?: string;
  className?: string;
}

// Animations
const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const matrixRain = keyframes`
  0% { transform: translateY(-100%); opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { transform: translateY(100vh); opacity: 0; }
`;

const pulse = keyframes`
  0%, 100% { opacity: 0.3; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.1); }
`;

const glitchEffect = keyframes`
  0%, 100% { transform: translate(0); }
  10% { transform: translate(-2px, -2px); }
  20% { transform: translate(2px, 2px); }
  30% { transform: translate(-2px, 2px); }
  40% { transform: translate(2px, -2px); }
  50% { transform: translate(-2px, -2px); }
  60% { transform: translate(2px, 2px); }
  70% { transform: translate(-2px, 2px); }
  80% { transform: translate(2px, -2px); }
  90% { transform: translate(-2px, -2px); }
`;

const hologramFlicker = keyframes`
  0%, 100% { opacity: 1; filter: hue-rotate(0deg); }
  25% { opacity: 0.8; filter: hue-rotate(90deg); }
  50% { opacity: 0.9; filter: hue-rotate(180deg); }
  75% { opacity: 0.7; filter: hue-rotate(270deg); }
`;

// Size variants
const sizeVariants = {
  sm: css`
    width: 24px;
    height: 24px;
    font-size: ${props => props.theme.typography.fontSize.sm};
  `,
  md: css`
    width: 48px;
    height: 48px;
    font-size: ${props => props.theme.typography.fontSize.md};
  `,
  lg: css`
    width: 72px;
    height: 72px;
    font-size: ${props => props.theme.typography.fontSize.lg};
  `,
};

// Color variants
const colorVariants = {
  blue: css`
    color: ${props => props.theme.colors.neonBlue};
    border-color: ${props => props.theme.colors.neonBlue};
  `,
  pink: css`
    color: ${props => props.theme.colors.hotPink};
    border-color: ${props => props.theme.colors.hotPink};
  `,
  green: css`
    color: ${props => props.theme.colors.acidGreen};
    border-color: ${props => props.theme.colors.acidGreen};
  `,
  purple: css`
    color: ${props => props.theme.colors.electricPurple};
    border-color: ${props => props.theme.colors.electricPurple};
  `,
};

const LoaderContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: ${props => props.theme.spacing.md};
`;

// Spinner Loader
const SpinnerLoader = styled.div<{ size: string; color: string }>`
  ${props => sizeVariants[props.size as keyof typeof sizeVariants]}
  ${props => colorVariants[props.color as keyof typeof colorVariants]}
  
  border: 3px solid transparent;
  border-top: 3px solid currentColor;
  border-radius: 50%;
  animation: ${spin} 1s linear infinite;
  box-shadow: 0 0 20px currentColor;
  
  &::before {
    content: '';
    position: absolute;
    top: -3px;
    left: -3px;
    right: -3px;
    bottom: -3px;
    border: 1px solid currentColor;
    border-radius: 50%;
    opacity: 0.3;
  }
`;

// Matrix Loader
const MatrixContainer = styled.div<{ size: string }>`
  ${props => sizeVariants[props.size as keyof typeof sizeVariants]}
  position: relative;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid ${props => props.theme.colors.acidGreen};
`;

const MatrixColumn = styled.div<{ delay: number }>`
  position: absolute;
  top: 0;
  width: 2px;
  height: 100%;
  background: linear-gradient(transparent, ${props => props.theme.colors.acidGreen}, transparent);
  animation: ${matrixRain} 2s linear infinite;
  animation-delay: ${props => props.delay}s;
`;

// Pulse Loader
const PulseLoader = styled.div<{ size: string; color: string }>`
  ${props => sizeVariants[props.size as keyof typeof sizeVariants]}
  ${props => colorVariants[props.color as keyof typeof colorVariants]}
  
  border-radius: 50%;
  background: currentColor;
  animation: ${pulse} 1.5s ease-in-out infinite;
  box-shadow: 0 0 30px currentColor;
`;

// Glitch Loader
const GlitchLoader = styled.div<{ size: string; color: string }>`
  ${props => sizeVariants[props.size as keyof typeof sizeVariants]}
  ${props => colorVariants[props.color as keyof typeof colorVariants]}
  
  position: relative;
  background: currentColor;
  animation: ${glitchEffect} 0.5s infinite;
  
  &::before,
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: currentColor;
  }
  
  &::before {
    animation: ${glitchEffect} 0.5s infinite reverse;
    filter: hue-rotate(90deg);
    transform: translate(-2px, 0);
  }
  
  &::after {
    animation: ${glitchEffect} 0.5s infinite;
    filter: hue-rotate(180deg);
    transform: translate(2px, 0);
  }
`;

// Hologram Loader
const HologramLoader = styled.div<{ size: string }>`
  ${props => sizeVariants[props.size as keyof typeof sizeVariants]}
  
  position: relative;
  border: 2px solid ${props => props.theme.colors.neonBlue};
  border-radius: 50%;
  animation: ${hologramFlicker} 2s ease-in-out infinite;
  
  &::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 60%;
    height: 60%;
    background: ${props => props.theme.effects.primaryGradient};
    border-radius: 50%;
    transform: translate(-50%, -50%);
    animation: ${spin} 3s linear infinite;
  }
  
  &::after {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    border: 1px solid ${props => props.theme.colors.hotPink};
    border-radius: 50%;
    opacity: 0.5;
    animation: ${spin} 2s linear infinite reverse;
  }
`;

const LoaderText = styled(motion.div)`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  color: ${props => props.theme.colors.secondaryText};
  text-align: center;
  text-transform: uppercase;
  letter-spacing: 2px;
  
  &::after {
    content: '';
    animation: loading-dots 1.5s infinite;
  }
  
  @keyframes loading-dots {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
  }
`;

export const CyberpunkLoader: React.FC<CyberpunkLoaderProps> = ({
  variant = 'spinner',
  size = 'md',
  color = 'blue',
  text,
  className,
}) => {
  const renderLoader = () => {
    switch (variant) {
      case 'matrix':
        return (
          <MatrixContainer size={size}>
            {Array.from({ length: 8 }, (_, i) => (
              <MatrixColumn
                key={i}
                delay={i * 0.2}
                style={{ left: `${(i * 12.5)}%` }}
              />
            ))}
          </MatrixContainer>
        );
      
      case 'pulse':
        return <PulseLoader size={size} color={color} />;
      
      case 'glitch':
        return <GlitchLoader size={size} color={color} />;
      
      case 'hologram':
        return <HologramLoader size={size} />;
      
      default:
        return <SpinnerLoader size={size} color={color} />;
    }
  };

  return (
    <LoaderContainer className={className}>
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        {renderLoader()}
      </motion.div>
      
      {text && (
        <LoaderText
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          {text}
        </LoaderText>
      )}
    </LoaderContainer>
  );
};