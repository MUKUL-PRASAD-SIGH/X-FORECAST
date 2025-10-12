import React, { ReactNode } from 'react';
import styled, { css } from 'styled-components';
import { motion, MotionProps } from 'framer-motion';

interface CyberpunkCardProps extends MotionProps {
  children: ReactNode;
  variant?: 'default' | 'glass' | 'neon' | 'hologram';
  padding?: 'sm' | 'md' | 'lg';
  glitch?: boolean;
  hover?: boolean;
  className?: string;
}

const cardVariants = {
  default: css`
    background: ${props => props.theme.colors.cardBg};
    border: 1px solid ${props => props.theme.colors.neonBlue};
    backdrop-filter: blur(10px);
  `,
  glass: css`
    background: rgba(20, 20, 20, 0.3);
    border: 1px solid rgba(0, 255, 255, 0.3);
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  `,
  neon: css`
    background: ${props => props.theme.colors.cardBg};
    border: 2px solid ${props => props.theme.colors.neonBlue};
    box-shadow: ${props => props.theme.effects.neonGlow};
  `,
  hologram: css`
    background: linear-gradient(135deg, 
      rgba(0, 255, 255, 0.1) 0%, 
      rgba(255, 20, 147, 0.1) 50%, 
      rgba(57, 255, 20, 0.1) 100%
    );
    border: 1px solid rgba(0, 255, 255, 0.5);
    backdrop-filter: blur(15px);
    position: relative;
    
    &::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(45deg, 
        transparent 30%, 
        rgba(255, 255, 255, 0.1) 50%, 
        transparent 70%
      );
      transform: translateX(-100%);
      transition: transform 0.6s ease;
    }
    
    &:hover::before {
      transform: translateX(100%);
    }
  `,
};

const paddingVariants = {
  sm: css`
    padding: ${props => props.theme.spacing.sm};
  `,
  md: css`
    padding: ${props => props.theme.spacing.md};
  `,
  lg: css`
    padding: ${props => props.theme.spacing.lg};
  `,
};

const StyledCard = styled(motion.div)<CyberpunkCardProps>`
  border-radius: 8px;
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
  
  ${props => cardVariants[props.variant || 'default']}
  ${props => paddingVariants[props.padding || 'md']}
  
  ${props => props.hover && css`
    cursor: pointer;
    
    &:hover {
      transform: translateY(-4px);
      box-shadow: ${props.theme.effects.intenseGlow};
    }
  `}
  
  ${props => props.glitch && css`
    &:hover {
      animation: glitch 0.3s ease-in-out infinite alternate;
    }
  `}
  
  /* Cyberpunk corner accents */
  &::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 20px;
    height: 20px;
    border-top: 2px solid ${props => props.theme.colors.hotPink};
    border-right: 2px solid ${props => props.theme.colors.hotPink};
    opacity: 0.7;
  }
  
  &::before {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 20px;
    height: 20px;
    border-bottom: 2px solid ${props => props.theme.colors.acidGreen};
    border-left: 2px solid ${props => props.theme.colors.acidGreen};
    opacity: 0.7;
  }
`;

const CardContent = styled.div`
  position: relative;
  z-index: 1;
`;

export const CyberpunkCard: React.FC<CyberpunkCardProps> = ({
  children,
  variant = 'default',
  padding = 'md',
  glitch = false,
  hover = false,
  className,
  ...props
}) => {
  return (
    <StyledCard
      variant={variant}
      padding={padding}
      glitch={glitch}
      hover={hover}
      className={className}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      whileHover={hover ? { scale: 1.02 } : {}}
      {...props}
    >
      <CardContent>
        {children}
      </CardContent>
    </StyledCard>
  );
};