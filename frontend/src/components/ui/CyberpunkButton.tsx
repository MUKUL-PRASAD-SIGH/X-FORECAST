import React, { ReactNode, ButtonHTMLAttributes } from 'react';
import styled, { css } from 'styled-components';
import { motion } from 'framer-motion';

// Functional props interface
interface CyberpunkButtonFunctionalProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'size'> {
  children: ReactNode;
  onClick?: (e: React.MouseEvent<HTMLButtonElement>) => void;
  disabled?: boolean;
  className?: string;
}

// Styled props interface
interface CyberpunkButtonStyledProps {
  $variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  $size?: 'sm' | 'md' | 'lg';
  $loading?: boolean;
  $glitch?: boolean;
}

// Motion props interface
interface CyberpunkButtonMotionProps {
  whileHover?: any;
  whileTap?: any;
  whileFocus?: any;
  initial?: any;
  animate?: any;
  exit?: any;
  transition?: any;
}

// Combined interface
interface CyberpunkButtonProps extends CyberpunkButtonFunctionalProps, CyberpunkButtonStyledProps, CyberpunkButtonMotionProps {}

const buttonVariants = {
  primary: css`
    background: linear-gradient(135deg, ${props => props.theme.colors.neonBlue}, ${props => props.theme.colors.hotPink});
    color: ${props => props.theme.colors.darkBg};
    border: 2px solid transparent;
    
    &:hover {
      box-shadow: ${props => props.theme.effects.neonGlow};
      transform: translateY(-2px);
    }
  `,
  secondary: css`
    background: transparent;
    color: ${props => props.theme.colors.neonBlue};
    border: 2px solid ${props => props.theme.colors.neonBlue};
    
    &:hover {
      background: ${props => props.theme.colors.neonBlue};
      color: ${props => props.theme.colors.darkBg};
      box-shadow: ${props => props.theme.effects.softGlow};
    }
  `,
  danger: css`
    background: linear-gradient(135deg, ${props => props.theme.colors.error}, ${props => props.theme.colors.hotPink});
    color: ${props => props.theme.colors.primaryText};
    border: 2px solid transparent;
    
    &:hover {
      box-shadow: 0 0 20px rgba(255, 0, 64, 0.6);
      transform: translateY(-2px);
    }
  `,
  ghost: css`
    background: transparent;
    color: ${props => props.theme.colors.secondaryText};
    border: 1px solid ${props => props.theme.colors.secondaryText};
    
    &:hover {
      color: ${props => props.theme.colors.primaryText};
      border-color: ${props => props.theme.colors.neonBlue};
      box-shadow: ${props => props.theme.effects.softGlow};
    }
  `,
};

const sizeVariants = {
  sm: css`
    padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
    font-size: ${props => props.theme.typography.fontSize.sm};
  `,
  md: css`
    padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
    font-size: ${props => props.theme.typography.fontSize.md};
  `,
  lg: css`
    padding: ${props => props.theme.spacing.md} ${props => props.theme.spacing.lg};
    font-size: ${props => props.theme.typography.fontSize.lg};
  `,
};

const StyledButton = styled(motion.button).withConfig({
  shouldForwardProp: (prop) => {
    // Filter out transient props (prefixed with $) and styling-specific props
    return !prop.startsWith('$') && 
           !['variant', 'size', 'loading', 'glitch'].includes(prop);
  }
})<CyberpunkButtonStyledProps>`
  font-family: ${props => props.theme.typography.fontFamily.primary};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  text-transform: uppercase;
  letter-spacing: 1px;
  
  ${props => buttonVariants[props.$variant || 'primary']}
  ${props => sizeVariants[props.$size || 'md']}
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    
    &:hover {
      transform: none;
      box-shadow: none;
    }
  }
  
  ${props => props.$glitch && css`
    &:hover {
      animation: glitch 0.3s ease-in-out infinite alternate;
    }
  `}
  
  /* Cyberpunk scanning line effect */
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s ease;
  }
  
  &:hover::before {
    left: 100%;
  }
  
  /* Loading state */
  ${props => props.$loading && css`
    &::after {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      width: 16px;
      height: 16px;
      margin: -8px 0 0 -8px;
      border: 2px solid transparent;
      border-top: 2px solid currentColor;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  `}
`;

const LoadingText = styled.span<{ $loading?: boolean }>`
  opacity: ${props => props.$loading ? 0 : 1};
  transition: opacity 0.3s ease;
`;

export const CyberpunkButton: React.FC<CyberpunkButtonProps> = ({
  children,
  $variant = 'primary',
  $size = 'md',
  disabled = false,
  $loading = false,
  $glitch = false,
  onClick,
  className,
  // Separate motion props from HTML button props
  whileHover,
  whileTap,
  whileFocus,
  initial,
  animate,
  exit,
  transition,
  ...htmlProps
}) => {
  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    if (!disabled && !$loading && onClick) {
      onClick(e);
    }
  };

  // Motion props for framer-motion
  const motionProps = {
    whileHover: whileHover || (disabled || $loading ? {} : { scale: 1.05 }),
    whileTap: whileTap || (disabled || $loading ? {} : { scale: 0.95 }),
    whileFocus,
    initial,
    animate,
    exit,
    transition,
  };

  // Styled props
  const styledProps = {
    $variant,
    $size,
    $loading,
    $glitch,
  };

  return (
    <StyledButton
      {...styledProps}
      {...motionProps}
      disabled={disabled || $loading}
      onClick={handleClick}
      className={className}
      {...htmlProps}
    >
      <LoadingText $loading={$loading}>
        {children}
      </LoadingText>
    </StyledButton>
  );
};