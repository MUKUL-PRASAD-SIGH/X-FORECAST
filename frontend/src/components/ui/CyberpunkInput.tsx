import React, { useState, forwardRef } from 'react';
import styled, { css } from 'styled-components';
import { motion } from 'framer-motion';

interface CyberpunkInputProps {
  type?: 'text' | 'email' | 'password' | 'number' | 'search';
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  disabled?: boolean;
  error?: string;
  label?: string;
  icon?: React.ReactNode;
  glitch?: boolean;
  className?: string;
  variant?: string;
}

const InputContainer = styled.div`
  position: relative;
  width: 100%;
`;

const Label = styled.label`
  display: block;
  font-family: ${props => props.theme.typography.fontFamily.primary};
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.secondaryText};
  margin-bottom: ${props => props.theme.spacing.xs};
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const InputWrapper = styled.div`
  position: relative;
  display: flex;
  align-items: center;
`;

const StyledInput = styled(motion.input)<{ hasIcon?: boolean; hasError?: boolean; glitch?: boolean }>`
  width: 100%;
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  padding-left: ${props => props.hasIcon ? '40px' : props.theme.spacing.md};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.md};
  background: rgba(20, 20, 20, 0.8);
  border: 2px solid ${props => props.hasError ? props.theme.colors.error : props.theme.colors.neonBlue};
  border-radius: 4px;
  color: ${props => props.theme.colors.primaryText};
  transition: all 0.3s ease;
  
  &::placeholder {
    color: ${props => props.theme.colors.secondaryText};
    opacity: 0.7;
  }
  
  &:focus {
    outline: none;
    border-color: ${props => props.hasError ? props.theme.colors.error : props.theme.colors.hotPink};
    box-shadow: ${props => props.hasError ? 
      '0 0 20px rgba(255, 0, 64, 0.3)' : 
      props.theme.effects.softGlow
    };
    background: rgba(30, 30, 30, 0.9);
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    background: rgba(10, 10, 10, 0.5);
  }
  
  ${props => props.glitch && css`
    &:focus {
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
    height: 2px;
    background: linear-gradient(90deg, transparent, ${props => props.theme.colors.neonBlue}, transparent);
    transition: left 0.5s ease;
  }
  
  &:focus::before {
    left: 100%;
  }
`;

const IconContainer = styled.div`
  position: absolute;
  left: ${props => props.theme.spacing.sm};
  top: 50%;
  transform: translateY(-50%);
  color: ${props => props.theme.colors.secondaryText};
  z-index: 1;
  
  svg {
    width: 16px;
    height: 16px;
  }
`;

const ErrorMessage = styled(motion.div)`
  margin-top: ${props => props.theme.spacing.xs};
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.error};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  text-shadow: 0 0 10px rgba(255, 0, 64, 0.5);
`;

const GlowEffect = styled.div<{ focused: boolean; hasError?: boolean }>`
  position: absolute;
  top: -2px;
  left: -2px;
  right: -2px;
  bottom: -2px;
  border-radius: 6px;
  background: ${props => props.hasError ? 
    'linear-gradient(45deg, #FF0040, #FF1493)' : 
    'linear-gradient(45deg, #00FFFF, #FF1493, #39FF14)'
  };
  opacity: ${props => props.focused ? 0.3 : 0};
  transition: opacity 0.3s ease;
  z-index: -1;
  filter: blur(4px);
`;

export const CyberpunkInput = forwardRef<HTMLInputElement, CyberpunkInputProps>(({
  type = 'text',
  placeholder,
  value,
  onChange,
  disabled = false,
  error,
  label,
  icon,
  glitch = false,
  className,
  ...props
}, ref) => {
  const [focused, setFocused] = useState(false);
  const [internalValue, setInternalValue] = useState(value || '');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setInternalValue(newValue);
    if (onChange) {
      onChange(newValue);
    }
  };

  const handleFocus = () => {
    setFocused(true);
  };

  const handleBlur = () => {
    setFocused(false);
  };

  return (
    <InputContainer className={className}>
      {label && <Label>{label}</Label>}
      <InputWrapper>
        <GlowEffect focused={focused} hasError={!!error} />
        {icon && <IconContainer>{icon}</IconContainer>}
        <StyledInput
          ref={ref}
          type={type}
          placeholder={placeholder}
          value={value !== undefined ? value : internalValue}
          onChange={handleChange}
          onFocus={handleFocus}
          onBlur={handleBlur}
          disabled={disabled}
          hasIcon={!!icon}
          hasError={!!error}
          glitch={glitch}
          whileFocus={{ scale: 1.02 }}
          {...props}
        />
      </InputWrapper>
      {error && (
        <ErrorMessage
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
        >
          {error}
        </ErrorMessage>
      )}
    </InputContainer>
  );
});