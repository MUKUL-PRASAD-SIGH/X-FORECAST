import React, { useState } from 'react';
import styled, { css } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';

export interface NavigationItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  active?: boolean;
  onClick?: () => void;
  badge?: string | number;
}

interface CyberpunkNavigationProps {
  items: NavigationItem[];
  orientation?: 'horizontal' | 'vertical';
  variant?: 'primary' | 'minimal' | 'floating';
  className?: string;
}

const orientationStyles = {
  horizontal: css`
    flex-direction: row;
    width: 100%;
    height: auto;
  `,
  vertical: css`
    flex-direction: column;
    width: auto;
    height: 100%;
  `,
};

const variantStyles = {
  primary: css`
    background: ${props => props.theme.colors.cardBg};
    border: 1px solid ${props => props.theme.colors.neonBlue};
    backdrop-filter: blur(10px);
  `,
  minimal: css`
    background: transparent;
    border: none;
  `,
  floating: css`
    background: rgba(20, 20, 20, 0.9);
    border: 1px solid rgba(0, 255, 255, 0.3);
    backdrop-filter: blur(20px);
    box-shadow: ${props => props.theme.effects.softGlow};
    border-radius: 12px;
  `,
};

const NavigationContainer = styled(motion.nav).withConfig({
  shouldForwardProp: (prop) => !['$orientation', '$variant'].includes(prop)
})<{
  $orientation: 'horizontal' | 'vertical';
  $variant: 'primary' | 'minimal' | 'floating';
}>`
  display: flex;
  position: relative;
  padding: ${props => props.theme.spacing.sm};
  
  ${props => orientationStyles[props.$orientation]}
  ${props => variantStyles[props.$variant]}
`;

const NavigationList = styled.ul.withConfig({
  shouldForwardProp: (prop) => !['$orientation'].includes(prop)
})<{ $orientation: 'horizontal' | 'vertical' }>`
  display: flex;
  list-style: none;
  margin: 0;
  padding: 0;
  gap: ${props => props.theme.spacing.xs};
  
  ${props => orientationStyles[props.$orientation]}
`;

const NavigationItem = styled(motion.li).withConfig({
  shouldForwardProp: (prop) => !['$active', '$orientation'].includes(prop)
})<{ $active?: boolean; $orientation: 'horizontal' | 'vertical' }>`
  position: relative;
  
  ${props => props.$orientation === 'horizontal' ? css`
    flex: 1;
  ` : css`
    width: 100%;
  `}
`;

const NavigationLink = styled(motion.button).withConfig({
  shouldForwardProp: (prop) => !['$active'].includes(prop)
})<{ $active?: boolean }>`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${props => props.theme.spacing.xs};
  width: 100%;
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  background: transparent;
  border: 1px solid transparent;
  border-radius: 6px;
  color: ${props => props.$active ? props.theme.colors.neonBlue : props.theme.colors.secondaryText};
  font-family: ${props => props.theme.typography.fontFamily.primary};
  font-size: ${props => props.theme.typography.fontSize.sm};
  font-weight: ${props => props.theme.typography.fontWeight.medium};
  text-transform: uppercase;
  letter-spacing: 1px;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  
  &:hover {
    color: ${props => props.theme.colors.neonBlue};
    border-color: ${props => props.theme.colors.neonBlue};
    box-shadow: ${props => props.theme.effects.softGlow};
  }
  
  ${props => props.$active && css`
    background: rgba(0, 255, 255, 0.1);
    border-color: ${props.theme.colors.neonBlue};
    box-shadow: ${props.theme.effects.softGlow};
    text-shadow: 0 0 10px ${props.theme.colors.neonBlue};
  `}
  
  /* Cyberpunk scanning effect */
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.2), transparent);
    transition: left 0.5s ease;
  }
  
  &:hover::before {
    left: 100%;
  }
`;

const IconContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  
  svg {
    width: 16px;
    height: 16px;
  }
`;

const Badge = styled(motion.span)`
  position: absolute;
  top: -4px;
  right: -4px;
  background: ${props => props.theme.colors.hotPink};
  color: ${props => props.theme.colors.darkBg};
  font-size: ${props => props.theme.typography.fontSize.xs};
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  padding: 2px 6px;
  border-radius: 10px;
  min-width: 16px;
  height: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 0 10px ${props => props.theme.colors.hotPink};
`;

const ActiveIndicator = styled(motion.div).withConfig({
  shouldForwardProp: (prop) => !['$orientation'].includes(prop)
})<{ $orientation: 'horizontal' | 'vertical' }>`
  position: absolute;
  background: ${props => props.theme.colors.neonBlue};
  box-shadow: ${props => props.theme.effects.neonGlow};
  
  ${props => props.$orientation === 'horizontal' ? css`
    bottom: -1px;
    left: 0;
    right: 0;
    height: 2px;
  ` : css`
    top: 0;
    bottom: 0;
    left: -1px;
    width: 2px;
  `}
`;

export const CyberpunkNavigation: React.FC<CyberpunkNavigationProps> = ({
  items,
  orientation = 'horizontal',
  variant = 'primary',
  className,
}) => {
  const [activeItem, setActiveItem] = useState(
    items.find(item => item.active)?.id || items[0]?.id
  );

  const handleItemClick = (item: NavigationItem) => {
    setActiveItem(item.id);
    if (item.onClick) {
      item.onClick();
    }
  };

  return (
    <NavigationContainer
      $orientation={orientation}
      $variant={variant}
      className={className}
      initial={{ opacity: 0, y: orientation === 'horizontal' ? -20 : 0, x: orientation === 'vertical' ? -20 : 0 }}
      animate={{ opacity: 1, y: 0, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <NavigationList $orientation={orientation}>
        {items.map((item, index) => (
          <NavigationItem
            key={item.id}
            $active={activeItem === item.id}
            $orientation={orientation}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <NavigationLink
              $active={activeItem === item.id}
              onClick={() => handleItemClick(item)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {item.icon && (
                <IconContainer>
                  {item.icon}
                </IconContainer>
              )}
              <span>{item.label}</span>
              
              {item.badge && (
                <Badge
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                >
                  {item.badge}
                </Badge>
              )}
            </NavigationLink>
            
            <AnimatePresence>
              {activeItem === item.id && (
                <ActiveIndicator
                  $orientation={orientation}
                  layoutId="activeIndicator"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                />
              )}
            </AnimatePresence>
          </NavigationItem>
        ))}
      </NavigationList>
    </NavigationContainer>
  );
};