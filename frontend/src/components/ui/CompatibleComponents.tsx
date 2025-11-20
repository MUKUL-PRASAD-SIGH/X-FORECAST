// Backward-compatible wrapper components that handle prop transformation
// These components accept both legacy and new prop formats

import React from 'react';
import { CyberpunkButton as OriginalCyberpunkButton } from './CyberpunkButton';
import { CyberpunkCard as OriginalCyberpunkCard } from './CyberpunkCard';
import { CyberpunkInput } from './CyberpunkInput';
import { CyberpunkLoader } from './CyberpunkLoader';
import { CyberpunkNavigation } from './CyberpunkNavigation';
import { transformButtonProps, transformCardProps } from '../../utils/fixComponentProps';

// Backward-compatible CyberpunkButton
export const CyberpunkButton = React.forwardRef<HTMLButtonElement, any>((props, ref) => {
  const transformedProps = transformButtonProps(props);
  return <OriginalCyberpunkButton ref={ref} {...transformedProps} />;
});

CyberpunkButton.displayName = 'CyberpunkButton';

// Backward-compatible CyberpunkCard
export const CyberpunkCard = React.forwardRef<HTMLDivElement, any>((props, ref) => {
  const transformedProps = transformCardProps(props);
  return <OriginalCyberpunkCard ref={ref} {...transformedProps} />;
});

CyberpunkCard.displayName = 'CyberpunkCard';

// Re-export other components as-is (they don't need prop transformation)
export { CyberpunkInput, CyberpunkLoader, CyberpunkNavigation };

// Export NavigationItem type
export type { NavigationItem } from './CyberpunkNavigation';

// Export all component prop types for TypeScript users
export type {
  CyberpunkButtonProps,
  CyberpunkCardProps,
  CyberpunkInputProps,
  CyberpunkLoaderProps,
  CyberpunkNavigationProps,
} from '../../types/components';

// Utility function to create compatible components
export const createCompatibleComponent = <T extends Record<string, any>>(
  Component: React.ComponentType<T>,
  propTransformer?: (props: any) => any
) => {
  return React.forwardRef<any, T>((props, ref) => {
    const transformedProps = propTransformer ? propTransformer(props) : props;
    return React.createElement(Component, { ...transformedProps, ref });
  });
};

// Helper to check if component needs prop transformation
export const needsTransformation = (props: any): boolean => {
  const legacyProps = ['variant', 'size', 'padding', 'loading', 'glitch', 'hover'];
  return legacyProps.some(prop => prop in props && !props[`$${prop}`]);
};

// Development warning for legacy prop usage
export const warnLegacyProps = (componentName: string, props: any) => {
  if (process.env.NODE_ENV === 'development' && needsTransformation(props)) {
    const legacyProps = Object.keys(props).filter(key => 
      ['variant', 'size', 'padding', 'loading', 'glitch', 'hover'].includes(key)
    );
    
    if (legacyProps.length > 0) {
      console.warn(
        `${componentName}: Using legacy props [${legacyProps.join(', ')}]. ` +
        `Consider migrating to $ prefixed versions: [${legacyProps.map(p => `$${p}`).join(', ')}]`
      );
    }
  }
};

// Enhanced CyberpunkButton with development warnings
export const CyberpunkButtonWithWarnings = React.forwardRef<HTMLButtonElement, any>((props, ref) => {
  warnLegacyProps('CyberpunkButton', props);
  const transformedProps = transformButtonProps(props);
  return <OriginalCyberpunkButton ref={ref} {...transformedProps} />;
});

CyberpunkButtonWithWarnings.displayName = 'CyberpunkButtonWithWarnings';

// Enhanced CyberpunkCard with development warnings
export const CyberpunkCardWithWarnings = React.forwardRef<HTMLDivElement, any>((props, ref) => {
  warnLegacyProps('CyberpunkCard', props);
  const transformedProps = transformCardProps(props);
  return <OriginalCyberpunkCard ref={ref} {...transformedProps} />;
});

CyberpunkCardWithWarnings.displayName = 'CyberpunkCardWithWarnings';

// Context for enabling/disabling prop transformation warnings
export const PropWarningContext = React.createContext<{
  enableWarnings: boolean;
  setEnableWarnings: (enabled: boolean) => void;
}>({
  enableWarnings: process.env.NODE_ENV === 'development',
  setEnableWarnings: () => {},
});

// Provider for prop warning context
export const PropWarningProvider: React.FC<{
  children: React.ReactNode;
  enableWarnings?: boolean;
}> = ({ children, enableWarnings = process.env.NODE_ENV === 'development' }) => {
  const [warningsEnabled, setWarningsEnabled] = React.useState(enableWarnings);
  
  return (
    <PropWarningContext.Provider 
      value={{ 
        enableWarnings: warningsEnabled, 
        setEnableWarnings: setWarningsEnabled 
      }}
    >
      {children}
    </PropWarningContext.Provider>
  );
};

// Hook to use prop warnings
export const usePropWarnings = () => {
  return React.useContext(PropWarningContext);
};

// Conditional warning function
export const conditionalWarn = (componentName: string, props: any) => {
  const { enableWarnings } = React.useContext(PropWarningContext);
  if (enableWarnings) {
    warnLegacyProps(componentName, props);
  }
};