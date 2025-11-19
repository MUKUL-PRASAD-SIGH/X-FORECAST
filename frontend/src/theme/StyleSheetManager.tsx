import React, { ReactNode } from 'react';
import { StyleSheetManager } from 'styled-components';
import isPropValid from '@emotion/is-prop-valid';

interface PropFilterWrapperProps {
  children: ReactNode;
}

/**
 * Global StyleSheetManager configuration for prop filtering
 * Uses @emotion/is-prop-valid to filter out non-DOM props from styled-components
 */
export const PropFilterWrapper: React.FC<PropFilterWrapperProps> = ({ children }) => {
  // Custom shouldForwardProp function that combines @emotion/is-prop-valid
  // with additional filtering for transient props ($ prefix)
  const shouldForwardProp = (prop: string) => {
    // Don't forward transient props (those starting with $)
    if (prop.startsWith('$')) {
      return false;
    }
    
    // Don't forward common styling props that shouldn't be DOM attributes
    const stylingProps = [
      'variant',
      'size',
      'loading',
      'glitch',
      'hover',
      'focused',
      'hasError',
      'padding',
      'margin',
      'spacing',
      'elevation',
      'intensity',
      'animated',
      'pulse',
      'glow'
    ];
    
    if (stylingProps.includes(prop)) {
      return false;
    }
    
    // Use @emotion/is-prop-valid for standard HTML prop validation
    return isPropValid(prop);
  };

  return (
    <StyleSheetManager shouldForwardProp={shouldForwardProp}>
      {children}
    </StyleSheetManager>
  );
};

/**
 * Utility function for creating shouldForwardProp functions in individual styled components
 * @param additionalProps - Array of additional props to filter out
 * @returns shouldForwardProp function
 */
export const createShouldForwardProp = (additionalProps: string[] = []) => {
  return (prop: string) => {
    // Don't forward transient props (those starting with $)
    if (prop.startsWith('$')) {
      return false;
    }
    
    // Don't forward common styling props
    const commonStylingProps = [
      'variant',
      'size',
      'loading',
      'glitch',
      'hover',
      'focused',
      'hasError',
      'padding',
      'margin',
      'spacing',
      'elevation',
      'intensity',
      'animated',
      'pulse',
      'glow'
    ];
    
    const allFilteredProps = [...commonStylingProps, ...additionalProps];
    
    if (allFilteredProps.includes(prop)) {
      return false;
    }
    
    // Use @emotion/is-prop-valid for standard HTML prop validation
    return isPropValid(prop);
  };
};

export default PropFilterWrapper;