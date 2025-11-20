// Quick fixes for remaining TypeScript compilation issues
// This file provides runtime solutions for the most critical remaining errors

import React from 'react';

/**
 * Utility to automatically convert legacy props to modern format
 * Can be used as a higher-order component or directly in components
 */
export const convertLegacyProps = (props: any) => {
  const converted = { ...props };
  
  // Convert variant to $variant
  if ('variant' in converted && !('$variant' in converted)) {
    converted.$variant = converted.variant;
    delete converted.variant;
  }
  
  // Convert size to $size  
  if ('size' in converted && !('$size' in converted)) {
    converted.$size = converted.size;
    delete converted.size;
  }
  
  // Convert other common props
  const propMappings = {
    loading: '$loading',
    glitch: '$glitch',
    padding: '$padding',
    hover: '$hover'
  };
  
  Object.entries(propMappings).forEach(([oldProp, newProp]) => {
    if (oldProp in converted && !(newProp in converted)) {
      converted[newProp] = converted[oldProp];
      delete converted[oldProp];
    }
  });
  
  return converted;
};

/**
 * Type-safe prop converter for CyberpunkButton
 */
export const convertButtonProps = (props: any) => {
  return convertLegacyProps(props);
};

/**
 * Type-safe prop converter for CyberpunkCard
 */
export const convertCardProps = (props: any) => {
  return convertLegacyProps(props);
};

/**
 * Higher-order component that automatically converts props
 */
export const withLegacyPropSupport = <P extends object>(
  Component: React.ComponentType<P>
) => {
  return React.forwardRef<any, P>((props, ref) => {
    const convertedProps = convertLegacyProps(props);
    return React.createElement(Component, { ...convertedProps, ref });
  });
};

/**
 * Development helper to identify components that need prop conversion
 */
export const identifyLegacyProps = (props: any, componentName: string) => {
  const legacyProps = ['variant', 'size', 'loading', 'glitch', 'padding', 'hover'];
  const foundLegacy = legacyProps.filter(prop => prop in props);
  
  if (foundLegacy.length > 0 && process.env.NODE_ENV === 'development') {
    console.warn(
      `${componentName}: Found legacy props [${foundLegacy.join(', ')}]. ` +
      `Consider updating to [$${foundLegacy.join(', $')}]`
    );
  }
  
  return foundLegacy;
};

/**
 * Batch convert multiple component props
 */
export const batchConvertProps = (componentProps: Record<string, any>) => {
  const converted: Record<string, any> = {};
  
  Object.entries(componentProps).forEach(([componentName, props]) => {
    converted[componentName] = convertLegacyProps(props);
  });
  
  return converted;
};

/**
 * Runtime prop validation and conversion
 */
export const validateAndConvertProps = (props: any, componentName: string) => {
  // Identify legacy props
  identifyLegacyProps(props, componentName);
  
  // Convert props
  return convertLegacyProps(props);
};

export default {
  convertLegacyProps,
  convertButtonProps,
  convertCardProps,
  withLegacyPropSupport,
  identifyLegacyProps,
  batchConvertProps,
  validateAndConvertProps,
};