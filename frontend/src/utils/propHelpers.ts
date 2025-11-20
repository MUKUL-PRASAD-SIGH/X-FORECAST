// Utility functions for handling component props and transformations

/**
 * Transforms legacy prop names to new styled-component prop names
 * This helps with backward compatibility during the migration
 */
export const transformLegacyProps = <T extends Record<string, any>>(props: T): T => {
  const transformed = { ...props } as any;
  
  // Transform variant to $variant
  if ('variant' in transformed && !('$variant' in transformed)) {
    transformed.$variant = transformed.variant;
    delete transformed.variant;
  }
  
  // Transform size to $size
  if ('size' in transformed && !('$size' in transformed)) {
    transformed.$size = transformed.size;
    delete transformed.size;
  }
  
  // Transform padding to $padding
  if ('padding' in transformed && !('$padding' in transformed)) {
    transformed.$padding = transformed.padding;
    delete transformed.padding;
  }
  
  // Transform loading to $loading
  if ('loading' in transformed && !('$loading' in transformed)) {
    transformed.$loading = transformed.loading;
    delete transformed.loading;
  }
  
  // Transform glitch to $glitch
  if ('glitch' in transformed && !('$glitch' in transformed)) {
    transformed.$glitch = transformed.glitch;
    delete transformed.glitch;
  }
  
  // Transform hover to $hover
  if ('hover' in transformed && !('$hover' in transformed)) {
    transformed.$hover = transformed.hover;
    delete transformed.hover;
  }
  
  return transformed;
};

/**
 * Separates styled props from functional props
 */
export const separateStyledProps = <T extends Record<string, any>>(
  props: T
): { styledProps: Record<string, any>; functionalProps: Record<string, any> } => {
  const styledProps: Record<string, any> = {};
  const functionalProps: Record<string, any> = {};
  
  Object.entries(props).forEach(([key, value]) => {
    if (key.startsWith('$')) {
      styledProps[key] = value;
    } else {
      functionalProps[key] = value;
    }
  });
  
  return { styledProps, functionalProps };
};

/**
 * Filters out DOM-invalid props (those starting with $)
 */
export const filterDOMProps = <T extends Record<string, any>>(props: T): Partial<T> => {
  const filtered: Partial<T> = {};
  
  Object.entries(props).forEach(([key, value]) => {
    if (!key.startsWith('$') && !['variant', 'size', 'padding', 'loading', 'glitch', 'hover'].includes(key)) {
      (filtered as any)[key] = value;
    }
  });
  
  return filtered;
};

/**
 * Type guard to check if a value is a valid variant
 */
export const isValidVariant = (variant: any): variant is string => {
  const validVariants = ['primary', 'secondary', 'danger', 'ghost', 'default', 'glass', 'neon', 'hologram'];
  return typeof variant === 'string' && validVariants.includes(variant);
};

/**
 * Type guard to check if a value is a valid size
 */
export const isValidSize = (size: any): size is 'sm' | 'md' | 'lg' => {
  return typeof size === 'string' && ['sm', 'md', 'lg'].includes(size);
};

/**
 * Validates component props at runtime
 */
export const validateProps = <T extends Record<string, any>>(
  props: T,
  validators: Record<keyof T, (value: any) => boolean>
): { isValid: boolean; errors: string[] } => {
  const errors: string[] = [];
  
  Object.entries(validators).forEach(([key, validator]) => {
    const value = props[key as keyof T];
    if (value !== undefined && !validator(value)) {
      errors.push(`Invalid prop ${key}: ${value}`);
    }
  });
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

/**
 * Merges default props with user props
 */
export const mergeProps = <T extends Record<string, any>>(
  defaultProps: Partial<T>,
  userProps: Partial<T>
): T => {
  return { ...defaultProps, ...userProps } as T;
};

/**
 * Creates a shouldForwardProp function for styled-components
 */
export const createShouldForwardProp = (excludeProps: string[] = []) => {
  return (prop: string) => {
    // Always exclude $ prefixed props
    if (prop.startsWith('$')) return false;
    
    // Exclude specific props
    if (excludeProps.includes(prop)) return false;
    
    // Exclude common styled props that shouldn't be forwarded to DOM
    const commonStyledProps = ['variant', 'size', 'padding', 'loading', 'glitch', 'hover'];
    if (commonStyledProps.includes(prop)) return false;
    
    return true;
  };
};

/**
 * Converts legacy component props to new format
 */
export const convertLegacyButtonProps = (props: any) => {
  const converted = { ...props };
  
  // Convert variant prop
  if (converted.variant && !converted.$variant) {
    converted.$variant = converted.variant;
    delete converted.variant;
  }
  
  // Convert size prop
  if (converted.size && !converted.$size) {
    converted.$size = converted.size;
    delete converted.size;
  }
  
  // Convert loading prop
  if (converted.loading !== undefined && converted.$loading === undefined) {
    converted.$loading = converted.loading;
    delete converted.loading;
  }
  
  // Convert glitch prop
  if (converted.glitch !== undefined && converted.$glitch === undefined) {
    converted.$glitch = converted.glitch;
    delete converted.glitch;
  }
  
  return converted;
};

/**
 * Converts legacy card props to new format
 */
export const convertLegacyCardProps = (props: any) => {
  const converted = { ...props };
  
  // Convert variant prop
  if (converted.variant && !converted.$variant) {
    converted.$variant = converted.variant;
    delete converted.variant;
  }
  
  // Convert padding prop
  if (converted.padding && !converted.$padding) {
    converted.$padding = converted.padding;
    delete converted.padding;
  }
  
  // Convert hover prop
  if (converted.hover !== undefined && converted.$hover === undefined) {
    converted.$hover = converted.hover;
    delete converted.hover;
  }
  
  // Convert glitch prop
  if (converted.glitch !== undefined && converted.$glitch === undefined) {
    converted.$glitch = converted.glitch;
    delete converted.glitch;
  }
  
  return converted;
};

/**
 * Debug helper to log prop transformations
 */
export const debugProps = (componentName: string, originalProps: any, transformedProps: any) => {
  if (process.env.NODE_ENV === 'development') {
    console.group(`${componentName} Props Debug`);
    console.log('Original:', originalProps);
    console.log('Transformed:', transformedProps);
    console.groupEnd();
  }
};