// Utility script to fix component prop usage across the codebase
// This provides runtime prop transformation for backward compatibility

import React from 'react';

/**
 * Transforms component props to use $ prefixed versions for styled-components
 */
export const transformComponentProps = (props: any) => {
  const transformed = { ...props };
  
  // CyberpunkButton prop transformations
  if ('variant' in transformed && !('$variant' in transformed)) {
    transformed.$variant = transformed.variant;
    delete transformed.variant;
  }
  
  if ('size' in transformed && !('$size' in transformed)) {
    transformed.$size = transformed.size;
    delete transformed.size;
  }
  
  if ('loading' in transformed && !('$loading' in transformed)) {
    transformed.$loading = transformed.loading;
    delete transformed.loading;
  }
  
  if ('glitch' in transformed && !('$glitch' in transformed)) {
    transformed.$glitch = transformed.glitch;
    delete transformed.glitch;
  }
  
  // CyberpunkCard prop transformations
  if ('padding' in transformed && !('$padding' in transformed)) {
    transformed.$padding = transformed.padding;
    delete transformed.padding;
  }
  
  if ('hover' in transformed && !('$hover' in transformed)) {
    transformed.$hover = transformed.hover;
    delete transformed.hover;
  }
  
  return transformed;
};

/**
 * Higher-order component that automatically fixes props
 */
export const withPropFix = <P extends object>(
  Component: React.ComponentType<P>
): React.FC<P> => {
  const WrappedComponent: React.FC<P> = (props) => {
    const fixedProps = transformComponentProps(props);
    return React.createElement(Component, fixedProps);
  };
  return WrappedComponent;
};

/**
 * Creates a wrapper that handles both old and new prop formats
 */
export const createPropCompatibleComponent = <P extends object>(
  Component: React.ComponentType<P>,
  displayName?: string
) => {
  const CompatibleComponent = React.forwardRef<any, P>((props, ref) => {
    const compatibleProps = transformComponentProps(props);
    return React.createElement(Component, { ...compatibleProps, ref });
  });
  
  if (displayName) {
    CompatibleComponent.displayName = displayName;
  }
  
  return CompatibleComponent;
};

/**
 * Specific prop transformers for different component types
 */
export const transformButtonProps = (props: any) => {
  const transformed = { ...props };
  
  // Transform variant
  if (transformed.variant && !transformed.$variant) {
    transformed.$variant = transformed.variant;
    delete transformed.variant;
  }
  
  // Transform size
  if (transformed.size && !transformed.$size) {
    transformed.$size = transformed.size;
    delete transformed.size;
  }
  
  // Transform loading
  if (transformed.loading !== undefined && transformed.$loading === undefined) {
    transformed.$loading = transformed.loading;
    delete transformed.loading;
  }
  
  // Transform glitch
  if (transformed.glitch !== undefined && transformed.$glitch === undefined) {
    transformed.$glitch = transformed.glitch;
    delete transformed.glitch;
  }
  
  return transformed;
};

export const transformCardProps = (props: any) => {
  const transformed = { ...props };
  
  // Transform variant
  if (transformed.variant && !transformed.$variant) {
    transformed.$variant = transformed.variant;
    delete transformed.variant;
  }
  
  // Transform padding
  if (transformed.padding && !transformed.$padding) {
    transformed.$padding = transformed.padding;
    delete transformed.padding;
  }
  
  // Transform hover
  if (transformed.hover !== undefined && transformed.$hover === undefined) {
    transformed.$hover = transformed.hover;
    delete transformed.hover;
  }
  
  // Transform glitch
  if (transformed.glitch !== undefined && transformed.$glitch === undefined) {
    transformed.$glitch = transformed.glitch;
    delete transformed.glitch;
  }
  
  return transformed;
};

/**
 * Runtime prop fixer that can be applied to any component
 */
export class ComponentPropFixer {
  private static instance: ComponentPropFixer;
  private transformers: Map<string, (props: any) => any> = new Map();
  
  constructor() {
    // Register default transformers
    this.transformers.set('CyberpunkButton', transformButtonProps);
    this.transformers.set('CyberpunkCard', transformCardProps);
  }
  
  static getInstance(): ComponentPropFixer {
    if (!ComponentPropFixer.instance) {
      ComponentPropFixer.instance = new ComponentPropFixer();
    }
    return ComponentPropFixer.instance;
  }
  
  registerTransformer(componentName: string, transformer: (props: any) => any) {
    this.transformers.set(componentName, transformer);
  }
  
  transformProps(componentName: string, props: any) {
    const transformer = this.transformers.get(componentName);
    if (transformer) {
      return transformer(props);
    }
    return transformComponentProps(props); // Fallback to generic transformer
  }
  
  wrapComponent<P extends object>(
    componentName: string,
    Component: React.ComponentType<P>
  ): React.FC<P> {
    const WrappedComponent: React.FC<P> = (props) => {
      const transformedProps = this.transformProps(componentName, props);
      return React.createElement(Component, transformedProps);
    };
    return WrappedComponent;
  }
}

// Export singleton instance
export const propFixer = ComponentPropFixer.getInstance();

/**
 * Utility to check if props need transformation
 */
export const needsPropTransformation = (props: any): boolean => {
  const legacyProps = ['variant', 'size', 'padding', 'loading', 'glitch', 'hover'];
  return legacyProps.some(prop => prop in props);
};

/**
 * Development helper to warn about legacy prop usage
 */
export const warnAboutLegacyProps = (componentName: string, props: any) => {
  if (process.env.NODE_ENV === 'development' && needsPropTransformation(props)) {
    const legacyProps = ['variant', 'size', 'padding', 'loading', 'glitch', 'hover'];
    const foundLegacyProps = legacyProps.filter(prop => prop in props);
    
    console.warn(
      `${componentName}: Legacy props detected: ${foundLegacyProps.join(', ')}. ` +
      `Consider using $ prefixed versions: ${foundLegacyProps.map(p => `$${p}`).join(', ')}`
    );
  }
};

/**
 * Batch transform props for multiple components
 */
export const batchTransformProps = (
  componentProps: Record<string, any>
): Record<string, any> => {
  const transformed: Record<string, any> = {};
  
  Object.entries(componentProps).forEach(([componentName, props]) => {
    transformed[componentName] = propFixer.transformProps(componentName, props);
  });
  
  return transformed;
};

/**
 * Create a context for prop transformation
 */
export const PropTransformContext = React.createContext<{
  transformProps: (componentName: string, props: any) => any;
}>({
  transformProps: (componentName: string, props: any) => 
    propFixer.transformProps(componentName, props),
});

/**
 * Hook to use prop transformation
 */
export const usePropTransform = () => {
  const context = React.useContext(PropTransformContext);
  return context.transformProps;
};

/**
 * Provider component for prop transformation
 */
export const PropTransformProvider: React.FC<{
  children: React.ReactNode;
  customTransformers?: Record<string, (props: any) => any>;
}> = ({ children, customTransformers = {} }) => {
  const transformProps = React.useCallback((componentName: string, props: any) => {
    if (customTransformers[componentName]) {
      return customTransformers[componentName](props);
    }
    return propFixer.transformProps(componentName, props);
  }, [customTransformers]);
  
  return React.createElement(
    PropTransformContext.Provider,
    { value: { transformProps } },
    children
  );
};