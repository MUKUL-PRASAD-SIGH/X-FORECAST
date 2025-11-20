// Utility for migrating legacy props to new styled-component format
// This helps maintain backward compatibility while transitioning to $ prefixed props

import React, { ComponentType } from 'react';

/**
 * Higher-order component that automatically converts legacy props to new format
 */
export function withPropMigration<T extends Record<string, any>>(
  Component: ComponentType<T>,
  propMappings: Record<string, string> = {}
) {
  const defaultMappings = {
    variant: '$variant',
    size: '$size',
    padding: '$padding',
    loading: '$loading',
    glitch: '$glitch',
    hover: '$hover',
  };

  const allMappings = { ...defaultMappings, ...propMappings };

  return function MigratedComponent(props: any) {
    const migratedProps = { ...props };

    // Apply prop mappings
    Object.entries(allMappings).forEach(([oldProp, newProp]) => {
      if (oldProp in migratedProps && !(newProp in migratedProps)) {
        migratedProps[newProp] = migratedProps[oldProp];
        delete migratedProps[oldProp];
      }
    });

    return React.createElement(Component, migratedProps);
  };
}

/**
 * Transforms props object to use $ prefixed versions
 */
export function migrateProps<T extends Record<string, any>>(
  props: T,
  customMappings: Record<string, string> = {}
): T {
  const defaultMappings = {
    variant: '$variant',
    size: '$size',
    padding: '$padding',
    loading: '$loading',
    glitch: '$glitch',
    hover: '$hover',
  };

  const allMappings = { ...defaultMappings, ...customMappings };
  const migratedProps = { ...props };

  Object.entries(allMappings).forEach(([oldProp, newProp]) => {
    if (oldProp in migratedProps && !(newProp in migratedProps)) {
      migratedProps[newProp as keyof T] = migratedProps[oldProp as keyof T];
      delete migratedProps[oldProp as keyof T];
    }
  });

  return migratedProps;
}

/**
 * Creates a wrapper component that handles both legacy and new prop formats
 */
export function createCompatibleComponent<T extends Record<string, any>>(
  Component: ComponentType<T>,
  displayName?: string
) {
  const CompatibleComponent = (props: any) => {
    const migratedProps = migrateProps(props);
    return React.createElement(Component, migratedProps);
  };

  if (displayName) {
    CompatibleComponent.displayName = displayName;
  }

  return CompatibleComponent;
}

/**
 * Validates that props follow the new format
 */
export function validateNewPropFormat(props: Record<string, any>): {
  isValid: boolean;
  warnings: string[];
} {
  const warnings: string[] = [];
  const legacyProps = ['variant', 'size', 'padding', 'loading', 'glitch', 'hover'];

  legacyProps.forEach(prop => {
    if (prop in props) {
      warnings.push(`Legacy prop '${prop}' detected. Consider using '$${prop}' instead.`);
    }
  });

  return {
    isValid: warnings.length === 0,
    warnings,
  };
}

/**
 * Development helper to log prop migration warnings
 */
export function logPropMigrationWarnings(
  componentName: string,
  props: Record<string, any>
) {
  if (process.env.NODE_ENV === 'development') {
    const { warnings } = validateNewPropFormat(props);
    if (warnings.length > 0) {
      console.group(`${componentName} Prop Migration Warnings`);
      warnings.forEach(warning => console.warn(warning));
      console.groupEnd();
    }
  }
}

/**
 * Batch migrate multiple components
 */
export function migrateComponentProps(components: Record<string, ComponentType<any>>) {
  const migratedComponents: Record<string, ComponentType<any>> = {};

  Object.entries(components).forEach(([name, Component]) => {
    migratedComponents[name] = createCompatibleComponent(Component, `Migrated${name}`);
  });

  return migratedComponents;
}

/**
 * Type-safe prop migration for specific component types
 */
export interface ButtonPropMigration {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  glitch?: boolean;
}

export interface CardPropMigration {
  variant?: 'default' | 'glass' | 'neon' | 'hologram';
  padding?: 'sm' | 'md' | 'lg';
  hover?: boolean;
  glitch?: boolean;
}

export function migrateButtonProps(props: ButtonPropMigration & Record<string, any>) {
  return migrateProps(props, {
    variant: '$variant',
    size: '$size',
    loading: '$loading',
    glitch: '$glitch',
  });
}

export function migrateCardProps(props: CardPropMigration & Record<string, any>) {
  return migrateProps(props, {
    variant: '$variant',
    padding: '$padding',
    hover: '$hover',
    glitch: '$glitch',
  });
}

/**
 * Runtime prop converter for dynamic usage
 */
export class PropMigrator {
  private mappings: Record<string, string>;

  constructor(customMappings: Record<string, string> = {}) {
    this.mappings = {
      variant: '$variant',
      size: '$size',
      padding: '$padding',
      loading: '$loading',
      glitch: '$glitch',
      hover: '$hover',
      ...customMappings,
    };
  }

  migrate<T extends Record<string, any>>(props: T): T {
    const migratedProps = { ...props };

    Object.entries(this.mappings).forEach(([oldProp, newProp]) => {
      if (oldProp in migratedProps && !(newProp in migratedProps)) {
        migratedProps[newProp as keyof T] = migratedProps[oldProp as keyof T];
        delete migratedProps[oldProp as keyof T];
      }
    });

    return migratedProps;
  }

  addMapping(oldProp: string, newProp: string) {
    this.mappings[oldProp] = newProp;
  }

  removeMapping(oldProp: string) {
    delete this.mappings[oldProp];
  }

  getMappings() {
    return { ...this.mappings };
  }
}

// Create default migrator instance
export const defaultPropMigrator = new PropMigrator();

/**
 * Utility to create shouldForwardProp function that handles migrated props
 */
export function createMigrationAwareShouldForwardProp(
  excludeProps: string[] = []
): (prop: string) => boolean {
  return (prop: string) => {
    // Never forward $ prefixed props
    if (prop.startsWith('$')) return false;
    
    // Never forward legacy styled props
    const legacyStyledProps = ['variant', 'size', 'padding', 'loading', 'glitch', 'hover'];
    if (legacyStyledProps.includes(prop)) return false;
    
    // Never forward custom excluded props
    if (excludeProps.includes(prop)) return false;
    
    // Forward everything else
    return true;
  };
}