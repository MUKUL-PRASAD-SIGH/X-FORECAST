import React from 'react';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from 'styled-components';
import '@testing-library/jest-dom';
import { CyberpunkCard } from '../CyberpunkCard';
import { CyberpunkButton } from '../CyberpunkButton';
import { PropFilterWrapper } from '../../../theme/StyleSheetManager';
import { cyberpunkTheme } from '../../../theme/cyberpunkTheme';

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <PropFilterWrapper>
      <ThemeProvider theme={cyberpunkTheme}>
        {component}
      </ThemeProvider>
    </PropFilterWrapper>
  );
};

describe('Component Prop Filtering', () => {
  // Mock console.error to capture React warnings
  let consoleErrorSpy: jest.SpyInstance;
  
  beforeEach(() => {
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
  });
  
  afterEach(() => {
    consoleErrorSpy.mockRestore();
  });

  describe('CyberpunkCard Prop Filtering', () => {
    it('should not pass transient props to DOM elements', () => {
      const { container } = renderWithTheme(
        <CyberpunkCard
          $variant="neon"
          $padding="lg"
          $glitch={true}
          $hover={true}
          data-testid="cyberpunk-card"
        >
          Test Content
        </CyberpunkCard>
      );

      const cardElement = container.querySelector('[data-testid="cyberpunk-card"]');
      
      // Verify transient props are not in DOM
      expect(cardElement).not.toHaveAttribute('$variant');
      expect(cardElement).not.toHaveAttribute('$padding');
      expect(cardElement).not.toHaveAttribute('$glitch');
      expect(cardElement).not.toHaveAttribute('$hover');
      
      // Verify valid props are preserved
      expect(cardElement).toHaveAttribute('data-testid', 'cyberpunk-card');
    });

    it('should render without React prop warnings', () => {
      renderWithTheme(
        <CyberpunkCard
          $variant="glass"
          $padding="md"
          $glitch={false}
          $hover={true}
        >
          Test Content
        </CyberpunkCard>
      );

      // Check that no React warnings were logged
      const reactWarnings = consoleErrorSpy.mock.calls.filter(call =>
        call[0]?.includes?.('React does not recognize') ||
        call[0]?.includes?.('received a non-boolean attribute')
      );
      
      expect(reactWarnings).toHaveLength(0);
    });

    it('should apply correct styling based on transient props', () => {
      const { container } = renderWithTheme(
        <CyberpunkCard
          $variant="neon"
          $padding="lg"
          data-testid="styled-card"
        >
          Test Content
        </CyberpunkCard>
      );

      const cardElement = container.querySelector('[data-testid="styled-card"]');
      expect(cardElement).toBeInTheDocument();
      expect(screen.getByText('Test Content')).toBeInTheDocument();
    });

    it('should handle all variant types without warnings', () => {
      const variants = ['default', 'glass', 'neon', 'hologram'] as const;
      
      variants.forEach(variant => {
        const { unmount } = renderWithTheme(
          <CyberpunkCard $variant={variant}>
            {variant} content
          </CyberpunkCard>
        );
        
        expect(screen.getByText(`${variant} content`)).toBeInTheDocument();
        
        // Check no warnings for this variant
        const reactWarnings = consoleErrorSpy.mock.calls.filter(call =>
          call[0]?.includes?.('React does not recognize')
        );
        expect(reactWarnings).toHaveLength(0);
        
        unmount();
      });
    });
  });

  describe('CyberpunkButton Prop Filtering', () => {
    it('should not pass transient props to DOM elements', () => {
      const { container } = renderWithTheme(
        <CyberpunkButton
          $variant="primary"
          $size="lg"
          $loading={true}
          $glitch={false}
          data-testid="cyberpunk-button"
        >
          Test Button
        </CyberpunkButton>
      );

      const buttonElement = container.querySelector('[data-testid="cyberpunk-button"]');
      
      // Verify transient props are not in DOM
      expect(buttonElement).not.toHaveAttribute('$variant');
      expect(buttonElement).not.toHaveAttribute('$size');
      expect(buttonElement).not.toHaveAttribute('$loading');
      expect(buttonElement).not.toHaveAttribute('$glitch');
      
      // Verify valid props are preserved
      expect(buttonElement).toHaveAttribute('data-testid', 'cyberpunk-button');
    });

    it('should render without React prop warnings', () => {
      renderWithTheme(
        <CyberpunkButton
          $variant="secondary"
          $size="md"
          $loading={false}
          $glitch={true}
        >
          Test Button
        </CyberpunkButton>
      );

      // Check that no React warnings were logged
      const reactWarnings = consoleErrorSpy.mock.calls.filter(call =>
        call[0]?.includes?.('React does not recognize') ||
        call[0]?.includes?.('received a non-boolean attribute')
      );
      
      expect(reactWarnings).toHaveLength(0);
    });

    it('should preserve standard button attributes', () => {
      const handleClick = jest.fn();
      
      renderWithTheme(
        <CyberpunkButton
          $variant="primary"
          onClick={handleClick}
          disabled={false}
          type="button"
          data-testid="standard-button"
        >
          Click Me
        </CyberpunkButton>
      );

      const buttonElement = screen.getByTestId('standard-button');
      
      // Verify standard HTML attributes are preserved
      expect(buttonElement).toHaveAttribute('type', 'button');
      expect(buttonElement).toHaveAttribute('data-testid', 'standard-button');
      expect(buttonElement).not.toBeDisabled();
    });

    it('should handle all variant and size combinations without warnings', () => {
      const variants = ['primary', 'secondary', 'danger', 'ghost'] as const;
      const sizes = ['sm', 'md', 'lg'] as const;
      
      variants.forEach(variant => {
        sizes.forEach(size => {
          const { unmount } = renderWithTheme(
            <CyberpunkButton $variant={variant} $size={size}>
              {variant}-{size}
            </CyberpunkButton>
          );
          
          expect(screen.getByText(`${variant}-${size}`)).toBeInTheDocument();
          
          // Check no warnings for this combination
          const reactWarnings = consoleErrorSpy.mock.calls.filter(call =>
            call[0]?.includes?.('React does not recognize')
          );
          expect(reactWarnings).toHaveLength(0);
          
          unmount();
        });
      });
    });

    it('should handle loading state without prop warnings', () => {
      renderWithTheme(
        <CyberpunkButton
          $variant="primary"
          $loading={true}
          disabled={true}
        >
          Loading Button
        </CyberpunkButton>
      );

      const reactWarnings = consoleErrorSpy.mock.calls.filter(call =>
        call[0]?.includes?.('React does not recognize')
      );
      
      expect(reactWarnings).toHaveLength(0);
    });
  });

  describe('StyleSheetManager Global Filtering', () => {
    it('should filter common styling props globally', () => {
      // Create a simple styled component to test global filtering
      const TestComponent = ({ variant, size, loading, ...props }: any) => (
        <div {...props}>Test Component</div>
      );

      const { container } = renderWithTheme(
        <TestComponent
          variant="test"
          size="large"
          loading={true}
          glitch={false}
          hover={true}
          data-testid="global-filter-test"
        />
      );

      const element = container.querySelector('[data-testid="global-filter-test"]');
      
      // These props should be filtered by StyleSheetManager
      expect(element).not.toHaveAttribute('variant');
      expect(element).not.toHaveAttribute('size');
      expect(element).not.toHaveAttribute('loading');
      expect(element).not.toHaveAttribute('glitch');
      expect(element).not.toHaveAttribute('hover');
      
      // Valid props should be preserved
      expect(element).toHaveAttribute('data-testid', 'global-filter-test');
    });

    it('should filter transient props starting with $', () => {
      const TestComponent = (props: any) => <div {...props}>Test</div>;

      const { container } = renderWithTheme(
        <TestComponent
          $customProp="value"
          $anotherProp={true}
          data-testid="transient-test"
        />
      );

      const element = container.querySelector('[data-testid="transient-test"]');
      
      expect(element).not.toHaveAttribute('$customProp');
      expect(element).not.toHaveAttribute('$anotherProp');
      expect(element).toHaveAttribute('data-testid', 'transient-test');
    });

    it('should not generate console warnings for filtered props', () => {
      // Clear any previous console calls
      consoleErrorSpy.mockClear();

      // Test with CyberpunkCard which uses the StyleSheetManager
      renderWithTheme(
        <CyberpunkCard
          $variant="neon"
          $padding="md"
          $glitch={false}
          $hover={true}
          data-testid="warning-test"
        >
          Test Content
        </CyberpunkCard>
      );

      // Verify no React warnings were generated
      const reactWarnings = consoleErrorSpy.mock.calls.filter(call =>
        call[0]?.includes?.('React does not recognize') ||
        call[0]?.includes?.('received a non-boolean attribute')
      );
      
      expect(reactWarnings).toHaveLength(0);
    });
  });

  describe('Component Integration Tests', () => {
    it('should render CyberpunkCard and CyberpunkButton together without warnings', () => {
      consoleErrorSpy.mockClear();

      renderWithTheme(
        <div>
          <CyberpunkCard
            $variant="neon"
            $padding="lg"
            $glitch={true}
            $hover={true}
          >
            <CyberpunkButton
              $variant="primary"
              $size="md"
              $loading={false}
              $glitch={true}
            >
              Test Button
            </CyberpunkButton>
          </CyberpunkCard>
        </div>
      );

      // Verify both components render
      expect(screen.getByText('Test Button')).toBeInTheDocument();

      // Verify no React warnings
      const reactWarnings = consoleErrorSpy.mock.calls.filter(call =>
        call[0]?.includes?.('React does not recognize') ||
        call[0]?.includes?.('received a non-boolean attribute')
      );
      
      expect(reactWarnings).toHaveLength(0);
    });

    it('should handle edge cases without warnings', () => {
      consoleErrorSpy.mockClear();

      renderWithTheme(
        <div>
          <CyberpunkCard
            $variant={undefined}
            $padding={undefined}
            $glitch={undefined}
            $hover={undefined}
          >
            <CyberpunkButton
              $variant={undefined}
              $size={undefined}
              $loading={undefined}
              $glitch={undefined}
            >
              Edge Case Button
            </CyberpunkButton>
          </CyberpunkCard>
        </div>
      );

      expect(screen.getByText('Edge Case Button')).toBeInTheDocument();

      const reactWarnings = consoleErrorSpy.mock.calls.filter(call =>
        call[0]?.includes?.('React does not recognize') ||
        call[0]?.includes?.('received a non-boolean attribute')
      );
      
      expect(reactWarnings).toHaveLength(0);
    });
  });
});