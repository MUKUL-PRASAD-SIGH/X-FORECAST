// Cyberpunk Theme Configuration
export interface CyberpunkColors {
  // Primary cyberpunk colors
  neonBlue: string;
  hotPink: string;
  acidGreen: string;
  electricPurple: string;
  cyberYellow: string;
  
  // Background colors
  darkBg: string;
  darkerBg: string;
  cardBg: string;
  overlayBg: string;
  
  // Text colors
  primaryText: string;
  secondaryText: string;
  accentText: string;
  
  // Status colors
  success: string;
  warning: string;
  error: string;
  info: string;
  
  // Glow effects
  glowBlue: string;
  glowPink: string;
  glowGreen: string;
  glowPurple: string;
}

export interface CyberpunkEffects {
  // Box shadows for glow effects
  neonGlow: string;
  softGlow: string;
  intenseGlow: string;
  
  // Gradients
  primaryGradient: string;
  secondaryGradient: string;
  backgroundGradient: string;
  
  // Animations
  glitchAnimation: string;
  pulseAnimation: string;
  scanlineAnimation: string;
  
  // Filters
  glitchFilter: string;
  hologramFilter: string;
}

export interface CyberpunkTypography {
  fontFamily: {
    primary: string;
    mono: string;
    display: string;
  };
  fontSize: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
    xxl: string;
    display: string;
  };
  fontWeight: {
    light: number;
    normal: number;
    medium: number;
    bold: number;
    black: number;
  };
}

export interface CyberpunkSpacing {
  xs: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
  xxl: string;
}

export interface CyberpunkTheme {
  colors: CyberpunkColors;
  effects: CyberpunkEffects;
  typography: CyberpunkTypography;
  spacing: CyberpunkSpacing;
  breakpoints: {
    mobile: string;
    tablet: string;
    desktop: string;
    wide: string;
  };
}

// Main cyberpunk theme object
export const cyberpunkTheme: CyberpunkTheme = {
  colors: {
    // Primary cyberpunk colors
    neonBlue: '#00FFFF',
    hotPink: '#FF1493',
    acidGreen: '#39FF14',
    electricPurple: '#BF00FF',
    cyberYellow: '#FFFF00',
    
    // Background colors
    darkBg: '#0A0A0A',
    darkerBg: '#050505',
    cardBg: 'rgba(20, 20, 20, 0.8)',
    overlayBg: 'rgba(0, 0, 0, 0.9)',
    
    // Text colors
    primaryText: '#FFFFFF',
    secondaryText: '#B0B0B0',
    accentText: '#00FFFF',
    
    // Status colors
    success: '#39FF14',
    warning: '#FFFF00',
    error: '#FF0040',
    info: '#00FFFF',
    
    // Glow effects
    glowBlue: 'rgba(0, 255, 255, 0.5)',
    glowPink: 'rgba(255, 20, 147, 0.5)',
    glowGreen: 'rgba(57, 255, 20, 0.5)',
    glowPurple: 'rgba(191, 0, 255, 0.5)',
  },
  
  effects: {
    // Box shadows for glow effects
    neonGlow: '0 0 20px rgba(0, 255, 255, 0.6), 0 0 40px rgba(0, 255, 255, 0.4), 0 0 60px rgba(0, 255, 255, 0.2)',
    softGlow: '0 0 10px rgba(0, 255, 255, 0.3), 0 0 20px rgba(0, 255, 255, 0.2)',
    intenseGlow: '0 0 30px rgba(0, 255, 255, 0.8), 0 0 60px rgba(0, 255, 255, 0.6), 0 0 90px rgba(0, 255, 255, 0.4)',
    
    // Gradients
    primaryGradient: 'linear-gradient(135deg, #00FFFF 0%, #FF1493 50%, #39FF14 100%)',
    secondaryGradient: 'linear-gradient(45deg, #BF00FF 0%, #00FFFF 100%)',
    backgroundGradient: 'linear-gradient(180deg, #0A0A0A 0%, #050505 100%)',
    
    // Animations
    glitchAnimation: `
      @keyframes glitch {
        0% { transform: translate(0); }
        20% { transform: translate(-2px, 2px); }
        40% { transform: translate(-2px, -2px); }
        60% { transform: translate(2px, 2px); }
        80% { transform: translate(2px, -2px); }
        100% { transform: translate(0); }
      }
    `,
    pulseAnimation: `
      @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
      }
    `,
    scanlineAnimation: `
      @keyframes scanline {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100vh); }
      }
    `,
    
    // Filters
    glitchFilter: 'hue-rotate(90deg) saturate(150%) contrast(120%)',
    hologramFilter: 'drop-shadow(0 0 10px rgba(0, 255, 255, 0.5)) brightness(110%) contrast(120%)',
  },
  
  typography: {
    fontFamily: {
      primary: '"Orbitron", "Roboto", sans-serif',
      mono: '"Fira Code", "Courier New", monospace',
      display: '"Exo 2", "Orbitron", sans-serif',
    },
    fontSize: {
      xs: '0.75rem',
      sm: '0.875rem',
      md: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      xxl: '1.5rem',
      display: '2.5rem',
    },
    fontWeight: {
      light: 300,
      normal: 400,
      medium: 500,
      bold: 700,
      black: 900,
    },
  },
  
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    xxl: '3rem',
  },
  
  breakpoints: {
    mobile: '768px',
    tablet: '1024px',
    desktop: '1440px',
    wide: '1920px',
  },
};

// CSS custom properties for global use
export const cyberpunkCSSVariables = `
  :root {
    --color-neon-blue: ${cyberpunkTheme.colors.neonBlue};
    --color-hot-pink: ${cyberpunkTheme.colors.hotPink};
    --color-acid-green: ${cyberpunkTheme.colors.acidGreen};
    --color-electric-purple: ${cyberpunkTheme.colors.electricPurple};
    --color-cyber-yellow: ${cyberpunkTheme.colors.cyberYellow};
    
    --color-dark-bg: ${cyberpunkTheme.colors.darkBg};
    --color-darker-bg: ${cyberpunkTheme.colors.darkerBg};
    --color-card-bg: ${cyberpunkTheme.colors.cardBg};
    
    --color-primary-text: ${cyberpunkTheme.colors.primaryText};
    --color-secondary-text: ${cyberpunkTheme.colors.secondaryText};
    --color-accent-text: ${cyberpunkTheme.colors.accentText};
    
    --effect-neon-glow: ${cyberpunkTheme.effects.neonGlow};
    --effect-soft-glow: ${cyberpunkTheme.effects.softGlow};
    --effect-intense-glow: ${cyberpunkTheme.effects.intenseGlow};
    
    --gradient-primary: ${cyberpunkTheme.effects.primaryGradient};
    --gradient-secondary: ${cyberpunkTheme.effects.secondaryGradient};
    --gradient-background: ${cyberpunkTheme.effects.backgroundGradient};
    
    --font-primary: ${cyberpunkTheme.typography.fontFamily.primary};
    --font-mono: ${cyberpunkTheme.typography.fontFamily.mono};
    --font-display: ${cyberpunkTheme.typography.fontFamily.display};
  }
  
  ${cyberpunkTheme.effects.glitchAnimation}
  ${cyberpunkTheme.effects.pulseAnimation}
  ${cyberpunkTheme.effects.scanlineAnimation}
`;