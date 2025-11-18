import React, { useState, useEffect, useRef } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { motion } from 'framer-motion';

// Animation keyframes
const neonPulse = keyframes`
  0%, 100% { 
    box-shadow: 
      0 0 20px rgba(0, 255, 255, 0.4),
      0 0 40px rgba(0, 255, 255, 0.2),
      inset 0 0 20px rgba(0, 255, 255, 0.1);
  }
  50% { 
    box-shadow: 
      0 0 40px rgba(0, 255, 255, 0.8),
      0 0 80px rgba(0, 255, 255, 0.4),
      inset 0 0 40px rgba(0, 255, 255, 0.2);
  }
`;

const gaugeGlow = keyframes`
  0%, 100% { 
    filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.6));
  }
  50% { 
    filter: drop-shadow(0 0 20px rgba(0, 255, 255, 0.9));
  }
`;

const sparkle = keyframes`
  0%, 100% { opacity: 0; transform: scale(0); }
  50% { opacity: 1; transform: scale(1); }
`;

const rotateGlow = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

// Styled components
const GaugeContainer = styled(motion.div)<{ size: number }>`
  position: relative;
  width: ${props => props.size}px;
  height: ${props => props.size}px;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const GaugeBackground = styled.div<{ size: number }>`
  position: absolute;
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: 
    radial-gradient(circle at 30% 30%, rgba(0, 255, 255, 0.1) 0%, transparent 50%),
    radial-gradient(circle at 70% 70%, rgba(255, 20, 147, 0.1) 0%, transparent 50%),
    conic-gradient(from 0deg, 
      rgba(0, 0, 0, 0.9) 0deg,
      rgba(20, 20, 40, 0.9) 180deg,
      rgba(0, 0, 0, 0.9) 360deg
    );
  border: 3px solid rgba(0, 255, 255, 0.3);
  animation: ${neonPulse} 3s infinite;
`;

const GaugeTrack = styled.svg<{ size: number }>`
  position: absolute;
  width: 100%;
  height: 100%;
  transform: rotate(-90deg);
  animation: ${gaugeGlow} 2s infinite;
`;

const GaugeProgress = styled.circle<{ 
  circumference: number; 
  progress: number;
  status: 'excellent' | 'good' | 'average' | 'poor' | 'critical';
}>`
  fill: none;
  stroke-width: 8;
  stroke-linecap: round;
  stroke-dasharray: ${props => props.circumference};
  stroke-dashoffset: ${props => props.circumference * (1 - props.progress)};
  stroke: ${props => {
    switch (props.status) {
      case 'excellent': return 'url(#excellentGradient)';
      case 'good': return 'url(#goodGradient)';
      case 'average': return 'url(#averageGradient)';
      case 'poor': return 'url(#poorGradient)';
      case 'critical': return 'url(#criticalGradient)';
      default: return props.theme.colors.secondaryText;
    }
  }};
  transition: stroke-dashoffset 1s ease-in-out;
  filter: drop-shadow(0 0 8px ${props => {
    switch (props.status) {
      case 'excellent': return '#39FF14';
      case 'good': return '#00FFFF';
      case 'average': return '#FFFF00';
      case 'poor': return '#FF8C00';
      case 'critical': return '#FF0040';
      default: return '#B0B0B0';
    }
  }});
`;

const GaugeCenter = styled.div<{ size: number }>`
  position: relative;
  z-index: 10;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: ${props => props.size * 0.6}px;
  height: ${props => props.size * 0.6}px;
  border-radius: 50%;
  background: 
    radial-gradient(circle at center, 
      rgba(0, 0, 0, 0.9) 0%, 
      rgba(20, 20, 40, 0.8) 70%, 
      rgba(0, 0, 0, 0.9) 100%
    );
  border: 2px solid rgba(0, 255, 255, 0.5);
  backdrop-filter: blur(10px);
`;

const GaugeValue = styled.div<{ 
  status: 'excellent' | 'good' | 'average' | 'poor' | 'critical';
  size: number;
}>`
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.size * 0.08}px;
  font-weight: ${props => props.theme.typography.fontWeight.black};
  color: ${props => {
    switch (props.status) {
      case 'excellent': return props.theme.colors.acidGreen;
      case 'good': return props.theme.colors.neonBlue;
      case 'average': return props.theme.colors.cyberYellow;
      case 'poor': return '#FF8C00';
      case 'critical': return props.theme.colors.error;
      default: return props.theme.colors.primaryText;
    }
  }};
  text-shadow: 
    0 0 10px currentColor,
    0 0 20px currentColor,
    0 0 30px currentColor;
  text-align: center;
  line-height: 1;
  margin-bottom: ${props => props.size * 0.02}px;
`;

const GaugeLabel = styled.div<{ size: number }>`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.size * 0.04}px;
  color: ${props => props.theme.colors.secondaryText};
  text-transform: uppercase;
  letter-spacing: 1px;
  text-align: center;
  opacity: 0.8;
`;

const GaugeStatus = styled.div<{ 
  status: 'excellent' | 'good' | 'average' | 'poor' | 'critical';
  size: number;
}>`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.size * 0.03}px;
  font-weight: ${props => props.theme.typography.fontWeight.bold};
  color: ${props => {
    switch (props.status) {
      case 'excellent': return props.theme.colors.acidGreen;
      case 'good': return props.theme.colors.neonBlue;
      case 'average': return props.theme.colors.cyberYellow;
      case 'poor': return '#FF8C00';
      case 'critical': return props.theme.colors.error;
      default: return props.theme.colors.primaryText;
    }
  }};
  text-transform: uppercase;
  letter-spacing: 1px;
  text-align: center;
  margin-top: ${props => props.size * 0.02}px;
`;

const SparkleEffect = styled.div<{ size: number }>`
  position: absolute;
  width: 100%;
  height: 100%;
  pointer-events: none;
  
  .sparkle {
    position: absolute;
    width: 4px;
    height: 4px;
    background: #00FFFF;
    border-radius: 50%;
    animation: ${sparkle} 2s infinite;
    box-shadow: 0 0 6px #00FFFF;
  }
`;

const RotatingRing = styled.div<{ size: number; active: boolean }>`
  position: absolute;
  width: ${props => props.size * 1.1}px;
  height: ${props => props.size * 1.1}px;
  border: 1px solid rgba(0, 255, 255, 0.2);
  border-radius: 50%;
  border-top-color: rgba(0, 255, 255, 0.6);
  border-right-color: rgba(255, 20, 147, 0.4);
  
  ${props => props.active && css`
    animation: ${rotateGlow} 4s linear infinite;
  `}
`;

const GaugeMarkers = styled.div<{ size: number }>`
  position: absolute;
  width: 100%;
  height: 100%;
  
  .marker {
    position: absolute;
    width: 2px;
    height: ${props => props.size * 0.08}px;
    background: rgba(0, 255, 255, 0.4);
    transform-origin: center ${props => props.size * 0.5}px;
  }
  
  .marker.major {
    width: 3px;
    height: ${props => props.size * 0.12}px;
    background: rgba(0, 255, 255, 0.6);
  }
`;

// Interfaces
export interface PerformanceGaugeProps {
  value: number;
  min?: number;
  max?: number;
  label: string;
  unit?: string;
  size?: number;
  showSparkles?: boolean;
  showRotatingRing?: boolean;
  animated?: boolean;
  thresholds?: {
    excellent: number;
    good: number;
    average: number;
    poor: number;
  };
}

export const PerformanceGauge: React.FC<PerformanceGaugeProps> = ({
  value,
  min = 0,
  max = 100,
  label,
  unit = '%',
  size = 200,
  showSparkles = true,
  showRotatingRing = true,
  animated = true,
  thresholds = {
    excellent: 90,
    good: 75,
    average: 60,
    poor: 40
  }
}) => {
  const [displayValue, setDisplayValue] = useState(min);
  const [sparkles, setSparkles] = useState<Array<{ id: number; x: number; y: number; delay: number }>>([]);
  const gaugeRef = useRef<HTMLDivElement>(null);

  // Animate value changes
  useEffect(() => {
    if (!animated) {
      setDisplayValue(value);
      return;
    }

    const startValue = displayValue;
    const endValue = Math.max(min, Math.min(max, value));
    const duration = 1000;
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function
      const easeOutCubic = 1 - Math.pow(1 - progress, 3);
      const currentValue = startValue + (endValue - startValue) * easeOutCubic;
      
      setDisplayValue(currentValue);
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [value, min, max, animated, displayValue]);

  // Generate sparkles
  useEffect(() => {
    if (!showSparkles) return;

    const generateSparkles = () => {
      const newSparkles = Array.from({ length: 8 }, (_, i) => {
        const angle = (i / 8) * 2 * Math.PI;
        const radius = size * 0.4;
        const x = 50 + (Math.cos(angle) * radius) / size * 100;
        const y = 50 + (Math.sin(angle) * radius) / size * 100;
        
        return {
          id: i,
          x,
          y,
          delay: i * 0.25
        };
      });
      
      setSparkles(newSparkles);
    };

    generateSparkles();
  }, [size, showSparkles]);

  const getStatus = (): 'excellent' | 'good' | 'average' | 'poor' | 'critical' => {
    if (displayValue >= thresholds.excellent) return 'excellent';
    if (displayValue >= thresholds.good) return 'good';
    if (displayValue >= thresholds.average) return 'average';
    if (displayValue >= thresholds.poor) return 'poor';
    return 'critical';
  };

  const getProgress = () => {
    return Math.max(0, Math.min(1, (displayValue - min) / (max - min)));
  };

  const getCircumference = () => {
    const radius = (size - 20) / 2;
    return 2 * Math.PI * radius;
  };

  const formatValue = () => {
    if (unit === '%') {
      return Math.round(displayValue);
    }
    return displayValue.toFixed(1);
  };

  const getStatusText = () => {
    const status = getStatus();
    return status.charAt(0).toUpperCase() + status.slice(1);
  };

  // Generate gauge markers
  const generateMarkers = () => {
    const markers = [];
    const totalMarkers = 20;
    
    for (let i = 0; i <= totalMarkers; i++) {
      const angle = (i / totalMarkers) * 360 - 90;
      const isMajor = i % 5 === 0;
      
      markers.push(
        <div
          key={i}
          className={`marker ${isMajor ? 'major' : ''}`}
          style={{
            transform: `rotate(${angle}deg)`,
            left: '50%',
            top: 0,
            marginLeft: '-1px'
          }}
        />
      );
    }
    
    return markers;
  };

  const radius = (size - 20) / 2;
  const circumference = getCircumference();
  const status = getStatus();

  return (
    <GaugeContainer
      ref={gaugeRef}
      size={size}
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.8, type: "spring", stiffness: 100 }}
      whileHover={{ scale: 1.05 }}
    >
      {showRotatingRing && (
        <RotatingRing size={size} active={animated} />
      )}
      
      <GaugeBackground size={size} />
      
      <GaugeMarkers size={size}>
        {generateMarkers()}
      </GaugeMarkers>
      
      <GaugeTrack size={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <linearGradient id="excellentGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#39FF14" />
            <stop offset="100%" stopColor="#7FFF00" />
          </linearGradient>
          <linearGradient id="goodGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#00FFFF" />
            <stop offset="100%" stopColor="#0080FF" />
          </linearGradient>
          <linearGradient id="averageGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#FFFF00" />
            <stop offset="100%" stopColor="#FFD700" />
          </linearGradient>
          <linearGradient id="poorGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#FF8C00" />
            <stop offset="100%" stopColor="#FF6347" />
          </linearGradient>
          <linearGradient id="criticalGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#FF0040" />
            <stop offset="100%" stopColor="#DC143C" />
          </linearGradient>
        </defs>
        
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth="8"
        />
        
        {/* Progress track */}
        <GaugeProgress
          cx={size / 2}
          cy={size / 2}
          r={radius}
          circumference={circumference}
          progress={getProgress()}
          status={status}
        />
      </GaugeTrack>
      
      {showSparkles && (
        <SparkleEffect size={size}>
          {sparkles.map(sparkle => (
            <div
              key={sparkle.id}
              className="sparkle"
              style={{
                left: `${sparkle.x}%`,
                top: `${sparkle.y}%`,
                animationDelay: `${sparkle.delay}s`
              }}
            />
          ))}
        </SparkleEffect>
      )}
      
      <GaugeCenter size={size}>
        <GaugeValue status={status} size={size}>
          {formatValue()}{unit}
        </GaugeValue>
        <GaugeLabel size={size}>
          {label}
        </GaugeLabel>
        <GaugeStatus status={status} size={size}>
          {getStatusText()}
        </GaugeStatus>
      </GaugeCenter>
    </GaugeContainer>
  );
};

export default PerformanceGauge;