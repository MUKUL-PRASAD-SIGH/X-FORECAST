import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import HolographicForecastChart3D from './HolographicForecastChart3D';
import AnimatedWeightEvolution from './AnimatedWeightEvolution';
import CyberpunkLoadingAnimation from './CyberpunkLoadingAnimation';

const DemoContainer = styled.div`
  padding: 20px;
  background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
  min-height: 100vh;
  color: #ffffff;
  font-family: 'Courier New', monospace;
`;

const DemoHeader = styled.div`
  text-align: center;
  margin-bottom: 30px;
`;

const DemoTitle = styled.h1`
  color: #00ffff;
  font-size: 2.5rem;
  margin-bottom: 10px;
  text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
  animation: titleGlow 3s ease-in-out infinite;
  
  @keyframes titleGlow {
    0%, 100% { text-shadow: 0 0 20px rgba(0, 255, 255, 0.5); }
    50% { text-shadow: 0 0 30px rgba(0, 255, 255, 0.8); }
  }
`;

const DemoSubtitle = styled.p`
  color: #ff1493;
  font-size: 1.2rem;
  margin-bottom: 20px;
`;

const VisualizationSection = styled(motion.div)`
  margin-bottom: 40px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(0, 255, 255, 0.2);
  border-radius: 12px;
  padding: 20px;
  backdrop-filter: blur(5px);
`;

const SectionTitle = styled.h2`
  color: #39ff14;
  font-size: 1.5rem;
  margin-bottom: 15px;
  text-shadow: 0 0 10px rgba(57, 255, 20, 0.5);
`;

const SectionDescription = styled.p`
  color: #ffffff;
  margin-bottom: 20px;
  opacity: 0.8;
  line-height: 1.6;
`;

const ControlPanel = styled.div`
  display: flex;
  gap: 15px;
  margin-bottom: 20px;
  flex-wrap: wrap;
`;

const ControlButton = styled.button<{ active?: boolean }>`
  background: ${props => props.active ? 'rgba(0, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.5)'};
  border: 1px solid ${props => props.active ? '#00ffff' : 'rgba(0, 255, 255, 0.3)'};
  color: ${props => props.active ? '#00ffff' : 'rgba(255, 255, 255, 0.7)'};
  padding: 8px 16px;
  border-radius: 6px;
  font-family: 'Courier New', monospace;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(4px);
  
  &:hover {
    border-color: #ff1493;
    color: #ff1493;
    box-shadow: 0 0 10px rgba(255, 20, 147, 0.3);
  }
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
`;

const StatCard = styled.div`
  background: rgba(0, 0, 0, 0.6);
  border: 1px solid rgba(255, 20, 147, 0.3);
  border-radius: 8px;
  padding: 15px;
  text-align: center;
`;

const StatValue = styled.div`
  color: #ff1493;
  font-size: 1.8rem;
  font-weight: bold;
  margin-bottom: 5px;
`;

const StatLabel = styled.div`
  color: #ffffff;
  font-size: 0.9rem;
  opacity: 0.8;
`;

// Generate mock data
const generateForecastData = () => {
  const data = [];
  const baseValue = 1000;
  
  // Historical data (last 12 months)
  for (let i = 0; i < 12; i++) {
    const date = new Date();
    date.setMonth(date.getMonth() - (12 - i));
    data.push({
      date: date.toISOString().split('T')[0],
      historical: baseValue + Math.sin(i * 0.5) * 200 + Math.random() * 100,
      ensemble: baseValue + Math.sin(i * 0.5) * 200 + Math.random() * 100
    });
  }
  
  // Forecast data (next 6 months)
  for (let i = 0; i < 6; i++) {
    const date = new Date();
    date.setMonth(date.getMonth() + i + 1);
    const baseForcast = baseValue + Math.sin((12 + i) * 0.5) * 200;
    
    data.push({
      date: date.toISOString().split('T')[0],
      arima: baseForcast + Math.random() * 150 - 75,
      ets: baseForcast + Math.random() * 150 - 75,
      xgboost: baseForcast + Math.random() * 150 - 75,
      lstm: baseForcast + Math.random() * 150 - 75,
      croston: baseForcast + Math.random() * 150 - 75,
      ensemble: baseForcast + Math.random() * 100 - 50,
      confidence_lower: baseForcast - 100 - Math.random() * 50,
      confidence_upper: baseForcast + 100 + Math.random() * 50
    });
  }
  
  return data;
};

const generateWeightEvolutionData = () => {
  const data = [];
  const models = ['arima', 'ets', 'xgboost', 'lstm', 'croston'] as const;
  
  for (let i = 0; i < 20; i++) {
    const timestamp = new Date(Date.now() - (20 - i) * 24 * 60 * 60 * 1000).toISOString();
    const weights: any = {};
    const performance: any = {};
    
    // Generate random weights that sum to 1
    const rawWeights = models.map(() => Math.random());
    const sum = rawWeights.reduce((a, b) => a + b, 0);
    
    models.forEach((model, index) => {
      weights[model] = rawWeights[index] / sum;
      performance[model] = 0.6 + Math.random() * 0.4; // Performance between 0.6 and 1.0
    });
    
    data.push({ timestamp, weights, performance });
  }
  
  return data;
};

const generateTrainingStatus = (isTraining: boolean) => {
  const models = ['arima', 'ets', 'xgboost', 'lstm', 'croston'];
  return models.map(model => ({
    model,
    status: isTraining 
      ? (Math.random() > 0.7 ? 'training' : Math.random() > 0.5 ? 'completed' : 'pending')
      : 'completed',
    progress: Math.random() * 100,
    eta: isTraining ? `${Math.floor(Math.random() * 5) + 1}m` : undefined,
    accuracy: Math.random() * 0.3 + 0.7
  })) as any;
};

export const Advanced3DVisualizationDemo: React.FC = () => {
  const [forecastData] = useState(generateForecastData());
  const [weightEvolutionData] = useState(generateWeightEvolutionData());
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState(generateTrainingStatus(false));
  
  // Demo controls
  const [showIndividualModels, setShowIndividualModels] = useState(true);
  const [showConfidenceIntervals, setShowConfidenceIntervals] = useState(true);
  const [enableParticleEffects, setEnableParticleEffects] = useState(true);
  const [showPerformanceIndicators, setShowPerformanceIndicators] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);
  
  // Simulate training process
  useEffect(() => {
    if (isTraining) {
      const interval = setInterval(() => {
        setTrainingProgress(prev => {
          const newProgress = prev + Math.random() * 5;
          if (newProgress >= 100) {
            setIsTraining(false);
            setTrainingStatus(generateTrainingStatus(false));
            return 100;
          }
          return newProgress;
        });
        
        setTrainingStatus(generateTrainingStatus(true));
      }, 500);
      
      return () => clearInterval(interval);
    }
  }, [isTraining]);
  
  const startTraining = () => {
    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingStatus(generateTrainingStatus(true));
  };
  
  return (
    <DemoContainer>
      <DemoHeader>
        <DemoTitle>Advanced 3D Forecast Visualizations</DemoTitle>
        <DemoSubtitle>Cyberpunk-styled ensemble forecasting with holographic effects</DemoSubtitle>
      </DemoHeader>
      
      <StatsGrid>
        <StatCard>
          <StatValue>{forecastData.filter(d => d.historical).length}</StatValue>
          <StatLabel>Historical Points</StatLabel>
        </StatCard>
        <StatCard>
          <StatValue>{forecastData.filter(d => !d.historical).length}</StatValue>
          <StatLabel>Forecast Points</StatLabel>
        </StatCard>
        <StatCard>
          <StatValue>{weightEvolutionData.length}</StatValue>
          <StatLabel>Weight Evolution Steps</StatLabel>
        </StatCard>
        <StatCard>
          <StatValue>5</StatValue>
          <StatLabel>Ensemble Models</StatLabel>
        </StatCard>
      </StatsGrid>
      
      {/* 3D Holographic Forecast Chart */}
      <VisualizationSection
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <SectionTitle>🔮 3D Holographic Forecast Chart</SectionTitle>
        <SectionDescription>
          Interactive 3D visualization of ensemble forecasts with holographic effects, 
          animated data points, and 3D probability clouds for confidence intervals.
        </SectionDescription>
        
        <ControlPanel>
          <ControlButton
            active={showIndividualModels}
            onClick={() => setShowIndividualModels(!showIndividualModels)}
          >
            Individual Models: {showIndividualModels ? 'ON' : 'OFF'}
          </ControlButton>
          <ControlButton
            active={showConfidenceIntervals}
            onClick={() => setShowConfidenceIntervals(!showConfidenceIntervals)}
          >
            Confidence Intervals: {showConfidenceIntervals ? 'ON' : 'OFF'}
          </ControlButton>
          <ControlButton
            active={enableParticleEffects}
            onClick={() => setEnableParticleEffects(!enableParticleEffects)}
          >
            Particle Effects: {enableParticleEffects ? 'ON' : 'OFF'}
          </ControlButton>
          <ControlButton
            active={autoRotate}
            onClick={() => setAutoRotate(!autoRotate)}
          >
            Auto Rotate: {autoRotate ? 'ON' : 'OFF'}
          </ControlButton>
        </ControlPanel>
        
        <HolographicForecastChart3D
          data={forecastData}
          showIndividualModels={showIndividualModels}
          showConfidenceIntervals={showConfidenceIntervals}
          enableParticleEffects={enableParticleEffects}
          autoRotate={autoRotate}
          width={800}
          height={600}
        />
      </VisualizationSection>
      
      {/* Animated Weight Evolution */}
      <VisualizationSection
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <SectionTitle>⚖️ Animated Weight Evolution</SectionTitle>
        <SectionDescription>
          Real-time visualization of model weight changes over time with particle effects, 
          performance indicators, and animated transitions showing how the ensemble adapts.
        </SectionDescription>
        
        <ControlPanel>
          <ControlButton
            active={showPerformanceIndicators}
            onClick={() => setShowPerformanceIndicators(!showPerformanceIndicators)}
          >
            Performance Rings: {showPerformanceIndicators ? 'ON' : 'OFF'}
          </ControlButton>
          <ControlButton
            active={enableParticleEffects}
            onClick={() => setEnableParticleEffects(!enableParticleEffects)}
          >
            Weight Particles: {enableParticleEffects ? 'ON' : 'OFF'}
          </ControlButton>
          <ControlButton
            active={autoRotate}
            onClick={() => setAutoRotate(!autoRotate)}
          >
            Auto Rotate: {autoRotate ? 'ON' : 'OFF'}
          </ControlButton>
        </ControlPanel>
        
        <AnimatedWeightEvolution
          data={weightEvolutionData}
          enableParticleEffects={enableParticleEffects}
          showPerformanceIndicators={showPerformanceIndicators}
          autoRotate={autoRotate}
          animationSpeed={1.5}
          width={800}
          height={600}
        />
      </VisualizationSection>
      
      {/* Cyberpunk Loading Animation */}
      <VisualizationSection
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <SectionTitle>🤖 Cyberpunk Training Animation</SectionTitle>
        <SectionDescription>
          Futuristic loading animation for model training with neural network visualization, 
          data flow particles, and real-time training progress indicators.
        </SectionDescription>
        
        <ControlPanel>
          <ControlButton onClick={startTraining} active={isTraining}>
            {isTraining ? 'Training...' : 'Start Training'}
          </ControlButton>
          <ControlButton
            active={enableParticleEffects}
            onClick={() => setEnableParticleEffects(!enableParticleEffects)}
          >
            Data Particles: {enableParticleEffects ? 'ON' : 'OFF'}
          </ControlButton>
        </ControlPanel>
        
        <CyberpunkLoadingAnimation
          modelStatuses={trainingStatus}
          overallProgress={trainingProgress}
          isTraining={isTraining}
          trainingMessage={isTraining ? "Training Ensemble Models..." : "Training Complete"}
          width={800}
          height={600}
          showParticles={enableParticleEffects}
          animationIntensity={1.2}
        />
      </VisualizationSection>
    </DemoContainer>
  );
};

export default Advanced3DVisualizationDemo;