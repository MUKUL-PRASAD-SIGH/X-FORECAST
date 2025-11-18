import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import HolographicForecastChart3D from './HolographicForecastChart3D';
import AnimatedWeightEvolution from './AnimatedWeightEvolution';
import CyberpunkLoadingAnimation from './CyberpunkLoadingAnimation';
import PerformanceMonitor from '../performance/PerformanceMonitor';

interface EnsembleData {
  historical: Array<{ date: string; value: number }>;
  forecasts: {
    arima: Array<{ date: string; value: number }>;
    ets: Array<{ date: string; value: number }>;
    xgboost: Array<{ date: string; value: number }>;
    lstm: Array<{ date: string; value: number }>;
    croston: Array<{ date: string; value: number }>;
    ensemble: Array<{ date: string; value: number }>;
  };
  confidenceIntervals: {
    lower: Array<{ date: string; value: number }>;
    upper: Array<{ date: string; value: number }>;
  };
  weights: Array<{
    timestamp: string;
    weights: {
      arima: number;
      ets: number;
      xgboost: number;
      lstm: number;
      croston: number;
    };
    performance: {
      arima: number;
      ets: number;
      xgboost: number;
      lstm: number;
      croston: number;
    };
  }>;
  trainingStatus?: {
    isTraining: boolean;
    progress: number;
    modelStatuses: Array<{
      model: string;
      status: 'pending' | 'training' | 'completed' | 'failed';
      progress: number;
      eta?: string;
      accuracy?: number;
    }>;
  };
}

interface EnsembleForecast3DVisualizationProps {
  data: EnsembleData;
  width?: number;
  height?: number;
  defaultView?: 'forecast' | 'weights' | 'training';
  showPerformanceMonitor?: boolean;
}

const VisualizationContainer = styled.div`
  background: linear-gradient(135deg, rgba(0, 0, 0, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 12px;
  padding: 20px;
  margin: 20px 0;
  backdrop-filter: blur(10px);
`;

const ViewSelector = styled.div`
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  justify-content: center;
`;

const ViewButton = styled.button<{ active: boolean }>`
  background: ${props => props.active ? 'rgba(0, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.5)'};
  border: 1px solid ${props => props.active ? '#00ffff' : 'rgba(0, 255, 255, 0.3)'};
  color: ${props => props.active ? '#00ffff' : 'rgba(255, 255, 255, 0.7)'};
  padding: 10px 20px;
  border-radius: 6px;
  font-family: 'Courier New', monospace;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(4px);
  
  &:hover {
    border-color: #ff1493;
    color: #ff1493;
    box-shadow: 0 0 10px rgba(255, 20, 147, 0.3);
  }
`;

const ViewTitle = styled.h3`
  color: #00ffff;
  text-align: center;
  margin-bottom: 15px;
  font-family: 'Courier New', monospace;
  font-size: 1.2rem;
  text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
`;

export const EnsembleForecast3DVisualization: React.FC<EnsembleForecast3DVisualizationProps> = ({
  data,
  width = 800,
  height = 600,
  defaultView = 'forecast',
  showPerformanceMonitor = true
}) => {
  const [currentView, setCurrentView] = useState<'forecast' | 'weights' | 'training'>(defaultView);
  
  // Transform data for 3D forecast chart
  const forecastData = React.useMemo(() => {
    const combined = [];
    
    // Add historical data
    data.historical.forEach(point => {
      combined.push({
        date: point.date,
        historical: point.value,
        ensemble: point.value
      });
    });
    
    // Add forecast data
    data.forecasts.ensemble.forEach((point, index) => {
      combined.push({
        date: point.date,
        arima: data.forecasts.arima[index]?.value,
        ets: data.forecasts.ets[index]?.value,
        xgboost: data.forecasts.xgboost[index]?.value,
        lstm: data.forecasts.lstm[index]?.value,
        croston: data.forecasts.croston[index]?.value,
        ensemble: point.value,
        confidence_lower: data.confidenceIntervals.lower[index]?.value,
        confidence_upper: data.confidenceIntervals.upper[index]?.value
      });
    });
    
    return combined;
  }, [data]);
  
  const renderCurrentView = () => {
    switch (currentView) {
      case 'forecast':
        return (
          <>
            <ViewTitle>🔮 3D Holographic Ensemble Forecast</ViewTitle>
            <HolographicForecastChart3D
              data={forecastData}
              showIndividualModels={true}
              showConfidenceIntervals={true}
              enableParticleEffects={true}
              autoRotate={false}
              width={width}
              height={height}
            />
          </>
        );
      
      case 'weights':
        return (
          <>
            <ViewTitle>⚖️ Model Weight Evolution</ViewTitle>
            <AnimatedWeightEvolution
              data={data.weights}
              enableParticleEffects={true}
              showPerformanceIndicators={true}
              autoRotate={false}
              animationSpeed={1}
              width={width}
              height={height}
            />
          </>
        );
      
      case 'training':
        return (
          <>
            <ViewTitle>🤖 Model Training Status</ViewTitle>
            <CyberpunkLoadingAnimation
              modelStatuses={data.trainingStatus?.modelStatuses || []}
              overallProgress={data.trainingStatus?.progress || 100}
              isTraining={data.trainingStatus?.isTraining || false}
              trainingMessage={
                data.trainingStatus?.isTraining 
                  ? "Training Ensemble Models..." 
                  : "All Models Trained Successfully"
              }
              width={width}
              height={height}
              showParticles={true}
              animationIntensity={1}
            />
          </>
        );
      
      default:
        return null;
    }
  };
  
  return (
    <>
      <VisualizationContainer>
        <ViewSelector>
          <ViewButton
            active={currentView === 'forecast'}
            onClick={() => setCurrentView('forecast')}
          >
            3D Forecast
          </ViewButton>
          <ViewButton
            active={currentView === 'weights'}
            onClick={() => setCurrentView('weights')}
          >
            Weight Evolution
          </ViewButton>
          <ViewButton
            active={currentView === 'training'}
            onClick={() => setCurrentView('training')}
          >
            Training Status
          </ViewButton>
        </ViewSelector>
        
        <motion.div
          key={currentView}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          {renderCurrentView()}
        </motion.div>
      </VisualizationContainer>
      
      {/* Performance Monitor */}
      {showPerformanceMonitor && (
        <PerformanceMonitor
          position="top-right"
          minimized={false}
          showControls={true}
        />
      )}
    </>
  );
};

export default EnsembleForecast3DVisualization;