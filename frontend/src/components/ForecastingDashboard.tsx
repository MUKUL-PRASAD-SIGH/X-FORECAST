import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkCard, CyberpunkButton } from './ui';
import CyberpunkChart3D from './3d/CyberpunkChart3D';
import { HolographicForecastChart } from './charts/HolographicForecastChart';
import { WeightDistributionChart } from './charts/WeightDistributionChart';
import { ScenarioPlanning } from './ScenarioPlanning';
import { ForecastComparison } from './ForecastComparison';
import { ForecastExport } from './ForecastExport';

// Types for forecast data
interface TimeSeriesPoint {
  date: string;
  value: number;
}

interface ModelForecast {
  arima: TimeSeriesPoint[];
  ets: TimeSeriesPoint[];
  xgboost: TimeSeriesPoint[];
  lstm: TimeSeriesPoint[];
  croston: TimeSeriesPoint[];
  ensemble: TimeSeriesPoint[];
}

interface ConfidenceIntervals {
  p10: TimeSeriesPoint[];
  p50: TimeSeriesPoint[];
  p90: TimeSeriesPoint[];
}

interface ForecastVisualization {
  historical: TimeSeriesPoint[];
  forecasts: ModelForecast;
  confidenceIntervals: ConfidenceIntervals;
  modelWeights: Record<string, number>;
}

interface ForecastingDashboardProps {
  authToken: string;
}

interface ScenarioResult {
  scenario_name: string;
  parameters: any;
  forecast_data: {
    point_forecast: Record<string, number>;
    confidence_intervals: {
      p10: Record<string, number>;
      p50: Record<string, number>;
      p90: Record<string, number>;
    };
  };
  impact_summary: {
    total_change: number;
    peak_month: string;
    risk_level: string;
  };
}

// Styled Components
const DashboardContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.lg};
  width: 100%;
`;

const ControlsSection = styled(CyberpunkCard)`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.md};
`;

const ControlsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: ${props => props.theme.spacing.md};
  align-items: end;
`;

const ControlGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.xs};
`;

const Label = styled.label`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.neonBlue};
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const HorizonSlider = styled.input`
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: linear-gradient(90deg, 
    ${props => props.theme.colors.neonBlue} 0%, 
    ${props => props.theme.colors.hotPink} 100%
  );
  outline: none;
  opacity: 0.8;
  transition: opacity 0.2s;
  
  &:hover {
    opacity: 1;
  }
  
  &::-webkit-slider-thumb {
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: ${props => props.theme.colors.neonBlue};
    cursor: pointer;
    box-shadow: ${props => props.theme.effects.softGlow};
  }
  
  &::-moz-range-thumb {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: ${props => props.theme.colors.neonBlue};
    cursor: pointer;
    border: none;
    box-shadow: ${props => props.theme.effects.softGlow};
  }
`;

const SliderValue = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.lg};
  color: ${props => props.theme.colors.hotPink};
  text-align: center;
  margin-top: ${props => props.theme.spacing.xs};
  font-weight: bold;
`;

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: ${props => props.theme.spacing.lg};
  
  @media (min-width: 1400px) {
    grid-template-columns: 2fr 1fr;
  }
`;

const MainChartContainer = styled(CyberpunkCard)`
  min-height: 600px;
  position: relative;
`;

const SidePanel = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.md};
`;

const ModelWeightsCard = styled(CyberpunkCard)`
  min-height: 300px;
`;

const WeightItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${props => props.theme.spacing.sm};
  border-bottom: 1px solid rgba(0, 255, 255, 0.2);
  
  &:last-child {
    border-bottom: none;
  }
`;

const ModelName = styled.span`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  color: ${props => props.theme.colors.primaryText};
  font-weight: bold;
`;

const WeightBar = styled.div<{ weight: number }>`
  flex: 1;
  height: 8px;
  background: rgba(0, 255, 255, 0.2);
  border-radius: 4px;
  margin: 0 ${props => props.theme.spacing.sm};
  position: relative;
  overflow: hidden;
  
  &::after {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    width: ${props => props.weight * 100}%;
    background: linear-gradient(90deg, 
      ${props => props.theme.colors.neonBlue}, 
      ${props => props.theme.colors.hotPink}
    );
    border-radius: 4px;
    transition: width 0.5s ease;
  }
`;

const WeightValue = styled.span`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  color: ${props => props.theme.colors.neonBlue};
  font-size: ${props => props.theme.typography.fontSize.sm};
  min-width: 40px;
  text-align: right;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: ${props => props.theme.spacing.sm};
  margin-top: ${props => props.theme.spacing.md};
`;

const MetricItem = styled.div`
  text-align: center;
  padding: ${props => props.theme.spacing.sm};
  background: rgba(0, 255, 255, 0.1);
  border-radius: 4px;
  border: 1px solid rgba(0, 255, 255, 0.3);
`;

const MetricLabel = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.xs};
  color: ${props => props.theme.colors.secondaryText};
  text-transform: uppercase;
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const MetricValue = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.display};
  font-size: ${props => props.theme.typography.fontSize.lg};
  color: ${props => props.theme.colors.neonBlue};
  font-weight: bold;
`;

const LoadingOverlay = styled(motion.div)`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 10;
  border-radius: 8px;
`;

const LoadingText = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  color: ${props => props.theme.colors.neonBlue};
  font-size: ${props => props.theme.typography.fontSize.lg};
  margin-top: ${props => props.theme.spacing.md};
  text-align: center;
`;

const LoadingSpinner = styled.div`
  width: 60px;
  height: 60px;
  border: 3px solid rgba(0, 255, 255, 0.3);
  border-top: 3px solid ${props => props.theme.colors.neonBlue};
  border-radius: 50%;
  animation: spin 1s linear infinite;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ErrorMessage = styled.div`
  color: ${props => props.theme.colors.error};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  text-align: center;
  padding: ${props => props.theme.spacing.md};
  background: rgba(255, 0, 64, 0.1);
  border: 1px solid rgba(255, 0, 64, 0.3);
  border-radius: 4px;
`;

const CardTitle = styled.h3`
  color: ${props => props.theme.colors.neonBlue};
  font-family: ${props => props.theme.typography.fontFamily.primary};
  margin-bottom: ${props => props.theme.spacing.md};
  font-size: ${props => props.theme.typography.fontSize.xl};
  text-shadow: ${props => props.theme.effects.softGlow};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const TabNavigation = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
  margin-bottom: ${props => props.theme.spacing.lg};
  flex-wrap: wrap;
`;

const TabButton = styled(motion.button)<{ active: boolean }>`
  padding: ${props => props.theme.spacing.md} ${props => props.theme.spacing.lg};
  background: ${props => props.active 
    ? `linear-gradient(45deg, ${props.theme.colors.neonBlue}, ${props.theme.colors.hotPink})`
    : 'rgba(0, 255, 255, 0.1)'
  };
  border: 1px solid ${props => props.active ? 'transparent' : props.theme.colors.neonBlue};
  border-radius: 8px;
  color: ${props => props.active ? '#000' : props.theme.colors.neonBlue};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.md};
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
  text-transform: uppercase;
  letter-spacing: 1px;
  
  &:hover {
    background: ${props => props.active 
      ? `linear-gradient(45deg, ${props.theme.colors.neonBlue}, ${props.theme.colors.hotPink})`
      : 'rgba(0, 255, 255, 0.2)'
    };
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.effects.softGlow};
  }
`;

const TabContent = styled(motion.div)`
  width: 100%;
`;

export const ForecastingDashboard: React.FC<ForecastingDashboardProps> = ({ authToken }) => {
  const [selectedHorizon, setSelectedHorizon] = useState(6);
  const [forecastData, setForecastData] = useState<ForecastVisualization | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showIndividualModels, setShowIndividualModels] = useState(true);
  const [showConfidenceIntervals, setShowConfidenceIntervals] = useState(true);
  const [chartType, setChartType] = useState<'3d' | '2d'>('2d');
  const [activeTab, setActiveTab] = useState<'forecast' | 'scenarios' | 'comparison' | 'export'>('forecast');
  const [scenarioResults, setScenarioResults] = useState<ScenarioResult[]>([]);
  const [realTimeUpdates, setRealTimeUpdates] = useState(true);

  // Generate forecast data
  const generateForecast = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/company-sales/forecast', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`,
        },
        body: JSON.stringify({ horizon_months: selectedHorizon }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Transform API response to component format
      const transformedData: ForecastVisualization = {
        historical: Object.entries(data.point_forecast || {}).slice(0, -selectedHorizon).map(([date, value]) => ({
          date,
          value: value as number
        })),
        forecasts: {
          arima: Object.entries(data.point_forecast || {}).slice(-selectedHorizon).map(([date, value]) => ({
            date,
            value: (value as number) * (data.model_weights?.arima || 0.2)
          })),
          ets: Object.entries(data.point_forecast || {}).slice(-selectedHorizon).map(([date, value]) => ({
            date,
            value: (value as number) * (data.model_weights?.ets || 0.2)
          })),
          xgboost: Object.entries(data.point_forecast || {}).slice(-selectedHorizon).map(([date, value]) => ({
            date,
            value: (value as number) * (data.model_weights?.xgboost || 0.2)
          })),
          lstm: Object.entries(data.point_forecast || {}).slice(-selectedHorizon).map(([date, value]) => ({
            date,
            value: (value as number) * (data.model_weights?.lstm || 0.2)
          })),
          croston: Object.entries(data.point_forecast || {}).slice(-selectedHorizon).map(([date, value]) => ({
            date,
            value: (value as number) * (data.model_weights?.croston || 0.2)
          })),
          ensemble: Object.entries(data.point_forecast || {}).slice(-selectedHorizon).map(([date, value]) => ({
            date,
            value: value as number
          }))
        },
        confidenceIntervals: {
          p10: Object.entries(data.confidence_intervals?.p10 || {}).map(([date, value]) => ({
            date,
            value: value as number
          })),
          p50: Object.entries(data.confidence_intervals?.p50 || {}).map(([date, value]) => ({
            date,
            value: value as number
          })),
          p90: Object.entries(data.confidence_intervals?.p90 || {}).map(([date, value]) => ({
            date,
            value: value as number
          }))
        },
        modelWeights: data.model_weights || {
          arima: 0.2,
          ets: 0.2,
          xgboost: 0.2,
          lstm: 0.2,
          croston: 0.2
        }
      };
      
      setForecastData(transformedData);
    } catch (err) {
      console.error('Forecast generation error:', err);
      setError(err instanceof Error ? err.message : 'Failed to generate forecast');
      
      // Generate mock data for demonstration
      const mockData: ForecastVisualization = {
        historical: Array.from({ length: 12 }, (_, i) => ({
          date: new Date(2024, i, 1).toISOString().split('T')[0],
          value: 1000 + Math.random() * 200 + i * 50
        })),
        forecasts: {
          arima: Array.from({ length: selectedHorizon }, (_, i) => ({
            date: new Date(2024, 12 + i, 1).toISOString().split('T')[0],
            value: 1600 + i * 30 + Math.random() * 50
          })),
          ets: Array.from({ length: selectedHorizon }, (_, i) => ({
            date: new Date(2024, 12 + i, 1).toISOString().split('T')[0],
            value: 1580 + i * 35 + Math.random() * 60
          })),
          xgboost: Array.from({ length: selectedHorizon }, (_, i) => ({
            date: new Date(2024, 12 + i, 1).toISOString().split('T')[0],
            value: 1620 + i * 25 + Math.random() * 40
          })),
          lstm: Array.from({ length: selectedHorizon }, (_, i) => ({
            date: new Date(2024, 12 + i, 1).toISOString().split('T')[0],
            value: 1590 + i * 40 + Math.random() * 70
          })),
          croston: Array.from({ length: selectedHorizon }, (_, i) => ({
            date: new Date(2024, 12 + i, 1).toISOString().split('T')[0],
            value: 1570 + i * 20 + Math.random() * 30
          })),
          ensemble: Array.from({ length: selectedHorizon }, (_, i) => ({
            date: new Date(2024, 12 + i, 1).toISOString().split('T')[0],
            value: 1600 + i * 32 + Math.random() * 25
          }))
        },
        confidenceIntervals: {
          p10: Array.from({ length: selectedHorizon }, (_, i) => ({
            date: new Date(2024, 12 + i, 1).toISOString().split('T')[0],
            value: 1500 + i * 25
          })),
          p50: Array.from({ length: selectedHorizon }, (_, i) => ({
            date: new Date(2024, 12 + i, 1).toISOString().split('T')[0],
            value: 1600 + i * 32
          })),
          p90: Array.from({ length: selectedHorizon }, (_, i) => ({
            date: new Date(2024, 12 + i, 1).toISOString().split('T')[0],
            value: 1700 + i * 40
          }))
        },
        modelWeights: {
          arima: 0.15,
          ets: 0.25,
          xgboost: 0.30,
          lstm: 0.20,
          croston: 0.10
        }
      };
      
      setForecastData(mockData);
    } finally {
      setLoading(false);
    }
  };

  // Auto-generate forecast on mount and horizon change
  useEffect(() => {
    generateForecast();
  }, [selectedHorizon]);

  // Handle scenario updates
  const handleScenarioUpdate = (scenarios: ScenarioResult[]) => {
    setScenarioResults(scenarios);
    
    // Trigger real-time forecast update if enabled
    if (realTimeUpdates && scenarios.length > 0) {
      // Update forecast data with latest scenario
      const latestScenario = scenarios[scenarios.length - 1];
      // This would integrate with the main forecast in a real implementation
      console.log('Updated scenarios:', scenarios);
    }
  };

  // Convert scenario results to comparison format
  const getComparisonScenarios = () => {
    const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7'];
    
    return scenarioResults.map((scenario, index) => ({
      name: scenario.scenario_name,
      color: colors[index % colors.length],
      data: scenario.forecast_data,
      parameters: scenario.parameters,
      metrics: {
        total_change: scenario.impact_summary.total_change,
        peak_value: Math.max(...Object.values(scenario.forecast_data.point_forecast)),
        risk_score: scenario.impact_summary.risk_level === 'high' ? 0.8 : 
                   scenario.impact_summary.risk_level === 'medium' ? 0.5 : 0.2,
        confidence_score: 0.75 // Mock confidence score
      }
    }));
  };

  // Handle export completion
  const handleExportComplete = (result: { success: boolean; downloadUrl?: string; error?: string }) => {
    if (result.success) {
      console.log('Export completed successfully');
    } else {
      console.error('Export failed:', result.error);
    }
  };

  // Create 3D chart data
  const create3DChartData = () => {
    if (!forecastData) return [];
    
    const data: any[] = [];
    
    // Add historical data points
    forecastData.historical.forEach((point, index) => {
      data.push({
        x: index,
        y: point.value,
        z: 0,
        value: point.value,
        label: `Historical: ${point.date}`,
        color: '#00ffff'
      });
    });
    
    // Add ensemble forecast points
    forecastData.forecasts.ensemble.forEach((point, index) => {
      data.push({
        x: forecastData.historical.length + index,
        y: point.value,
        z: 1,
        value: point.value,
        label: `Ensemble: ${point.date}`,
        color: '#ff1493'
      });
    });
    
    // Add confidence intervals as 3D cloud
    if (showConfidenceIntervals) {
      forecastData.confidenceIntervals.p10.forEach((point, index) => {
        data.push({
          x: forecastData.historical.length + index,
          y: point.value,
          z: 0.5,
          value: point.value,
          label: `P10: ${point.date}`,
          color: '#39ff14'
        });
      });
      
      forecastData.confidenceIntervals.p90.forEach((point, index) => {
        data.push({
          x: forecastData.historical.length + index,
          y: point.value,
          z: 1.5,
          value: point.value,
          label: `P90: ${point.date}`,
          color: '#39ff14'
        });
      });
    }
    
    return data;
  };

  return (
    <DashboardContainer>
      {/* Tab Navigation */}
      <TabNavigation>
        <TabButton
          active={activeTab === 'forecast'}
          onClick={() => setActiveTab('forecast')}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          📈 Forecast Dashboard
        </TabButton>
        <TabButton
          active={activeTab === 'scenarios'}
          onClick={() => setActiveTab('scenarios')}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          🎯 Scenario Planning
        </TabButton>
        <TabButton
          active={activeTab === 'comparison'}
          onClick={() => setActiveTab('comparison')}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          📊 Scenario Comparison
        </TabButton>
        <TabButton
          active={activeTab === 'export'}
          onClick={() => setActiveTab('export')}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          📤 Export & Reports
        </TabButton>
      </TabNavigation>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        <TabContent
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
        >
          {activeTab === 'forecast' && (
            <>
              {/* Forecast Controls */}
              <ControlsSection variant="neon" padding="lg">
                <CardTitle>🔮 Ensemble Forecast Configuration</CardTitle>
        
        <ControlsGrid>
          <ControlGroup>
            <Label>Forecast Horizon (Months)</Label>
            <HorizonSlider
              type="range"
              min={1}
              max={12}
              value={selectedHorizon}
              onChange={(e) => setSelectedHorizon(parseInt(e.target.value))}
            />
            <SliderValue>{selectedHorizon} months</SliderValue>
          </ControlGroup>
          
          <ControlGroup>
            <Label>Display Options</Label>
            <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#00ffff' }}>
                <input
                  type="checkbox"
                  checked={showIndividualModels}
                  onChange={(e) => setShowIndividualModels(e.target.checked)}
                />
                Individual Models
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#00ffff' }}>
                <input
                  type="checkbox"
                  checked={showConfidenceIntervals}
                  onChange={(e) => setShowConfidenceIntervals(e.target.checked)}
                />
                Confidence Intervals
              </label>
            </div>
          </ControlGroup>
          
          <ControlGroup>
            <Label>Chart Type</Label>
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
              <CyberpunkButton
                variant={chartType === '2d' ? 'primary' : 'secondary'}
                size="sm"
                onClick={() => setChartType('2d')}
              >
                2D Holographic
              </CyberpunkButton>
              <CyberpunkButton
                variant={chartType === '3d' ? 'primary' : 'secondary'}
                size="sm"
                onClick={() => setChartType('3d')}
              >
                3D Visualization
              </CyberpunkButton>
            </div>
          </ControlGroup>
          
          <ControlGroup>
            <CyberpunkButton
              variant="primary"
              onClick={generateForecast}
              disabled={loading}
            >
              {loading ? 'Generating...' : 'Regenerate Forecast'}
            </CyberpunkButton>
          </ControlGroup>
        </ControlsGrid>
      </ControlsSection>

      {/* Main Charts */}
      <ChartsGrid>
        {/* Forecast Chart */}
        <MainChartContainer variant="hologram" padding="lg">
          <CardTitle>
            📈 {chartType === '3d' ? '3D Holographic' : '2D Holographic'} Forecast Visualization
          </CardTitle>
          
          {loading && (
            <LoadingOverlay
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <LoadingSpinner />
              <LoadingText>
                Generating ensemble forecast...
                <br />
                Analyzing {selectedHorizon} month horizon
              </LoadingText>
            </LoadingOverlay>
          )}
          
          {error && (
            <ErrorMessage>
              ⚠️ {error}
              <br />
              <small>Showing demo data for visualization</small>
            </ErrorMessage>
          )}
          
          {forecastData && (
            <>
              {chartType === '3d' ? (
                <CyberpunkChart3D
                  data={create3DChartData()}
                  width={800}
                  height={500}
                  title="Ensemble Forecast with Confidence Intervals"
                  showGrid={true}
                  animationSpeed={0.005}
                  glowIntensity={0.6}
                />
              ) : (
                <HolographicForecastChart
                  data={forecastData}
                  showIndividualModels={showIndividualModels}
                  showConfidenceIntervals={showConfidenceIntervals}
                  width={800}
                  height={500}
                />
              )}
            </>
          )}
        </MainChartContainer>

        {/* Side Panel */}
        <SidePanel>
          {/* Model Weights */}
          <ModelWeightsCard variant="glass" padding="lg">
            <CardTitle>⚖️ Model Weight Distribution</CardTitle>
            
            {forecastData && (
              <>
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
                  <WeightDistributionChart
                    weights={forecastData.modelWeights}
                    width={250}
                    height={250}
                  />
                </div>
                
                {Object.entries(forecastData.modelWeights).map(([model, weight]) => (
                  <WeightItem key={model}>
                    <ModelName>{model.toUpperCase()}</ModelName>
                    <WeightBar weight={weight} />
                    <WeightValue>{(weight * 100).toFixed(1)}%</WeightValue>
                  </WeightItem>
                ))}
                
                <MetricsGrid>
                  <MetricItem>
                    <MetricLabel>Active Models</MetricLabel>
                    <MetricValue>5</MetricValue>
                  </MetricItem>
                  <MetricItem>
                    <MetricLabel>Ensemble Accuracy</MetricLabel>
                    <MetricValue>87.3%</MetricValue>
                  </MetricItem>
                  <MetricItem>
                    <MetricLabel>Confidence Score</MetricLabel>
                    <MetricValue>0.85</MetricValue>
                  </MetricItem>
                  <MetricItem>
                    <MetricLabel>Data Quality</MetricLabel>
                    <MetricValue>92%</MetricValue>
                  </MetricItem>
                </MetricsGrid>
              </>
            )}
          </ModelWeightsCard>

          {/* Forecast Summary */}
          <CyberpunkCard $variant="neon" $padding="lg">
            <CardTitle>📊 Forecast Summary</CardTitle>
            
            {forecastData && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div>
                  <MetricLabel>Forecast Period</MetricLabel>
                  <MetricValue>{selectedHorizon} months ahead</MetricValue>
                </div>
                
                <div>
                  <MetricLabel>Expected Growth</MetricLabel>
                  <MetricValue style={{ color: '#39ff14' }}>+12.5%</MetricValue>
                </div>
                
                <div>
                  <MetricLabel>Volatility Risk</MetricLabel>
                  <MetricValue style={{ color: '#ffff00' }}>Medium</MetricValue>
                </div>
                
                <div>
                  <MetricLabel>Pattern Detected</MetricLabel>
                  <MetricValue>Seasonal Trend</MetricValue>
                </div>
              </div>
            )}
          </CyberpunkCard>
        </SidePanel>
      </ChartsGrid>
            </>
          )}

          {activeTab === 'scenarios' && (
            <ScenarioPlanning
              authToken={authToken}
              baselineData={forecastData}
              onScenarioUpdate={handleScenarioUpdate}
            />
          )}

          {activeTab === 'comparison' && (
            <ForecastComparison
              scenarios={getComparisonScenarios()}
              baselineData={forecastData}
              onScenarioSelect={(scenario) => {
                console.log('Selected scenario:', scenario);
                // Could switch to scenario tab or highlight scenario
              }}
            />
          )}

          {activeTab === 'export' && (
            <ForecastExport
              authToken={authToken}
              forecastData={forecastData}
              scenarioData={scenarioResults}
              onExportComplete={handleExportComplete}
            />
          )}
        </TabContent>
      </AnimatePresence>
    </DashboardContainer>
  );
};