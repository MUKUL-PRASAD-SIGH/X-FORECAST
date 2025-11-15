import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { CyberpunkCard, CyberpunkButton } from './ui';

interface ScenarioParameters {
  seasonality_factor: number;
  trend_adjustment: number;
  volatility_modifier: number;
  external_factors: number;
  market_conditions: 'optimistic' | 'neutral' | 'pessimistic';
}

interface ScenarioResult {
  scenario_name: string;
  parameters: ScenarioParameters;
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

interface ScenarioPlanningProps {
  authToken: string;
  baselineData?: any;
  onScenarioUpdate?: (scenarios: ScenarioResult[]) => void;
}

const ScenarioContainer = styled(CyberpunkCard)`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.lg};
`;

const ParameterGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: ${props => props.theme.spacing.md};
`;

const ParameterCard = styled(CyberpunkCard)`
  padding: ${props => props.theme.spacing.md};
`;

const ParameterLabel = styled.label`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.neonBlue};
  text-transform: uppercase;
  letter-spacing: 1px;
  display: block;
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const ParameterSlider = styled.input`
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
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: ${props => props.theme.colors.neonBlue};
    cursor: pointer;
    box-shadow: ${props => props.theme.effects.softGlow};
  }
  
  &::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: ${props => props.theme.colors.neonBlue};
    cursor: pointer;
    border: none;
    box-shadow: ${props => props.theme.effects.softGlow};
  }
`;

const ParameterValue = styled.div`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.md};
  color: ${props => props.theme.colors.hotPink};
  text-align: center;
  margin-top: ${props => props.theme.spacing.xs};
  font-weight: bold;
`;

const MarketConditionSelect = styled.select`
  width: 100%;
  padding: ${props => props.theme.spacing.sm};
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid ${props => props.theme.colors.neonBlue};
  border-radius: 4px;
  color: ${props => props.theme.colors.primaryText};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.hotPink};
    box-shadow: ${props => props.theme.effects.softGlow};
  }
  
  option {
    background: rgba(0, 0, 0, 0.9);
    color: ${props => props.theme.colors.primaryText};
  }
`;

const ScenarioTabs = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
  margin-bottom: ${props => props.theme.spacing.md};
  flex-wrap: wrap;
`;

const ScenarioTab = styled(motion.button)<{ active: boolean }>`
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  background: ${props => props.active 
    ? `linear-gradient(45deg, ${props.theme.colors.neonBlue}, ${props.theme.colors.hotPink})`
    : 'rgba(0, 255, 255, 0.1)'
  };
  border: 1px solid ${props => props.active ? 'transparent' : props.theme.colors.neonBlue};
  border-radius: 4px;
  color: ${props => props.active ? '#000' : props.theme.colors.neonBlue};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: ${props => props.active 
      ? `linear-gradient(45deg, ${props.theme.colors.neonBlue}, ${props.theme.colors.hotPink})`
      : 'rgba(0, 255, 255, 0.2)'
    };
    transform: translateY(-2px);
  }
`;

const ActionButtons = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  justify-content: center;
  flex-wrap: wrap;
`;

const ImpactSummary = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: ${props => props.theme.spacing.sm};
  margin-top: ${props => props.theme.spacing.md};
`;

const ImpactMetric = styled.div`
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

export const ScenarioPlanning: React.FC<ScenarioPlanningProps> = ({ 
  authToken, 
  baselineData,
  onScenarioUpdate 
}) => {
  const [activeScenario, setActiveScenario] = useState<string>('scenario_1');
  const [scenarios, setScenarios] = useState<Record<string, ScenarioParameters>>({
    scenario_1: {
      seasonality_factor: 1.0,
      trend_adjustment: 1.0,
      volatility_modifier: 1.0,
      external_factors: 1.0,
      market_conditions: 'neutral'
    },
    scenario_2: {
      seasonality_factor: 1.2,
      trend_adjustment: 1.1,
      volatility_modifier: 0.8,
      external_factors: 1.1,
      market_conditions: 'optimistic'
    },
    scenario_3: {
      seasonality_factor: 0.8,
      trend_adjustment: 0.9,
      volatility_modifier: 1.3,
      external_factors: 0.9,
      market_conditions: 'pessimistic'
    }
  });
  
  const [scenarioResults, setScenarioResults] = useState<ScenarioResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [realTimeUpdates, setRealTimeUpdates] = useState(true);

  // Update scenario parameters
  const updateParameter = (parameter: keyof ScenarioParameters, value: any) => {
    setScenarios(prev => ({
      ...prev,
      [activeScenario]: {
        ...prev[activeScenario],
        [parameter]: value
      }
    }));
    
    // Trigger real-time update if enabled
    if (realTimeUpdates) {
      generateScenarioForecast(activeScenario, {
        ...scenarios[activeScenario],
        [parameter]: value
      });
    }
  };

  // Generate forecast for specific scenario
  const generateScenarioForecast = async (scenarioName: string, parameters: ScenarioParameters) => {
    try {
      const response = await fetch('/api/company-sales/scenario-forecast', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          scenario_name: scenarioName,
          parameters: parameters,
          horizon_months: 6
        }),
      });

      if (response.ok) {
        const result = await response.json();
        
        // Update scenario results
        setScenarioResults(prev => {
          const updated = prev.filter(s => s.scenario_name !== scenarioName);
          return [...updated, result];
        });
        
        // Notify parent component
        if (onScenarioUpdate) {
          const updatedResults = scenarioResults.filter(s => s.scenario_name !== scenarioName);
          onScenarioUpdate([...updatedResults, result]);
        }
      } else {
        console.warn('Scenario forecast failed, using mock data');
        generateMockScenarioResult(scenarioName, parameters);
      }
    } catch (error) {
      console.warn('Scenario forecast error, using mock data:', error);
      generateMockScenarioResult(scenarioName, parameters);
    }
  };

  // Generate mock scenario result for demonstration
  const generateMockScenarioResult = (scenarioName: string, parameters: ScenarioParameters) => {
    const baseValue = 1000;
    const months = 6;
    
    const mockResult: ScenarioResult = {
      scenario_name: scenarioName,
      parameters: parameters,
      forecast_data: {
        point_forecast: {},
        confidence_intervals: {
          p10: {},
          p50: {},
          p90: {}
        }
      },
      impact_summary: {
        total_change: (parameters.trend_adjustment - 1) * 100,
        peak_month: new Date(2024, 5, 1).toISOString().split('T')[0],
        risk_level: parameters.volatility_modifier > 1.2 ? 'high' : 
                   parameters.volatility_modifier < 0.8 ? 'low' : 'medium'
      }
    };
    
    // Generate mock forecast data
    for (let i = 0; i < months; i++) {
      const date = new Date(2024, i, 1).toISOString().split('T')[0];
      const trendEffect = Math.pow(parameters.trend_adjustment, i / 12);
      const seasonalEffect = 1 + 0.1 * Math.sin(i * Math.PI / 6) * parameters.seasonality_factor;
      const volatilityEffect = 1 + (Math.random() - 0.5) * 0.2 * parameters.volatility_modifier;
      const externalEffect = parameters.external_factors;
      
      let marketMultiplier = 1;
      if (parameters.market_conditions === 'optimistic') marketMultiplier = 1.1;
      if (parameters.market_conditions === 'pessimistic') marketMultiplier = 0.9;
      
      const forecastValue = baseValue * trendEffect * seasonalEffect * volatilityEffect * externalEffect * marketMultiplier;
      
      mockResult.forecast_data.point_forecast[date] = forecastValue;
      mockResult.forecast_data.confidence_intervals.p10[date] = forecastValue * 0.85;
      mockResult.forecast_data.confidence_intervals.p50[date] = forecastValue;
      mockResult.forecast_data.confidence_intervals.p90[date] = forecastValue * 1.15;
    }
    
    // Update scenario results
    setScenarioResults(prev => {
      const updated = prev.filter(s => s.scenario_name !== scenarioName);
      return [...updated, mockResult];
    });
    
    // Notify parent component
    if (onScenarioUpdate) {
      const updatedResults = scenarioResults.filter(s => s.scenario_name !== scenarioName);
      onScenarioUpdate([...updatedResults, mockResult]);
    }
  };

  // Generate all scenarios
  const generateAllScenarios = async () => {
    setLoading(true);
    
    for (const [scenarioName, parameters] of Object.entries(scenarios)) {
      await generateScenarioForecast(scenarioName, parameters);
    }
    
    setLoading(false);
  };

  // Reset to baseline
  const resetToBaseline = () => {
    setScenarios({
      scenario_1: {
        seasonality_factor: 1.0,
        trend_adjustment: 1.0,
        volatility_modifier: 1.0,
        external_factors: 1.0,
        market_conditions: 'neutral'
      },
      scenario_2: {
        seasonality_factor: 1.2,
        trend_adjustment: 1.1,
        volatility_modifier: 0.8,
        external_factors: 1.1,
        market_conditions: 'optimistic'
      },
      scenario_3: {
        seasonality_factor: 0.8,
        trend_adjustment: 0.9,
        volatility_modifier: 1.3,
        external_factors: 0.9,
        market_conditions: 'pessimistic'
      }
    });
    setScenarioResults([]);
  };

  // Get current scenario result
  const currentScenarioResult = scenarioResults.find(s => s.scenario_name === activeScenario);

  return (
    <ScenarioContainer variant="hologram" padding="lg">
      <CardTitle>🎯 Scenario Planning & Parameter Adjustment</CardTitle>
      
      {/* Scenario Tabs */}
      <ScenarioTabs>
        {Object.keys(scenarios).map((scenarioName) => (
          <ScenarioTab
            key={scenarioName}
            active={activeScenario === scenarioName}
            onClick={() => setActiveScenario(scenarioName)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {scenarioName.replace('_', ' ').toUpperCase()}
          </ScenarioTab>
        ))}
      </ScenarioTabs>

      {/* Parameter Controls */}
      <ParameterGrid>
        <ParameterCard variant="glass">
          <ParameterLabel>Seasonality Factor</ParameterLabel>
          <ParameterSlider
            type="range"
            min={0.5}
            max={2.0}
            step={0.1}
            value={scenarios[activeScenario].seasonality_factor}
            onChange={(e) => updateParameter('seasonality_factor', parseFloat(e.target.value))}
          />
          <ParameterValue>
            {scenarios[activeScenario].seasonality_factor.toFixed(1)}x
          </ParameterValue>
        </ParameterCard>

        <ParameterCard variant="glass">
          <ParameterLabel>Trend Adjustment</ParameterLabel>
          <ParameterSlider
            type="range"
            min={0.5}
            max={2.0}
            step={0.1}
            value={scenarios[activeScenario].trend_adjustment}
            onChange={(e) => updateParameter('trend_adjustment', parseFloat(e.target.value))}
          />
          <ParameterValue>
            {scenarios[activeScenario].trend_adjustment.toFixed(1)}x
          </ParameterValue>
        </ParameterCard>

        <ParameterCard variant="glass">
          <ParameterLabel>Volatility Modifier</ParameterLabel>
          <ParameterSlider
            type="range"
            min={0.3}
            max={2.0}
            step={0.1}
            value={scenarios[activeScenario].volatility_modifier}
            onChange={(e) => updateParameter('volatility_modifier', parseFloat(e.target.value))}
          />
          <ParameterValue>
            {scenarios[activeScenario].volatility_modifier.toFixed(1)}x
          </ParameterValue>
        </ParameterCard>

        <ParameterCard variant="glass">
          <ParameterLabel>External Factors</ParameterLabel>
          <ParameterSlider
            type="range"
            min={0.5}
            max={1.5}
            step={0.05}
            value={scenarios[activeScenario].external_factors}
            onChange={(e) => updateParameter('external_factors', parseFloat(e.target.value))}
          />
          <ParameterValue>
            {scenarios[activeScenario].external_factors.toFixed(2)}x
          </ParameterValue>
        </ParameterCard>

        <ParameterCard variant="glass">
          <ParameterLabel>Market Conditions</ParameterLabel>
          <MarketConditionSelect
            value={scenarios[activeScenario].market_conditions}
            onChange={(e) => updateParameter('market_conditions', e.target.value as 'optimistic' | 'neutral' | 'pessimistic')}
          >
            <option value="pessimistic">Pessimistic</option>
            <option value="neutral">Neutral</option>
            <option value="optimistic">Optimistic</option>
          </MarketConditionSelect>
        </ParameterCard>

        <ParameterCard variant="glass">
          <ParameterLabel>Real-time Updates</ParameterLabel>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#00ffff', marginTop: '1rem' }}>
            <input
              type="checkbox"
              checked={realTimeUpdates}
              onChange={(e) => setRealTimeUpdates(e.target.checked)}
            />
            Auto-update forecasts
          </label>
        </ParameterCard>
      </ParameterGrid>

      {/* Impact Summary */}
      {currentScenarioResult && (
        <ImpactSummary>
          <ImpactMetric>
            <MetricLabel>Total Change</MetricLabel>
            <MetricValue style={{ color: currentScenarioResult.impact_summary.total_change >= 0 ? '#39ff14' : '#ff0040' }}>
              {currentScenarioResult.impact_summary.total_change >= 0 ? '+' : ''}
              {currentScenarioResult.impact_summary.total_change.toFixed(1)}%
            </MetricValue>
          </ImpactMetric>
          
          <ImpactMetric>
            <MetricLabel>Peak Month</MetricLabel>
            <MetricValue>
              {new Date(currentScenarioResult.impact_summary.peak_month).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
            </MetricValue>
          </ImpactMetric>
          
          <ImpactMetric>
            <MetricLabel>Risk Level</MetricLabel>
            <MetricValue style={{ 
              color: currentScenarioResult.impact_summary.risk_level === 'high' ? '#ff0040' : 
                     currentScenarioResult.impact_summary.risk_level === 'medium' ? '#ffff00' : '#39ff14'
            }}>
              {currentScenarioResult.impact_summary.risk_level.toUpperCase()}
            </MetricValue>
          </ImpactMetric>
        </ImpactSummary>
      )}

      {/* Action Buttons */}
      <ActionButtons>
        <CyberpunkButton
          variant="primary"
          onClick={generateAllScenarios}
          disabled={loading}
        >
          {loading ? 'Generating...' : 'Generate All Scenarios'}
        </CyberpunkButton>
        
        <CyberpunkButton
          variant="secondary"
          onClick={resetToBaseline}
        >
          Reset to Baseline
        </CyberpunkButton>
        
        <CyberpunkButton
          variant="secondary"
          onClick={() => generateScenarioForecast(activeScenario, scenarios[activeScenario])}
        >
          Update Current Scenario
        </CyberpunkButton>
      </ActionButtons>
    </ScenarioContainer>
  );
};