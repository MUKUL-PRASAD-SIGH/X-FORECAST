import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import CyberpunkChart3D from './3d/CyberpunkChart3D';

interface PatternData {
  timestamp: string;
  value: number;
  trend?: number;
  seasonal?: number;
  residual?: number;
  volatility?: number;
  anomaly?: boolean;
  forecast?: number;
  confidence_upper?: number;
  confidence_lower?: number;
}

interface PatternAnalysis {
  pattern_type: string;
  trend_strength: number;
  seasonal_strength: number;
  volatility_level: string;
  anomaly_score: number;
  business_impact: {
    risk_level: string;
    revenue_impact_score: number;
    recommendations: string[];
  };
}

interface PatternVisualizationProps {
  data: PatternData[];
  analysis: PatternAnalysis;
  title?: string;
  height?: number;
  showControls?: boolean;
  cyberpunkMode?: boolean;
}

const PatternVisualization: React.FC<PatternVisualizationProps> = ({
  data,
  analysis,
  title = "Pattern Analysis Dashboard",
  height = 400,
  showControls = true,
  cyberpunkMode = true
}) => {
  const [activeView, setActiveView] = useState<'overview' | 'decomposition' | '3d' | 'volatility'>('overview');
  const [showAnomalies, setShowAnomalies] = useState(true);
  const [showForecast, setShowForecast] = useState(true);
  const [glowIntensity, setGlowIntensity] = useState(0.5);

  // Prepare data for different visualizations
  const chartData = data.map((item, index) => ({
    ...item,
    index,
    date: new Date(item.timestamp).toLocaleDateString(),
    anomalyValue: item.anomaly ? item.value : null
  }));

  // Prepare 3D data
  const chart3DData = data.map((item, index) => ({
    x: index,
    y: item.value,
    z: item.volatility || 0,
    value: item.value,
    label: item.timestamp,
    color: item.anomaly ? '#ff0080' : undefined
  }));

  // Get risk color based on analysis
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return '#00ff00';
      case 'medium': return '#ffff00';
      case 'high': return '#ff8000';
      case 'critical': return '#ff0000';
      default: return '#00ffff';
    }
  };

  // Custom tooltip for cyberpunk styling
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-black bg-opacity-90 border border-cyan-400 rounded p-3 shadow-lg shadow-cyan-400/20">
          <p className="text-cyan-400 font-semibold">{`Date: ${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {`${entry.dataKey}: ${entry.value?.toFixed(2)}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  // Render overview chart
  const renderOverviewChart = () => (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#004444" />
        <XAxis 
          dataKey="date" 
          stroke="#00ffff"
          tick={{ fill: '#00ffff', fontSize: 12 }}
        />
        <YAxis 
          stroke="#00ffff"
          tick={{ fill: '#00ffff', fontSize: 12 }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke="#00ffff" 
          strokeWidth={2}
          dot={{ fill: '#00ffff', strokeWidth: 2, r: 4 }}
          name="Actual Values"
        />
        
        {showForecast && (
          <>
            <Line 
              type="monotone" 
              dataKey="forecast" 
              stroke="#ff00ff" 
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ fill: '#ff00ff', strokeWidth: 2, r: 3 }}
              name="Forecast"
            />
            <Line 
              type="monotone" 
              dataKey="confidence_upper" 
              stroke="#ff00ff" 
              strokeWidth={1}
              strokeOpacity={0.5}
              dot={false}
              name="Upper Confidence"
            />
            <Line 
              type="monotone" 
              dataKey="confidence_lower" 
              stroke="#ff00ff" 
              strokeWidth={1}
              strokeOpacity={0.5}
              dot={false}
              name="Lower Confidence"
            />
          </>
        )}
        
        {showAnomalies && (
          <Line 
            type="monotone" 
            dataKey="anomalyValue" 
            stroke="#ff0080" 
            strokeWidth={0}
            dot={{ fill: '#ff0080', strokeWidth: 3, r: 6 }}
            name="Anomalies"
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );

  // Render decomposition chart
  const renderDecompositionChart = () => (
    <div className="grid grid-cols-1 gap-4">
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#004444" />
            <XAxis dataKey="date" stroke="#00ffff" tick={{ fill: '#00ffff', fontSize: 10 }} />
            <YAxis stroke="#00ffff" tick={{ fill: '#00ffff', fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="trend" stroke="#00ff00" strokeWidth={2} dot={false} name="Trend" />
          </LineChart>
        </ResponsiveContainer>
        <div className="text-center text-green-400 text-sm mt-1">Trend Component</div>
      </div>
      
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#004444" />
            <XAxis dataKey="date" stroke="#00ffff" tick={{ fill: '#00ffff', fontSize: 10 }} />
            <YAxis stroke="#00ffff" tick={{ fill: '#00ffff', fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="seasonal" stroke="#ffff00" strokeWidth={2} dot={false} name="Seasonal" />
          </LineChart>
        </ResponsiveContainer>
        <div className="text-center text-yellow-400 text-sm mt-1">Seasonal Component</div>
      </div>
      
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#004444" />
            <XAxis dataKey="date" stroke="#00ffff" tick={{ fill: '#00ffff', fontSize: 10 }} />
            <YAxis stroke="#00ffff" tick={{ fill: '#00ffff', fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="residual" stroke="#ff8000" fill="#ff8000" fillOpacity={0.3} name="Residual" />
          </AreaChart>
        </ResponsiveContainer>
        <div className="text-center text-orange-400 text-sm mt-1">Residual Component</div>
      </div>
    </div>
  );

  // Render volatility chart
  const renderVolatilityChart = () => (
    <div className="space-y-4">
      <ResponsiveContainer width="100%" height={height * 0.6}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#004444" />
          <XAxis dataKey="date" stroke="#00ffff" tick={{ fill: '#00ffff', fontSize: 12 }} />
          <YAxis stroke="#00ffff" tick={{ fill: '#00ffff', fontSize: 12 }} />
          <Tooltip content={<CustomTooltip />} />
          <Area 
            type="monotone" 
            dataKey="volatility" 
            stroke="#ff0080" 
            fill="#ff0080" 
            fillOpacity={0.4}
            name="Volatility"
          />
        </AreaChart>
      </ResponsiveContainer>
      
      {/* Volatility Risk Alert */}
      <div className={`p-4 rounded-lg border-2 ${
        analysis.volatility_level === 'high' || analysis.volatility_level === 'extreme' 
          ? 'border-red-500 bg-red-900 bg-opacity-20' 
          : analysis.volatility_level === 'medium'
          ? 'border-yellow-500 bg-yellow-900 bg-opacity-20'
          : 'border-green-500 bg-green-900 bg-opacity-20'
      }`}>
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-lg font-semibold text-white">Volatility Risk Level</h4>
            <p className={`text-2xl font-bold ${
              analysis.volatility_level === 'high' || analysis.volatility_level === 'extreme' 
                ? 'text-red-400' 
                : analysis.volatility_level === 'medium'
                ? 'text-yellow-400'
                : 'text-green-400'
            }`}>
              {analysis.volatility_level.toUpperCase()}
            </p>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-400">Risk Score</div>
            <div className="text-xl font-bold text-white">
              {(analysis.business_impact.revenue_impact_score * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="bg-black bg-opacity-80 border border-cyan-500 rounded-lg p-6 shadow-lg shadow-cyan-500/20">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-cyan-400">{title}</h2>
        <div className="flex items-center space-x-4">
          <div className={`px-3 py-1 rounded-full text-sm font-semibold border-2`}
               style={{ 
                 borderColor: getRiskColor(analysis.business_impact.risk_level),
                 color: getRiskColor(analysis.business_impact.risk_level)
               }}>
            {analysis.pattern_type.replace('_', ' ').toUpperCase()}
          </div>
          <div className={`px-3 py-1 rounded-full text-sm font-semibold border-2`}
               style={{ 
                 borderColor: getRiskColor(analysis.business_impact.risk_level),
                 color: getRiskColor(analysis.business_impact.risk_level)
               }}>
            Risk: {analysis.business_impact.risk_level.toUpperCase()}
          </div>
        </div>
      </div>

      {/* Controls */}
      {showControls && (
        <div className="flex flex-wrap items-center justify-between mb-6 p-4 bg-gray-900 bg-opacity-50 rounded-lg">
          <div className="flex space-x-2">
            {['overview', 'decomposition', '3d', 'volatility'].map((view) => (
              <button
                key={view}
                onClick={() => setActiveView(view as any)}
                className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  activeView === view
                    ? 'bg-cyan-500 text-black'
                    : 'bg-gray-700 text-cyan-400 hover:bg-gray-600'
                }`}
              >
                {view.charAt(0).toUpperCase() + view.slice(1)}
              </button>
            ))}
          </div>
          
          <div className="flex items-center space-x-4">
            <label className="flex items-center space-x-2 text-cyan-400">
              <input
                type="checkbox"
                checked={showAnomalies}
                onChange={(e) => setShowAnomalies(e.target.checked)}
                className="form-checkbox text-cyan-500"
              />
              <span>Show Anomalies</span>
            </label>
            
            <label className="flex items-center space-x-2 text-cyan-400">
              <input
                type="checkbox"
                checked={showForecast}
                onChange={(e) => setShowForecast(e.target.checked)}
                className="form-checkbox text-cyan-500"
              />
              <span>Show Forecast</span>
            </label>
            
            {activeView === '3d' && (
              <div className="flex items-center space-x-2 text-cyan-400">
                <span>Glow:</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={glowIntensity}
                  onChange={(e) => setGlowIntensity(parseFloat(e.target.value))}
                  className="w-20"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Main Visualization */}
      <div className="mb-6">
        {activeView === 'overview' && renderOverviewChart()}
        {activeView === 'decomposition' && renderDecompositionChart()}
        {activeView === '3d' && (
          <div className="flex justify-center">
            <CyberpunkChart3D 
              data={chart3DData}
              width={800}
              height={height}
              title="3D Pattern Analysis"
              glowIntensity={glowIntensity}
            />
          </div>
        )}
        {activeView === 'volatility' && renderVolatilityChart()}
      </div>

      {/* Pattern Insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-900 bg-opacity-50 p-4 rounded-lg border border-green-500">
          <h4 className="text-green-400 font-semibold mb-2">Trend Strength</h4>
          <div className="text-2xl font-bold text-white">
            {(analysis.trend_strength * 100).toFixed(0)}%
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
            <div 
              className="bg-green-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${analysis.trend_strength * 100}%` }}
            />
          </div>
        </div>
        
        <div className="bg-gray-900 bg-opacity-50 p-4 rounded-lg border border-yellow-500">
          <h4 className="text-yellow-400 font-semibold mb-2">Seasonal Strength</h4>
          <div className="text-2xl font-bold text-white">
            {(analysis.seasonal_strength * 100).toFixed(0)}%
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
            <div 
              className="bg-yellow-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${analysis.seasonal_strength * 100}%` }}
            />
          </div>
        </div>
        
        <div className="bg-gray-900 bg-opacity-50 p-4 rounded-lg border border-red-500">
          <h4 className="text-red-400 font-semibold mb-2">Anomaly Score</h4>
          <div className="text-2xl font-bold text-white">
            {(analysis.anomaly_score * 100).toFixed(0)}%
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
            <div 
              className="bg-red-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${analysis.anomaly_score * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Business Recommendations */}
      {analysis.business_impact.recommendations.length > 0 && (
        <div className="bg-gray-900 bg-opacity-50 p-4 rounded-lg border border-cyan-500">
          <h4 className="text-cyan-400 font-semibold mb-3">Business Recommendations</h4>
          <ul className="space-y-2">
            {analysis.business_impact.recommendations.map((recommendation, index) => (
              <li key={index} className="flex items-start space-x-2 text-gray-300">
                <span className="text-cyan-400 mt-1">•</span>
                <span>{recommendation}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default PatternVisualization;