import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

interface SeasonalData {
  timestamp: string;
  original: number;
  trend: number;
  seasonal: number;
  residual: number;
  monthlyVariation?: number;
}

interface SeasonalDecompositionChartProps {
  data: SeasonalData[];
  title?: string;
  height?: number;
  showMonthlyVariation?: boolean;
  cyberpunkMode?: boolean;
}

const SeasonalDecompositionChart: React.FC<SeasonalDecompositionChartProps> = ({
  data,
  title = "Seasonal Decomposition Analysis",
  height = 600,
  showMonthlyVariation = true,
  cyberpunkMode = true
}) => {
  const [selectedComponent, setSelectedComponent] = useState<'all' | 'trend' | 'seasonal' | 'residual'>('all');
  const [showGrid, setShowGrid] = useState(true);

  // Calculate monthly variation statistics
  const monthlyStats = useMemo(() => {
    if (!showMonthlyVariation || data.length === 0) return null;

    const monthlyData: { [key: number]: number[] } = {};
    
    data.forEach(item => {
      const date = new Date(item.timestamp);
      const month = date.getMonth();
      if (!monthlyData[month]) monthlyData[month] = [];
      monthlyData[month].push(item.seasonal);
    });

    return Object.keys(monthlyData).map(month => ({
      month: parseInt(month),
      monthName: new Date(2023, parseInt(month), 1).toLocaleDateString('en-US', { month: 'short' }),
      average: monthlyData[parseInt(month)].reduce((a, b) => a + b, 0) / monthlyData[parseInt(month)].length,
      min: Math.min(...monthlyData[parseInt(month)]),
      max: Math.max(...monthlyData[parseInt(month)]),
      variation: Math.max(...monthlyData[parseInt(month)]) - Math.min(...monthlyData[parseInt(month)])
    }));
  }, [data, showMonthlyVariation]);

  // Prepare chart data
  const chartData = data.map((item, index) => ({
    ...item,
    index,
    date: new Date(item.timestamp).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
  }));

  // Color scheme for cyberpunk mode
  const colors = cyberpunkMode ? {
    original: '#00ffff',
    trend: '#00ff00',
    seasonal: '#ffff00',
    residual: '#ff8000',
    grid: '#004444',
    text: '#00ffff',
    background: 'rgba(0, 0, 0, 0.8)'
  } : {
    original: '#2563eb',
    trend: '#16a34a',
    seasonal: '#eab308',
    residual: '#dc2626',
    grid: '#e5e7eb',
    text: '#374151',
    background: 'rgba(255, 255, 255, 0.9)'
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className={`p-3 rounded-lg border shadow-lg ${
          cyberpunkMode 
            ? 'bg-black bg-opacity-90 border-cyan-400 shadow-cyan-400/20' 
            : 'bg-white border-gray-300 shadow-gray-300/20'
        }`}>
          <p className={`font-semibold mb-2 ${cyberpunkMode ? 'text-cyan-400' : 'text-gray-700'}`}>
            {`Date: ${label}`}
          </p>
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

  // Render individual component chart
  const renderComponentChart = (component: string, dataKey: string, color: string, chartHeight: number) => (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-2">
        <h4 className={`text-lg font-semibold ${cyberpunkMode ? 'text-white' : 'text-gray-800'}`}>
          {component} Component
        </h4>
        <div className="flex items-center space-x-2">
          <div 
            className="w-4 h-4 rounded"
            style={{ backgroundColor: color }}
          />
          <span className={`text-sm ${cyberpunkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            {component}
          </span>
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height={chartHeight}>
        {dataKey === 'residual' ? (
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke={showGrid ? colors.grid : 'transparent'} />
            <XAxis 
              dataKey="date" 
              stroke={colors.text}
              tick={{ fill: colors.text, fontSize: 11 }}
            />
            <YAxis 
              stroke={colors.text}
              tick={{ fill: colors.text, fontSize: 11 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area 
              type="monotone" 
              dataKey={dataKey} 
              stroke={color} 
              fill={color}
              fillOpacity={0.3}
              strokeWidth={2}
            />
          </AreaChart>
        ) : (
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke={showGrid ? colors.grid : 'transparent'} />
            <XAxis 
              dataKey="date" 
              stroke={colors.text}
              tick={{ fill: colors.text, fontSize: 11 }}
            />
            <YAxis 
              stroke={colors.text}
              tick={{ fill: colors.text, fontSize: 11 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line 
              type="monotone" 
              dataKey={dataKey} 
              stroke={color} 
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  );

  // Render all components overlay
  const renderOverlayChart = () => (
    <ResponsiveContainer width="100%" height={height * 0.6}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke={showGrid ? colors.grid : 'transparent'} />
        <XAxis 
          dataKey="date" 
          stroke={colors.text}
          tick={{ fill: colors.text, fontSize: 12 }}
        />
        <YAxis 
          stroke={colors.text}
          tick={{ fill: colors.text, fontSize: 12 }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        
        <Line 
          type="monotone" 
          dataKey="original" 
          stroke={colors.original} 
          strokeWidth={3}
          dot={false}
          name="Original"
        />
        <Line 
          type="monotone" 
          dataKey="trend" 
          stroke={colors.trend} 
          strokeWidth={2}
          dot={false}
          name="Trend"
        />
        <Line 
          type="monotone" 
          dataKey="seasonal" 
          stroke={colors.seasonal} 
          strokeWidth={2}
          dot={false}
          name="Seasonal"
        />
      </LineChart>
    </ResponsiveContainer>
  );

  // Render monthly variation chart
  const renderMonthlyVariationChart = () => {
    if (!monthlyStats) return null;

    return (
      <div className="mt-6">
        <h4 className={`text-lg font-semibold mb-4 ${cyberpunkMode ? 'text-white' : 'text-gray-800'}`}>
          Monthly Seasonal Variation
        </h4>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={monthlyStats}>
            <CartesianGrid strokeDasharray="3 3" stroke={showGrid ? colors.grid : 'transparent'} />
            <XAxis 
              dataKey="monthName" 
              stroke={colors.text}
              tick={{ fill: colors.text, fontSize: 12 }}
            />
            <YAxis 
              stroke={colors.text}
              tick={{ fill: colors.text, fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area 
              type="monotone" 
              dataKey="max" 
              stroke={colors.seasonal} 
              fill={colors.seasonal}
              fillOpacity={0.2}
              name="Max"
            />
            <Area 
              type="monotone" 
              dataKey="min" 
              stroke={colors.seasonal} 
              fill={colors.seasonal}
              fillOpacity={0.2}
              name="Min"
            />
            <Line 
              type="monotone" 
              dataKey="average" 
              stroke={colors.seasonal} 
              strokeWidth={3}
              dot={{ fill: colors.seasonal, strokeWidth: 2, r: 4 }}
              name="Average"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className={`p-6 rounded-lg border shadow-lg ${
      cyberpunkMode 
        ? 'bg-black bg-opacity-80 border-cyan-500 shadow-cyan-500/20' 
        : 'bg-white border-gray-300 shadow-gray-300/20'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className={`text-2xl font-bold ${cyberpunkMode ? 'text-cyan-400' : 'text-gray-800'}`}>
          {title}
        </h2>
        <div className="flex items-center space-x-4">
          <label className={`flex items-center space-x-2 ${cyberpunkMode ? 'text-cyan-400' : 'text-gray-600'}`}>
            <input
              type="checkbox"
              checked={showGrid}
              onChange={(e) => setShowGrid(e.target.checked)}
              className="form-checkbox text-cyan-500"
            />
            <span>Show Grid</span>
          </label>
        </div>
      </div>

      {/* Component Selection */}
      <div className="flex flex-wrap gap-2 mb-6">
        {['all', 'trend', 'seasonal', 'residual'].map((component) => (
          <button
            key={component}
            onClick={() => setSelectedComponent(component as any)}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              selectedComponent === component
                ? cyberpunkMode 
                  ? 'bg-cyan-500 text-black' 
                  : 'bg-blue-500 text-white'
                : cyberpunkMode
                  ? 'bg-gray-700 text-cyan-400 hover:bg-gray-600'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {component.charAt(0).toUpperCase() + component.slice(1)}
          </button>
        ))}
      </div>

      {/* Main Chart */}
      <div className="mb-6">
        {selectedComponent === 'all' && renderOverlayChart()}
        {selectedComponent === 'trend' && renderComponentChart('Trend', 'trend', colors.trend, height * 0.6)}
        {selectedComponent === 'seasonal' && renderComponentChart('Seasonal', 'seasonal', colors.seasonal, height * 0.6)}
        {selectedComponent === 'residual' && renderComponentChart('Residual', 'residual', colors.residual, height * 0.6)}
      </div>

      {/* Decomposed Components (when showing all) */}
      {selectedComponent === 'all' && (
        <div className="grid grid-cols-1 gap-4">
          {renderComponentChart('Trend', 'trend', colors.trend, 120)}
          {renderComponentChart('Seasonal', 'seasonal', colors.seasonal, 120)}
          {renderComponentChart('Residual', 'residual', colors.residual, 120)}
        </div>
      )}

      {/* Monthly Variation */}
      {showMonthlyVariation && selectedComponent === 'seasonal' && renderMonthlyVariationChart()}

      {/* Statistics Panel */}
      <div className={`mt-6 p-4 rounded-lg ${
        cyberpunkMode 
          ? 'bg-gray-900 bg-opacity-50 border border-cyan-500' 
          : 'bg-gray-50 border border-gray-200'
      }`}>
        <h4 className={`font-semibold mb-3 ${cyberpunkMode ? 'text-cyan-400' : 'text-gray-800'}`}>
          Decomposition Statistics
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {['original', 'trend', 'seasonal', 'residual'].map((component) => {
            const values = data.map(d => d[component as keyof SeasonalData] as number);
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);
            
            return (
              <div key={component} className="text-center">
                <div className={`text-sm ${cyberpunkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  {component.charAt(0).toUpperCase() + component.slice(1)}
                </div>
                <div className={`text-lg font-bold ${cyberpunkMode ? 'text-white' : 'text-gray-800'}`}>
                  μ: {mean.toFixed(2)}
                </div>
                <div className={`text-sm ${cyberpunkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  σ: {std.toFixed(2)}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default SeasonalDecompositionChart;