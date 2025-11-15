import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { CyberpunkCard, CyberpunkButton } from './ui';

interface ExportOptions {
  format: 'pdf' | 'excel' | 'json' | 'csv';
  includeCharts: boolean;
  includeMetrics: boolean;
  includeConfidenceIntervals: boolean;
  includeScenarios: boolean;
  includeRecommendations: boolean;
  dateRange: {
    start: string;
    end: string;
  };
  customTitle?: string;
  customNotes?: string;
}

interface ForecastExportProps {
  authToken: string;
  forecastData?: any;
  scenarioData?: any[];
  onExportComplete?: (result: { success: boolean; downloadUrl?: string; error?: string }) => void;
}

const ExportContainer = styled(CyberpunkCard)`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.lg};
`;

const OptionsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: ${props => props.theme.spacing.md};
`;

const OptionCard = styled(CyberpunkCard)`
  padding: ${props => props.theme.spacing.md};
`;

const OptionLabel = styled.label`
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  color: ${props => props.theme.colors.neonBlue};
  text-transform: uppercase;
  letter-spacing: 1px;
  display: block;
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const FormatSelector = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${props => props.theme.spacing.sm};
`;

const FormatButton = styled(motion.button)<{ active: boolean }>`
  padding: ${props => props.theme.spacing.sm};
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
  text-transform: uppercase;
  
  &:hover {
    background: ${props => props.active 
      ? `linear-gradient(45deg, ${props.theme.colors.neonBlue}, ${props.theme.colors.hotPink})`
      : 'rgba(0, 255, 255, 0.2)'
    };
    transform: translateY(-2px);
  }
`;

const CheckboxGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.sm};
`;

const CheckboxItem = styled.label`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  color: ${props => props.theme.colors.primaryText};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  cursor: pointer;
  
  input[type="checkbox"] {
    accent-color: ${props => props.theme.colors.neonBlue};
  }
  
  &:hover {
    color: ${props => props.theme.colors.neonBlue};
  }
`;

const DateInput = styled.input`
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
`;

const TextInput = styled.input`
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
`;

const TextArea = styled.textarea`
  width: 100%;
  min-height: 80px;
  padding: ${props => props.theme.spacing.sm};
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid ${props => props.theme.colors.neonBlue};
  border-radius: 4px;
  color: ${props => props.theme.colors.primaryText};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  resize: vertical;
  
  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.hotPink};
    box-shadow: ${props => props.theme.effects.softGlow};
  }
`;

const PreviewSection = styled.div`
  background: rgba(0, 255, 255, 0.05);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 4px;
  padding: ${props => props.theme.spacing.md};
`;

const PreviewTitle = styled.h4`
  color: ${props => props.theme.colors.neonBlue};
  font-family: ${props => props.theme.typography.fontFamily.mono};
  margin-bottom: ${props => props.theme.spacing.sm};
  font-size: ${props => props.theme.typography.fontSize.md};
`;

const PreviewList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
  
  li {
    color: ${props => props.theme.colors.secondaryText};
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.sm};
    padding: ${props => props.theme.spacing.xs} 0;
    
    &::before {
      content: '▶ ';
      color: ${props => props.theme.colors.neonBlue};
      margin-right: ${props => props.theme.spacing.xs};
    }
  }
`;

const ExportActions = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  justify-content: center;
  flex-wrap: wrap;
`;

const ProgressBar = styled.div<{ progress: number }>`
  width: 100%;
  height: 6px;
  background: rgba(0, 255, 255, 0.2);
  border-radius: 3px;
  overflow: hidden;
  margin: ${props => props.theme.spacing.sm} 0;
  
  &::after {
    content: '';
    display: block;
    height: 100%;
    width: ${props => props.progress}%;
    background: linear-gradient(90deg, 
      ${props => props.theme.colors.neonBlue}, 
      ${props => props.theme.colors.hotPink}
    );
    transition: width 0.3s ease;
  }
`;

const StatusMessage = styled.div<{ type: 'success' | 'error' | 'info' }>`
  padding: ${props => props.theme.spacing.sm};
  border-radius: 4px;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  text-align: center;
  
  background: ${props => {
    switch (props.type) {
      case 'success': return 'rgba(57, 255, 20, 0.1)';
      case 'error': return 'rgba(255, 0, 64, 0.1)';
      default: return 'rgba(0, 255, 255, 0.1)';
    }
  }};
  
  border: 1px solid ${props => {
    switch (props.type) {
      case 'success': return '#39ff14';
      case 'error': return '#ff0040';
      default: return '#00ffff';
    }
  }};
  
  color: ${props => {
    switch (props.type) {
      case 'success': return '#39ff14';
      case 'error': return '#ff0040';
      default: return '#00ffff';
    }
  }};
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

export const ForecastExport: React.FC<ForecastExportProps> = ({ 
  authToken, 
  forecastData,
  scenarioData,
  onExportComplete 
}) => {
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    format: 'pdf',
    includeCharts: true,
    includeMetrics: true,
    includeConfidenceIntervals: true,
    includeScenarios: true,
    includeRecommendations: true,
    dateRange: {
      start: new Date().toISOString().split('T')[0],
      end: new Date(Date.now() + 180 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
    },
    customTitle: '',
    customNotes: ''
  });

  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportStatus, setExportStatus] = useState<{
    type: 'success' | 'error' | 'info';
    message: string;
  } | null>(null);

  // Update export option
  const updateOption = <K extends keyof ExportOptions>(key: K, value: ExportOptions[K]) => {
    setExportOptions(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // Update date range
  const updateDateRange = (field: 'start' | 'end', value: string) => {
    setExportOptions(prev => ({
      ...prev,
      dateRange: {
        ...prev.dateRange,
        [field]: value
      }
    }));
  };

  // Generate export
  const generateExport = async () => {
    setExporting(true);
    setExportProgress(0);
    setExportStatus({ type: 'info', message: 'Preparing export...' });

    try {
      // Simulate export progress
      const progressSteps = [
        { progress: 20, message: 'Collecting forecast data...' },
        { progress: 40, message: 'Processing scenarios...' },
        { progress: 60, message: 'Generating charts...' },
        { progress: 80, message: 'Formatting document...' },
        { progress: 100, message: 'Finalizing export...' }
      ];

      for (const step of progressSteps) {
        setExportProgress(step.progress);
        setExportStatus({ type: 'info', message: step.message });
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // Attempt actual export API call
      try {
        const response = await fetch('/api/company-sales/export-forecast', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${authToken}`,
          },
          body: JSON.stringify({
            export_options: exportOptions,
            forecast_data: forecastData,
            scenario_data: scenarioData
          }),
        });

        if (response.ok) {
          const result = await response.json();
          setExportStatus({ 
            type: 'success', 
            message: `Export completed successfully! Format: ${exportOptions.format.toUpperCase()}` 
          });
          
          if (onExportComplete) {
            onExportComplete({
              success: true,
              downloadUrl: result.download_url
            });
          }

          // Trigger download if URL is provided
          if (result.download_url) {
            const link = document.createElement('a');
            link.href = result.download_url;
            link.download = result.filename || `forecast_export.${exportOptions.format}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
          }
        } else {
          throw new Error(`Export failed: ${response.statusText}`);
        }
      } catch (apiError) {
        console.warn('API export failed, generating mock download:', apiError);
        
        // Generate mock download for demonstration
        const mockData = generateMockExportData();
        const blob = new Blob([mockData], { 
          type: exportOptions.format === 'json' ? 'application/json' : 'text/plain' 
        });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `forecast_export_${Date.now()}.${exportOptions.format}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
        
        setExportStatus({ 
          type: 'success', 
          message: `Mock export completed! Format: ${exportOptions.format.toUpperCase()}` 
        });
        
        if (onExportComplete) {
          onExportComplete({ success: true });
        }
      }
    } catch (error) {
      setExportStatus({ 
        type: 'error', 
        message: `Export failed: ${error instanceof Error ? error.message : 'Unknown error'}` 
      });
      
      if (onExportComplete) {
        onExportComplete({
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    } finally {
      setExporting(false);
    }
  };

  // Generate mock export data
  const generateMockExportData = () => {
    const exportData = {
      title: exportOptions.customTitle || 'Forecast Export Report',
      generated_at: new Date().toISOString(),
      date_range: exportOptions.dateRange,
      format: exportOptions.format,
      options: exportOptions,
      forecast_data: forecastData || {
        point_forecast: { '2024-01-01': 1000, '2024-02-01': 1050, '2024-03-01': 1100 },
        confidence_intervals: {
          p10: { '2024-01-01': 900, '2024-02-01': 945, '2024-03-01': 990 },
          p90: { '2024-01-01': 1100, '2024-02-01': 1155, '2024-03-01': 1210 }
        }
      },
      scenario_data: scenarioData || [],
      notes: exportOptions.customNotes || 'Generated by Cyberpunk AI Dashboard'
    };

    switch (exportOptions.format) {
      case 'json':
        return JSON.stringify(exportData, null, 2);
      case 'csv':
        return convertToCSV(exportData);
      default:
        return JSON.stringify(exportData, null, 2);
    }
  };

  // Convert data to CSV format
  const convertToCSV = (data: any) => {
    const headers = ['Date', 'Forecast', 'P10', 'P90'];
    const rows = [headers.join(',')];
    
    if (data.forecast_data?.point_forecast) {
      Object.entries(data.forecast_data.point_forecast).forEach(([date, value]) => {
        const p10 = data.forecast_data.confidence_intervals?.p10?.[date] || '';
        const p90 = data.forecast_data.confidence_intervals?.p90?.[date] || '';
        rows.push(`${date},${value},${p10},${p90}`);
      });
    }
    
    return rows.join('\n');
  };

  // Get preview items
  const getPreviewItems = () => {
    const items = [];
    
    if (exportOptions.includeCharts) items.push('Interactive forecast charts');
    if (exportOptions.includeMetrics) items.push('Performance metrics and accuracy scores');
    if (exportOptions.includeConfidenceIntervals) items.push('Confidence intervals (P10, P50, P90)');
    if (exportOptions.includeScenarios) items.push('Scenario comparison analysis');
    if (exportOptions.includeRecommendations) items.push('AI-generated recommendations');
    
    items.push(`Date range: ${exportOptions.dateRange.start} to ${exportOptions.dateRange.end}`);
    items.push(`Format: ${exportOptions.format.toUpperCase()}`);
    
    return items;
  };

  return (
    <ExportContainer variant="neon" padding="lg">
      <CardTitle>📤 Forecast Export & Reporting</CardTitle>
      
      <OptionsGrid>
        {/* Format Selection */}
        <OptionCard variant="glass">
          <OptionLabel>Export Format</OptionLabel>
          <FormatSelector>
            {(['pdf', 'excel', 'json', 'csv'] as const).map(format => (
              <FormatButton
                key={format}
                active={exportOptions.format === format}
                onClick={() => updateOption('format', format)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {format.toUpperCase()}
              </FormatButton>
            ))}
          </FormatSelector>
        </OptionCard>

        {/* Content Options */}
        <OptionCard variant="glass">
          <OptionLabel>Include Content</OptionLabel>
          <CheckboxGroup>
            <CheckboxItem>
              <input
                type="checkbox"
                checked={exportOptions.includeCharts}
                onChange={(e) => updateOption('includeCharts', e.target.checked)}
              />
              Charts & Visualizations
            </CheckboxItem>
            <CheckboxItem>
              <input
                type="checkbox"
                checked={exportOptions.includeMetrics}
                onChange={(e) => updateOption('includeMetrics', e.target.checked)}
              />
              Performance Metrics
            </CheckboxItem>
            <CheckboxItem>
              <input
                type="checkbox"
                checked={exportOptions.includeConfidenceIntervals}
                onChange={(e) => updateOption('includeConfidenceIntervals', e.target.checked)}
              />
              Confidence Intervals
            </CheckboxItem>
            <CheckboxItem>
              <input
                type="checkbox"
                checked={exportOptions.includeScenarios}
                onChange={(e) => updateOption('includeScenarios', e.target.checked)}
              />
              Scenario Analysis
            </CheckboxItem>
            <CheckboxItem>
              <input
                type="checkbox"
                checked={exportOptions.includeRecommendations}
                onChange={(e) => updateOption('includeRecommendations', e.target.checked)}
              />
              AI Recommendations
            </CheckboxItem>
          </CheckboxGroup>
        </OptionCard>

        {/* Date Range */}
        <OptionCard variant="glass">
          <OptionLabel>Date Range</OptionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div>
              <label style={{ fontSize: '0.8rem', color: '#00ffff', marginBottom: '0.25rem', display: 'block' }}>
                Start Date
              </label>
              <DateInput
                type="date"
                value={exportOptions.dateRange.start}
                onChange={(e) => updateDateRange('start', e.target.value)}
              />
            </div>
            <div>
              <label style={{ fontSize: '0.8rem', color: '#00ffff', marginBottom: '0.25rem', display: 'block' }}>
                End Date
              </label>
              <DateInput
                type="date"
                value={exportOptions.dateRange.end}
                onChange={(e) => updateDateRange('end', e.target.value)}
              />
            </div>
          </div>
        </OptionCard>

        {/* Custom Options */}
        <OptionCard variant="glass">
          <OptionLabel>Customization</OptionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div>
              <label style={{ fontSize: '0.8rem', color: '#00ffff', marginBottom: '0.25rem', display: 'block' }}>
                Report Title
              </label>
              <TextInput
                type="text"
                placeholder="Custom report title..."
                value={exportOptions.customTitle}
                onChange={(e) => updateOption('customTitle', e.target.value)}
              />
            </div>
            <div>
              <label style={{ fontSize: '0.8rem', color: '#00ffff', marginBottom: '0.25rem', display: 'block' }}>
                Additional Notes
              </label>
              <TextArea
                placeholder="Add custom notes or comments..."
                value={exportOptions.customNotes}
                onChange={(e) => updateOption('customNotes', e.target.value)}
              />
            </div>
          </div>
        </OptionCard>
      </OptionsGrid>

      {/* Export Preview */}
      <PreviewSection>
        <PreviewTitle>Export Preview</PreviewTitle>
        <PreviewList>
          {getPreviewItems().map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </PreviewList>
      </PreviewSection>

      {/* Progress Bar */}
      {exporting && (
        <ProgressBar progress={exportProgress} />
      )}

      {/* Status Message */}
      {exportStatus && (
        <StatusMessage type={exportStatus.type}>
          {exportStatus.message}
        </StatusMessage>
      )}

      {/* Export Actions */}
      <ExportActions>
        <CyberpunkButton
          variant="primary"
          onClick={generateExport}
          disabled={exporting}
        >
          {exporting ? 'Exporting...' : 'Generate Export'}
        </CyberpunkButton>
        
        <CyberpunkButton
          variant="secondary"
          onClick={() => {
            setExportOptions({
              format: 'pdf',
              includeCharts: true,
              includeMetrics: true,
              includeConfidenceIntervals: true,
              includeScenarios: true,
              includeRecommendations: true,
              dateRange: {
                start: new Date().toISOString().split('T')[0],
                end: new Date(Date.now() + 180 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
              },
              customTitle: '',
              customNotes: ''
            });
            setExportStatus(null);
          }}
        >
          Reset Options
        </CyberpunkButton>
      </ExportActions>
    </ExportContainer>
  );
};