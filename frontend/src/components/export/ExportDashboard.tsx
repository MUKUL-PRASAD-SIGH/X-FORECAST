import React, { useState, useEffect } from 'react';
import styled, { keyframes } from 'styled-components';

// Cyberpunk animations
const glowPulse = keyframes`
  0%, 100% { box-shadow: 0 0 5px #00ffff, 0 0 10px #00ffff, 0 0 15px #00ffff; }
  50% { box-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff; }
`;

const scanLine = keyframes`
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
`;

const matrixRain = keyframes`
  0% { transform: translateY(-100%); opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { transform: translateY(100vh); opacity: 0; }
`;

// Styled components
const ExportContainer = styled.div`
  background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
  border: 2px solid #00ffff;
  border-radius: 15px;
  padding: 2rem;
  margin: 1rem;
  position: relative;
  overflow: hidden;
  animation: ${glowPulse} 3s ease-in-out infinite;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00ffff, transparent);
    animation: ${scanLine} 3s linear infinite;
  }
`;

const Title = styled.h2`
  color: #00ffff;
  font-family: 'Orbitron', monospace;
  font-size: 2rem;
  text-align: center;
  margin-bottom: 2rem;
  text-shadow: 0 0 10px #00ffff;
  position: relative;

  &::after {
    content: '';
    position: absolute;
    bottom: -5px;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00ffff, transparent);
  }
`;

const ExportGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-bottom: 2rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ExportSection = styled.div`
  background: rgba(0, 255, 255, 0.05);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 10px;
  padding: 1.5rem;
  position: relative;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ffff, transparent);
  }
`;

const SectionTitle = styled.h3`
  color: #00ffff;
  font-family: 'Orbitron', monospace;
  font-size: 1.2rem;
  margin-bottom: 1rem;
  text-shadow: 0 0 5px #00ffff;
`;

const CheckboxGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-bottom: 1rem;
`;

const CheckboxLabel = styled.label`
  display: flex;
  align-items: center;
  color: #ffffff;
  font-family: 'Roboto Mono', monospace;
  cursor: pointer;
  transition: all 0.3s ease;

  &:hover {
    color: #00ffff;
    text-shadow: 0 0 5px #00ffff;
  }

  input[type="checkbox"] {
    margin-right: 0.5rem;
    accent-color: #00ffff;
  }
`;

const FormatSelector = styled.select`
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid #00ffff;
  border-radius: 5px;
  color: #00ffff;
  font-family: 'Roboto Mono', monospace;
  padding: 0.5rem;
  width: 100%;
  margin-bottom: 1rem;

  &:focus {
    outline: none;
    box-shadow: 0 0 10px #00ffff;
  }

  option {
    background: #000000;
    color: #00ffff;
  }
`;

const CustomInput = styled.input`
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid #00ffff;
  border-radius: 5px;
  color: #00ffff;
  font-family: 'Roboto Mono', monospace;
  padding: 0.5rem;
  width: 100%;
  margin-bottom: 1rem;

  &:focus {
    outline: none;
    box-shadow: 0 0 10px #00ffff;
  }

  &::placeholder {
    color: rgba(0, 255, 255, 0.5);
  }
`;

const ExportButton = styled.button`
  background: linear-gradient(45deg, #00ffff, #0080ff);
  border: none;
  border-radius: 10px;
  color: #000000;
  font-family: 'Orbitron', monospace;
  font-size: 1.1rem;
  font-weight: bold;
  padding: 1rem 2rem;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  width: 100%;
  margin-top: 1rem;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 255, 255, 0.4);
  }

  &:active {
    transform: translateY(0);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
  }

  &:hover::before {
    left: 100%;
  }
`;

const StatusMessage = styled.div<{ type: 'success' | 'error' | 'info' }>`
  background: ${props => 
    props.type === 'success' ? 'rgba(0, 255, 0, 0.1)' :
    props.type === 'error' ? 'rgba(255, 0, 0, 0.1)' :
    'rgba(0, 255, 255, 0.1)'
  };
  border: 1px solid ${props => 
    props.type === 'success' ? '#00ff00' :
    props.type === 'error' ? '#ff0000' :
    '#00ffff'
  };
  border-radius: 5px;
  color: ${props => 
    props.type === 'success' ? '#00ff00' :
    props.type === 'error' ? '#ff0000' :
    '#00ffff'
  };
  font-family: 'Roboto Mono', monospace;
  padding: 1rem;
  margin-top: 1rem;
  text-align: center;
`;

const ProgressBar = styled.div`
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid #00ffff;
  border-radius: 10px;
  height: 20px;
  margin-top: 1rem;
  overflow: hidden;
  position: relative;
`;

const ProgressFill = styled.div<{ progress: number }>`
  background: linear-gradient(90deg, #00ffff, #0080ff);
  height: 100%;
  width: ${props => props.progress}%;
  transition: width 0.3s ease;
  position: relative;

  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    animation: ${scanLine} 2s linear infinite;
  }
`;

const DownloadLink = styled.a`
  background: linear-gradient(45deg, #00ff00, #00cc00);
  border: none;
  border-radius: 10px;
  color: #000000;
  font-family: 'Orbitron', monospace;
  font-size: 1rem;
  font-weight: bold;
  padding: 0.8rem 1.5rem;
  text-decoration: none;
  display: inline-block;
  margin-top: 1rem;
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 255, 0, 0.4);
  }
`;

// Interfaces
interface ExportOptions {
  format: string;
  include_forecasts: boolean;
  include_performance: boolean;
  include_insights: boolean;
  include_metadata: boolean;
  include_charts: boolean;
  custom_title: string;
  horizon_months: number;
}

interface ExportResponse {
  success: boolean;
  filename: string;
  download_url: string;
  file_size_bytes: number;
  format: string;
  generated_at: string;
  expires_at: string;
  metadata: any;
}

interface ExportDashboardProps {
  companyId: string;
  className?: string;
}

const ExportDashboard: React.FC<ExportDashboardProps> = ({ companyId, className }) => {
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    format: 'json',
    include_forecasts: true,
    include_performance: true,
    include_insights: true,
    include_metadata: true,
    include_charts: false,
    custom_title: '',
    horizon_months: 6
  });

  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportResult, setExportResult] = useState<ExportResponse | null>(null);
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error' | 'info', message: string } | null>(null);
  const [supportedFormats, setSupportedFormats] = useState<any[]>([]);

  useEffect(() => {
    loadSupportedFormats();
  }, []);

  const loadSupportedFormats = async () => {
    try {
      const response = await fetch('/api/export/formats');
      if (response.ok) {
        const data = await response.json();
        setSupportedFormats(data.supported_formats || []);
      }
    } catch (error) {
      console.error('Failed to load supported formats:', error);
    }
  };

  const handleExportOptionChange = (key: keyof ExportOptions, value: any) => {
    setExportOptions(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const simulateProgress = () => {
    setExportProgress(0);
    const interval = setInterval(() => {
      setExportProgress(prev => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + Math.random() * 15;
      });
    }, 200);
    return interval;
  };

  const handleExport = async (exportType: 'comprehensive' | 'forecast' | 'performance' | 'insights' | 'detailed-performance' = 'comprehensive') => {
    setIsExporting(true);
    setStatusMessage({ type: 'info', message: 'Generating comprehensive report...' });
    setExportResult(null);
    
    const progressInterval = simulateProgress();

    try {
      // Determine endpoint based on export type
      let endpoint = '/api/export/comprehensive-report';
      let exportMessage = 'Generating comprehensive report...';
      
      switch (exportType) {
        case 'forecast':
          endpoint = '/api/export/forecast-only';
          exportMessage = 'Generating forecast report...';
          break;
        case 'performance':
          endpoint = '/api/export/performance-report';
          exportMessage = 'Generating performance report...';
          break;
        case 'insights':
          endpoint = '/api/export/insights-report';
          exportMessage = 'Generating insights report...';
          break;
        case 'detailed-performance':
          endpoint = '/api/export/model-performance-detailed';
          exportMessage = 'Generating detailed performance analysis...';
          break;
        default:
          endpoint = '/api/export/comprehensive-report';
          exportMessage = 'Generating comprehensive report...';
      }
      
      setStatusMessage({ type: 'info', message: exportMessage });

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${companyId}`
        },
        body: JSON.stringify(exportOptions)
      });

      clearInterval(progressInterval);
      setExportProgress(100);

      if (response.ok) {
        const result: ExportResponse = await response.json();
        setExportResult(result);
        
        // Enhanced success message with metadata
        const metadata = result.metadata;
        let successMessage = `Report generated successfully! File size: ${(result.file_size_bytes / 1024 / 1024).toFixed(2)} MB`;
        
        if (metadata?.generation_time_seconds) {
          successMessage += ` (Generated in ${metadata.generation_time_seconds.toFixed(1)}s)`;
        }
        
        if (metadata?.data_points_included) {
          successMessage += ` | Data points: ${metadata.data_points_included.toLocaleString()}`;
        }
        
        if (metadata?.export_features?.comprehensive_formatting) {
          successMessage += ` | Enhanced formatting enabled`;
        }
        
        setStatusMessage({ 
          type: 'success', 
          message: successMessage
        });
      } else {
        const error = await response.json();
        setStatusMessage({ 
          type: 'error', 
          message: `Export failed: ${error.detail || 'Unknown error'}` 
        });
      }
    } catch (error) {
      clearInterval(progressInterval);
      setExportProgress(0);
      setStatusMessage({ 
        type: 'error', 
        message: `Export failed: ${error instanceof Error ? error.message : 'Network error'}` 
      });
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <ExportContainer className={className}>
      <Title>📊 EXPORT CONTROL CENTER</Title>
      
      <ExportGrid>
        <ExportSection>
          <SectionTitle>📋 Report Components</SectionTitle>
          <CheckboxGroup>
            <CheckboxLabel>
              <input
                type="checkbox"
                checked={exportOptions.include_forecasts}
                onChange={(e) => handleExportOptionChange('include_forecasts', e.target.checked)}
              />
              Ensemble Forecasts & Predictions
            </CheckboxLabel>
            <CheckboxLabel>
              <input
                type="checkbox"
                checked={exportOptions.include_performance}
                onChange={(e) => handleExportOptionChange('include_performance', e.target.checked)}
              />
              Model Performance Metrics
            </CheckboxLabel>
            <CheckboxLabel>
              <input
                type="checkbox"
                checked={exportOptions.include_insights}
                onChange={(e) => handleExportOptionChange('include_insights', e.target.checked)}
              />
              AI Business Insights
            </CheckboxLabel>
            <CheckboxLabel>
              <input
                type="checkbox"
                checked={exportOptions.include_metadata}
                onChange={(e) => handleExportOptionChange('include_metadata', e.target.checked)}
              />
              Model Metadata & Configuration
            </CheckboxLabel>
            <CheckboxLabel>
              <input
                type="checkbox"
                checked={exportOptions.include_charts}
                onChange={(e) => handleExportOptionChange('include_charts', e.target.checked)}
              />
              Chart Data (where supported)
            </CheckboxLabel>
          </CheckboxGroup>
        </ExportSection>

        <ExportSection>
          <SectionTitle>⚙️ Export Configuration</SectionTitle>
          
          <label style={{ color: '#00ffff', fontFamily: 'Roboto Mono', fontSize: '0.9rem' }}>
            Export Format:
          </label>
          <FormatSelector
            value={exportOptions.format}
            onChange={(e) => handleExportOptionChange('format', e.target.value)}
          >
            {supportedFormats.map(format => (
              <option key={format.format} value={format.format}>
                {format.format.toUpperCase()} - {format.description}
              </option>
            ))}
            {supportedFormats.length === 0 && (
              <>
                <option value="json">JSON - Complete data structure</option>
                <option value="excel">EXCEL - Multi-sheet workbook</option>
                <option value="pdf">PDF - Formatted report</option>
              </>
            )}
          </FormatSelector>

          <label style={{ color: '#00ffff', fontFamily: 'Roboto Mono', fontSize: '0.9rem' }}>
            Custom Report Title:
          </label>
          <CustomInput
            type="text"
            placeholder="Enter custom report title (optional)"
            value={exportOptions.custom_title}
            onChange={(e) => handleExportOptionChange('custom_title', e.target.value)}
          />

          <label style={{ color: '#00ffff', fontFamily: 'Roboto Mono', fontSize: '0.9rem' }}>
            Forecast Horizon (months):
          </label>
          <CustomInput
            type="number"
            min="1"
            max="24"
            value={exportOptions.horizon_months}
            onChange={(e) => handleExportOptionChange('horizon_months', parseInt(e.target.value) || 6)}
          />
        </ExportSection>
      </ExportGrid>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
        <ExportButton
          onClick={() => handleExport('comprehensive')}
          disabled={isExporting}
        >
          {isExporting ? '🔄 GENERATING...' : '🚀 COMPREHENSIVE REPORT'}
        </ExportButton>
        
        <ExportButton
          onClick={() => handleExport('forecast')}
          disabled={isExporting}
          style={{ background: 'linear-gradient(45deg, #ff00ff, #8000ff)' }}
        >
          {isExporting ? '🔄 GENERATING...' : '📈 FORECAST ONLY'}
        </ExportButton>
        
        <ExportButton
          onClick={() => handleExport('performance')}
          disabled={isExporting}
          style={{ background: 'linear-gradient(45deg, #ffff00, #ff8000)' }}
        >
          {isExporting ? '🔄 GENERATING...' : '⚡ PERFORMANCE REPORT'}
        </ExportButton>
        
        <ExportButton
          onClick={() => handleExport('insights')}
          disabled={isExporting}
          style={{ background: 'linear-gradient(45deg, #00ff00, #00ff80)' }}
        >
          {isExporting ? '🔄 GENERATING...' : '🧠 INSIGHTS REPORT'}
        </ExportButton>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1rem', marginTop: '1rem' }}>
        <ExportButton
          onClick={() => window.open('/shareable-reports', '_blank')}
          disabled={isExporting}
          style={{ background: 'linear-gradient(45deg, #00ffff, #ff00ff)' }}
        >
          🔗 CREATE SHAREABLE REPORTS
        </ExportButton>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1rem', marginTop: '1rem' }}>
        <ExportButton
          onClick={() => handleExport('detailed-performance')}
          disabled={isExporting}
          style={{ background: 'linear-gradient(45deg, #ff8000, #ff0080)' }}
        >
          {isExporting ? '🔄 GENERATING...' : '📊 DETAILED PERFORMANCE & WEIGHT EVOLUTION'}
        </ExportButton>
      </div>

      {isExporting && (
        <ProgressBar>
          <ProgressFill progress={exportProgress} />
        </ProgressBar>
      )}

      {statusMessage && (
        <StatusMessage type={statusMessage.type}>
          {statusMessage.message}
        </StatusMessage>
      )}

      {exportResult && (
        <div style={{ textAlign: 'center' }}>
          <DownloadLink
            href={exportResult.download_url}
            download={exportResult.filename}
          >
            📥 DOWNLOAD {exportResult.format.toUpperCase()} REPORT
          </DownloadLink>
          
          {/* Enhanced metadata display */}
          <div style={{ 
            background: 'rgba(0, 255, 255, 0.1)',
            border: '1px solid rgba(0, 255, 255, 0.3)',
            borderRadius: '5px',
            padding: '1rem',
            marginTop: '1rem',
            textAlign: 'left'
          }}>
            <div style={{ 
              color: '#00ffff', 
              fontFamily: 'Orbitron', 
              fontSize: '1rem',
              marginBottom: '0.5rem',
              textAlign: 'center'
            }}>
              📊 EXPORT METADATA
            </div>
            
            <div style={{ 
              color: '#ffffff', 
              fontFamily: 'Roboto Mono', 
              fontSize: '0.8rem',
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '0.5rem'
            }}>
              <div>File Size: {(exportResult.file_size_bytes / 1024 / 1024).toFixed(2)} MB</div>
              <div>Format: {exportResult.format.toUpperCase()}</div>
              
              {exportResult.metadata?.generation_time_seconds && (
                <div>Generation Time: {exportResult.metadata.generation_time_seconds.toFixed(1)}s</div>
              )}
              
              {exportResult.metadata?.data_points_included && (
                <div>Data Points: {exportResult.metadata.data_points_included.toLocaleString()}</div>
              )}
              
              {exportResult.metadata?.export_id && (
                <div>Export ID: {exportResult.metadata.export_id.slice(-8)}</div>
              )}
              
              {exportResult.metadata?.data_quality_score && (
                <div>Data Quality: {(exportResult.metadata.data_quality_score * 100).toFixed(1)}%</div>
              )}
            </div>
            
            {exportResult.metadata?.export_features && (
              <div style={{ marginTop: '0.5rem' }}>
                <div style={{ 
                  color: '#ff00ff', 
                  fontFamily: 'Roboto Mono', 
                  fontSize: '0.8rem',
                  marginBottom: '0.3rem'
                }}>
                  Features Enabled:
                </div>
                <div style={{ 
                  color: '#ffffff', 
                  fontFamily: 'Roboto Mono', 
                  fontSize: '0.7rem',
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '0.5rem'
                }}>
                  {exportResult.metadata.export_features.comprehensive_formatting && (
                    <span style={{ background: 'rgba(0, 255, 0, 0.2)', padding: '2px 6px', borderRadius: '3px' }}>
                      ✓ Comprehensive Formatting
                    </span>
                  )}
                  {exportResult.metadata.export_features.cyberpunk_styling && (
                    <span style={{ background: 'rgba(255, 0, 255, 0.2)', padding: '2px 6px', borderRadius: '3px' }}>
                      ✓ Cyberpunk Styling
                    </span>
                  )}
                  {exportResult.metadata.export_features.professional_pdf && (
                    <span style={{ background: 'rgba(255, 255, 0, 0.2)', padding: '2px 6px', borderRadius: '3px' }}>
                      ✓ Professional PDF
                    </span>
                  )}
                  {exportResult.metadata.export_features.advanced_excel && (
                    <span style={{ background: 'rgba(0, 255, 255, 0.2)', padding: '2px 6px', borderRadius: '3px' }}>
                      ✓ Advanced Excel
                    </span>
                  )}
                </div>
              </div>
            )}
            
            <div style={{ 
              color: '#00ffff', 
              fontFamily: 'Roboto Mono', 
              fontSize: '0.7rem', 
              marginTop: '0.5rem',
              textAlign: 'center',
              borderTop: '1px solid rgba(0, 255, 255, 0.3)',
              paddingTop: '0.5rem'
            }}>
              Expires: {new Date(exportResult.expires_at).toLocaleString()}
            </div>
          </div>
        </div>
      )}
    </ExportContainer>
  );
};

export default ExportDashboard;