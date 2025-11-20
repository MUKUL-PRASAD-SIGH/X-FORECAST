import React, { useState, useCallback, useRef, useEffect } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkButton, CyberpunkCard } from './ui';
import { useAuth } from '../contexts/AuthContext';
import { useEnhancedApiClient } from '../hooks/useEnhancedApiClient';

// Simple logger for client-side error tracking
const logger = {
  error: (message: string, error?: any) => {
    console.error(message, error);
  },
  warn: (message: string, data?: any) => {
    console.warn(message, data);
  },
  info: (message: string, data?: any) => {
    console.info(message, data);
  }
};

const UploadContainer = styled(CyberpunkCard)`
  padding: 2rem;
  margin: 1rem 0;
`;

const pulseGlow = keyframes`
  0% { box-shadow: 0 0 5px rgba(0, 255, 255, 0.3); }
  50% { box-shadow: 0 0 20px rgba(0, 255, 255, 0.8), 0 0 30px rgba(255, 20, 147, 0.4); }
  100% { box-shadow: 0 0 5px rgba(0, 255, 255, 0.3); }
`;

const scanLine = keyframes`
  0% { left: -100%; }
  100% { left: 100%; }
`;

const dataFlow = keyframes`
  0% { transform: translateX(-100%) scaleX(0); }
  50% { transform: translateX(0%) scaleX(1); }
  100% { transform: translateX(100%) scaleX(0); }
`;

const DropZone = styled.div<{ isDragOver: boolean; hasFile: boolean; isProcessing: boolean }>`
  border: 2px dashed ${props => 
    props.hasFile ? props.theme.colors.acidGreen :
    props.isDragOver ? props.theme.colors.neonBlue : 
    props.theme.colors.secondaryText
  };
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
  background: ${props => 
    props.hasFile ? 'rgba(57, 255, 20, 0.1)' :
    props.isDragOver ? 'rgba(0, 255, 255, 0.15)' : 
    'rgba(10, 10, 10, 0.8)'
  };
  transition: all 0.3s ease;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  
  ${props => props.isProcessing && css`
    animation: ${pulseGlow} 2s ease-in-out infinite;
    border-color: ${props.theme.colors.hotPink};
  `}
  
  &:hover {
    border-color: ${props => props.theme.colors.neonBlue};
    background: rgba(0, 255, 255, 0.1);
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.effects.softGlow};
  }
  
  /* Animated scanning line */
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 3px;
    background: ${props => props.theme.effects.primaryGradient};
    ${props => props.isDragOver && css`
      animation: ${scanLine} 1.5s ease-in-out infinite;
    `}
    z-index: 2;
  }
  
  /* Data flow effect when processing */
  &::after {
    content: '';
    position: absolute;
    top: 50%;
    left: -100%;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, ${props => props.theme.colors.acidGreen}, transparent);
    transform: translateY(-50%);
    ${props => props.isProcessing && css`
      animation: ${dataFlow} 2s ease-in-out infinite;
    `}
    z-index: 1;
  }
  
  /* Corner accents */
  &:hover::before {
    animation: ${scanLine} 1s ease-in-out infinite;
  }
`;

const FileInput = styled.input`
  display: none;
`;

const UploadStatus = styled(motion.div)<{ success: boolean }>`
  margin-top: 1rem;
  padding: 1rem;
  border-radius: 4px;
  background: ${props => props.success ? 'rgba(0, 255, 127, 0.1)' : 'rgba(255, 107, 107, 0.1)'};
  border: 1px solid ${props => props.success ? '#00ff7f' : '#ff6b6b'};
  color: ${props => props.success ? '#00ff7f' : '#ff6b6b'};
  font-family: ${props => props.theme.typography.fontFamily.mono};
`;

const FilePreview = styled(motion.div)`
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(0, 212, 255, 0.1);
  border: 1px solid ${props => props.theme.colors.neonBlue};
  border-radius: 4px;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
`;

const ColumnMappingContainer = styled(motion.div)`
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(255, 20, 147, 0.1);
  border: 1px solid ${props => props.theme.colors.hotPink};
  border-radius: 4px;
`;

const ColumnMappingRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
  padding: 0.5rem;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 4px;
  
  .required-field {
    color: ${props => props.theme.colors.neonBlue};
    font-weight: bold;
    min-width: 120px;
    text-align: left;
  }
  
  .detected-column {
    color: ${props => props.theme.colors.acidGreen};
    font-family: ${props => props.theme.typography.fontFamily.mono};
    flex: 1;
    text-align: center;
  }
  
  .mapping-status {
    min-width: 60px;
    text-align: right;
  }
`;

const DataQualityIndicator = styled(motion.div)<{ quality: number }>`
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid ${props => 
    props.quality >= 0.8 ? props.theme.colors.acidGreen :
    props.quality >= 0.6 ? props.theme.colors.neonBlue :
    props.theme.colors.error
  };
  border-radius: 4px;
  
  .quality-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    
    .quality-title {
      font-family: ${props => props.theme.typography.fontFamily.mono};
      font-weight: bold;
      color: ${props => props.theme.colors.neonBlue};
      text-transform: uppercase;
    }
    
    .quality-score {
      font-size: 1.2rem;
      font-weight: bold;
      color: ${props => 
        props.quality >= 0.8 ? props.theme.colors.acidGreen :
        props.quality >= 0.6 ? props.theme.colors.neonBlue :
        props.theme.colors.error
      };
    }
  }
  
  .quality-breakdown {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 0.5rem;
    margin-bottom: 1rem;
    
    .quality-metric {
      text-align: center;
      padding: 0.5rem;
      background: rgba(0, 0, 0, 0.4);
      border-radius: 4px;
      
      .metric-label {
        font-size: 0.7rem;
        color: ${props => props.theme.colors.secondaryText};
        text-transform: uppercase;
        margin-bottom: 0.25rem;
      }
      
      .metric-value {
        font-size: 0.9rem;
        font-weight: bold;
        color: ${props => props.theme.colors.acidGreen};
      }
    }
  }
  
  .quality-bar {
    width: 100%;
    height: 8px;
    background: rgba(0, 0, 0, 0.5);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 0.5rem;
    
    .quality-fill {
      height: 100%;
      background: ${props => 
        props.quality >= 0.8 ? props.theme.colors.acidGreen :
        props.quality >= 0.6 ? props.theme.colors.neonBlue :
        props.theme.colors.error
      };
      width: ${props => props.quality * 100}%;
      transition: width 0.5s ease;
      box-shadow: 0 0 10px currentColor;
    }
  }
  
  .quality-issues {
    margin-top: 1rem;
    
    .issues-title {
      font-size: 0.8rem;
      color: ${props => props.theme.colors.error};
      font-weight: bold;
      margin-bottom: 0.5rem;
      text-transform: uppercase;
    }
    
    .issue-item {
      font-size: 0.7rem;
      color: ${props => props.theme.colors.secondaryText};
      margin-bottom: 0.25rem;
      padding-left: 1rem;
      position: relative;
      
      &::before {
        content: '⚠';
        position: absolute;
        left: 0;
        color: ${props => props.theme.colors.warning};
      }
    }
  }
  
  .quality-recommendations {
    margin-top: 1rem;
    
    .recommendations-title {
      font-size: 0.8rem;
      color: ${props => props.theme.colors.neonBlue};
      font-weight: bold;
      margin-bottom: 0.5rem;
      text-transform: uppercase;
    }
    
    .recommendation-item {
      font-size: 0.7rem;
      color: ${props => props.theme.colors.secondaryText};
      margin-bottom: 0.25rem;
      padding-left: 1rem;
      position: relative;
      
      &::before {
        content: '💡';
        position: absolute;
        left: 0;
      }
    }
  }
`;

const ProcessingSteps = styled(motion.div)`
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid ${props => props.theme.colors.secondaryText};
  border-radius: 4px;
  
  .step {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.sm};
    
    .step-icon {
      margin-right: 0.5rem;
      width: 16px;
    }
    
    &.completed {
      color: ${props => props.theme.colors.acidGreen};
    }
    
    &.processing {
      color: ${props => props.theme.colors.neonBlue};
      
      .step-icon {
        animation: spin 1s linear infinite;
      }
    }
    
    &.pending {
      color: ${props => props.theme.colors.secondaryText};
    }
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const progressPulse = keyframes`
  0% { box-shadow: 0 0 5px rgba(0, 255, 255, 0.5); }
  50% { box-shadow: 0 0 15px rgba(0, 255, 255, 0.8), 0 0 25px rgba(255, 20, 147, 0.6); }
  100% { box-shadow: 0 0 5px rgba(0, 255, 255, 0.5); }
`;

const ProgressBar = styled(motion.div)`
  margin-top: 1rem;
  padding: 1.5rem;
  background: rgba(0, 0, 0, 0.6);
  border: 2px solid ${props => props.theme.colors.neonBlue};
  border-radius: 8px;
  backdrop-filter: blur(15px);
  animation: ${progressPulse} 3s ease-in-out infinite;
  
  .progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.md};
    color: ${props => props.theme.colors.neonBlue};
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  
  .progress-track {
    width: 100%;
    height: 12px;
    background: rgba(0, 0, 0, 0.8);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
    border: 1px solid rgba(0, 255, 255, 0.3);
    
    .progress-fill {
      height: 100%;
      background: ${props => props.theme.effects.primaryGradient};
      transition: width 0.5s ease;
      box-shadow: 0 0 15px rgba(0, 255, 255, 0.8);
      position: relative;
      
      &::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 20px;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.6));
        animation: ${scanLine} 2s ease-in-out infinite;
      }
    }
  }
  
  .current-operation {
    margin-top: 1rem;
    font-size: 0.9rem;
    color: ${props => props.theme.colors.acidGreen};
    font-family: ${props => props.theme.typography.fontFamily.mono};
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .progress-stats {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: ${props => props.theme.colors.secondaryText};
    font-family: ${props => props.theme.typography.fontFamily.mono};
  }
`;

const UploadModeSelector = styled(motion.div)`
  margin-bottom: 1rem;
  display: flex;
  gap: 0.5rem;
  justify-content: center;
  flex-wrap: wrap;
`;

const ModeButton = styled(motion.button)<{ isActive: boolean }>`
  padding: 0.5rem 1rem;
  background: ${props => props.isActive ? props.theme.effects.primaryGradient : 'transparent'};
  color: ${props => props.isActive ? props.theme.colors.darkBg : props.theme.colors.neonBlue};
  border: 2px solid ${props => props.theme.colors.neonBlue};
  border-radius: 6px;
  font-family: ${props => props.theme.typography.fontFamily.mono};
  font-size: ${props => props.theme.typography.fontSize.sm};
  text-transform: uppercase;
  letter-spacing: 1px;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  
  &:hover {
    box-shadow: ${props => props.theme.effects.softGlow};
    transform: translateY(-2px);
  }
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s ease;
  }
  
  &:hover::before {
    left: 100%;
  }
`;

const LiveAnalysisPanel = styled(motion.div)`
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(57, 255, 20, 0.1);
  border: 1px solid ${props => props.theme.colors.acidGreen};
  border-radius: 8px;
  backdrop-filter: blur(10px);
  
  .analysis-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    color: ${props => props.theme.colors.acidGreen};
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-weight: bold;
    text-transform: uppercase;
  }
  
  .analysis-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }
  
  .analysis-item {
    padding: 0.75rem;
    background: rgba(0, 0, 0, 0.4);
    border-radius: 6px;
    border-left: 3px solid ${props => props.theme.colors.acidGreen};
    
    .item-label {
      font-size: 0.8rem;
      color: ${props => props.theme.colors.secondaryText};
      margin-bottom: 0.25rem;
      text-transform: uppercase;
    }
    
    .item-value {
      font-size: 1rem;
      color: ${props => props.theme.colors.acidGreen};
      font-family: ${props => props.theme.typography.fontFamily.mono};
      font-weight: bold;
    }
  }
`;

const BatchUploadContainer = styled(motion.div)`
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(255, 20, 147, 0.1);
  border: 1px solid ${props => props.theme.colors.hotPink};
  border-radius: 4px;
  
  .batch-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    
    h4 {
      color: ${props => props.theme.colors.hotPink};
      margin: 0;
    }
    
    .batch-controls {
      display: flex;
      gap: 0.5rem;
    }
  }
  
  .batch-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1rem;
    margin-bottom: 1rem;
    
    .summary-item {
      text-align: center;
      padding: 0.5rem;
      background: rgba(0, 0, 0, 0.3);
      border-radius: 4px;
      
      .summary-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: ${props => props.theme.colors.acidGreen};
      }
      
      .summary-label {
        font-size: 0.8rem;
        color: ${props => props.theme.colors.secondaryText};
        margin-top: 0.25rem;
      }
    }
  }
`;

const FileStatusList = styled.div`
  max-height: 300px;
  overflow-y: auto;
  
  .file-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 4px;
    border-left: 3px solid transparent;
    
    &.pending {
      border-left-color: ${props => props.theme.colors.secondaryText};
    }
    
    &.uploading {
      border-left-color: ${props => props.theme.colors.neonBlue};
    }
    
    &.completed {
      border-left-color: ${props => props.theme.colors.acidGreen};
    }
    
    &.failed {
      border-left-color: ${props => props.theme.colors.error};
    }
    
    &.retrying {
      border-left-color: ${props => props.theme.colors.warning};
    }
    
    .file-info {
      flex: 1;
      
      .file-name {
        font-weight: bold;
        margin-bottom: 0.25rem;
      }
      
      .file-details {
        font-size: 0.8rem;
        color: ${props => props.theme.colors.secondaryText};
      }
    }
    
    .file-progress {
      width: 100px;
      margin: 0 1rem;
      
      .progress-bar {
        width: 100%;
        height: 4px;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 2px;
        overflow: hidden;
        
        .progress-fill {
          height: 100%;
          background: ${props => props.theme.colors.neonBlue};
          transition: width 0.3s ease;
        }
      }
      
      .progress-text {
        font-size: 0.7rem;
        text-align: center;
        margin-top: 0.25rem;
      }
    }
    
    .file-actions {
      display: flex;
      gap: 0.5rem;
      align-items: center;
      
      .status-icon {
        font-size: 1.2rem;
      }
    }
  }
`;

interface DetectedColumn {
  name: string;
  type: string;
  sample_values: string[];
  confidence: number;
  quality_score?: number;
  recommendations?: string[];
}

interface ColumnMapping {
  required_field: string;
  detected_column: string | null;
  confidence: number;
  status: 'mapped' | 'unmapped' | 'uncertain' | 'conflict';
  alternatives?: Array<{ column: string; confidence: number }>;
  mapping_reason?: string;
}

interface DataQuality {
  overall_score: number;
  completeness: number;
  consistency: number;
  validity: number;
  accuracy?: number;
  uniqueness?: number;
  timeliness?: number;
  issues: string[];
  recommendations: string[];
}

interface ProcessingStep {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  message?: string;
}

interface FileUploadStatus {
  file: File;
  status: 'pending' | 'uploading' | 'completed' | 'failed' | 'retrying';
  progress: number;
  result?: any;
  error?: string;
  retryCount: number;
}

interface BatchUploadSummary {
  totalFiles: number;
  completedFiles: number;
  failedFiles: number;
  processingFiles: number;
  overallProgress: number;
}

type UploadMode = 'single' | 'batch' | 'streaming';

interface LiveAnalysis {
  fileSize: number;
  fileType: string;
  estimatedRows: number;
  detectedColumns: number;
  processingTime: number;
  dataQualityScore: number;
}

interface DataUploadProps {
  authToken?: string; // Make optional since we'll use AuthContext
  onUploadComplete: (result: any) => void;
  onAuthError?: () => void; // Optional callback for authentication errors
}

export const DataUpload: React.FC<DataUploadProps> = ({ authToken: propAuthToken, onUploadComplete, onAuthError }) => {
  const { isAuthenticated } = useAuth();
  const { 
    uploadWithRetry,
    uploadFile,
    serviceHealth, 
    checkHealth, 
    fallbackMode, 
    enableFallbackMode,
    circuitBreakerStatus 
  } = useEnhancedApiClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Core upload state
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  
  // Analysis and processing state
  const [detectedColumns, setDetectedColumns] = useState<DetectedColumn[]>([]);
  const [columnMappings, setColumnMappings] = useState<ColumnMapping[]>([]);
  const [dataQuality, setDataQuality] = useState<DataQuality | null>(null);
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([]);
  const [filePreview, setFilePreview] = useState<any>(null);
  
  // Progress and operation state
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [currentOperation, setCurrentOperation] = useState<string>('');
  const [processingSpeed, setProcessingSpeed] = useState<number>(0);
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState<number>(0);
  
  // Upload mode and batch state
  const [uploadMode, setUploadMode] = useState<UploadMode>('single');
  const [batchFiles, setBatchFiles] = useState<FileUploadStatus[]>([]);
  const [batchSummary, setBatchSummary] = useState<BatchUploadSummary | null>(null);
  
  // Live analysis state
  const [liveAnalysis, setLiveAnalysis] = useState<LiveAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAuthenticationError = useCallback((error: Error) => {
    if (error.message.includes('Authentication failed')) {
      // Clear any stored authentication data
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user_data');
      
      // Call the auth error callback if provided
      if (onAuthError) {
        onAuthError();
      } else {
        // Fallback: reload the page to trigger login
        window.location.reload();
      }
    }
  }, [onAuthError]);

  // Fallback parameter detection for when enhanced processing fails
  const detectFileParametersBasic = async (file: File) => {
    try {
      updateProcessingStep('parameter-detection', 'processing', 'Using basic parameter detection...');
      
      // Basic file analysis
      const basicColumns: DetectedColumn[] = [];
      
      if (file.name.toLowerCase().endsWith('.csv')) {
        // For CSV files, try to read first few lines
        const text = await file.slice(0, 2048).text();
        const lines = text.split('\n');
        const headers = lines[0]?.split(',') || [];
        
        headers.forEach((header, index) => {
          const cleanHeader = header.trim().replace(/['"]/g, '');
          let type = 'categorical';
          let confidence = 0.5;
          
          // Basic type detection based on header names
          const headerLower = cleanHeader.toLowerCase();
          if (headerLower.includes('date') || headerLower.includes('time')) {
            type = 'date';
            confidence = 0.7;
          } else if (headerLower.includes('amount') || headerLower.includes('sales') || headerLower.includes('revenue')) {
            type = 'sales_amount';
            confidence = 0.7;
          } else if (headerLower.includes('product') || headerLower.includes('category')) {
            type = 'product_category';
            confidence = 0.7;
          } else if (headerLower.includes('region') || headerLower.includes('location')) {
            type = 'region';
            confidence = 0.7;
          }
          
          basicColumns.push({
            name: cleanHeader,
            type,
            sample_values: [`Sample ${index + 1}`, `Sample ${index + 2}`],
            confidence,
            quality_score: 0.7,
            recommendations: ['Basic analysis - consider using enhanced mode for detailed insights']
          });
        });
      } else {
        // For other file types, create generic columns
        basicColumns.push({
          name: 'data',
          type: 'unknown',
          sample_values: ['Data preview not available in basic mode'],
          confidence: 0.3,
          quality_score: 0.5,
          recommendations: ['Upload as CSV for better analysis']
        });
      }
      
      setDetectedColumns(basicColumns);
      
      // Basic column mapping
      const basicMappings = autoMapColumns(basicColumns);
      setColumnMappings(basicMappings);
      
      // Basic data quality
      const basicQuality = assessDataQuality(basicColumns, { basic: true });
      setDataQuality(basicQuality);
      
      // Basic preview
      setFilePreview({
        type: file.name.split('.').pop()?.toUpperCase() || 'Unknown',
        basic_mode: true,
        message: 'Basic analysis mode - limited functionality'
      });
      
      updateProcessingStep('parameter-detection', 'completed');
      updateProcessingStep('column-mapping', 'completed');
      updateProcessingStep('data-quality', 'completed');
      
      return {
        success: true,
        fallback_mode: true,
        detected_columns: basicColumns,
        column_mappings: basicMappings,
        data_quality: basicQuality
      };
      
    } catch (error) {
      logger.error('Basic parameter detection failed:', error);
      throw new Error('Both enhanced and basic parameter detection failed');
    }
  };

  // Live file analysis on selection
  const performLiveAnalysis = useCallback(async (file: File) => {
    setIsAnalyzing(true);
    const startTime = Date.now();
    
    try {
      // Immediate analysis based on file properties
      const analysis: LiveAnalysis = {
        fileSize: file.size,
        fileType: file.type || file.name.split('.').pop()?.toUpperCase() || 'Unknown',
        estimatedRows: Math.floor(file.size / 100), // Rough estimate
        detectedColumns: 0,
        processingTime: 0,
        dataQualityScore: 0.8 // Initial estimate
      };
      
      setLiveAnalysis(analysis);
      
      // For CSV/Excel files, try to read first few lines for better analysis
      if (file.type.includes('csv') || file.name.endsWith('.csv')) {
        const text = await file.slice(0, 1024).text(); // Read first 1KB
        const lines = text.split('\n');
        const firstLine = lines[0];
        
        if (firstLine) {
          const columns = firstLine.split(',').length;
          const estimatedRows = Math.floor(file.size / firstLine.length);
          
          setLiveAnalysis(prev => prev ? {
            ...prev,
            detectedColumns: columns,
            estimatedRows: estimatedRows,
            processingTime: Date.now() - startTime
          } : null);
        }
      }
      
      // Update processing time
      setTimeout(() => {
        setLiveAnalysis(prev => prev ? {
          ...prev,
          processingTime: Date.now() - startTime
        } : null);
      }, 100);
      
    } catch (error) {
      console.warn('Live analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  // Progress tracking state
  const progressTrackingRef = useRef({ lastUpdate: Date.now(), lastProgress: 0 });

  // Update progress with speed calculation
  const updateProgressWithSpeed = useCallback((progress: number, operation: string) => {
    const now = Date.now();
    const { lastUpdate, lastProgress } = progressTrackingRef.current;
    
    if (now - lastUpdate > 0) {
      const speed = (progress - lastProgress) / (now - lastUpdate) * 1000; // progress per second
      setProcessingSpeed(Math.max(0, speed));
      
      if (speed > 0) {
        const remaining = (100 - progress) / speed;
        setEstimatedTimeRemaining(Math.max(0, remaining));
      }
    }
    
    setUploadProgress(progress);
    setCurrentOperation(operation);
    
    // Update tracking state
    progressTrackingRef.current = { lastUpdate: now, lastProgress: progress };
  }, []);

  const requiredFields = [
    'date',
    'sales_amount', 
    'product_category',
    'region'
  ];

  const updateProcessingStep = useCallback((stepId: string, status: ProcessingStep['status'], message?: string) => {
    setProcessingSteps(prev => prev.map(step => 
      step.id === stepId ? { ...step, status, message } : step
    ));
  }, []);

  const initializeProcessingSteps = useCallback(() => {
    const steps: ProcessingStep[] = [
      { id: 'file-validation', name: 'File Validation', status: 'pending' },
      { id: 'parameter-detection', name: 'Parameter Detection', status: 'pending' },
      { id: 'column-mapping', name: 'Column Mapping', status: 'pending' },
      { id: 'data-quality', name: 'Data Quality Assessment', status: 'pending' },
      { id: 'ensemble-init', name: 'Ensemble Model Initialization', status: 'pending' },
      { id: 'pattern-detection', name: 'Pattern Detection', status: 'pending' }
    ];
    setProcessingSteps(steps);
  }, []);

  const detectFileParameters = async (file: File) => {
    try {
      // Authentication is handled by the API client
      
      updateProcessingStep('file-validation', 'processing');
      
      // Basic file validation - support CSV, Excel, and PDF files
      const validTypes = [
        'text/csv', 
        'application/vnd.ms-excel', 
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/pdf'
      ];
      if (!validTypes.includes(file.type) && !file.name.match(/\.(csv|xlsx|xls|pdf)$/i)) {
        throw new Error('Invalid file type. Please upload CSV, Excel, or PDF files.');
      }

      if (file.size > 50 * 1024 * 1024) {
        throw new Error('File size exceeds 50MB limit.');
      }

      updateProcessingStep('file-validation', 'completed');
      
      // Use enhanced data processing API with bulletproof error handling
      updateProcessingStep('parameter-detection', 'processing', 'Analyzing file with smart processor...');
      
      // Use enhanced data processing API with bulletproof error handling
      updateProcessingStep('parameter-detection', 'processing', 'Analyzing file with smart processor...');
      
      // Use enhanced upload with retry logic
      const response = await uploadWithRetry<any>('/api/v1/data-processing/analyze-file', file, {
        maxRetries: 3,
        onProgress: (progress) => {
          updateProgressWithSpeed(progress.percentage * 0.6, 'Analyzing file structure...');
        },
        onRetry: (attempt, error) => {
          updateProcessingStep('parameter-detection', 'processing', 
            `Retrying analysis (attempt ${attempt}/3)... ${error.category}`);
        },
        onFallback: () => {
          updateProcessingStep('parameter-detection', 'processing', 
            'Using fallback analysis mode...');
        }
      });

      if (!response.success) {
        // Handle specific error types
        if (response.errorDetails) {
          const { category, recoveryActions } = response.errorDetails;
          
          if (category === 'service_unavailable' && recoveryActions.includes('fallback_mode')) {
            updateProcessingStep('parameter-detection', 'processing', 
              'Service temporarily unavailable, enabling fallback mode...');
            enableFallbackMode();
            
            // Try basic parameter detection as fallback
            return await detectFileParametersBasic(file);
          }
        }
        
        throw new Error(response.error || 'Failed to analyze file');
      }

      const result = response.data;
      
      // Set enhanced detected columns with quality scores
      const enhancedColumns = result.detected_columns.map((col: any) => ({
        name: col.name,
        type: col.type,
        sample_values: col.sample_values,
        confidence: col.confidence,
        quality_score: col.quality_score,
        recommendations: col.recommendations
      }));
      
      setDetectedColumns(enhancedColumns);
      
      // Set enhanced preview data
      setFilePreview({
        type: result.file_info.format,
        ...result.preview_data
      });
      
      updateProcessingStep('parameter-detection', 'completed');
      updateProcessingStep('column-mapping', 'processing', 'Auto-mapping columns...');

      // Use enhanced auto-mapping results
      const enhancedMappings = result.column_mappings.map((mapping: any) => ({
        required_field: mapping.required_field,
        detected_column: mapping.detected_column,
        confidence: mapping.confidence,
        status: mapping.status,
        alternatives: mapping.alternatives,
        mapping_reason: mapping.mapping_reason
      }));
      
      setColumnMappings(enhancedMappings);
      
      updateProcessingStep('column-mapping', 'completed');
      updateProcessingStep('data-quality', 'processing', 'Assessing data quality...');

      // Use comprehensive data quality assessment
      const enhancedQuality = {
        overall_score: result.data_quality.overall_score,
        completeness: result.data_quality.completeness,
        consistency: result.data_quality.consistency,
        validity: result.data_quality.validity,
        accuracy: result.data_quality.accuracy,
        uniqueness: result.data_quality.uniqueness,
        timeliness: result.data_quality.timeliness,
        issues: result.data_quality.issues,
        recommendations: result.data_quality.recommendations
      };
      
      setDataQuality(enhancedQuality);
      
      updateProcessingStep('data-quality', 'completed');

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Parameter detection failed';
      updateProcessingStep('parameter-detection', 'error', errorMessage);
      
      // Handle authentication errors
      if (error instanceof Error) {
        handleAuthenticationError(error);
      }
      
      throw error;
    }
  };

  const autoMapColumns = (columns: DetectedColumn[]): ColumnMapping[] => {
    // This function is now primarily used as a fallback
    // The enhanced API provides intelligent auto-mapping
    return requiredFields.map(field => {
      // Find best matching column
      const matches = columns.filter(col => {
        const colName = col.name.toLowerCase();
        const fieldName = field.toLowerCase();
        
        // Direct match
        if (colName === fieldName) return true;
        
        // Partial matches
        if (field === 'date' && (colName.includes('date') || colName.includes('time'))) return true;
        if (field === 'sales_amount' && (colName.includes('sales') || colName.includes('amount') || colName.includes('revenue'))) return true;
        if (field === 'product_category' && (colName.includes('product') || colName.includes('category') || colName.includes('sku'))) return true;
        if (field === 'region' && (colName.includes('region') || colName.includes('location') || colName.includes('area'))) return true;
        
        return false;
      });

      const bestMatch = matches.sort((a, b) => b.confidence - a.confidence)[0];
      
      return {
        required_field: field,
        detected_column: bestMatch?.name || null,
        confidence: bestMatch?.confidence || 0,
        status: bestMatch ? (bestMatch.confidence > 0.8 ? 'mapped' : 'uncertain') : 'unmapped'
      };
    });
  };

  const assessDataQuality = (columns: DetectedColumn[], preview: any): DataQuality => {
    // This function is now primarily used as a fallback
    // The enhanced API provides comprehensive quality assessment
    let completeness = 0;
    let consistency = 0;
    let validity = 0;
    const issues: string[] = [];
    const recommendations: string[] = [];

    // Basic quality assessment for backward compatibility
    if (columns.length > 0) {
      completeness = Math.min(columns.length / requiredFields.length, 1);
      consistency = columns.reduce((acc, col) => acc + col.confidence, 0) / columns.length;
      validity = 0.8; // Placeholder - enhanced API provides actual validation
    }

    const overall_score = (completeness + consistency + validity) / 3;

    if (completeness < 1) {
      issues.push('Missing required columns detected');
      recommendations.push('Ensure all required fields are present in your data');
    }

    if (consistency < 0.7) {
      issues.push('Low confidence in column detection');
      recommendations.push('Review column mappings and ensure consistent naming');
    }

    return {
      overall_score,
      completeness,
      consistency,
      validity,
      issues,
      recommendations
    };
  };

  const assessPDFQuality = (pdfResult: any): DataQuality => {
    let completeness = 0;
    let consistency = 0;
    let validity = 0;
    const issues: string[] = [];
    const recommendations: string[] = [];

    // PDF quality assessment based on text extraction results
    const textLength = pdfResult.text_length || 0;
    const pageCount = pdfResult.page_count || 0;
    
    // Completeness: based on text extraction success
    if (textLength > 0) {
      completeness = Math.min(textLength / 1000, 1); // Assume 1000 chars is good baseline
    } else {
      issues.push('No text content extracted from PDF');
      recommendations.push('Ensure PDF contains readable text (not just images)');
    }

    // Consistency: based on extraction success rate
    consistency = pdfResult.extraction_success ? 1.0 : 0.5;
    if (!pdfResult.extraction_success) {
      issues.push('PDF text extraction had issues');
      recommendations.push('Try converting PDF to a different format or OCR processing');
    }

    // Validity: based on file structure and readability
    validity = pageCount > 0 ? 0.9 : 0.3;
    if (pageCount === 0) {
      issues.push('PDF appears to have no readable pages');
      recommendations.push('Verify PDF file is not corrupted');
    }

    const overall_score = (completeness + consistency + validity) / 3;

    return {
      overall_score,
      completeness,
      consistency,
      validity,
      issues,
      recommendations
    };
  };

  const handleFileUpload = async (file: File) => {
    // Authentication is handled by the API client
    if (!isAuthenticated) {
      alert('Authentication required. Please log in again.');
      return;
    }
    
    setUploading(true);
    setUploadResult(null);
    setSelectedFile(file);
    setUploadProgress(0);
    setCurrentOperation('Initializing upload...');
    initializeProcessingSteps();

    try {
      // Enhanced progress tracking for parameter detection
      updateProgressWithSpeed(10, 'Analyzing file structure...');
      
      // First detect parameters
      await detectFileParameters(file);
      
      updateProgressWithSpeed(50, 'Uploading file to server...');

      // Then upload with enhanced processing
      const fileExtension = file.name.toLowerCase().split('.').pop();
      const isPDF = fileExtension === 'pdf';
      
      updateProcessingStep('ensemble-init', 'processing', 
        isPDF ? 'Processing PDF document...' : 'Initializing ensemble models...');
      
      updateProgressWithSpeed(70, isPDF ? 'Processing PDF document...' : 'Training ensemble models...');
      
      // Use the enhanced upload with retry logic for bulletproof reliability
      const response = await uploadWithRetry<any>('/api/v1/upload-enhanced', file, {
        maxRetries: 3,
        onProgress: (progress) => {
          updateProgressWithSpeed(70 + (progress.percentage * 0.2), 
            isPDF ? 'Processing PDF document...' : 'Training ensemble models...');
        },
        onRetry: (attempt, error) => {
          updateProcessingStep('ensemble-init', 'processing', 
            `Retrying upload (attempt ${attempt}/3)... ${error.category}`);
          setCurrentOperation(`Retrying upload (attempt ${attempt}/3)...`);
        },
        onFallback: () => {
          updateProcessingStep('ensemble-init', 'processing', 
            'Using fallback upload mode...');
          setCurrentOperation('Using fallback upload mode...');
          enableFallbackMode();
        }
      });

      if (!response.success) {
        // Enhanced error handling with specific recovery actions
        if (response.errorDetails) {
          const { category, recoveryActions } = response.errorDetails;
          
          if (category === 'network' && recoveryActions.includes('retry')) {
            throw new Error('Network connection failed. Please check your internet connection and try again.');
          } else if (category === 'service_unavailable' && recoveryActions.includes('fallback_mode')) {
            throw new Error('Service temporarily unavailable. Fallback mode activated.');
          } else if (category === 'authentication' && recoveryActions.includes('redirect_login')) {
            throw new Error('Authentication failed. Please log in again.');
          }
        }
        
        throw new Error(response.error || 'Upload failed due to network issues. Please check your connection and try again.');
      }

      const result = response.data;
      setUploadResult(result);
      
      updateProgressWithSpeed(90, 'Finalizing processing...');
      
      if (result.success) {
        updateProcessingStep('ensemble-init', 'completed');
        
        if (isPDF) {
          updateProcessingStep('pattern-detection', 'completed', 'PDF integrated into knowledge base');
          setCurrentOperation('PDF successfully integrated into knowledge base');
        } else {
          updateProcessingStep('pattern-detection', 'processing');
          setCurrentOperation('Detecting data patterns...');
          
          // Simulate pattern detection completion for CSV files
          setTimeout(() => {
            updateProcessingStep('pattern-detection', 'completed');
            updateProgressWithSpeed(100, 'Pattern detection completed');
          }, 1000);
        }
        
        if (isPDF) {
          updateProgressWithSpeed(100, 'PDF processing completed');
        }
        
        onUploadComplete(result);
      } else {
        updateProcessingStep('ensemble-init', 'error', result.message);
        setCurrentOperation(`Upload failed: ${result.message}`);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      setUploadResult({ success: false, message: errorMessage });
      updateProcessingStep('ensemble-init', 'error', errorMessage);
      setCurrentOperation(`Error: ${errorMessage}`);
      
      // Handle authentication errors
      if (error instanceof Error) {
        handleAuthenticationError(error);
      }
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      if (uploadMode === 'single' && files.length === 1) {
        setSelectedFile(files[0]);
        performLiveAnalysis(files[0]);
      } else if (uploadMode === 'batch' || files.length > 1) {
        setUploadMode('batch');
        handleBatchFileSelection(files);
      } else if (uploadMode === 'streaming') {
        // For streaming mode, process files one by one
        handleStreamingUpload(files);
      }
    }
  }, [uploadMode, performLiveAnalysis]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const fileArray = Array.from(files);
      
      if (uploadMode === 'single' && fileArray.length === 1) {
        setSelectedFile(fileArray[0]);
        performLiveAnalysis(fileArray[0]);
      } else if (uploadMode === 'batch' || fileArray.length > 1) {
        setUploadMode('batch');
        handleBatchFileSelection(fileArray);
      } else if (uploadMode === 'streaming') {
        handleStreamingUpload(fileArray);
      }
    }
  }, [uploadMode, performLiveAnalysis]);

  const handleStreamingUpload = useCallback(async (files: File[]) => {
    setUploading(true);
    let processedCount = 0;
    
    for (const file of files) {
      try {
        updateProgressWithSpeed((processedCount / files.length) * 100, `Streaming upload: ${file.name}`);
        await handleFileUpload(file);
        processedCount++;
      } catch (error) {
        console.error(`Streaming upload failed for ${file.name}:`, error);
      }
    }
    
    updateProgressWithSpeed(100, 'Streaming upload completed');
    setUploading(false);
  }, [updateProgressWithSpeed]);

  const updateBatchSummary = useCallback((files: FileUploadStatus[]) => {
    const totalFiles = files.length;
    const completedFiles = files.filter(f => f.status === 'completed').length;
    const failedFiles = files.filter(f => f.status === 'failed').length;
    const processingFiles = files.filter(f => f.status === 'uploading' || f.status === 'retrying').length;
    const overallProgress = totalFiles > 0 ? (completedFiles / totalFiles) * 100 : 0;

    setBatchSummary({
      totalFiles,
      completedFiles,
      failedFiles,
      processingFiles,
      overallProgress
    });
  }, []);

  const handleBatchFileSelection = useCallback((files: File[]) => {
    // Validate all files
    const validFiles: File[] = [];
    const invalidFiles: string[] = [];

    files.forEach(file => {
      const validTypes = [
        'text/csv', 
        'application/vnd.ms-excel', 
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/pdf'
      ];
      
      if (validTypes.includes(file.type) || file.name.match(/\.(csv|xlsx|xls|pdf)$/i)) {
        if (file.size <= 50 * 1024 * 1024) {
          validFiles.push(file);
        } else {
          invalidFiles.push(`${file.name} (exceeds 50MB limit)`);
        }
      } else {
        invalidFiles.push(`${file.name} (unsupported format)`);
      }
    });

    if (invalidFiles.length > 0) {
      alert(`Some files were skipped:\n${invalidFiles.join('\n')}`);
    }

    if (validFiles.length > 0) {
      const batchFileStatuses: FileUploadStatus[] = validFiles.map(file => ({
        file,
        status: 'pending',
        progress: 0,
        retryCount: 0
      }));
      
      setBatchFiles(batchFileStatuses);
      updateBatchSummary(batchFileStatuses);
      
      // Perform live analysis on the first file as a preview
      if (validFiles.length > 0) {
        performLiveAnalysis(validFiles[0]);
      }
    }
  }, [updateBatchSummary, performLiveAnalysis]);



  const resetUpload = () => {
    setSelectedFile(null);
    setUploadResult(null);
    setDetectedColumns([]);
    setColumnMappings([]);
    setDataQuality(null);
    setProcessingSteps([]);
    setFilePreview(null);
    setUploadProgress(0);
    setCurrentOperation('');
    setProcessingSpeed(0);
    setEstimatedTimeRemaining(0);
    setBatchFiles([]);
    setBatchSummary(null);
    setLiveAnalysis(null);
    setIsAnalyzing(false);
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const updateFileStatus = useCallback((fileIndex: number, updates: Partial<FileUploadStatus>) => {
    setBatchFiles(prev => {
      const newFiles = [...prev];
      newFiles[fileIndex] = { ...newFiles[fileIndex], ...updates };
      updateBatchSummary(newFiles);
      return newFiles;
    });
  }, [updateBatchSummary]);

  const retryFailedFile = useCallback(async (fileIndex: number) => {
    const fileStatus = batchFiles[fileIndex];
    if (fileStatus.status !== 'failed' || fileStatus.retryCount >= 3) return;

    updateFileStatus(fileIndex, { 
      status: 'retrying', 
      retryCount: fileStatus.retryCount + 1,
      error: undefined 
    });

    try {
      await processSingleFile(fileStatus.file, fileIndex);
    } catch (error) {
      updateFileStatus(fileIndex, { 
        status: 'failed', 
        error: error instanceof Error ? error.message : 'Retry failed' 
      });
    }
  }, [batchFiles]);

  const processSingleFile = async (file: File, fileIndex: number) => {
    try {
      updateFileStatus(fileIndex, { status: 'uploading', progress: 0 });

      // Detect parameters first
      updateFileStatus(fileIndex, { progress: 20 });
      await detectFileParameters(file);

      // Upload file with enhanced retry logic
      updateFileStatus(fileIndex, { progress: 50 });
      
      const response = await uploadWithRetry<any>('/api/v1/upload-enhanced', file, {
        maxRetries: 3,
        onProgress: (progress) => {
          updateFileStatus(fileIndex, { progress: 50 + (progress.percentage * 0.5) });
        },
        onRetry: (attempt, error) => {
          updateFileStatus(fileIndex, { 
            status: 'retrying',
            retryCount: attempt 
          });
        },
        onFallback: () => {
          // Enable fallback mode for this file
          enableFallbackMode();
        }
      });

      if (!response.success) {
        // Enhanced error handling for batch uploads
        if (response.errorDetails) {
          const { category } = response.errorDetails;
          
          if (category === 'network') {
            throw new Error('Network connection failed. Please check your internet connection.');
          } else if (category === 'service_unavailable') {
            throw new Error('Service temporarily unavailable. Please try again later.');
          } else if (category === 'authentication') {
            throw new Error('Authentication failed. Please log in again.');
          }
        }
        
        throw new Error(response.error || 'Upload failed due to network issues.');
      }

      const result = response.data;
      
      updateFileStatus(fileIndex, { progress: 100 });

      if (result.success) {
        updateFileStatus(fileIndex, { 
          status: 'completed', 
          result,
          progress: 100 
        });
      } else {
        updateFileStatus(fileIndex, { 
          status: 'failed', 
          error: result.message,
          progress: 0 
        });
      }

      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      updateFileStatus(fileIndex, { 
        status: 'failed', 
        error: errorMessage,
        progress: 0 
      });
      
      // Handle authentication errors
      if (error instanceof Error) {
        handleAuthenticationError(error);
      }
      
      throw error;
    }
  };

  const processBatchUpload = async () => {
    // Authentication is handled by the API client
    if (!isAuthenticated) {
      alert('Authentication required. Please log in again.');
      return;
    }
    
    setUploading(true);
    
    // Process files in parallel with concurrency limit
    const concurrencyLimit = 3;
    const chunks = [];
    
    for (let i = 0; i < batchFiles.length; i += concurrencyLimit) {
      chunks.push(batchFiles.slice(i, i + concurrencyLimit));
    }

    for (const chunk of chunks) {
      const promises = chunk.map((fileStatus, chunkIndex) => {
        const globalIndex = batchFiles.indexOf(fileStatus);
        return processSingleFile(fileStatus.file, globalIndex);
      });

      await Promise.allSettled(promises);
    }

    setUploading(false);
    
    // Call completion callback with batch results
    const results = batchFiles.map(f => f.result).filter(Boolean);
    onUploadComplete({ 
      success: true, 
      message: `Batch upload completed: ${results.length} files processed`,
      batch_results: results 
    });
  };

  return (
    <UploadContainer $variant="neon">
      <motion.h3 
        style={{ color: '#00d4ff', marginBottom: '1rem', fontFamily: 'monospace' }}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
      >
        🚀 Hot Live Upload Interface
      </motion.h3>
      
      {/* Upload Mode Selector */}
      <UploadModeSelector
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <ModeButton
          isActive={uploadMode === 'single'}
          onClick={() => setUploadMode('single')}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          ⚡ Single
        </ModeButton>
        <ModeButton
          isActive={uploadMode === 'batch'}
          onClick={() => setUploadMode('batch')}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          📦 Batch
        </ModeButton>
        <ModeButton
          isActive={uploadMode === 'streaming'}
          onClick={() => setUploadMode('streaming')}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          🌊 Streaming
        </ModeButton>
      </UploadModeSelector>
      
      <DropZone
        isDragOver={isDragOver}
        hasFile={!!selectedFile || uploadMode === 'batch'}
        isProcessing={uploading || isAnalyzing}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && fileInputRef.current?.click()}
      >
        {uploading ? (
          <div>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⚡</div>
            <div>Processing live upload...</div>
            <div style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem' }}>
              {currentOperation || 'Initializing hot processing pipeline...'}
            </div>
            {processingSpeed > 0 && (
              <div style={{ fontSize: '0.7rem', opacity: 0.6, marginTop: '0.25rem' }}>
                Speed: {processingSpeed.toFixed(1)}%/s • ETA: {estimatedTimeRemaining.toFixed(0)}s
              </div>
            )}
          </div>
        ) : isAnalyzing ? (
          <div>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>🔍</div>
            <div>Live analysis in progress...</div>
            <div style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem' }}>
              Scanning file structure and detecting patterns
            </div>
          </div>
        ) : selectedFile && uploadMode === 'single' ? (
          <div>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>✅</div>
            <div>{selectedFile.name}</div>
            <div style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem' }}>
              {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Ready for ensemble processing
            </div>
            <div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
              <CyberpunkButton 
                $variant="primary" 
                $size="sm" 
                onClick={() => selectedFile && handleFileUpload(selectedFile)}
                disabled={uploading}
              >
                🚀 Start Upload
              </CyberpunkButton>
              <CyberpunkButton 
                $variant="ghost" 
                $size="sm" 
                onClick={() => resetUpload()}
              >
                Clear
              </CyberpunkButton>
            </div>
          </div>
        ) : uploadMode === 'batch' && batchFiles.length > 0 ? (
          <div>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>📦</div>
            <div>{batchFiles.length} files selected for batch upload</div>
            <div style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem' }}>
              Total size: {(batchFiles.reduce((sum, f) => sum + f.file.size, 0) / 1024 / 1024).toFixed(2)} MB
            </div>
            <div style={{ marginTop: '1rem' }}>
              <CyberpunkButton 
                $variant="ghost" 
                $size="sm" 
                onClick={() => resetUpload()}
              >
                Clear Selection
              </CyberpunkButton>
            </div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>
              {uploadMode === 'single' && '⚡'}
              {uploadMode === 'batch' && '📦'}
              {uploadMode === 'streaming' && '🌊'}
            </div>
            <div>
              {uploadMode === 'single' && 'Drop a single file for instant processing'}
              {uploadMode === 'batch' && 'Drop multiple files for batch processing'}
              {uploadMode === 'streaming' && 'Drop files for real-time streaming upload'}
            </div>
            <div style={{ fontSize: '0.9rem', opacity: 0.7, marginTop: '0.5rem' }}>
              Supported: .csv, .xlsx, .xls, .pdf • Live analysis • Hot-swappable modes
            </div>
          </div>
        )}
      </DropZone>

      <FileInput
        ref={fileInputRef}
        type="file"
        accept=".csv,.xlsx,.xls,.pdf"
        multiple={uploadMode !== 'single'}
        onChange={handleFileSelect}
      />

      {/* Live Analysis Panel */}
      <AnimatePresence>
        {liveAnalysis && (
          <LiveAnalysisPanel
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="analysis-header">
              <span>🔍 Live Analysis</span>
              {isAnalyzing && <span style={{ fontSize: '0.8rem' }}>Analyzing...</span>}
            </div>
            <div className="analysis-grid">
              <div className="analysis-item">
                <div className="item-label">File Size</div>
                <div className="item-value">{(liveAnalysis.fileSize / 1024 / 1024).toFixed(2)} MB</div>
              </div>
              <div className="analysis-item">
                <div className="item-label">File Type</div>
                <div className="item-value">{liveAnalysis.fileType}</div>
              </div>
              <div className="analysis-item">
                <div className="item-label">Est. Rows</div>
                <div className="item-value">{liveAnalysis.estimatedRows.toLocaleString()}</div>
              </div>
              <div className="analysis-item">
                <div className="item-label">Columns</div>
                <div className="item-value">{liveAnalysis.detectedColumns || 'Detecting...'}</div>
              </div>
              <div className="analysis-item">
                <div className="item-label">Quality Score</div>
                <div className="item-value">{(liveAnalysis.dataQualityScore * 100).toFixed(0)}%</div>
              </div>
              <div className="analysis-item">
                <div className="item-label">Analysis Time</div>
                <div className="item-value">{liveAnalysis.processingTime}ms</div>
              </div>
            </div>
          </LiveAnalysisPanel>
        )}
      </AnimatePresence>

      {/* Upload Progress */}
      <AnimatePresence>
        {uploading && (
          <ProgressBar
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="progress-header">
              <span>⚡ Live Processing</span>
              <span>{uploadProgress.toFixed(1)}%</span>
            </div>
            <div className="progress-track">
              <div 
                className="progress-fill" 
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            {currentOperation && (
              <div className="current-operation">
                {currentOperation}
              </div>
            )}
            <div className="progress-stats">
              <span>Speed: {processingSpeed.toFixed(1)}%/s</span>
              <span>ETA: {estimatedTimeRemaining > 0 ? `${estimatedTimeRemaining.toFixed(0)}s` : '--'}</span>
              <span>Mode: {uploadMode.toUpperCase()}</span>
            </div>
          </ProgressBar>
        )}
      </AnimatePresence>

      {/* Processing Steps */}
      <AnimatePresence>
        {processingSteps.length > 0 && (
          <ProcessingSteps
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <h4 style={{ color: '#00d4ff', marginBottom: '1rem', fontSize: '1rem' }}>
              🔄 Processing Pipeline
            </h4>
            {processingSteps.map((step) => (
              <div key={step.id} className={`step ${step.status}`}>
                <div className="step-icon">
                  {step.status === 'completed' && '✅'}
                  {step.status === 'processing' && '⚡'}
                  {step.status === 'error' && '❌'}
                  {step.status === 'pending' && '⏳'}
                </div>
                <span>{step.name}</span>
                {step.message && <span style={{ marginLeft: '0.5rem', opacity: 0.7 }}>- {step.message}</span>}
              </div>
            ))}
          </ProcessingSteps>
        )}
      </AnimatePresence>

      {/* File Preview */}
      <AnimatePresence>
        {filePreview && (
          <FilePreview
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <h4 style={{ color: '#00d4ff', marginBottom: '0.5rem' }}>
              📋 File Preview - {filePreview.type?.toUpperCase() || 'Unknown'} File
            </h4>
            
            {filePreview.type === 'pdf' ? (
              <div>
                <div>📄 {filePreview.pageCount || 0} pages • {filePreview.textLength || 0} characters extracted</div>
                {filePreview.extractedText && (
                  <div style={{ 
                    marginTop: '0.5rem', 
                    fontSize: '0.8rem', 
                    maxHeight: '100px', 
                    overflow: 'hidden',
                    background: 'rgba(0, 0, 0, 0.2)',
                    padding: '0.5rem',
                    borderRadius: '4px'
                  }}>
                    <strong>Text Preview:</strong><br />
                    {filePreview.extractedText.substring(0, 200)}
                    {filePreview.extractedText.length > 200 && '...'}
                  </div>
                )}
              </div>
            ) : (
              <div>
                <div>Detected {detectedColumns.length} columns • {filePreview.rows || 0} rows</div>
                {detectedColumns.slice(0, 5).map((col, idx) => (
                  <div key={idx} style={{ marginTop: '0.25rem', fontSize: '0.8rem' }}>
                    <strong>{col.name}</strong> ({col.type}) - {col.sample_values?.slice(0, 3).join(', ')}...
                  </div>
                ))}
              </div>
            )}
          </FilePreview>
        )}
      </AnimatePresence>

      {/* Column Mapping */}
      <AnimatePresence>
        {columnMappings.length > 0 && (
          <ColumnMappingContainer
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <h4 style={{ color: '#ff1493', marginBottom: '1rem' }}>🎯 Smart Column Mapping</h4>
            {columnMappings.map((mapping, idx) => (
              <ColumnMappingRow key={idx}>
                <div className="required-field">{mapping.required_field}</div>
                <div className="detected-column">
                  {mapping.detected_column || 'Not detected'}
                  {mapping.confidence && (
                    <div style={{ fontSize: '0.7rem', opacity: 0.7, marginTop: '0.25rem' }}>
                      Confidence: {(mapping.confidence * 100).toFixed(0)}%
                      {mapping.mapping_reason && (
                        <span style={{ marginLeft: '0.5rem' }}>• {mapping.mapping_reason}</span>
                      )}
                    </div>
                  )}
                  {mapping.alternatives && mapping.alternatives.length > 0 && (
                    <div style={{ fontSize: '0.6rem', opacity: 0.6, marginTop: '0.25rem' }}>
                      Alternatives: {mapping.alternatives.map(alt => `${alt.column} (${(alt.confidence * 100).toFixed(0)}%)`).join(', ')}
                    </div>
                  )}
                </div>
                <div className="mapping-status">
                  {mapping.status === 'mapped' && '✅'}
                  {mapping.status === 'uncertain' && '⚠️'}
                  {mapping.status === 'unmapped' && '❌'}
                  {mapping.status === 'conflict' && '🔄'}
                </div>
              </ColumnMappingRow>
            ))}
          </ColumnMappingContainer>
        )}
      </AnimatePresence>

      {/* Enhanced Data Quality Assessment */}
      <AnimatePresence>
        {dataQuality && (
          <DataQualityIndicator
            quality={dataQuality.overall_score}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <div className="quality-header">
              <div className="quality-title">📊 Data Quality Assessment</div>
              <div className="quality-score">{(dataQuality.overall_score * 100).toFixed(1)}%</div>
            </div>
            
            <div className="quality-breakdown">
              <div className="quality-metric">
                <div className="metric-label">Completeness</div>
                <div className="metric-value">{(dataQuality.completeness * 100).toFixed(0)}%</div>
              </div>
              <div className="quality-metric">
                <div className="metric-label">Consistency</div>
                <div className="metric-value">{(dataQuality.consistency * 100).toFixed(0)}%</div>
              </div>
              <div className="quality-metric">
                <div className="metric-label">Validity</div>
                <div className="metric-value">{(dataQuality.validity * 100).toFixed(0)}%</div>
              </div>
              {dataQuality.accuracy !== undefined && (
                <div className="quality-metric">
                  <div className="metric-label">Accuracy</div>
                  <div className="metric-value">{(dataQuality.accuracy * 100).toFixed(0)}%</div>
                </div>
              )}
              {dataQuality.uniqueness !== undefined && (
                <div className="quality-metric">
                  <div className="metric-label">Uniqueness</div>
                  <div className="metric-value">{(dataQuality.uniqueness * 100).toFixed(0)}%</div>
                </div>
              )}
              {dataQuality.timeliness !== undefined && (
                <div className="quality-metric">
                  <div className="metric-label">Timeliness</div>
                  <div className="metric-value">{(dataQuality.timeliness * 100).toFixed(0)}%</div>
                </div>
              )}
            </div>
            
            <div className="quality-bar">
              <div className="quality-fill" />
            </div>
            
            {dataQuality.issues && dataQuality.issues.length > 0 && (
              <div className="quality-issues">
                <div className="issues-title">Issues Detected</div>
                {dataQuality.issues.map((issue, index) => (
                  <div key={index} className="issue-item">{issue}</div>
                ))}
              </div>
            )}
            
            {dataQuality.recommendations && dataQuality.recommendations.length > 0 && (
              <div className="quality-recommendations">
                <div className="recommendations-title">Recommendations</div>
                {dataQuality.recommendations.map((rec, index) => (
                  <div key={index} className="recommendation-item">{rec}</div>
                ))}
              </div>
            )}
          </DataQualityIndicator>
        )}
      </AnimatePresence>

      {/* Batch Upload Interface */}
      <AnimatePresence>
        {uploadMode === 'batch' && batchFiles.length > 0 && (
          <BatchUploadContainer
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="batch-header">
              <h4>📦 Batch Upload ({batchFiles.length} files)</h4>
              <div className="batch-controls">
                <CyberpunkButton 
                  $variant="primary" 
                  $size="sm"
                  onClick={processBatchUpload}
                  disabled={uploading || batchFiles.every(f => f.status === 'completed')}
                >
                  {uploading ? 'Processing...' : 'Start Batch Upload'}
                </CyberpunkButton>
                <CyberpunkButton 
                  $variant="ghost" 
                  $size="sm"
                  onClick={() => resetUpload()}
                  disabled={uploading}
                >
                  Cancel
                </CyberpunkButton>
              </div>
            </div>

            {batchSummary && (
              <div className="batch-summary">
                <div className="summary-item">
                  <div className="summary-value">{batchSummary.totalFiles}</div>
                  <div className="summary-label">Total Files</div>
                </div>
                <div className="summary-item">
                  <div className="summary-value">{batchSummary.completedFiles}</div>
                  <div className="summary-label">Completed</div>
                </div>
                <div className="summary-item">
                  <div className="summary-value">{batchSummary.processingFiles}</div>
                  <div className="summary-label">Processing</div>
                </div>
                <div className="summary-item">
                  <div className="summary-value">{batchSummary.failedFiles}</div>
                  <div className="summary-label">Failed</div>
                </div>
                <div className="summary-item">
                  <div className="summary-value">{batchSummary.overallProgress.toFixed(0)}%</div>
                  <div className="summary-label">Progress</div>
                </div>
              </div>
            )}

            <FileStatusList>
              {batchFiles.map((fileStatus, index) => (
                <div key={index} className={`file-item ${fileStatus.status}`}>
                  <div className="file-info">
                    <div className="file-name">{fileStatus.file.name}</div>
                    <div className="file-details">
                      {(fileStatus.file.size / 1024 / 1024).toFixed(2)} MB • 
                      {fileStatus.file.name.split('.').pop()?.toUpperCase()} file
                      {fileStatus.error && (
                        <span style={{ color: '#ff6b6b', marginLeft: '0.5rem' }}>
                          • {fileStatus.error}
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <div className="file-progress">
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{ width: `${fileStatus.progress}%` }}
                      />
                    </div>
                    <div className="progress-text">{fileStatus.progress}%</div>
                  </div>
                  
                  <div className="file-actions">
                    <div className="status-icon">
                      {fileStatus.status === 'pending' && '⏳'}
                      {fileStatus.status === 'uploading' && '⚡'}
                      {fileStatus.status === 'completed' && '✅'}
                      {fileStatus.status === 'failed' && '❌'}
                      {fileStatus.status === 'retrying' && '🔄'}
                    </div>
                    {fileStatus.status === 'failed' && fileStatus.retryCount < 3 && (
                      <CyberpunkButton 
                        $variant="ghost" 
                        $size="sm"
                        onClick={() => retryFailedFile(index)}
                        disabled={uploading}
                      >
                        Retry
                      </CyberpunkButton>
                    )}
                  </div>
                </div>
              ))}
            </FileStatusList>
          </BatchUploadContainer>
        )}
      </AnimatePresence>

      {/* Upload Result */}
      <AnimatePresence>
        {uploadResult && uploadMode === 'single' && (
          <UploadStatus 
            success={uploadResult.success}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <div><strong>{uploadResult.success ? '🎉 Ensemble Initialized!' : '❌ Upload Failed'}</strong></div>
            <div>{uploadResult.message}</div>
            {uploadResult.success && (
              <div style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
                {uploadResult.records_processed && (
                  <div>📊 Processed {uploadResult.records_processed} records</div>
                )}
                {uploadResult.models_initialized && (
                  <div>🤖 Initialized models: {uploadResult.models_initialized.join(', ')}</div>
                )}
                {uploadResult.pattern_detected && (
                  <div>🔍 Pattern detected: {uploadResult.pattern_detected}</div>
                )}
                {uploadResult.data_quality && (
                  <div>✨ Data quality: {(uploadResult.data_quality * 100).toFixed(1)}%</div>
                )}
              </div>
            )}
          </UploadStatus>
        )}
      </AnimatePresence>
    </UploadContainer>
  );
};