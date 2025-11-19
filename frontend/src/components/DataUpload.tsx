import React, { useState, useCallback } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CyberpunkButton, CyberpunkCard } from './ui';

const UploadContainer = styled(CyberpunkCard)`
  padding: 2rem;
  margin: 1rem 0;
`;

const DropZone = styled.div<{ isDragOver: boolean; hasFile: boolean }>`
  border: 2px dashed ${props => 
    props.hasFile ? props.theme.colors.acidGreen :
    props.isDragOver ? props.theme.colors.neonBlue : 
    props.theme.colors.secondaryText
  };
  border-radius: 8px;
  padding: 2rem;
  text-align: center;
  background: ${props => 
    props.hasFile ? 'rgba(0, 255, 127, 0.1)' :
    props.isDragOver ? 'rgba(0, 212, 255, 0.1)' : 
    'transparent'
  };
  transition: all 0.3s ease;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  
  &:hover {
    border-color: ${props => props.theme.colors.neonBlue};
    background: rgba(0, 212, 255, 0.05);
  }
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 2px;
    background: ${props => props.theme.effects.primaryGradient};
    ${props => props.isDragOver && css`
      animation: scan 1s infinite;
    `}
  }
  
  @keyframes scan {
    0% { left: -100%; }
    100% { left: 100%; }
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

const ProgressBar = styled(motion.div)`
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid ${props => props.theme.colors.neonBlue};
  border-radius: 4px;
  
  .progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    font-family: ${props => props.theme.typography.fontFamily.mono};
    font-size: ${props => props.theme.typography.fontSize.sm};
    color: ${props => props.theme.colors.neonBlue};
  }
  
  .progress-track {
    width: 100%;
    height: 8px;
    background: rgba(0, 0, 0, 0.5);
    border-radius: 4px;
    overflow: hidden;
    
    .progress-fill {
      height: 100%;
      background: ${props => props.theme.effects.primaryGradient};
      transition: width 0.3s ease;
      box-shadow: 0 0 10px ${props => props.theme.colors.neonBlue};
    }
  }
  
  .current-operation {
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: ${props => props.theme.colors.secondaryText};
    font-style: italic;
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
}

interface ColumnMapping {
  required_field: string;
  detected_column: string | null;
  confidence: number;
  status: 'mapped' | 'unmapped' | 'uncertain';
}

interface DataQuality {
  overall_score: number;
  completeness: number;
  consistency: number;
  validity: number;
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

interface DataUploadProps {
  authToken: string;
  onUploadComplete: (result: any) => void;
  onAuthError?: () => void; // Optional callback for authentication errors
}

export const DataUpload: React.FC<DataUploadProps> = ({ authToken, onUploadComplete, onAuthError }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [detectedColumns, setDetectedColumns] = useState<DetectedColumn[]>([]);
  const [columnMappings, setColumnMappings] = useState<ColumnMapping[]>([]);
  const [dataQuality, setDataQuality] = useState<DataQuality | null>(null);
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([]);
  const [filePreview, setFilePreview] = useState<any>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [currentOperation, setCurrentOperation] = useState<string>('');
  const [batchFiles, setBatchFiles] = useState<FileUploadStatus[]>([]);
  const [batchMode, setBatchMode] = useState<boolean>(false);
  const [batchSummary, setBatchSummary] = useState<BatchUploadSummary | null>(null);

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
      // Validate authentication token
      if (!authToken) {
        throw new Error('Authentication required. Please log in again.');
      }
      
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
      
      // Determine file type for different processing workflows
      const fileExtension = file.name.toLowerCase().split('.').pop();
      const isPDF = fileExtension === 'pdf';
      
      if (isPDF) {
        // PDF processing workflow
        updateProcessingStep('parameter-detection', 'processing', 'Extracting PDF content...');
        
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://localhost:8000/api/v1/pdf/extract-preview', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${authToken}`,
          },
          body: formData,
        });

        if (response.status === 401) {
          throw new Error('Authentication failed. Please log in again.');
        }
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Failed to extract PDF content');
        }

        const result = await response.json();
        
        // Set PDF-specific preview data
        setFilePreview({
          type: 'pdf',
          pageCount: result.page_count || 0,
          textLength: result.text_length || 0,
          extractedText: result.preview_text || '',
          metadata: result.metadata || {}
        });
        
        // PDF files don't have columns, so skip column mapping
        setDetectedColumns([]);
        setColumnMappings([]);
        
        updateProcessingStep('parameter-detection', 'completed');
        updateProcessingStep('column-mapping', 'completed', 'Skipped for PDF files');
        updateProcessingStep('data-quality', 'processing');

        // Assess PDF quality
        const quality = assessPDFQuality(result);
        setDataQuality(quality);
        
        updateProcessingStep('data-quality', 'completed');

        return result;
      } else {
        // CSV/Excel processing workflow (existing logic)
        updateProcessingStep('parameter-detection', 'processing');

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://localhost:8000/api/company-sales/detect-parameters', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${authToken}`,
          },
          body: formData,
        });

        if (response.status === 401) {
          throw new Error('Authentication failed. Please log in again.');
        }
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Failed to detect file parameters');
        }

        const result = await response.json();
        
        setDetectedColumns(result.detected_columns || []);
        setFilePreview({
          type: 'csv',
          ...result.preview
        });
        
        updateProcessingStep('parameter-detection', 'completed');
        updateProcessingStep('column-mapping', 'processing');

        // Auto-map columns
        const mappings = autoMapColumns(result.detected_columns || []);
        setColumnMappings(mappings);
        
        updateProcessingStep('column-mapping', 'completed');
        updateProcessingStep('data-quality', 'processing');

        // Assess data quality
        const quality = assessDataQuality(result.detected_columns || [], result.preview || {});
        setDataQuality(quality);
        
        updateProcessingStep('data-quality', 'completed');

        return result;
      }
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
    let completeness = 0;
    let consistency = 0;
    let validity = 0;
    const issues: string[] = [];
    const recommendations: string[] = [];

    // Basic quality assessment
    if (columns.length > 0) {
      completeness = Math.min(columns.length / requiredFields.length, 1);
      consistency = columns.reduce((acc, col) => acc + col.confidence, 0) / columns.length;
      validity = 0.8; // Placeholder - would need actual data validation
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
    // Validate authentication token
    if (!authToken) {
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
      // Progress tracking for parameter detection
      setUploadProgress(10);
      setCurrentOperation('Analyzing file structure...');
      
      // First detect parameters
      await detectFileParameters(file);
      
      setUploadProgress(50);
      setCurrentOperation('Uploading file to server...');

      // Then upload with enhanced processing
      const fileExtension = file.name.toLowerCase().split('.').pop();
      const isPDF = fileExtension === 'pdf';
      
      updateProcessingStep('ensemble-init', 'processing', 
        isPDF ? 'Processing PDF document...' : 'Initializing ensemble models...');
      
      setCurrentOperation(isPDF ? 'Processing PDF document...' : 'Training ensemble models...');
      setUploadProgress(70);
      
      const formData = new FormData();
      formData.append('file', file);

      // Use the enhanced upload endpoint that handles both CSV and PDF
      const response = await fetch('http://localhost:8000/api/v1/upload-enhanced', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
        },
        body: formData,
      });

      if (response.status === 401) {
        throw new Error('Authentication failed. Please log in again.');
      }

      const result = await response.json();
      setUploadResult(result);
      
      setUploadProgress(90);
      setCurrentOperation('Finalizing processing...');
      
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
            setCurrentOperation('Pattern detection completed');
            setUploadProgress(100);
          }, 1000);
        }
        
        if (isPDF) {
          setUploadProgress(100);
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
      if (files.length === 1) {
        handleFileUpload(files[0]);
      } else {
        handleBatchFileSelection(files);
      }
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const fileArray = Array.from(files);
      if (fileArray.length === 1) {
        handleFileUpload(fileArray[0]);
      } else {
        handleBatchFileSelection(fileArray);
      }
    }
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
      setBatchMode(true);
      const batchFileStatuses: FileUploadStatus[] = validFiles.map(file => ({
        file,
        status: 'pending',
        progress: 0,
        retryCount: 0
      }));
      
      setBatchFiles(batchFileStatuses);
      updateBatchSummary(batchFileStatuses);
    }
  }, [updateBatchSummary]);

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
    setBatchFiles([]);
    setBatchMode(false);
    setBatchSummary(null);
  };

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

      // Upload file
      updateFileStatus(fileIndex, { progress: 50 });
      
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/api/v1/upload-enhanced', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
        },
        body: formData,
      });

      if (response.status === 401) {
        throw new Error('Authentication failed. Please log in again.');
      }

      const result = await response.json();
      
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
    // Validate authentication token
    if (!authToken) {
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
    <UploadContainer variant="neon">
      <motion.h3 
        style={{ color: '#00d4ff', marginBottom: '1rem', fontFamily: 'monospace' }}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
      >
        🤖 Enhanced Ensemble Data Upload
      </motion.h3>
      
      <DropZone
        isDragOver={isDragOver}
        hasFile={!!selectedFile || batchMode}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && document.getElementById('file-input')?.click()}
      >
        {uploading ? (
          <div>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⚡</div>
            <div>Processing ensemble initialization...</div>
            <div style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem' }}>
              Detecting patterns and training models
            </div>
          </div>
        ) : selectedFile && !batchMode ? (
          <div>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>✅</div>
            <div>{selectedFile.name}</div>
            <div style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem' }}>
              {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Ready for ensemble processing
            </div>
            <div style={{ marginTop: '1rem' }}>
              <CyberpunkButton 
                variant="ghost" 
                size="sm" 
                onClick={() => resetUpload()}
              >
                Choose Different File
              </CyberpunkButton>
            </div>
          </div>
        ) : batchMode ? (
          <div>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>📦</div>
            <div>{batchFiles.length} files selected for batch upload</div>
            <div style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: '0.5rem' }}>
              Total size: {(batchFiles.reduce((sum, f) => sum + f.file.size, 0) / 1024 / 1024).toFixed(2)} MB
            </div>
            <div style={{ marginTop: '1rem' }}>
              <CyberpunkButton 
                variant="ghost" 
                size="sm" 
                onClick={() => resetUpload()}
              >
                Clear Selection
              </CyberpunkButton>
            </div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>📊</div>
            <div>Drop your CSV/Excel/PDF files here or click to browse</div>
            <div style={{ fontSize: '0.9rem', opacity: 0.7, marginTop: '0.5rem' }}>
              Supported: .csv, .xlsx, .xls, .pdf • Single or multiple files • Auto-detects parameters and processes documents
            </div>
          </div>
        )}
      </DropZone>

      <FileInput
        id="file-input"
        type="file"
        accept=".csv,.xlsx,.xls,.pdf"
        multiple
        onChange={handleFileSelect}
      />

      {/* Upload Progress */}
      <AnimatePresence>
        {uploading && (
          <ProgressBar
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="progress-header">
              <span>⚡ Upload Progress</span>
              <span>{uploadProgress}%</span>
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
            <h4 style={{ color: '#ff1493', marginBottom: '1rem' }}>🎯 Column Mapping</h4>
            {columnMappings.map((mapping, idx) => (
              <ColumnMappingRow key={idx}>
                <div className="required-field">{mapping.required_field}</div>
                <div className="detected-column">
                  {mapping.detected_column || 'Not detected'}
                </div>
                <div className="mapping-status">
                  {mapping.status === 'mapped' && '✅'}
                  {mapping.status === 'uncertain' && '⚠️'}
                  {mapping.status === 'unmapped' && '❌'}
                </div>
              </ColumnMappingRow>
            ))}
          </ColumnMappingContainer>
        )}
      </AnimatePresence>

      {/* Data Quality Assessment */}
      <AnimatePresence>
        {dataQuality && (
          <DataQualityIndicator
            quality={dataQuality.overall_score}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <h4 style={{ marginBottom: '0.5rem' }}>
              📊 Data Quality Score: {(dataQuality.overall_score * 100).toFixed(1)}%
            </h4>
            <div className="quality-bar">
              <div className="quality-fill" />
            </div>
            <div style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
              Completeness: {(dataQuality.completeness * 100).toFixed(0)}% • 
              Consistency: {(dataQuality.consistency * 100).toFixed(0)}% • 
              Validity: {(dataQuality.validity * 100).toFixed(0)}%
            </div>
            {dataQuality.issues.length > 0 && (
              <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#ff6b6b' }}>
                Issues: {dataQuality.issues.join(', ')}
              </div>
            )}
          </DataQualityIndicator>
        )}
      </AnimatePresence>

      {/* Batch Upload Interface */}
      <AnimatePresence>
        {batchMode && batchFiles.length > 0 && (
          <BatchUploadContainer
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="batch-header">
              <h4>📦 Batch Upload ({batchFiles.length} files)</h4>
              <div className="batch-controls">
                <CyberpunkButton 
                  variant="primary" 
                  size="sm"
                  onClick={processBatchUpload}
                  disabled={uploading || batchFiles.every(f => f.status === 'completed')}
                >
                  {uploading ? 'Processing...' : 'Start Batch Upload'}
                </CyberpunkButton>
                <CyberpunkButton 
                  variant="ghost" 
                  size="sm"
                  onClick={() => setBatchMode(false)}
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
                        variant="ghost" 
                        size="xs"
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
        {uploadResult && !batchMode && (
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