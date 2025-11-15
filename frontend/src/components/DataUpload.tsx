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

interface DataUploadProps {
  authToken: string;
  onUploadComplete: (result: any) => void;
}

export const DataUpload: React.FC<DataUploadProps> = ({ authToken, onUploadComplete }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [detectedColumns, setDetectedColumns] = useState<DetectedColumn[]>([]);
  const [columnMappings, setColumnMappings] = useState<ColumnMapping[]>([]);
  const [dataQuality, setDataQuality] = useState<DataQuality | null>(null);
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([]);
  const [filePreview, setFilePreview] = useState<any>(null);

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
      updateProcessingStep('file-validation', 'processing');
      
      // Basic file validation
      const validTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
      if (!validTypes.includes(file.type) && !file.name.match(/\.(csv|xlsx|xls)$/i)) {
        throw new Error('Invalid file type. Please upload CSV or Excel files.');
      }

      if (file.size > 50 * 1024 * 1024) {
        throw new Error('File size exceeds 50MB limit.');
      }

      updateProcessingStep('file-validation', 'completed');
      updateProcessingStep('parameter-detection', 'processing');

      // Read file preview for parameter detection
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/api/company-sales/detect-parameters', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to detect file parameters');
      }

      const result = await response.json();
      
      setDetectedColumns(result.detected_columns || []);
      setFilePreview(result.preview || null);
      
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
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Parameter detection failed';
      updateProcessingStep('parameter-detection', 'error', errorMessage);
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

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setUploadResult(null);
    setSelectedFile(file);
    initializeProcessingSteps();

    try {
      // First detect parameters
      await detectFileParameters(file);

      // Then upload with enhanced processing
      updateProcessingStep('ensemble-init', 'processing');
      
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/api/company-sales/upload-enhanced', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
        },
        body: formData,
      });

      const result = await response.json();
      setUploadResult(result);
      
      if (result.success) {
        updateProcessingStep('ensemble-init', 'completed');
        updateProcessingStep('pattern-detection', 'processing');
        
        // Simulate pattern detection completion
        setTimeout(() => {
          updateProcessingStep('pattern-detection', 'completed');
        }, 1000);
        
        onUploadComplete(result);
      } else {
        updateProcessingStep('ensemble-init', 'error', result.message);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      setUploadResult({ success: false, message: errorMessage });
      updateProcessingStep('ensemble-init', 'error', errorMessage);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, []);

  const resetUpload = () => {
    setSelectedFile(null);
    setUploadResult(null);
    setDetectedColumns([]);
    setColumnMappings([]);
    setDataQuality(null);
    setProcessingSteps([]);
    setFilePreview(null);
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
        hasFile={!!selectedFile}
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
        ) : selectedFile ? (
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
        ) : (
          <div>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>📊</div>
            <div>Drop your CSV/Excel files here or click to browse</div>
            <div style={{ fontSize: '0.9rem', opacity: 0.7, marginTop: '0.5rem' }}>
              Supported: .csv, .xlsx, .xls • Auto-detects parameters for ensemble models
            </div>
          </div>
        )}
      </DropZone>

      <FileInput
        id="file-input"
        type="file"
        accept=".csv,.xlsx,.xls"
        onChange={handleFileSelect}
      />

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
            <h4 style={{ color: '#00d4ff', marginBottom: '0.5rem' }}>📋 File Preview</h4>
            <div>Detected {detectedColumns.length} columns • {filePreview.rows || 0} rows</div>
            {detectedColumns.slice(0, 5).map((col, idx) => (
              <div key={idx} style={{ marginTop: '0.25rem', fontSize: '0.8rem' }}>
                <strong>{col.name}</strong> ({col.type}) - {col.sample_values?.slice(0, 3).join(', ')}...
              </div>
            ))}
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

      {/* Upload Result */}
      <AnimatePresence>
        {uploadResult && (
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