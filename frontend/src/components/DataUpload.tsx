import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { CyberpunkButton, CyberpunkCard } from './ui';

const UploadContainer = styled(CyberpunkCard)`
  padding: 2rem;
  margin: 1rem 0;
`;

const DropZone = styled.div<{ isDragOver: boolean }>`
  border: 2px dashed ${props => props.isDragOver ? props.theme.colors.neonBlue : props.theme.colors.secondaryText};
  border-radius: 8px;
  padding: 2rem;
  text-align: center;
  background: ${props => props.isDragOver ? 'rgba(0, 212, 255, 0.1)' : 'transparent'};
  transition: all 0.3s ease;
  cursor: pointer;
  
  &:hover {
    border-color: ${props => props.theme.colors.neonBlue};
    background: rgba(0, 212, 255, 0.05);
  }
`;

const FileInput = styled.input`
  display: none;
`;

const UploadStatus = styled.div<{ success: boolean }>`
  margin-top: 1rem;
  padding: 1rem;
  border-radius: 4px;
  background: ${props => props.success ? 'rgba(0, 255, 127, 0.1)' : 'rgba(255, 107, 107, 0.1)'};
  border: 1px solid ${props => props.success ? '#00ff7f' : '#ff6b6b'};
  color: ${props => props.success ? '#00ff7f' : '#ff6b6b'};
`;

interface DataUploadProps {
  authToken: string;
  onUploadComplete: () => void;
}

export const DataUpload: React.FC<DataUploadProps> = ({ authToken, onUploadComplete }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setUploadResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('data_type', 'sales');

    try {
      const response = await fetch('http://localhost:8000/api/v1/auth/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
        },
        body: formData,
      });

      const result = await response.json();
      setUploadResult(result);
      
      if (result.success) {
        onUploadComplete();
      }
    } catch (error) {
      setUploadResult({ success: false, message: 'Upload failed' });
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  return (
    <UploadContainer variant="neon">
      <h3 style={{ color: '#00d4ff', marginBottom: '1rem' }}>📊 Upload Your Business Data</h3>
      
      <DropZone
        isDragOver={isDragOver}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input')?.click()}
      >
        {uploading ? (
          <div>⏳ Uploading and processing...</div>
        ) : (
          <div>
            <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>📁</div>
            <div>Drop your CSV/Excel files here or click to browse</div>
            <div style={{ fontSize: '0.9rem', opacity: 0.7, marginTop: '0.5rem' }}>
              Supported: .csv, .xlsx, .xls
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

      {uploadResult && (
        <UploadStatus success={uploadResult.success}>
          <div><strong>{uploadResult.success ? '✅ Success!' : '❌ Error'}</strong></div>
          <div>{uploadResult.message}</div>
          {uploadResult.success && (
            <div style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
              Processed {uploadResult.processed_records} records with {(uploadResult.data_quality_score * 100).toFixed(1)}% quality score
            </div>
          )}
        </UploadStatus>
      )}
    </UploadContainer>
  );
};