import React from 'react';
import styled from 'styled-components';
import { Scatter } from 'recharts';
// import { CompanyPriority } from '../../types';

interface CompanyPriority {
  id: string;
  name: string;
  impact: number;
  effort: number;
}

interface PriorityMatrixProps {
  priorities?: CompanyPriority[];
  recommendations?: any[];
  onRecommendationClick?: (id: string) => void;
  onPriorityFilter?: (filter: string) => void;
}

const MatrixContainer = styled.div`
  background: ${props => props.theme.colors.darkerBg};
  border-radius: 8px;
  padding: 1.5rem;
  height: 400px;
  width: 100%;
`;

const PriorityMatrix: React.FC<PriorityMatrixProps> = ({ 
  priorities = [], 
  recommendations = [],
  onRecommendationClick,
  onPriorityFilter 
}) => {
  return (
    <MatrixContainer>
      <div style={{ padding: '2rem', textAlign: 'center', color: '#00d4ff' }}>
        📊 Priority Matrix Visualization
        <br />
        <small>Recommendations: {recommendations.length} | Priorities: {priorities.length}</small>
      </div>
    </MatrixContainer>
  );
};

export default PriorityMatrix;

export default PriorityMatrix;