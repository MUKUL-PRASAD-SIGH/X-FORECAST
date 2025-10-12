import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { CyberpunkButton, CyberpunkInput, CyberpunkCard } from '../ui';

const AuthContainer = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 2000;
`;

const AuthCard = styled(CyberpunkCard)`
  width: 400px;
  padding: 2rem;
  text-align: center;
`;

const Title = styled.h2`
  color: ${props => props.theme.colors.neonBlue};
  margin-bottom: 1.5rem;
  font-family: ${props => props.theme.typography.fontFamily.display};
`;

const Form = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const SwitchText = styled.p`
  color: ${props => props.theme.colors.secondaryText};
  margin-top: 1rem;
  
  button {
    background: none;
    border: none;
    color: ${props => props.theme.colors.neonBlue};
    cursor: pointer;
    text-decoration: underline;
  }
`;

interface LoginFormProps {
  onLogin: (token: string, user: any) => void;
}

export const LoginForm: React.FC<LoginFormProps> = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    company_name: '',
    business_type: 'retail',
    industry: 'retail'
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e?: React.FormEvent | React.MouseEvent) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const endpoint = isLogin ? '/api/v1/auth/login' : '/api/v1/auth/register';
      
      // Add timeout to prevent infinite loading
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      const data = await response.json();

      if (response.ok) {
        if (isLogin) {
          localStorage.setItem('auth_token', data.token);
          onLogin(data.token, data.user);
        } else {
          setIsLogin(true);
          setError('✅ Registration successful! Please login.');
        }
      } else {
        setError(data.detail || 'Authentication failed');
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setError('⏰ Request timeout. Is the backend server running on port 8000?');
      } else {
        setError('❌ Backend server not responding. Please start the server with: py -m uvicorn src.api.main:app --reload --port 8000');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthContainer
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <AuthCard variant="neon">
        <Title>{isLogin ? '🔐 Login' : '📝 Register'}</Title>
        
        <Form onSubmit={(e) => handleSubmit(e)}>
          <CyberpunkInput
            type="email"
            placeholder="Email"
            value={formData.email}
            onChange={(value) => setFormData({...formData, email: value})}
          />
          
          <CyberpunkInput
            type="password"
            placeholder="Password"
            value={formData.password}
            onChange={(value) => setFormData({...formData, password: value})}
          />
          
          {!isLogin && (
            <>
              <CyberpunkInput
                placeholder="Company Name"
                value={formData.company_name}
                onChange={(value) => setFormData({...formData, company_name: value})}
              />
              
              <select
                value={formData.business_type}
                onChange={(e) => setFormData({...formData, business_type: e.target.value})}
                style={{
                  background: '#1a1a2e',
                  border: '1px solid #00d4ff',
                  color: '#fff',
                  padding: '0.75rem',
                  borderRadius: '4px'
                }}
              >
                <option value="retail">Retail Store</option>
                <option value="supermarket">Supermarket</option>
                <option value="restaurant">Restaurant</option>
                <option value="ecommerce">E-commerce</option>
                <option value="wholesale">Wholesale</option>
              </select>
            </>
          )}
          
          {error && (
            <div style={{ color: '#ff6b6b', fontSize: '0.9rem' }}>
              {error}
            </div>
          )}
          
          <CyberpunkButton
            variant="primary"
            disabled={loading}
            onClick={handleSubmit}
          >
            {loading ? 'Processing...' : (isLogin ? 'Login' : 'Register')}
          </CyberpunkButton>
        </Form>
        
        <SwitchText>
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button type="button" onClick={() => setIsLogin(!isLogin)}>
            {isLogin ? 'Register' : 'Login'}
          </button>
        </SwitchText>
      </AuthCard>
    </AuthContainer>
  );
};