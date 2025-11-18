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
const ShareableContainer = styled.div`
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

const TabContainer = styled.div`
  display: flex;
  margin-bottom: 2rem;
  border-bottom: 1px solid rgba(0, 255, 255, 0.3);
`;

const Tab = styled.button<{ active: boolean }>`
  background: ${props => props.active ? 'rgba(0, 255, 255, 0.2)' : 'transparent'};
  border: none;
  color: ${props => props.active ? '#00ffff' : '#ffffff'};
  font-family: 'Orbitron', monospace;
  font-size: 1rem;
  padding: 1rem 2rem;
  cursor: pointer;
  transition: all 0.3s ease;
  border-bottom: 2px solid ${props => props.active ? '#00ffff' : 'transparent'};

  &:hover {
    background: rgba(0, 255, 255, 0.1);
    color: #00ffff;
  }
`;

const ContentSection = styled.div`
  background: rgba(0, 255, 255, 0.05);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 10px;
  padding: 2rem;
  margin-bottom: 2rem;
`;

const FormGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-bottom: 2rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const FormGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const Label = styled.label`
  color: #00ffff;
  font-family: 'Roboto Mono', monospace;
  font-size: 0.9rem;
  font-weight: bold;
`;

const Input = styled.input`
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid #00ffff;
  border-radius: 5px;
  color: #00ffff;
  font-family: 'Roboto Mono', monospace;
  padding: 0.75rem;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    box-shadow: 0 0 10px #00ffff;
    border-color: #ff00ff;
  }

  &::placeholder {
    color: rgba(0, 255, 255, 0.5);
  }
`;

const Select = styled.select`
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid #00ffff;
  border-radius: 5px;
  color: #00ffff;
  font-family: 'Roboto Mono', monospace;
  padding: 0.75rem;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    box-shadow: 0 0 10px #00ffff;
    border-color: #ff00ff;
  }

  option {
    background: #000000;
    color: #00ffff;
  }
`;

const TextArea = styled.textarea`
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid #00ffff;
  border-radius: 5px;
  color: #00ffff;
  font-family: 'Roboto Mono', monospace;
  padding: 0.75rem;
  min-height: 100px;
  resize: vertical;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    box-shadow: 0 0 10px #00ffff;
    border-color: #ff00ff;
  }

  &::placeholder {
    color: rgba(0, 255, 255, 0.5);
  }
`;

const CheckboxGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
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

const ActionButton = styled.button<{ variant?: 'primary' | 'secondary' | 'danger' }>`
  background: ${props => 
    props.variant === 'danger' ? 'linear-gradient(45deg, #ff0000, #cc0000)' :
    props.variant === 'secondary' ? 'linear-gradient(45deg, #ff00ff, #8000ff)' :
    'linear-gradient(45deg, #00ffff, #0080ff)'
  };
  border: none;
  border-radius: 10px;
  color: #000000;
  font-family: 'Orbitron', monospace;
  font-size: 1rem;
  font-weight: bold;
  padding: 1rem 2rem;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;

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

const ShareableLink = styled.div`
  background: rgba(0, 255, 0, 0.1);
  border: 1px solid #00ff00;
  border-radius: 10px;
  padding: 1.5rem;
  margin-top: 1rem;
`;

const LinkDisplay = styled.div`
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid #00ffff;
  border-radius: 5px;
  color: #00ffff;
  font-family: 'Roboto Mono', monospace;
  padding: 1rem;
  margin: 1rem 0;
  word-break: break-all;
  position: relative;
`;

const CopyButton = styled.button`
  background: linear-gradient(45deg, #00ff00, #00cc00);
  border: none;
  border-radius: 5px;
  color: #000000;
  font-family: 'Orbitron', monospace;
  font-size: 0.8rem;
  font-weight: bold;
  padding: 0.5rem 1rem;
  cursor: pointer;
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  transition: all 0.3s ease;

  &:hover {
    transform: scale(1.05);
  }
`;

const ScheduleList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const ScheduleItem = styled.div`
  background: rgba(0, 255, 255, 0.05);
  border: 1px solid rgba(0, 255, 255, 0.3);
  border-radius: 10px;
  padding: 1.5rem;
  position: relative;
`;

const ScheduleHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
`;

const ScheduleTitle = styled.h4`
  color: #00ffff;
  font-family: 'Orbitron', monospace;
  margin: 0;
`;

const ScheduleStatus = styled.span<{ active: boolean }>`
  background: ${props => props.active ? 'rgba(0, 255, 0, 0.2)' : 'rgba(255, 0, 0, 0.2)'};
  border: 1px solid ${props => props.active ? '#00ff00' : '#ff0000'};
  border-radius: 15px;
  color: ${props => props.active ? '#00ff00' : '#ff0000'};
  font-family: 'Roboto Mono', monospace;
  font-size: 0.8rem;
  padding: 0.25rem 0.75rem;
`;

const ScheduleDetails = styled.div`
  color: #ffffff;
  font-family: 'Roboto Mono', monospace;
  font-size: 0.9rem;
  line-height: 1.5;
`;

const TemplateGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-top: 1rem;
`;

const TemplateCard = styled.div<{ selected: boolean }>`
  background: ${props => props.selected ? 'rgba(0, 255, 255, 0.2)' : 'rgba(0, 255, 255, 0.05)'};
  border: 2px solid ${props => props.selected ? '#00ffff' : 'rgba(0, 255, 255, 0.3)'};
  border-radius: 10px;
  padding: 1.5rem;
  cursor: pointer;
  transition: all 0.3s ease;

  &:hover {
    border-color: #00ffff;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
  }
`;

const TemplateTitle = styled.h4`
  color: #00ffff;
  font-family: 'Orbitron', monospace;
  margin: 0 0 0.5rem 0;
`;

const TemplateDescription = styled.p`
  color: #ffffff;
  font-family: 'Roboto Mono', monospace;
  font-size: 0.9rem;
  margin: 0 0 1rem 0;
`;

const TemplateStakeholder = styled.span`
  background: rgba(255, 0, 255, 0.2);
  border: 1px solid #ff00ff;
  border-radius: 15px;
  color: #ff00ff;
  font-family: 'Roboto Mono', monospace;
  font-size: 0.8rem;
  padding: 0.25rem 0.75rem;
`;

// Interfaces
interface ShareableReportRequest {
  report_type: string;
  format: string;
  include_forecasts: boolean;
  include_performance: boolean;
  include_insights: boolean;
  include_metadata: boolean;
  include_charts: boolean;
  custom_title: string;
  horizon_months: number;
  stakeholder_type: string;
  template_id: string;
  expiration_hours: number;
  password_protected: boolean;
  allow_downloads: boolean;
  embed_interactive_charts: boolean;
}

interface ScheduledReportRequest {
  report_config: ShareableReportRequest;
  schedule_type: string;
  schedule_time: string;
  recipients: string[];
  subject_template: string;
  message_template: string;
  start_date: string;
  end_date: string;
  timezone: string;
  active: boolean;
}

interface ReportTemplate {
  template_id: string;
  template_name: string;
  description: string;
  stakeholder_type: string;
  sections: string[];
  styling: any;
  default_config: any;
  created_at: string;
  updated_at: string;
}

interface ShareableReportsDashboardProps {
  companyId: string;
  className?: string;
}

const ShareableReportsDashboard: React.FC<ShareableReportsDashboardProps> = ({ companyId, className }) => {
  const [activeTab, setActiveTab] = useState<'create' | 'schedule' | 'templates' | 'analytics'>('create');
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error' | 'info', message: string } | null>(null);
  
  // Create report state
  const [reportRequest, setReportRequest] = useState<ShareableReportRequest>({
    report_type: 'comprehensive',
    format: 'json',
    include_forecasts: true,
    include_performance: true,
    include_insights: true,
    include_metadata: true,
    include_charts: true,
    custom_title: '',
    horizon_months: 6,
    stakeholder_type: 'executive',
    template_id: 'executive-summary',
    expiration_hours: 72,
    password_protected: false,
    allow_downloads: true,
    embed_interactive_charts: true
  });

  // Schedule report state
  const [scheduleRequest, setScheduleRequest] = useState<ScheduledReportRequest>({
    report_config: reportRequest,
    schedule_type: 'weekly',
    schedule_time: '09:00',
    recipients: [],
    subject_template: 'Weekly Forecast Report - {company_id} - {date}',
    message_template: 'Your scheduled forecast report is ready. Access it here: {share_url}',
    start_date: '',
    end_date: '',
    timezone: 'UTC',
    active: true
  });

  const [shareableLink, setShareableLink] = useState<any>(null);
  const [templates, setTemplates] = useState<ReportTemplate[]>([]);
  const [schedules, setSchedules] = useState<any[]>([]);
  const [analytics, setAnalytics] = useState<any>(null);
  const [recipientInput, setRecipientInput] = useState('');

  useEffect(() => {
    loadTemplates();
    loadSchedules();
    loadAnalytics();
  }, []);

  const loadTemplates = async () => {
    try {
      const response = await fetch('/api/shareable-reports/templates');
      if (response.ok) {
        const data = await response.json();
        setTemplates(data.templates || []);
      }
    } catch (error) {
      console.error('Failed to load templates:', error);
    }
  };

  const loadSchedules = async () => {
    try {
      const response = await fetch('/api/shareable-reports/schedules', {
        headers: {
          'Authorization': `Bearer ${companyId}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setSchedules(data.schedules || []);
      }
    } catch (error) {
      console.error('Failed to load schedules:', error);
    }
  };

  const loadAnalytics = async () => {
    try {
      const response = await fetch('/api/shareable-reports/analytics', {
        headers: {
          'Authorization': `Bearer ${companyId}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setAnalytics(data.analytics);
      }
    } catch (error) {
      console.error('Failed to load analytics:', error);
    }
  };

  const handleCreateShareableReport = async () => {
    setIsLoading(true);
    setStatusMessage({ type: 'info', message: 'Creating shareable report...' });

    try {
      const response = await fetch('/api/shareable-reports/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${companyId}`
        },
        body: JSON.stringify(reportRequest)
      });

      if (response.ok) {
        const result = await response.json();
        setShareableLink(result);
        setStatusMessage({ 
          type: 'success', 
          message: `Shareable report created successfully! ${result.password ? 'Password: ' + result.password : ''}` 
        });
      } else {
        const error = await response.json();
        setStatusMessage({ 
          type: 'error', 
          message: `Failed to create shareable report: ${error.detail}` 
        });
      }
    } catch (error) {
      setStatusMessage({ 
        type: 'error', 
        message: `Network error: ${error instanceof Error ? error.message : 'Unknown error'}` 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleScheduleReport = async () => {
    if (scheduleRequest.recipients.length === 0) {
      setStatusMessage({ type: 'error', message: 'Please add at least one recipient' });
      return;
    }

    setIsLoading(true);
    setStatusMessage({ type: 'info', message: 'Scheduling report...' });

    try {
      const response = await fetch('/api/shareable-reports/schedule', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${companyId}`
        },
        body: JSON.stringify({
          ...scheduleRequest,
          report_config: reportRequest
        })
      });

      if (response.ok) {
        const result = await response.json();
        setStatusMessage({ 
          type: 'success', 
          message: `Report scheduled successfully! Next run: ${new Date(result.next_run).toLocaleString()}` 
        });
        loadSchedules(); // Refresh schedules
      } else {
        const error = await response.json();
        setStatusMessage({ 
          type: 'error', 
          message: `Failed to schedule report: ${error.detail}` 
        });
      }
    } catch (error) {
      setStatusMessage({ 
        type: 'error', 
        message: `Network error: ${error instanceof Error ? error.message : 'Unknown error'}` 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteSchedule = async (scheduleId: string) => {
    try {
      const response = await fetch(`/api/shareable-reports/schedule/${scheduleId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${companyId}`
        }
      });

      if (response.ok) {
        setStatusMessage({ type: 'success', message: 'Schedule deleted successfully' });
        loadSchedules(); // Refresh schedules
      } else {
        const error = await response.json();
        setStatusMessage({ type: 'error', message: `Failed to delete schedule: ${error.detail}` });
      }
    } catch (error) {
      setStatusMessage({ type: 'error', message: 'Network error occurred' });
    }
  };

  const handleTemplateSelect = (template: ReportTemplate) => {
    setReportRequest(prev => ({
      ...prev,
      template_id: template.template_id,
      stakeholder_type: template.stakeholder_type,
      ...template.default_config
    }));
  };

  const addRecipient = () => {
    if (recipientInput.trim() && !scheduleRequest.recipients.includes(recipientInput.trim())) {
      setScheduleRequest(prev => ({
        ...prev,
        recipients: [...prev.recipients, recipientInput.trim()]
      }));
      setRecipientInput('');
    }
  };

  const removeRecipient = (email: string) => {
    setScheduleRequest(prev => ({
      ...prev,
      recipients: prev.recipients.filter(r => r !== email)
    }));
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setStatusMessage({ type: 'success', message: 'Link copied to clipboard!' });
  };

  const renderCreateTab = () => (
    <ContentSection>
      <h3 style={{ color: '#00ffff', fontFamily: 'Orbitron', marginBottom: '1.5rem' }}>
        🔗 Create Shareable Report
      </h3>
      
      <FormGrid>
        <FormGroup>
          <Label>Report Type</Label>
          <Select
            value={reportRequest.report_type}
            onChange={(e) => setReportRequest(prev => ({ ...prev, report_type: e.target.value }))}
          >
            <option value="comprehensive">Comprehensive Report</option>
            <option value="forecast">Forecast Only</option>
            <option value="performance">Performance Only</option>
            <option value="insights">Insights Only</option>
          </Select>
        </FormGroup>

        <FormGroup>
          <Label>Export Format</Label>
          <Select
            value={reportRequest.format}
            onChange={(e) => setReportRequest(prev => ({ ...prev, format: e.target.value }))}
          >
            <option value="json">JSON</option>
            <option value="excel">Excel</option>
            <option value="pdf">PDF</option>
          </Select>
        </FormGroup>

        <FormGroup>
          <Label>Stakeholder Type</Label>
          <Select
            value={reportRequest.stakeholder_type}
            onChange={(e) => setReportRequest(prev => ({ ...prev, stakeholder_type: e.target.value }))}
          >
            <option value="executive">Executive</option>
            <option value="analyst">Analyst</option>
            <option value="technical">Technical</option>
          </Select>
        </FormGroup>

        <FormGroup>
          <Label>Expiration (hours)</Label>
          <Input
            type="number"
            min="1"
            max="168"
            value={reportRequest.expiration_hours}
            onChange={(e) => setReportRequest(prev => ({ ...prev, expiration_hours: parseInt(e.target.value) || 72 }))}
          />
        </FormGroup>

        <FormGroup>
          <Label>Custom Title</Label>
          <Input
            type="text"
            placeholder="Enter custom report title (optional)"
            value={reportRequest.custom_title}
            onChange={(e) => setReportRequest(prev => ({ ...prev, custom_title: e.target.value }))}
          />
        </FormGroup>

        <FormGroup>
          <Label>Forecast Horizon (months)</Label>
          <Input
            type="number"
            min="1"
            max="24"
            value={reportRequest.horizon_months}
            onChange={(e) => setReportRequest(prev => ({ ...prev, horizon_months: parseInt(e.target.value) || 6 }))}
          />
        </FormGroup>
      </FormGrid>

      <FormGroup>
        <Label>Report Components</Label>
        <CheckboxGroup>
          <CheckboxLabel>
            <input
              type="checkbox"
              checked={reportRequest.include_forecasts}
              onChange={(e) => setReportRequest(prev => ({ ...prev, include_forecasts: e.target.checked }))}
            />
            Include Forecasts
          </CheckboxLabel>
          <CheckboxLabel>
            <input
              type="checkbox"
              checked={reportRequest.include_performance}
              onChange={(e) => setReportRequest(prev => ({ ...prev, include_performance: e.target.checked }))}
            />
            Include Performance Metrics
          </CheckboxLabel>
          <CheckboxLabel>
            <input
              type="checkbox"
              checked={reportRequest.include_insights}
              onChange={(e) => setReportRequest(prev => ({ ...prev, include_insights: e.target.checked }))}
            />
            Include Business Insights
          </CheckboxLabel>
          <CheckboxLabel>
            <input
              type="checkbox"
              checked={reportRequest.include_metadata}
              onChange={(e) => setReportRequest(prev => ({ ...prev, include_metadata: e.target.checked }))}
            />
            Include Metadata
          </CheckboxLabel>
          <CheckboxLabel>
            <input
              type="checkbox"
              checked={reportRequest.include_charts}
              onChange={(e) => setReportRequest(prev => ({ ...prev, include_charts: e.target.checked }))}
            />
            Include Interactive Charts
          </CheckboxLabel>
        </CheckboxGroup>
      </FormGroup>

      <FormGroup>
        <Label>Security Options</Label>
        <CheckboxGroup>
          <CheckboxLabel>
            <input
              type="checkbox"
              checked={reportRequest.password_protected}
              onChange={(e) => setReportRequest(prev => ({ ...prev, password_protected: e.target.checked }))}
            />
            Password Protected
          </CheckboxLabel>
          <CheckboxLabel>
            <input
              type="checkbox"
              checked={reportRequest.allow_downloads}
              onChange={(e) => setReportRequest(prev => ({ ...prev, allow_downloads: e.target.checked }))}
            />
            Allow Downloads
          </CheckboxLabel>
          <CheckboxLabel>
            <input
              type="checkbox"
              checked={reportRequest.embed_interactive_charts}
              onChange={(e) => setReportRequest(prev => ({ ...prev, embed_interactive_charts: e.target.checked }))}
            />
            Embed Interactive Charts
          </CheckboxLabel>
        </CheckboxGroup>
      </FormGroup>

      <ActionButton
        onClick={handleCreateShareableReport}
        disabled={isLoading}
      >
        {isLoading ? '🔄 CREATING...' : '🚀 CREATE SHAREABLE REPORT'}
      </ActionButton>

      {shareableLink && (
        <ShareableLink>
          <h4 style={{ color: '#00ff00', fontFamily: 'Orbitron', marginBottom: '1rem' }}>
            ✅ Shareable Report Created!
          </h4>
          
          <LinkDisplay>
            {window.location.origin}{shareableLink.share_url}
            <CopyButton onClick={() => copyToClipboard(`${window.location.origin}${shareableLink.share_url}`)}>
              📋 COPY
            </CopyButton>
          </LinkDisplay>

          <div style={{ 
            color: '#ffffff', 
            fontFamily: 'Roboto Mono', 
            fontSize: '0.9rem',
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '1rem'
          }}>
            <div>Share ID: {shareableLink.share_id}</div>
            <div>Expires: {new Date(shareableLink.expires_at).toLocaleString()}</div>
            {shareableLink.password && (
              <div>Password: <strong>{shareableLink.password}</strong></div>
            )}
            <div>Interactive: {shareableLink.access_controls.interactive_charts ? 'Yes' : 'No'}</div>
          </div>
        </ShareableLink>
      )}
    </ContentSection>
  );

  const renderScheduleTab = () => (
    <ContentSection>
      <h3 style={{ color: '#00ffff', fontFamily: 'Orbitron', marginBottom: '1.5rem' }}>
        ⏰ Schedule Automated Reports
      </h3>
      
      <FormGrid>
        <FormGroup>
          <Label>Schedule Type</Label>
          <Select
            value={scheduleRequest.schedule_type}
            onChange={(e) => setScheduleRequest(prev => ({ ...prev, schedule_type: e.target.value }))}
          >
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
            <option value="quarterly">Quarterly</option>
          </Select>
        </FormGroup>

        <FormGroup>
          <Label>Schedule Time</Label>
          <Input
            type="time"
            value={scheduleRequest.schedule_time}
            onChange={(e) => setScheduleRequest(prev => ({ ...prev, schedule_time: e.target.value }))}
          />
        </FormGroup>

        <FormGroup>
          <Label>Start Date</Label>
          <Input
            type="date"
            value={scheduleRequest.start_date}
            onChange={(e) => setScheduleRequest(prev => ({ ...prev, start_date: e.target.value }))}
          />
        </FormGroup>

        <FormGroup>
          <Label>End Date (optional)</Label>
          <Input
            type="date"
            value={scheduleRequest.end_date}
            onChange={(e) => setScheduleRequest(prev => ({ ...prev, end_date: e.target.value }))}
          />
        </FormGroup>
      </FormGrid>

      <FormGroup>
        <Label>Email Recipients</Label>
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
          <Input
            type="email"
            placeholder="Enter email address"
            value={recipientInput}
            onChange={(e) => setRecipientInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && addRecipient()}
            style={{ flex: 1 }}
          />
          <ActionButton variant="secondary" onClick={addRecipient}>
            ADD
          </ActionButton>
        </div>
        
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {scheduleRequest.recipients.map((email, index) => (
            <span
              key={index}
              style={{
                background: 'rgba(0, 255, 255, 0.2)',
                border: '1px solid #00ffff',
                borderRadius: '15px',
                color: '#00ffff',
                fontFamily: 'Roboto Mono',
                fontSize: '0.8rem',
                padding: '0.25rem 0.75rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}
            >
              {email}
              <button
                onClick={() => removeRecipient(email)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#ff0000',
                  cursor: 'pointer',
                  fontSize: '0.8rem'
                }}
              >
                ✕
              </button>
            </span>
          ))}
        </div>
      </FormGroup>

      <FormGroup>
        <Label>Email Subject Template</Label>
        <Input
          type="text"
          placeholder="Use {company_id}, {date} placeholders"
          value={scheduleRequest.subject_template}
          onChange={(e) => setScheduleRequest(prev => ({ ...prev, subject_template: e.target.value }))}
        />
      </FormGroup>

      <FormGroup>
        <Label>Email Message Template</Label>
        <TextArea
          placeholder="Use {company_id}, {date}, {share_url} placeholders"
          value={scheduleRequest.message_template}
          onChange={(e) => setScheduleRequest(prev => ({ ...prev, message_template: e.target.value }))}
        />
      </FormGroup>

      <ActionButton
        onClick={handleScheduleReport}
        disabled={isLoading}
      >
        {isLoading ? '⏰ SCHEDULING...' : '📅 SCHEDULE REPORT'}
      </ActionButton>

      {schedules.length > 0 && (
        <div style={{ marginTop: '2rem' }}>
          <h4 style={{ color: '#00ffff', fontFamily: 'Orbitron', marginBottom: '1rem' }}>
            Active Schedules
          </h4>
          <ScheduleList>
            {schedules.map((schedule) => (
              <ScheduleItem key={schedule.schedule_id}>
                <ScheduleHeader>
                  <ScheduleTitle>
                    {schedule.config.schedule_type.charAt(0).toUpperCase() + schedule.config.schedule_type.slice(1)} Report
                  </ScheduleTitle>
                  <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <ScheduleStatus active={schedule.active}>
                      {schedule.active ? 'ACTIVE' : 'INACTIVE'}
                    </ScheduleStatus>
                    <ActionButton
                      variant="danger"
                      onClick={() => handleDeleteSchedule(schedule.schedule_id)}
                      style={{ padding: '0.5rem 1rem', fontSize: '0.8rem' }}
                    >
                      DELETE
                    </ActionButton>
                  </div>
                </ScheduleHeader>
                
                <ScheduleDetails>
                  <div>Next Run: {new Date(schedule.next_run).toLocaleString()}</div>
                  <div>Recipients: {schedule.config.recipients.join(', ')}</div>
                  <div>Run Count: {schedule.run_count}</div>
                  {schedule.last_run && (
                    <div>Last Run: {new Date(schedule.last_run).toLocaleString()}</div>
                  )}
                </ScheduleDetails>
              </ScheduleItem>
            ))}
          </ScheduleList>
        </div>
      )}
    </ContentSection>
  );

  const renderTemplatesTab = () => (
    <ContentSection>
      <h3 style={{ color: '#00ffff', fontFamily: 'Orbitron', marginBottom: '1.5rem' }}>
        📋 Report Templates
      </h3>
      
      <TemplateGrid>
        {templates.map((template) => (
          <TemplateCard
            key={template.template_id}
            selected={reportRequest.template_id === template.template_id}
            onClick={() => handleTemplateSelect(template)}
          >
            <TemplateTitle>{template.template_name}</TemplateTitle>
            <TemplateDescription>{template.description}</TemplateDescription>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <TemplateStakeholder>
                {template.stakeholder_type.toUpperCase()}
              </TemplateStakeholder>
              <div style={{ 
                color: '#00ffff', 
                fontFamily: 'Roboto Mono', 
                fontSize: '0.8rem' 
              }}>
                {template.sections.length} sections
              </div>
            </div>
          </TemplateCard>
        ))}
      </TemplateGrid>
    </ContentSection>
  );

  const renderAnalyticsTab = () => (
    <ContentSection>
      <h3 style={{ color: '#00ffff', fontFamily: 'Orbitron', marginBottom: '1.5rem' }}>
        📊 Sharing Analytics
      </h3>
      
      {analytics && (
        <div>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
            gap: '1rem',
            marginBottom: '2rem'
          }}>
            <div style={{
              background: 'rgba(0, 255, 255, 0.1)',
              border: '1px solid #00ffff',
              borderRadius: '10px',
              padding: '1.5rem',
              textAlign: 'center'
            }}>
              <div style={{ 
                color: '#00ffff', 
                fontSize: '2rem', 
                fontWeight: 'bold',
                fontFamily: 'Orbitron'
              }}>
                {analytics.total_shared_reports}
              </div>
              <div style={{ color: '#ffffff', fontFamily: 'Roboto Mono', fontSize: '0.9rem' }}>
                Total Shared Reports
              </div>
            </div>

            <div style={{
              background: 'rgba(255, 0, 255, 0.1)',
              border: '1px solid #ff00ff',
              borderRadius: '10px',
              padding: '1.5rem',
              textAlign: 'center'
            }}>
              <div style={{ 
                color: '#ff00ff', 
                fontSize: '2rem', 
                fontWeight: 'bold',
                fontFamily: 'Orbitron'
              }}>
                {analytics.total_views}
              </div>
              <div style={{ color: '#ffffff', fontFamily: 'Roboto Mono', fontSize: '0.9rem' }}>
                Total Views
              </div>
            </div>

            <div style={{
              background: 'rgba(255, 255, 0, 0.1)',
              border: '1px solid #ffff00',
              borderRadius: '10px',
              padding: '1.5rem',
              textAlign: 'center'
            }}>
              <div style={{ 
                color: '#ffff00', 
                fontSize: '2rem', 
                fontWeight: 'bold',
                fontFamily: 'Orbitron'
              }}>
                {analytics.average_views_per_report.toFixed(1)}
              </div>
              <div style={{ color: '#ffffff', fontFamily: 'Roboto Mono', fontSize: '0.9rem' }}>
                Avg Views per Report
              </div>
            </div>
          </div>

          {analytics.reports && analytics.reports.length > 0 && (
            <div>
              <h4 style={{ color: '#00ffff', fontFamily: 'Orbitron', marginBottom: '1rem' }}>
                Recent Shared Reports
              </h4>
              <div style={{ 
                background: 'rgba(0, 0, 0, 0.5)',
                border: '1px solid rgba(0, 255, 255, 0.3)',
                borderRadius: '10px',
                overflow: 'hidden'
              }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ background: 'rgba(0, 255, 255, 0.2)' }}>
                      <th style={{ 
                        color: '#00ffff', 
                        fontFamily: 'Orbitron', 
                        padding: '1rem',
                        textAlign: 'left'
                      }}>
                        Share ID
                      </th>
                      <th style={{ 
                        color: '#00ffff', 
                        fontFamily: 'Orbitron', 
                        padding: '1rem',
                        textAlign: 'left'
                      }}>
                        Type
                      </th>
                      <th style={{ 
                        color: '#00ffff', 
                        fontFamily: 'Orbitron', 
                        padding: '1rem',
                        textAlign: 'left'
                      }}>
                        Views
                      </th>
                      <th style={{ 
                        color: '#00ffff', 
                        fontFamily: 'Orbitron', 
                        padding: '1rem',
                        textAlign: 'left'
                      }}>
                        Created
                      </th>
                      <th style={{ 
                        color: '#00ffff', 
                        fontFamily: 'Orbitron', 
                        padding: '1rem',
                        textAlign: 'left'
                      }}>
                        Last Accessed
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {analytics.reports.slice(0, 10).map((report: any, index: number) => (
                      <tr key={report.share_id} style={{ 
                        borderBottom: '1px solid rgba(0, 255, 255, 0.1)',
                        background: index % 2 === 0 ? 'rgba(0, 255, 255, 0.05)' : 'transparent'
                      }}>
                        <td style={{ 
                          color: '#ffffff', 
                          fontFamily: 'Roboto Mono', 
                          padding: '0.75rem',
                          fontSize: '0.8rem'
                        }}>
                          {report.share_id.slice(-8)}
                        </td>
                        <td style={{ 
                          color: '#ffffff', 
                          fontFamily: 'Roboto Mono', 
                          padding: '0.75rem'
                        }}>
                          {report.report_type}
                        </td>
                        <td style={{ 
                          color: '#ffffff', 
                          fontFamily: 'Roboto Mono', 
                          padding: '0.75rem'
                        }}>
                          {report.access_count}
                        </td>
                        <td style={{ 
                          color: '#ffffff', 
                          fontFamily: 'Roboto Mono', 
                          padding: '0.75rem',
                          fontSize: '0.8rem'
                        }}>
                          {new Date(report.created_at).toLocaleDateString()}
                        </td>
                        <td style={{ 
                          color: '#ffffff', 
                          fontFamily: 'Roboto Mono', 
                          padding: '0.75rem',
                          fontSize: '0.8rem'
                        }}>
                          {report.last_accessed ? new Date(report.last_accessed).toLocaleDateString() : 'Never'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </ContentSection>
  );

  return (
    <ShareableContainer className={className}>
      <Title>🔗 SHAREABLE REPORTS CONTROL CENTER</Title>
      
      <TabContainer>
        <Tab 
          active={activeTab === 'create'} 
          onClick={() => setActiveTab('create')}
        >
          CREATE REPORT
        </Tab>
        <Tab 
          active={activeTab === 'schedule'} 
          onClick={() => setActiveTab('schedule')}
        >
          SCHEDULE REPORTS
        </Tab>
        <Tab 
          active={activeTab === 'templates'} 
          onClick={() => setActiveTab('templates')}
        >
          TEMPLATES
        </Tab>
        <Tab 
          active={activeTab === 'analytics'} 
          onClick={() => setActiveTab('analytics')}
        >
          ANALYTICS
        </Tab>
      </TabContainer>

      {activeTab === 'create' && renderCreateTab()}
      {activeTab === 'schedule' && renderScheduleTab()}
      {activeTab === 'templates' && renderTemplatesTab()}
      {activeTab === 'analytics' && renderAnalyticsTab()}

      {statusMessage && (
        <StatusMessage type={statusMessage.type}>
          {statusMessage.message}
        </StatusMessage>
      )}
    </ShareableContainer>
  );
};

export default ShareableReportsDashboard;