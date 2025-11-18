"""
Shareable Reporting System API
Provides shareable link generation, scheduled reports, and customizable report templates
for the ensemble forecasting system
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import os
import tempfile

# Import required components
try:
    from api.export_api import export_engine, ExportEngine, ExportRequest, ExportResponse
    from utils.export_formatters import comprehensive_formatter
    from models.ensemble_forecasting_engine import EnsembleForecastingEngine
    from company_sales.company_data_manager import CompanyDataManager
except ImportError as e:
    logging.warning(f"Import warning in shareable_reports_api: {e}")

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

router = APIRouter(prefix="/api/shareable-reports", tags=["Shareable Reports"])

# Pydantic models
class ShareableReportRequest(BaseModel):
    report_type: str = Field(..., description="Type of report: comprehensive, forecast, performance, insights")
    format: str = Field("json", description="Export format: json, excel, pdf")
    include_forecasts: bool = Field(True, description="Include forecast data")
    include_performance: bool = Field(True, description="Include model performance metrics")
    include_insights: bool = Field(True, description="Include business insights")
    include_metadata: bool = Field(True, description="Include model metadata")
    include_charts: bool = Field(True, description="Include interactive chart data")
    custom_title: Optional[str] = Field(None, description="Custom report title")
    horizon_months: Optional[int] = Field(6, description="Forecast horizon")
    stakeholder_type: str = Field("executive", description="Target stakeholder: executive, analyst, technical")
    template_id: Optional[str] = Field(None, description="Report template ID")
    expiration_hours: Optional[int] = Field(72, description="Link expiration in hours")
    password_protected: bool = Field(False, description="Enable password protection")
    allow_downloads: bool = Field(True, description="Allow file downloads")
    embed_interactive_charts: bool = Field(True, description="Embed interactive charts in HTML")

class ShareableReportResponse(BaseModel):
    success: bool
    share_id: str
    share_url: str
    qr_code_url: Optional[str]
    password: Optional[str]
    expires_at: str
    report_metadata: Dict[str, Any]
    access_controls: Dict[str, Any]

class ScheduledReportRequest(BaseModel):
    report_config: ShareableReportRequest
    schedule_type: str = Field(..., description="Schedule type: daily, weekly, monthly, quarterly")
    schedule_time: str = Field("09:00", description="Time to send report (HH:MM)")
    recipients: List[str] = Field(..., description="Email recipients")
    subject_template: Optional[str] = Field(None, description="Email subject template")
    message_template: Optional[str] = Field(None, description="Email message template")
    start_date: Optional[str] = Field(None, description="Schedule start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Schedule end date (YYYY-MM-DD)")
    timezone: str = Field("UTC", description="Timezone for scheduling")
    active: bool = Field(True, description="Schedule is active")

class ScheduledReportResponse(BaseModel):
    success: bool
    schedule_id: str
    next_run: str
    schedule_summary: Dict[str, Any]

class ReportTemplate(BaseModel):
    template_id: str
    template_name: str
    description: str
    stakeholder_type: str
    sections: List[str]
    styling: Dict[str, Any]
    default_config: Dict[str, Any]
    created_at: str
    updated_at: str

class ShareableReportManager:
    """Manages shareable reports, templates, and scheduling"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "shareable_reports"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Storage for shareable reports and schedules
        self.shared_reports = {}  # share_id -> report_data
        self.scheduled_reports = {}  # schedule_id -> schedule_config
        self.report_templates = {}  # template_id -> template_config
        
        # Initialize default templates
        self._initialize_default_templates()
        
        # Email configuration (would be loaded from environment)
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'localhost'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'smtp_username': os.getenv('SMTP_USERNAME', ''),
            'smtp_password': os.getenv('SMTP_PASSWORD', ''),
            'from_email': os.getenv('FROM_EMAIL', 'reports@company.com')
        }
    
    def _initialize_default_templates(self):
        """Initialize default report templates for different stakeholders"""
        
        # Executive template
        executive_template = {
            'template_id': 'executive-summary',
            'template_name': 'Executive Summary Report',
            'description': 'High-level summary for executives with key insights and recommendations',
            'stakeholder_type': 'executive',
            'sections': [
                'executive_summary',
                'key_findings',
                'growth_indicators',
                'risk_assessment',
                'top_recommendations',
                'forecast_summary'
            ],
            'styling': {
                'theme': 'professional',
                'color_scheme': 'corporate',
                'chart_style': 'clean',
                'font_family': 'Arial',
                'emphasis': 'insights'
            },
            'default_config': {
                'include_forecasts': True,
                'include_performance': False,
                'include_insights': True,
                'include_metadata': False,
                'include_charts': True,
                'horizon_months': 6
            },
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Analyst template
        analyst_template = {
            'template_id': 'analyst-detailed',
            'template_name': 'Detailed Analysis Report',
            'description': 'Comprehensive analysis for data analysts with detailed metrics and performance data',
            'stakeholder_type': 'analyst',
            'sections': [
                'forecast_analysis',
                'model_performance',
                'accuracy_metrics',
                'weight_evolution',
                'pattern_analysis',
                'confidence_intervals',
                'model_comparison',
                'recommendations'
            ],
            'styling': {
                'theme': 'analytical',
                'color_scheme': 'data_focused',
                'chart_style': 'detailed',
                'font_family': 'Roboto',
                'emphasis': 'metrics'
            },
            'default_config': {
                'include_forecasts': True,
                'include_performance': True,
                'include_insights': True,
                'include_metadata': True,
                'include_charts': True,
                'horizon_months': 12
            },
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Technical template
        technical_template = {
            'template_id': 'technical-deep-dive',
            'template_name': 'Technical Deep Dive Report',
            'description': 'Technical report for data scientists with model details and system metrics',
            'stakeholder_type': 'technical',
            'sections': [
                'model_architecture',
                'training_details',
                'performance_metrics',
                'drift_detection',
                'system_health',
                'model_versions',
                'configuration_details',
                'technical_recommendations'
            ],
            'styling': {
                'theme': 'technical',
                'color_scheme': 'cyberpunk',
                'chart_style': 'technical',
                'font_family': 'Roboto Mono',
                'emphasis': 'technical_details'
            },
            'default_config': {
                'include_forecasts': True,
                'include_performance': True,
                'include_insights': False,
                'include_metadata': True,
                'include_charts': True,
                'horizon_months': 6
            },
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        self.report_templates = {
            'executive-summary': executive_template,
            'analyst-detailed': analyst_template,
            'technical-deep-dive': technical_template
        }
    
    async def create_shareable_report(self, 
                                    company_id: str, 
                                    request: ShareableReportRequest) -> ShareableReportResponse:
        """Create a shareable report with embedded interactive charts"""
        try:
            # Generate unique share ID
            share_id = str(uuid.uuid4())
            
            # Apply template if specified
            if request.template_id and request.template_id in self.report_templates:
                template = self.report_templates[request.template_id]
                # Override request with template defaults
                for key, value in template['default_config'].items():
                    if not hasattr(request, key) or getattr(request, key) is None:
                        setattr(request, key, value)
            
            # Generate password if required
            password = None
            if request.password_protected:
                password = self._generate_password()
            
            # Create export request
            export_request = ExportRequest(
                format=request.format,
                include_forecasts=request.include_forecasts,
                include_performance=request.include_performance,
                include_insights=request.include_insights,
                include_metadata=request.include_metadata,
                include_charts=request.include_charts,
                custom_title=request.custom_title,
                horizon_months=request.horizon_months
            )
            
            # Generate report data
            report_data = await export_engine.generate_comprehensive_report(company_id, export_request)
            
            # Create interactive HTML report if charts are enabled
            html_report = None
            if request.embed_interactive_charts:
                html_report = await self._create_interactive_html_report(
                    report_data, request, company_id
                )
            
            # Calculate expiration
            expires_at = datetime.now() + timedelta(hours=request.expiration_hours)
            
            # Store shareable report
            self.shared_reports[share_id] = {
                'company_id': company_id,
                'request': request.dict(),
                'report_data': report_data,
                'html_report': html_report,
                'password': password,
                'created_at': datetime.now().isoformat(),
                'expires_at': expires_at.isoformat(),
                'access_count': 0,
                'last_accessed': None
            }
            
            # Generate share URL
            share_url = f"/api/shareable-reports/view/{share_id}"
            
            # Generate QR code URL (placeholder)
            qr_code_url = f"/api/shareable-reports/qr/{share_id}" if request.embed_interactive_charts else None
            
            return ShareableReportResponse(
                success=True,
                share_id=share_id,
                share_url=share_url,
                qr_code_url=qr_code_url,
                password=password,
                expires_at=expires_at.isoformat(),
                report_metadata={
                    'report_type': request.report_type,
                    'format': request.format,
                    'stakeholder_type': request.stakeholder_type,
                    'template_id': request.template_id,
                    'company_id': company_id,
                    'interactive_charts': request.embed_interactive_charts,
                    'components_included': {
                        'forecasts': request.include_forecasts,
                        'performance': request.include_performance,
                        'insights': request.include_insights,
                        'metadata': request.include_metadata,
                        'charts': request.include_charts
                    }
                },
                access_controls={
                    'password_protected': request.password_protected,
                    'allow_downloads': request.allow_downloads,
                    'expiration_hours': request.expiration_hours,
                    'interactive_charts': request.embed_interactive_charts
                }
            )
            
        except Exception as e:
            logger.error(f"Shareable report creation failed: {e}")
            raise
    
    async def _create_interactive_html_report(self, 
                                            report_data: Dict[str, Any], 
                                            request: ShareableReportRequest,
                                            company_id: str) -> str:
        """Create interactive HTML report with embedded charts"""
        
        # Apply template styling if specified
        template_styling = {}
        if request.template_id and request.template_id in self.report_templates:
            template_styling = self.report_templates[request.template_id]['styling']
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{request.custom_title or f'Ensemble Forecast Report - {company_id}'}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                {self._get_html_styles(template_styling)}
            </style>
        </head>
        <body>
            <div class="report-container">
                <header class="report-header">
                    <h1>{request.custom_title or f'Ensemble Forecast Report'}</h1>
                    <div class="report-meta">
                        <span>Company: {company_id}</span>
                        <span>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                        <span>Type: {request.stakeholder_type.title()}</span>
                    </div>
                </header>
                
                <main class="report-content">
                    {await self._generate_html_sections(report_data, request, template_styling)}
                </main>
                
                <footer class="report-footer">
                    <p>Generated by Ensemble Forecasting System | 
                       <span class="timestamp">{datetime.now().isoformat()}</span>
                    </p>
                </footer>
            </div>
            
            <script>
                {self._get_interactive_scripts(report_data)}
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def _get_html_styles(self, template_styling: Dict[str, Any]) -> str:
        """Generate CSS styles based on template"""
        
        # Base styles
        base_styles = """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }
        
        .report-container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        
        .report-header h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        
        .report-meta {
            display: flex;
            justify-content: center;
            gap: 2rem;
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        .report-content {
            padding: 2rem;
        }
        
        .section {
            margin-bottom: 3rem;
            padding: 1.5rem;
            border-radius: 8px;
            background: #fafafa;
            border-left: 4px solid #667eea;
        }
        
        .section h2 {
            color: #667eea;
            margin-bottom: 1rem;
            font-size: 1.8rem;
        }
        
        .chart-container {
            margin: 1.5rem 0;
            padding: 1rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        
        .report-footer {
            background: #333;
            color: white;
            text-align: center;
            padding: 1rem;
            font-size: 0.8rem;
        }
        
        .timestamp {
            opacity: 0.7;
        }
        """
        
        # Apply template-specific styles
        if template_styling.get('theme') == 'cyberpunk':
            base_styles += """
            body {
                background: #0a0a0a;
                color: #00ffff;
            }
            
            .report-container {
                background: #1a1a2e;
                border: 2px solid #00ffff;
            }
            
            .report-header {
                background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
                border-bottom: 2px solid #00ffff;
            }
            
            .section {
                background: rgba(0, 255, 255, 0.05);
                border-left: 4px solid #00ffff;
                border: 1px solid rgba(0, 255, 255, 0.3);
            }
            
            .section h2 {
                color: #00ffff;
                text-shadow: 0 0 10px #00ffff;
            }
            
            .metric-card {
                background: rgba(0, 255, 255, 0.1);
                border: 1px solid rgba(0, 255, 255, 0.3);
            }
            
            .metric-value {
                color: #00ffff;
                text-shadow: 0 0 5px #00ffff;
            }
            """
        
        return base_styles
    
    async def _generate_html_sections(self, 
                                    report_data: Dict[str, Any], 
                                    request: ShareableReportRequest,
                                    template_styling: Dict[str, Any]) -> str:
        """Generate HTML sections based on report data and template"""
        
        sections_html = ""
        
        # Executive Summary
        if report_data.get('insights', {}).get('executive_summary'):
            sections_html += f"""
            <section class="section">
                <h2>Executive Summary</h2>
                <p>{report_data['insights']['executive_summary']}</p>
            </section>
            """
        
        # Key Metrics
        if report_data.get('forecasts') or report_data.get('performance'):
            metrics_html = '<div class="metrics-grid">'
            
            # Forecast metrics
            if report_data.get('forecasts', {}).get('ensemble_forecast'):
                forecast_values = report_data['forecasts']['ensemble_forecast']['values']
                avg_forecast = np.mean(forecast_values)
                metrics_html += f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_forecast:.2f}</div>
                    <div class="metric-label">Average Forecast</div>
                </div>
                """
            
            # Performance metrics
            if report_data.get('performance', {}).get('current_performance'):
                current_perf = report_data['performance']['current_performance']
                if 'overall_accuracy' in current_perf:
                    metrics_html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{current_perf['overall_accuracy']:.2%}</div>
                        <div class="metric-label">Overall Accuracy</div>
                    </div>
                    """
            
            # Data quality
            if report_data.get('metadata', {}).get('data_quality_score'):
                quality_score = report_data['metadata']['data_quality_score']
                metrics_html += f"""
                <div class="metric-card">
                    <div class="metric-value">{quality_score:.2%}</div>
                    <div class="metric-label">Data Quality</div>
                </div>
                """
            
            metrics_html += '</div>'
            
            sections_html += f"""
            <section class="section">
                <h2>Key Metrics</h2>
                {metrics_html}
            </section>
            """
        
        # Interactive Forecast Chart
        if request.include_charts and report_data.get('forecasts', {}).get('ensemble_forecast'):
            sections_html += f"""
            <section class="section">
                <h2>Forecast Visualization</h2>
                <div class="chart-container">
                    <div id="forecast-chart" style="height: 400px;"></div>
                </div>
            </section>
            """
        
        # Model Performance Chart
        if request.include_charts and report_data.get('performance', {}).get('model_rankings'):
            sections_html += f"""
            <section class="section">
                <h2>Model Performance</h2>
                <div class="chart-container">
                    <div id="performance-chart" style="height: 300px;"></div>
                </div>
            </section>
            """
        
        # Key Findings
        if report_data.get('insights', {}).get('key_findings'):
            findings_html = "<ul>"
            for finding in report_data['insights']['key_findings'][:5]:
                findings_html += f"<li>{finding}</li>"
            findings_html += "</ul>"
            
            sections_html += f"""
            <section class="section">
                <h2>Key Findings</h2>
                {findings_html}
            </section>
            """
        
        # Recommendations
        if report_data.get('insights', {}).get('recommendations'):
            rec_html = ""
            for i, rec in enumerate(report_data['insights']['recommendations'][:5], 1):
                rec_html += f"""
                <div class="recommendation">
                    <h4>{i}. {rec.get('title', 'N/A')}</h4>
                    <p>{rec.get('description', 'N/A')}</p>
                    <small>Urgency: {rec.get('urgency', 'N/A')} | 
                           Confidence: {rec.get('confidence', 0):.2%}</small>
                </div>
                """
            
            sections_html += f"""
            <section class="section">
                <h2>Recommendations</h2>
                {rec_html}
            </section>
            """
        
        return sections_html
    
    def _get_interactive_scripts(self, report_data: Dict[str, Any]) -> str:
        """Generate JavaScript for interactive charts"""
        
        scripts = ""
        
        # Forecast chart
        if report_data.get('forecasts', {}).get('ensemble_forecast'):
            forecast_data = report_data['forecasts']['ensemble_forecast']
            dates = forecast_data['dates']
            values = forecast_data['values']
            
            scripts += f"""
            // Forecast Chart
            var forecastData = {{
                x: {json.dumps(dates)},
                y: {json.dumps(values)},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Ensemble Forecast',
                line: {{color: '#667eea', width: 3}},
                marker: {{size: 6}}
            }};
            
            var forecastLayout = {{
                title: 'Ensemble Forecast',
                xaxis: {{title: 'Date'}},
                yaxis: {{title: 'Forecast Value'}},
                showlegend: true,
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)'
            }};
            
            if (document.getElementById('forecast-chart')) {{
                Plotly.newPlot('forecast-chart', [forecastData], forecastLayout, {{responsive: true}});
            }}
            """
        
        # Performance chart
        if report_data.get('performance', {}).get('model_rankings'):
            model_rankings = report_data['performance']['model_rankings']
            model_names = [model.get('model_name', 'Unknown') for model in model_rankings[:5]]
            accuracies = [model.get('accuracy', 0) for model in model_rankings[:5]]
            
            scripts += f"""
            // Performance Chart
            var performanceData = {{
                x: {json.dumps(model_names)},
                y: {json.dumps(accuracies)},
                type: 'bar',
                marker: {{
                    color: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
                }}
            }};
            
            var performanceLayout = {{
                title: 'Model Performance Comparison',
                xaxis: {{title: 'Models'}},
                yaxis: {{title: 'Accuracy'}},
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)'
            }};
            
            if (document.getElementById('performance-chart')) {{
                Plotly.newPlot('performance-chart', [performanceData], performanceLayout, {{responsive: true}});
            }}
            """
        
        return scripts
    
    def _generate_password(self) -> str:
        """Generate a secure password for protected reports"""
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(12))
    
    async def schedule_report(self, 
                            company_id: str, 
                            request: ScheduledReportRequest) -> ScheduledReportResponse:
        """Schedule automated report generation and distribution"""
        try:
            schedule_id = str(uuid.uuid4())
            
            # Calculate next run time
            next_run = self._calculate_next_run(request.schedule_type, request.schedule_time)
            
            # Store scheduled report
            self.scheduled_reports[schedule_id] = {
                'company_id': company_id,
                'config': request.dict(),
                'created_at': datetime.now().isoformat(),
                'next_run': next_run.isoformat(),
                'last_run': None,
                'run_count': 0,
                'active': request.active
            }
            
            # Start background scheduler (in production, use proper task queue)
            asyncio.create_task(self._schedule_background_task(schedule_id))
            
            return ScheduledReportResponse(
                success=True,
                schedule_id=schedule_id,
                next_run=next_run.isoformat(),
                schedule_summary={
                    'schedule_type': request.schedule_type,
                    'schedule_time': request.schedule_time,
                    'recipients_count': len(request.recipients),
                    'report_type': request.report_config.report_type,
                    'active': request.active,
                    'timezone': request.timezone
                }
            )
            
        except Exception as e:
            logger.error(f"Report scheduling failed: {e}")
            raise
    
    def _calculate_next_run(self, schedule_type: str, schedule_time: str) -> datetime:
        """Calculate next run time based on schedule type"""
        now = datetime.now()
        hour, minute = map(int, schedule_time.split(':'))
        
        if schedule_type == 'daily':
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        elif schedule_type == 'weekly':
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            days_ahead = 0 - now.weekday()  # Monday = 0
            if days_ahead <= 0:
                days_ahead += 7
            next_run += timedelta(days=days_ahead)
        elif schedule_type == 'monthly':
            next_run = now.replace(day=1, hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                if next_run.month == 12:
                    next_run = next_run.replace(year=next_run.year + 1, month=1)
                else:
                    next_run = next_run.replace(month=next_run.month + 1)
        elif schedule_type == 'quarterly':
            # First day of next quarter
            current_quarter = (now.month - 1) // 3
            next_quarter_month = (current_quarter + 1) * 3 + 1
            if next_quarter_month > 12:
                next_quarter_month = 1
                next_run = now.replace(year=now.year + 1, month=next_quarter_month, day=1, 
                                     hour=hour, minute=minute, second=0, microsecond=0)
            else:
                next_run = now.replace(month=next_quarter_month, day=1, 
                                     hour=hour, minute=minute, second=0, microsecond=0)
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")
        
        return next_run
    
    async def _schedule_background_task(self, schedule_id: str):
        """Background task for scheduled report execution"""
        while schedule_id in self.scheduled_reports:
            schedule_data = self.scheduled_reports[schedule_id]
            
            if not schedule_data['active']:
                await asyncio.sleep(3600)  # Check every hour
                continue
            
            next_run = datetime.fromisoformat(schedule_data['next_run'])
            
            if datetime.now() >= next_run:
                try:
                    await self._execute_scheduled_report(schedule_id)
                except Exception as e:
                    logger.error(f"Scheduled report execution failed for {schedule_id}: {e}")
                
                # Calculate next run
                config = schedule_data['config']
                next_run = self._calculate_next_run(config['schedule_type'], config['schedule_time'])
                schedule_data['next_run'] = next_run.isoformat()
                schedule_data['last_run'] = datetime.now().isoformat()
                schedule_data['run_count'] += 1
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _execute_scheduled_report(self, schedule_id: str):
        """Execute a scheduled report and send to recipients"""
        schedule_data = self.scheduled_reports[schedule_id]
        company_id = schedule_data['company_id']
        config = schedule_data['config']
        
        # Generate report
        shareable_response = await self.create_shareable_report(
            company_id, 
            ShareableReportRequest(**config['report_config'])
        )
        
        # Send email to recipients
        await self._send_scheduled_report_email(
            schedule_data, 
            shareable_response
        )
    
    async def _send_scheduled_report_email(self, 
                                         schedule_data: Dict[str, Any], 
                                         report_response: ShareableReportResponse):
        """Send scheduled report via email"""
        try:
            config = schedule_data['config']
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(config['recipients'])
            
            # Subject
            subject = config.get('subject_template', 'Scheduled Forecast Report')
            subject = subject.replace('{company_id}', schedule_data['company_id'])
            subject = subject.replace('{date}', datetime.now().strftime('%Y-%m-%d'))
            msg['Subject'] = subject
            
            # Message body
            message = config.get('message_template', 
                               f"Your scheduled forecast report is ready. Access it here: {report_response.share_url}")
            message = message.replace('{company_id}', schedule_data['company_id'])
            message = message.replace('{share_url}', report_response.share_url)
            message = message.replace('{date}', datetime.now().strftime('%Y-%m-%d'))
            
            if report_response.password:
                message += f"\n\nPassword: {report_response.password}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            if self.email_config['smtp_username']:
                server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
                server.starttls()
                server.login(self.email_config['smtp_username'], self.email_config['smtp_password'])
                server.send_message(msg)
                server.quit()
                
                logger.info(f"Scheduled report email sent for schedule {schedule_data}")
            else:
                logger.warning("Email configuration not available, skipping email send")
                
        except Exception as e:
            logger.error(f"Failed to send scheduled report email: {e}")

# Initialize manager
shareable_manager = ShareableReportManager()

# Helper function
def get_company_id_from_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract company ID from authorization token"""
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

# API Endpoints

@router.post("/create", response_model=ShareableReportResponse)
async def create_shareable_report(
    request: ShareableReportRequest,
    company_id: str = Depends(get_company_id_from_token)
):
    """Create a shareable report with embedded interactive charts"""
    try:
        return await shareable_manager.create_shareable_report(company_id, request)
    except Exception as e:
        logger.error(f"Shareable report creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report creation failed: {str(e)}")

@router.get("/view/{share_id}")
async def view_shareable_report(share_id: str, password: Optional[str] = Query(None)):
    """View a shareable report"""
    try:
        if share_id not in shareable_manager.shared_reports:
            raise HTTPException(status_code=404, detail="Report not found or expired")
        
        report_data = shareable_manager.shared_reports[share_id]
        
        # Check expiration
        expires_at = datetime.fromisoformat(report_data['expires_at'])
        if datetime.now() > expires_at:
            del shareable_manager.shared_reports[share_id]
            raise HTTPException(status_code=410, detail="Report has expired")
        
        # Check password
        if report_data['password'] and password != report_data['password']:
            raise HTTPException(status_code=401, detail="Invalid password")
        
        # Update access tracking
        report_data['access_count'] += 1
        report_data['last_accessed'] = datetime.now().isoformat()
        
        # Return HTML report if available
        if report_data.get('html_report'):
            return HTMLResponse(content=report_data['html_report'])
        else:
            return JSONResponse(content=report_data['report_data'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report viewing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report viewing failed: {str(e)}")

@router.post("/schedule", response_model=ScheduledReportResponse)
async def schedule_report(
    request: ScheduledReportRequest,
    company_id: str = Depends(get_company_id_from_token)
):
    """Schedule automated report generation and distribution"""
    try:
        return await shareable_manager.schedule_report(company_id, request)
    except Exception as e:
        logger.error(f"Report scheduling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scheduling failed: {str(e)}")

@router.get("/templates")
async def get_report_templates():
    """Get available report templates"""
    try:
        templates = []
        for template_id, template_data in shareable_manager.report_templates.items():
            templates.append(ReportTemplate(**template_data))
        
        return {
            'success': True,
            'templates': templates,
            'total_templates': len(templates)
        }
    except Exception as e:
        logger.error(f"Template retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Template retrieval failed: {str(e)}")

@router.get("/schedules")
async def get_scheduled_reports(company_id: str = Depends(get_company_id_from_token)):
    """Get scheduled reports for a company"""
    try:
        company_schedules = []
        for schedule_id, schedule_data in shareable_manager.scheduled_reports.items():
            if schedule_data['company_id'] == company_id:
                company_schedules.append({
                    'schedule_id': schedule_id,
                    'config': schedule_data['config'],
                    'next_run': schedule_data['next_run'],
                    'last_run': schedule_data['last_run'],
                    'run_count': schedule_data['run_count'],
                    'active': schedule_data['active']
                })
        
        return {
            'success': True,
            'schedules': company_schedules,
            'total_schedules': len(company_schedules)
        }
    except Exception as e:
        logger.error(f"Schedule retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Schedule retrieval failed: {str(e)}")

@router.delete("/schedule/{schedule_id}")
async def delete_scheduled_report(
    schedule_id: str,
    company_id: str = Depends(get_company_id_from_token)
):
    """Delete a scheduled report"""
    try:
        if schedule_id not in shareable_manager.scheduled_reports:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        schedule_data = shareable_manager.scheduled_reports[schedule_id]
        if schedule_data['company_id'] != company_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        del shareable_manager.scheduled_reports[schedule_id]
        
        return {
            'success': True,
            'message': 'Schedule deleted successfully'
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Schedule deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.get("/analytics")
async def get_sharing_analytics(company_id: str = Depends(get_company_id_from_token)):
    """Get analytics for shared reports"""
    try:
        company_reports = []
        total_views = 0
        
        for share_id, report_data in shareable_manager.shared_reports.items():
            if report_data['company_id'] == company_id:
                company_reports.append({
                    'share_id': share_id,
                    'created_at': report_data['created_at'],
                    'access_count': report_data['access_count'],
                    'last_accessed': report_data['last_accessed'],
                    'report_type': report_data['request']['report_type'],
                    'stakeholder_type': report_data['request']['stakeholder_type']
                })
                total_views += report_data['access_count']
        
        return {
            'success': True,
            'analytics': {
                'total_shared_reports': len(company_reports),
                'total_views': total_views,
                'average_views_per_report': total_views / len(company_reports) if company_reports else 0,
                'reports': company_reports
            }
        }
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")