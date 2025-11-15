"""
Company Sales Forecasting Dashboard
Interactive dashboard for company-specific sales forecasting and analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import io
from typing import Dict, List, Optional, Any

# Import company sales components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from company_sales.company_data_manager import CompanyDataManager
    from company_sales.company_forecasting_engine import CompanyForecastingEngine
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from company_sales.company_data_manager import CompanyDataManager
    from company_sales.company_forecasting_engine import CompanyForecastingEngine

class CompanySalesDashboard:
    """Interactive dashboard for company sales forecasting"""
    
    def __init__(self):
        self.company_data_manager = CompanyDataManager()
        self.forecasting_engine = CompanyForecastingEngine(self.company_data_manager)
        
        # Initialize session state
        if 'company_id' not in st.session_state:
            st.session_state.company_id = None
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
    
    def run(self):
        """Run the dashboard application"""
        
        st.set_page_config(
            page_title="Company Sales Forecasting Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 0.75rem;
            border-radius: 0.25rem;
            border: 1px solid #c3e6cb;
        }
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            padding: 0.75rem;
            border-radius: 0.25rem;
            border: 1px solid #f5c6cb;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main header
        st.markdown('<div class="main-header">📈 Company Sales Forecasting Dashboard</div>', 
                   unsafe_allow_html=True)
        
        # Authentication and company selection
        if not st.session_state.authenticated:
            self._show_authentication()
        else:
            self._show_main_dashboard()
    
    def _show_authentication(self):
        """Show authentication and company selection"""
        
        st.markdown("### 🔐 Company Authentication")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Company selection/registration
            auth_option = st.radio(
                "Choose an option:",
                ["Login to Existing Company", "Register New Company"]
            )
            
            if auth_option == "Login to Existing Company":
                self._show_company_login()
            else:
                self._show_company_registration()
    
    def _show_company_login(self):
        """Show company login form"""
        
        st.markdown("#### Login to Your Company Account")
        
        # Get list of registered companies
        companies = list(self.company_data_manager.companies.keys())
        
        if not companies:
            st.warning("No companies registered yet. Please register a new company.")
            return
        
        company_id = st.selectbox(
            "Select your company:",
            options=companies,
            format_func=lambda x: f"{x} - {self.company_data_manager.companies[x].company_name}"
        )
        
        # Simple authentication (token input removed for local/dev convenience)
        st.info(
            "Authentication token input has been removed for local development. "
            "If authentication is enabled on the server, use the API login flow or unset DISABLE_AUTH."
        )

        if st.button("Login"):
            # In local/dev, allow login without a token to make it easy to test the dashboard.
            # In production, this should validate credentials or tokens.
            st.session_state.company_id = company_id
            st.session_state.authenticated = True
            st.success(f"Successfully logged in as {self.company_data_manager.companies[company_id].company_name}")
            st.experimental_rerun()
    
    def _show_company_registration(self):
        """Show company registration form"""
        
        st.markdown("#### Register New Company")
        
        with st.form("company_registration"):
            company_id = st.text_input("Company ID:", help="Unique identifier for your company")
            company_name = st.text_input("Company Name:")
            industry = st.selectbox(
                "Industry:",
                ["Retail", "E-commerce", "Manufacturing", "Technology", "Healthcare", 
                 "Finance", "Food & Beverage", "Automotive", "Other"]
            )
            
            # Custom requirements
            st.markdown("**Optional: Custom Data Requirements**")
            custom_min_months = st.number_input("Minimum months of data:", min_value=1, value=3)
            custom_max_file_size = st.number_input("Max file size (MB):", min_value=1, value=50)
            
            submitted = st.form_submit_button("Register Company")
            
            if submitted:
                if company_id and company_name:
                    try:
                        custom_requirements = {
                            'min_months': custom_min_months,
                            'max_file_size_mb': custom_max_file_size
                        }
                        
                        profile = self.company_data_manager.register_company(
                            company_id=company_id,
                            company_name=company_name,
                            industry=industry,
                            custom_requirements=custom_requirements
                        )
                        
                        st.success(f"Company '{company_name}' registered successfully!")
                        st.session_state.company_id = company_id
                        st.session_state.authenticated = True
                        st.experimental_rerun()
                        
                    except ValueError as e:
                        st.error(f"Registration failed: {str(e)}")
                else:
                    st.error("Please fill in all required fields")
    
    def _show_main_dashboard(self):
        """Show main dashboard interface"""
        
        # Sidebar
        with st.sidebar:
            self._show_sidebar()
        
        # Main content
        company_profile = self.company_data_manager.get_company_profile(st.session_state.company_id)
        
        if not company_profile:
            st.error("Company profile not found")
            return
        
        # Company header
        st.markdown(f"## 🏢 {company_profile.company_name}")
        st.markdown(f"**Industry:** {company_profile.industry} | **Company ID:** {company_profile.company_id}")
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", "📤 Upload Data", "🔮 Forecasting", 
            "📈 Analytics", "⚙️ Settings"
        ])
        
        with tab1:
            self._show_dashboard_overview()
        
        with tab2:
            self._show_data_upload()
        
        with tab3:
            self._show_forecasting()
        
        with tab4:
            self._show_analytics()
        
        with tab5:
            self._show_settings()
    
    def _show_sidebar(self):
        """Show sidebar with navigation and quick stats"""
        
        st.markdown("### 📋 Quick Stats")
        
        try:
            stats = self.company_data_manager.get_company_stats(st.session_state.company_id)
            
            st.metric("Total Uploads", stats['company_info']['total_uploads'])
            
            if stats['data_summary'].get('total_records', 0) > 0:
                st.metric("Total Records", stats['data_summary']['total_records'])
                st.metric("Product Categories", stats['data_summary']['categories'])
                st.metric("Regions", stats['data_summary']['regions'])
                
                if stats['data_summary'].get('avg_monthly_sales'):
                    st.metric("Avg Monthly Sales", f"${stats['data_summary']['avg_monthly_sales']:,.0f}")
        
        except Exception as e:
            st.error(f"Failed to load stats: {e}")
        
        st.markdown("---")
        
        # Model status
        st.markdown("### 🤖 Model Status")
        
        try:
            model_status = self.forecasting_engine.get_company_model_status(st.session_state.company_id)
            
            if model_status['status'] == 'initialized':
                st.success("✅ Models Initialized")
                st.write(f"**Pattern:** {model_status['pattern_detected']}")
                st.write(f"**Confidence:** {model_status['pattern_confidence']:.2f}")
                st.write(f"**Total Forecasts:** {model_status['total_forecasts']}")
            else:
                st.warning("⚠️ Models Not Initialized")
                st.write("Upload data to initialize models")
        
        except Exception:
            st.warning("⚠️ Models Not Available")
        
        st.markdown("---")
        
        # Logout
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.company_id = None
            st.experimental_rerun()
    
    def _show_dashboard_overview(self):
        """Show dashboard overview with key metrics and charts"""
        
        st.markdown("### 📊 Company Overview")
        
        try:
            # Load company data
            data = self.company_data_manager.load_company_data(st.session_state.company_id)
            
            if data.empty:
                st.info("📤 No data uploaded yet. Go to the 'Upload Data' tab to get started.")
                return
            
            # Aggregate monthly data
            monthly_data = self._aggregate_monthly_data(data)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sales = monthly_data['sales_amount'].sum()
                st.metric("Total Sales", f"${total_sales:,.0f}")
            
            with col2:
                avg_monthly = monthly_data['sales_amount'].mean()
                st.metric("Avg Monthly Sales", f"${avg_monthly:,.0f}")
            
            with col3:
                growth_rate = self._calculate_growth_rate(monthly_data['sales_amount'])
                st.metric("Growth Rate", f"{growth_rate:.1f}%")
            
            with col4:
                data_months = len(monthly_data)
                st.metric("Data Months", data_months)
            
            # Sales trend chart
            st.markdown("#### 📈 Sales Trend")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=monthly_data['date'],
                y=monthly_data['sales_amount'],
                mode='lines+markers',
                name='Monthly Sales',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Monthly Sales Trend",
                xaxis_title="Date",
                yaxis_title="Sales Amount ($)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Category and region breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                if 'product_category' in data.columns:
                    st.markdown("#### 🏷️ Sales by Category")
                    category_sales = data.groupby('product_category')['sales_amount'].sum().sort_values(ascending=False)
                    
                    fig_cat = px.pie(
                        values=category_sales.values,
                        names=category_sales.index,
                        title="Sales Distribution by Category"
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
            
            with col2:
                if 'region' in data.columns:
                    st.markdown("#### 🌍 Sales by Region")
                    region_sales = data.groupby('region')['sales_amount'].sum().sort_values(ascending=False)
                    
                    fig_reg = px.bar(
                        x=region_sales.index,
                        y=region_sales.values,
                        title="Sales by Region",
                        labels={'x': 'Region', 'y': 'Sales Amount ($)'}
                    )
                    st.plotly_chart(fig_reg, use_container_width=True)
        
        except Exception as e:
            st.error(f"Failed to load dashboard data: {e}")
    
    def _show_data_upload(self):
        """Show data upload interface"""
        
        st.markdown("### 📤 Upload Sales Data")
        
        # Data requirements
        try:
            requirements = self.company_data_manager.get_data_requirements_template(st.session_state.company_id)
            
            with st.expander("📋 Data Requirements & Template", expanded=False):
                st.markdown("#### Required Columns:")
                for col in requirements['data_requirements']['required_columns']:
                    st.write(f"• **{col}**")
                
                st.markdown("#### Optional Columns (improve accuracy):")
                for col in requirements['data_requirements']['optional_columns']:
                    st.write(f"• {col}")
                
                st.markdown("#### Sample Data Format:")
                sample_df = pd.DataFrame(requirements['sample_data'])
                st.dataframe(sample_df)
                
                st.markdown("#### Validation Rules:")
                for rule, description in requirements['validation_rules'].items():
                    st.write(f"• **{rule}:** {description}")
                
                # Download template
                csv_template = sample_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Template",
                    data=csv_template,
                    file_name="sales_data_template.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Failed to load data requirements: {e}")
            return
        
        # File upload
        st.markdown("#### Upload Your Sales Data")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json'],
            help="Upload your monthly sales data in CSV, Excel, or JSON format"
        )
        
        if uploaded_file is not None:
            try:
                # Parse file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.markdown("#### 👀 Data Preview")
                st.dataframe(df.head(10))
                
                st.markdown("#### 📊 Data Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", len(df))
                
                with col2:
                    st.metric("Columns", len(df.columns))
                
                with col3:
                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    st.metric("Missing Data", f"{missing_pct:.1f}%")
                
                # Validation
                is_valid, errors = self.company_data_manager.validate_upload_data(st.session_state.company_id, df)
                
                if errors:
                    st.markdown("#### ⚠️ Validation Issues")
                    for error in errors:
                        st.error(error)
                
                # Upload button
                if st.button("📤 Upload Data", disabled=not is_valid):
                    try:
                        file_path = self.company_data_manager.save_company_data(
                            company_id=st.session_state.company_id,
                            data=df,
                            upload_metadata={
                                'original_filename': uploaded_file.name,
                                'file_format': uploaded_file.name.split('.')[-1],
                                'upload_source': 'dashboard'
                            }
                        )
                        
                        st.success(f"✅ Data uploaded successfully! {len(df)} records processed.")
                        
                        # Initialize models if needed
                        if st.session_state.company_id not in self.forecasting_engine.company_models:
                            with st.spinner("Initializing forecasting models..."):
                                success = self.forecasting_engine.initialize_company_models(st.session_state.company_id)
                                if success:
                                    st.success("🤖 Forecasting models initialized successfully!")
                                else:
                                    st.warning("⚠️ Model initialization failed. You may need more data.")
                        
                        # Refresh the page to update stats
                        st.experimental_rerun()
                    
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
            
            except Exception as e:
                st.error(f"Failed to parse file: {e}")
    
    def _show_forecasting(self):
        """Show forecasting interface and results"""
        
        st.markdown("### 🔮 Sales Forecasting")
        
        # Check if models are initialized
        try:
            model_status = self.forecasting_engine.get_company_model_status(st.session_state.company_id)
            
            if model_status['status'] != 'initialized':
                st.warning("⚠️ Forecasting models not initialized. Please upload data first.")
                return
        
        except Exception:
            st.warning("⚠️ Forecasting not available. Please upload data first.")
            return
        
        # Forecast parameters
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("#### Forecast Settings")
            
            horizon_months = st.slider(
                "Forecast Horizon (months)",
                min_value=1,
                max_value=12,
                value=6,
                help="Number of months to forecast into the future"
            )
            
            include_confidence = st.checkbox(
                "Include Confidence Intervals",
                value=True,
                help="Show P10, P50, P90 confidence intervals"
            )
            
            if st.button("🔮 Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    try:
                        result = self.forecasting_engine.generate_forecast(
                            company_id=st.session_state.company_id,
                            horizon_months=horizon_months
                        )
                        
                        st.session_state.forecast_result = result
                        st.success("✅ Forecast generated successfully!")
                    
                    except Exception as e:
                        st.error(f"Forecast generation failed: {e}")
        
        with col2:
            # Display forecast results
            if hasattr(st.session_state, 'forecast_result'):
                self._display_forecast_results(st.session_state.forecast_result, include_confidence)
    
    def _display_forecast_results(self, result, include_confidence=True):
        """Display forecast results with charts and metrics"""
        
        st.markdown("#### 📊 Forecast Results")
        
        # Forecast metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not result.point_forecast.empty:
                total_forecast = result.point_forecast.sum()
                st.metric("Total Forecast", f"${total_forecast:,.0f}")
        
        with col2:
            if not result.point_forecast.empty:
                avg_monthly = result.point_forecast.mean()
                st.metric("Avg Monthly", f"${avg_monthly:,.0f}")
        
        with col3:
            if result.forecast_accuracy_metrics:
                mape = result.forecast_accuracy_metrics.get('ensemble_mape', 0)
                st.metric("Forecast MAPE", f"{mape:.1f}%")
        
        with col4:
            st.metric("Pattern Detected", result.pattern_detected.pattern_type.title())
        
        # Forecast chart
        if not result.point_forecast.empty:
            # Load historical data for context
            try:
                historical_data = self.company_data_manager.load_company_data(st.session_state.company_id)
                monthly_historical = self._aggregate_monthly_data(historical_data)
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=monthly_historical['date'],
                    y=monthly_historical['sales_amount'],
                    mode='lines+markers',
                    name='Historical Sales',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6)
                ))
                
                # Point forecast
                fig.add_trace(go.Scatter(
                    x=result.point_forecast.index,
                    y=result.point_forecast.values,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    marker=dict(size=8)
                ))
                
                # Confidence intervals
                if include_confidence and result.confidence_intervals:
                    if 'p90' in result.confidence_intervals and 'p10' in result.confidence_intervals:
                        fig.add_trace(go.Scatter(
                            x=result.confidence_intervals['p90'].index,
                            y=result.confidence_intervals['p90'].values,
                            mode='lines',
                            name='P90 (Upper)',
                            line=dict(color='rgba(255,127,14,0.3)', width=1),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=result.confidence_intervals['p10'].index,
                            y=result.confidence_intervals['p10'].values,
                            mode='lines',
                            name='P10-P90 Interval',
                            line=dict(color='rgba(255,127,14,0.3)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(255,127,14,0.2)'
                        ))
                
                fig.update_layout(
                    title="Sales Forecast with Historical Context",
                    xaxis_title="Date",
                    yaxis_title="Sales Amount ($)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Failed to create forecast chart: {e}")
        
        # Model performance and weights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Model Weights")
            
            if result.model_weights:
                weights_df = pd.DataFrame(
                    list(result.model_weights.items()),
                    columns=['Model', 'Weight']
                )
                weights_df['Weight'] = weights_df['Weight'].round(3)
                
                fig_weights = px.bar(
                    weights_df,
                    x='Model',
                    y='Weight',
                    title="Current Model Weights"
                )
                st.plotly_chart(fig_weights, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Model Performance")
            
            if result.model_performances:
                perf_data = []
                for perf in result.model_performances:
                    perf_data.append({
                        'Model': perf.model_name,
                        'MAE': perf.mae if pd.notna(perf.mae) else 0,
                        'MAPE': perf.mape if pd.notna(perf.mape) else 0,
                        'Weight': perf.weight
                    })
                
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, use_container_width=True)
        
        # Recommendations
        if result.recommendations:
            st.markdown("#### 💡 Business Recommendations")
            
            for i, recommendation in enumerate(result.recommendations, 1):
                st.write(f"{i}. {recommendation}")
        
        # Export forecast
        if st.button("📥 Export Forecast Data"):
            forecast_export = pd.DataFrame({
                'Date': result.point_forecast.index,
                'Forecast': result.point_forecast.values
            })
            
            if include_confidence and result.confidence_intervals:
                for level, series in result.confidence_intervals.items():
                    forecast_export[f'CI_{level.upper()}'] = series.values
            
            csv_data = forecast_export.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv_data,
                file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    def _show_analytics(self):
        """Show advanced analytics and insights"""
        
        st.markdown("### 📈 Advanced Analytics")
        
        try:
            # Load data
            data = self.company_data_manager.load_company_data(st.session_state.company_id)
            
            if data.empty:
                st.info("No data available for analytics. Please upload data first.")
                return
            
            monthly_data = self._aggregate_monthly_data(data)
            
            # Analytics tabs
            tab1, tab2, tab3 = st.tabs(["📊 Performance Analysis", "🔍 Pattern Analysis", "📈 Model Evolution"])
            
            with tab1:
                self._show_performance_analysis(monthly_data)
            
            with tab2:
                self._show_pattern_analysis(monthly_data)
            
            with tab3:
                self._show_model_evolution()
        
        except Exception as e:
            st.error(f"Failed to load analytics: {e}")
    
    def _show_performance_analysis(self, monthly_data):
        """Show performance analysis"""
        
        st.markdown("#### 📊 Sales Performance Analysis")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            volatility = monthly_data['sales_amount'].std() / monthly_data['sales_amount'].mean()
            st.metric("Volatility (CV)", f"{volatility:.2f}")
        
        with col2:
            growth_rate = self._calculate_growth_rate(monthly_data['sales_amount'])
            st.metric("Growth Rate", f"{growth_rate:.1f}%")
        
        with col3:
            trend_strength = self._calculate_trend_strength(monthly_data['sales_amount'])
            st.metric("Trend Strength", f"{trend_strength:.2f}")
        
        # Seasonality analysis
        if len(monthly_data) >= 12:
            st.markdown("#### 🔄 Seasonality Analysis")
            
            monthly_data['month'] = pd.to_datetime(monthly_data['date']).dt.month
            seasonal_pattern = monthly_data.groupby('month')['sales_amount'].mean()
            
            fig_seasonal = px.bar(
                x=seasonal_pattern.index,
                y=seasonal_pattern.values,
                title="Average Sales by Month",
                labels={'x': 'Month', 'y': 'Average Sales ($)'}
            )
            st.plotly_chart(fig_seasonal, use_container_width=True)
    
    def _show_pattern_analysis(self, monthly_data):
        """Show pattern analysis"""
        
        st.markdown("#### 🔍 Data Pattern Analysis")
        
        try:
            # Get pattern detection results
            sales_series = monthly_data.set_index('date')['sales_amount']
            pattern = self.forecasting_engine.pattern_detector.detect_pattern(sales_series)
            
            # Pattern summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Detected Pattern:**")
                st.write(f"• **Type:** {pattern.pattern_type.title()}")
                st.write(f"• **Confidence:** {pattern.confidence:.2f}")
                st.write(f"• **Seasonality Strength:** {pattern.seasonality_strength:.2f}")
                st.write(f"• **Trend Strength:** {pattern.trend_strength:.2f}")
                st.write(f"• **Intermittency Ratio:** {pattern.intermittency_ratio:.2f}")
                st.write(f"• **Volatility:** {pattern.volatility:.2f}")
            
            with col2:
                # Pattern visualization
                fig_pattern = go.Figure()
                
                fig_pattern.add_trace(go.Scatter(
                    x=sales_series.index,
                    y=sales_series.values,
                    mode='lines+markers',
                    name='Sales Data',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Add trend line if trending
                if pattern.trend_strength > 0.2:
                    x_numeric = np.arange(len(sales_series))
                    trend_coef = np.polyfit(x_numeric, sales_series.values, 1)
                    trend_line = np.poly1d(trend_coef)(x_numeric)
                    
                    fig_pattern.add_trace(go.Scatter(
                        x=sales_series.index,
                        y=trend_line,
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='red', dash='dash')
                    ))
                
                fig_pattern.update_layout(
                    title=f"Pattern: {pattern.pattern_type.title()}",
                    xaxis_title="Date",
                    yaxis_title="Sales Amount ($)",
                    height=400
                )
                
                st.plotly_chart(fig_pattern, use_container_width=True)
        
        except Exception as e:
            st.error(f"Pattern analysis failed: {e}")
    
    def _show_model_evolution(self):
        """Show model evolution and weight changes"""
        
        st.markdown("#### 📈 Model Evolution")
        
        try:
            # Get model performance history
            performance_history = self.forecasting_engine.company_performance_history.get(
                st.session_state.company_id, []
            )
            
            weight_history = self.forecasting_engine.company_weight_history.get(
                st.session_state.company_id, []
            )
            
            if not performance_history and not weight_history:
                st.info("No model evolution data available yet. Generate forecasts to see model performance over time.")
                return
            
            # Performance evolution
            if performance_history:
                st.markdown("##### 🎯 Performance Evolution")
                
                perf_data = []
                for perf in performance_history:
                    perf_data.append({
                        'Date': perf.evaluation_date,
                        'Model': perf.model_name,
                        'MAE': perf.mae if pd.notna(perf.mae) else None,
                        'MAPE': perf.mape if pd.notna(perf.mape) else None,
                        'Weight': perf.weight
                    })
                
                perf_df = pd.DataFrame(perf_data)
                
                if not perf_df.empty:
                    fig_perf = px.line(
                        perf_df,
                        x='Date',
                        y='MAPE',
                        color='Model',
                        title="Model Performance Over Time (MAPE)"
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)
            
            # Weight evolution
            if weight_history:
                st.markdown("##### ⚖️ Weight Evolution")
                
                weight_data = []
                for record in weight_history:
                    for model, weight in record.new_weights.items():
                        weight_data.append({
                            'Date': record.update_date,
                            'Model': model,
                            'Weight': weight,
                            'Reason': record.trigger_reason
                        })
                
                weight_df = pd.DataFrame(weight_data)
                
                if not weight_df.empty:
                    fig_weights = px.line(
                        weight_df,
                        x='Date',
                        y='Weight',
                        color='Model',
                        title="Model Weight Changes Over Time"
                    )
                    st.plotly_chart(fig_weights, use_container_width=True)
        
        except Exception as e:
            st.error(f"Model evolution analysis failed: {e}")
    
    def _show_settings(self):
        """Show settings and configuration"""
        
        st.markdown("### ⚙️ Settings & Configuration")
        
        try:
            profile = self.company_data_manager.get_company_profile(st.session_state.company_id)
            
            if not profile:
                st.error("Company profile not found")
                return
            
            # Company information
            st.markdown("#### 🏢 Company Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Company ID:** {profile.company_id}")
                st.write(f"**Company Name:** {profile.company_name}")
                st.write(f"**Industry:** {profile.industry}")
            
            with col2:
                st.write(f"**Created:** {profile.created_date.strftime('%Y-%m-%d')}")
                st.write(f"**Total Uploads:** {profile.total_uploads}")
                if profile.last_upload_date:
                    st.write(f"**Last Upload:** {profile.last_upload_date.strftime('%Y-%m-%d')}")
            
            # Adaptive configuration
            st.markdown("#### 🤖 Adaptive Forecasting Configuration")
            
            with st.form("config_update"):
                col1, col2 = st.columns(2)
                
                with col1:
                    adaptive_enabled = st.checkbox(
                        "Enable Adaptive Learning",
                        value=profile.adaptive_config.get('adaptive_learning_enabled', True)
                    )
                    
                    learning_window = st.number_input(
                        "Learning Window (months)",
                        min_value=1,
                        max_value=24,
                        value=profile.adaptive_config.get('learning_window_months', 6)
                    )
                    
                    min_weight = st.slider(
                        "Minimum Model Weight",
                        min_value=0.01,
                        max_value=0.5,
                        value=profile.adaptive_config.get('min_model_weight', 0.05),
                        step=0.01
                    )
                
                with col2:
                    update_frequency = st.selectbox(
                        "Weight Update Frequency",
                        options=['monthly', 'weekly', 'daily'],
                        index=['monthly', 'weekly', 'daily'].index(
                            profile.adaptive_config.get('weight_update_frequency', 'monthly')
                        )
                    )
                    
                    confidence_intervals = st.checkbox(
                        "Enable Confidence Intervals",
                        value=profile.adaptive_config.get('confidence_intervals_enabled', True)
                    )
                    
                    pattern_detection = st.checkbox(
                        "Enable Pattern Detection",
                        value=profile.adaptive_config.get('pattern_detection_enabled', True)
                    )
                
                if st.form_submit_button("💾 Update Configuration"):
                    config_updates = {
                        'adaptive_learning_enabled': adaptive_enabled,
                        'learning_window_months': learning_window,
                        'min_model_weight': min_weight,
                        'weight_update_frequency': update_frequency,
                        'confidence_intervals_enabled': confidence_intervals,
                        'pattern_detection_enabled': pattern_detection
                    }
                    
                    success = self.company_data_manager.update_company_config(
                        st.session_state.company_id, config_updates
                    )
                    
                    if success:
                        st.success("✅ Configuration updated successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("❌ Failed to update configuration")
            
            # Data management
            st.markdown("#### 📊 Data Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export All Data"):
                    try:
                        data = self.company_data_manager.load_company_data(st.session_state.company_id)
                        if not data.empty:
                            csv_data = data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Data CSV",
                                data=csv_data,
                                file_name=f"{profile.company_id}_sales_data.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No data to export")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            
            with col2:
                if st.button("📈 Export Model Performance"):
                    try:
                        performance_history = self.forecasting_engine.company_performance_history.get(
                            st.session_state.company_id, []
                        )
                        
                        if performance_history:
                            perf_data = []
                            for perf in performance_history:
                                perf_data.append({
                                    'Model': perf.model_name,
                                    'MAE': perf.mae,
                                    'MAPE': perf.mape,
                                    'RMSE': perf.rmse,
                                    'R_Squared': perf.r_squared,
                                    'Weight': perf.weight,
                                    'Evaluation_Date': perf.evaluation_date.isoformat(),
                                    'Data_Points': perf.data_points
                                })
                            
                            perf_df = pd.DataFrame(perf_data)
                            csv_data = perf_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download Performance CSV",
                                data=csv_data,
                                file_name=f"{profile.company_id}_model_performance.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No performance data to export")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
        
        except Exception as e:
            st.error(f"Failed to load settings: {e}")
    
    # Helper methods
    def _aggregate_monthly_data(self, data):
        """Aggregate data by month"""
        if 'date' not in data.columns or 'sales_amount' not in data.columns:
            return pd.DataFrame()
        
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        data['year_month'] = data['date'].dt.to_period('M')
        
        monthly_agg = data.groupby('year_month').agg({
            'sales_amount': 'sum',
            'units_sold': 'sum' if 'units_sold' in data.columns else lambda x: np.nan,
            'customer_count': 'sum' if 'customer_count' in data.columns else lambda x: np.nan
        }).reset_index()
        
        monthly_agg['date'] = monthly_agg['year_month'].dt.to_timestamp()
        monthly_agg = monthly_agg.drop('year_month', axis=1)
        monthly_agg = monthly_agg.sort_values('date').reset_index(drop=True)
        
        return monthly_agg
    
    def _calculate_growth_rate(self, series):
        """Calculate growth rate"""
        if len(series) < 2:
            return 0.0
        
        return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100
    
    def _calculate_trend_strength(self, series):
        """Calculate trend strength"""
        if len(series) < 3:
            return 0.0
        
        x = np.arange(len(series))
        correlation = np.corrcoef(x, series.values)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0

# Main application
def main():
    """Main application entry point"""
    dashboard = CompanySalesDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()