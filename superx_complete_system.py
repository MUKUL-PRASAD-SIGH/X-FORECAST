"""
Complete SuperX System with Login, Data Upload, and RAG-powered Chat
Persistent data storage with intelligent suggestions from uploaded CSV data
"""

import sys
import os
sys.path.append('src')

from src.auth.superx_auth import SuperXAuthenticator, UserRole, SubscriptionTier
from src.data_upload.superx_data_uploader import SuperXDataUploader, DataType
from src.rag.csv_knowledge_base import CSVKnowledgeBase
from src.ai_chatbot.conversational_ai import ConversationalAI
from src.dashboard.dynamic_dashboard import DynamicDashboard
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json

class SuperXCompleteSystem:
    """Complete SuperX System with Authentication, Upload, and RAG Chat"""
    
    def __init__(self):
        self.auth = SuperXAuthenticator()
        self.uploader = SuperXDataUploader()
        self.knowledge_base = CSVKnowledgeBase()
        self.chatbot = ConversationalAI()
        self.dashboard = DynamicDashboard(self.knowledge_base)
        
        self.current_token = None
        self.current_user = None
        
        # Enhanced chatbot with RAG capabilities
        self.chatbot.knowledge_base = self.knowledge_base
    
    def display_welcome(self):
        """Display enhanced welcome screen"""
        print("""
╭─────────────────────────────────────────────────────────────╮
│  🚀 **SUPERX AI PLATFORM** - COMPLETE SYSTEM 🤖            │
╰─────────────────────────────────────────────────────────────╯

🔐 **SECURE LOGIN & DATA UPLOAD**
• Enterprise authentication with role-based access
• Persistent CSV data storage with RAG capabilities
• Intelligent suggestions from your uploaded data
• Chat with your data using natural language

🧠 **AI-POWERED FEATURES**
• Upload CSV files and chat with your data
• Persistent storage - data survives restarts
• Smart suggestions based on your datasets
• Advanced analytics and forecasting

┌─ 👥 **DEMO ACCOUNTS** ─────────────────────────────────────┐
│                                                            │
│  🔑 **Admin** (admin/admin123)                             │
│     • Full access, 1GB upload limit                       │
│     • Advanced analytics and forecasting                  │
│                                                            │
│  🔑 **Manager** (manager/manager123)                       │
│     • Management features, 500MB upload limit             │
│     • Team analytics and reporting                        │
│                                                            │
│  🔑 **Analyst** (analyst/analyst123)                       │
│     • Analytics features, 200MB upload limit              │
│     • Data exploration and insights                       │
│                                                            │
└────────────────────────────────────────────────────────────┘

💡 **What makes this special:**
• Your data is remembered between sessions
• Upload multiple CSV files to build knowledge
• Ask questions like "What are my sales trends?"
• Get intelligent suggestions based on your data
""")
    
    def login_flow(self):
        """Enhanced login flow"""
        
        print("\n🔐 **LOGIN TO SUPERX PLATFORM**")
        print("-" * 50)
        
        while True:
            username = input("👤 Username: ").strip()
            password = input("🔒 Password: ").strip()
            
            if not username or not password:
                print("❌ Please enter both username and password")
                continue
            
            result = self.auth.login(username, password)
            
            if result["success"]:
                self.current_token = result["token"]
                self.current_user = result["user"]
                
                print(f"\n{result['message']}")
                self.display_user_dashboard()
                return True
            else:
                print(f"\n{result['message']}")
                retry = input("\n🔄 Try again? (y/n): ").lower()
                if retry != 'y':
                    return False
    
    def display_user_dashboard(self):
        """Enhanced user dashboard with data overview"""
        
        user_info = self.auth.get_user_info(self.current_token)
        user_datasets = self.knowledge_base.get_user_datasets(self.current_user.user_id)
        
        print(f"""
╭─────────────────────────────────────────────────────────────╮
│  👋 **WELCOME {user_info['username'].upper()}** - YOUR DATA DASHBOARD 📊      │
╰─────────────────────────────────────────────────────────────╯

┌─ 👤 **ACCOUNT INFO** ──────────────────────────────────────┐
│                                                            │
│  🏢 Company: {user_info['company_name']:<40} │
│  👤 Role: {user_info['role'].title():<47} │
│  💎 Plan: {user_info['subscription_tier'].title():<47} │
│  📊 Data: {user_info['data_uploaded_mb']}MB used / {user_info['data_upload_limit_mb']}MB limit              │
│                                                            │
└────────────────────────────────────────────────────────────┘

┌─ 📊 **YOUR DATA** ─────────────────────────────────────────┐
│                                                            │""")
        
        if user_datasets:
            print(f"│  📁 Datasets: {len(user_datasets)} files uploaded                    │")
            for i, dataset in enumerate(user_datasets[:3], 1):
                filename = dataset['filename'][:30] + "..." if len(dataset['filename']) > 30 else dataset['filename']
                print(f"│  {i}. {filename:<35} ({dataset['row_count']} rows)  │")
            
            if len(user_datasets) > 3:
                print(f"│     ... and {len(user_datasets) - 3} more datasets                        │")
        else:
            print("│  📁 No datasets uploaded yet                           │")
            print("│  💡 Upload CSV files to start chatting with your data │")
        
        print("""│                                                            │
└────────────────────────────────────────────────────────────┘

🎯 **AVAILABLE ACTIONS:**
• 'upload' - Upload CSV files to your knowledge base
• 'chat' - Chat with your data using natural language
• 'dashboard' - View live dashboard with your data metrics
• 'datasets' - View your uploaded datasets
• 'sample' - Create sample data for testing
• 'history' - View your query history
• 'logout' - Sign out
""")
    
    def handle_user_actions(self):
        """Enhanced user action handling"""
        
        while self.current_token:
            action = input("\n🎯 What would you like to do? ").lower().strip()
            
            if action == 'upload':
                self.data_upload_flow()
            elif action == 'chat':
                self.chat_with_data()
            elif action == 'dashboard':
                self.show_live_dashboard()
            elif action == 'datasets':
                self.show_datasets()
            elif action == 'sample':
                self.create_sample_data()
            elif action == 'history':
                self.show_query_history()
            elif action == 'logout':
                self.logout()
                break
            elif action in ['help', '?']:
                self.show_help()
            elif action in ['quit', 'exit']:
                break
            else:
                print("❓ Unknown command. Available: upload, chat, dashboard, datasets, sample, history, logout")
    
    def show_live_dashboard(self):
        """Display live dashboard with metrics from user's data"""
        
        print("\n🔄 Generating dashboard from your data...")
        
        # Generate dashboard data
        dashboard_data = self.dashboard.generate_dashboard_data(self.current_user.user_id)
        
        # Display dashboard
        self._display_dashboard(dashboard_data)
    
    def _display_dashboard(self, data: Dict[str, Any]):
        """Display the dashboard in a beautiful format"""
        
        user_info = data['user_info']
        metrics = data['metrics']
        
        print(f"""
╭─────────────────────────────────────────────────────────────╮
│  📊 **SUPERX LIVE DASHBOARD** - POWERED BY YOUR DATA 🚀    │
╰─────────────────────────────────────────────────────────────╯

🔗 **Data Source:** {user_info['datasets_count']} uploaded datasets | {user_info['total_records']:,} total records

┌─ 👥 **CUSTOMER METRICS** ──────────────────────────────────┐
│                                                            │
│  Total Customers    {metrics['customers']['total']:,}                           │
│  Growth Rate        {metrics['customers']['growth_rate']:+.1f}% {self._get_arrow(metrics['customers']['growth_direction'])}                    │
│                                                            │
│  Retention Rate     {metrics['retention']['rate']:.1f}%                        │
│  Change             {metrics['retention']['change']:+.1f}% {self._get_arrow(metrics['retention']['direction'])}                    │
│                                                            │
└────────────────────────────────────────────────────────────┘

┌─ 💰 **REVENUE METRICS** ───────────────────────────────────┐
│                                                            │
│  Revenue Growth     {metrics['revenue_growth']['rate']:+.1f}% {self._get_arrow(metrics['revenue_growth']['direction'])}                    │
│  Forecast Accuracy  {metrics['forecast_accuracy']['rate']:.1f}%                        │
│  Improvement        {metrics['forecast_accuracy']['improvement']:+.1f}% {self._get_arrow(metrics['forecast_accuracy']['direction'])}                    │
│                                                            │
└────────────────────────────────────────────────────────────┘

┌─ 🔧 **SYSTEM STATUS** ─────────────────────────────────────┐
│                                                            │
│  System Health      {metrics['system_health']['score']:.1f}% ({metrics['system_health']['status']})              │
│  Uptime            {data['system_status']['uptime']:.1f}%                           │
│  Active Alerts      {metrics['alerts']['active']} {self._get_arrow(metrics['alerts']['direction'])}                              │
│  Last Update       {data['system_status']['last_update']}                    │
│                                                            │
└────────────────────────────────────────────────────────────┘""")
        
        # Show top products if available
        if data['insights']['top_products']:
            print(f"""
┌─ 🏆 **TOP PERFORMING PRODUCTS** ───────────────────────────┐
│                                                            │""")
            
            for product in data['insights']['top_products'][:5]:
                name = product['name'][:30]
                revenue = product['revenue']
                print(f"│  {product['rank']}. {name:<30} ${revenue:>10,.2f}     │")
            
            print("""│                                                            │
└────────────────────────────────────────────────────────────┘""")
        
        # Show forecasts
        forecasts = data['forecasts']
        if forecasts['next_month']:
            print(f"""
┌─ 🔮 **FORECASTS FROM YOUR DATA** ──────────────────────────┐
│                                                            │
│  Next Month:                                               │""")
            
            for metric, value in forecasts['next_month'].items():
                if isinstance(value, (int, float)):
                    if 'revenue' in metric:
                        print(f"│    💰 {metric.title()}: ${value:,.2f}                           │")
                    else:
                        print(f"│    📊 {metric.title()}: {value:,}                              │")
            
            print(f"""│                                                            │
│  Confidence Level: {forecasts['confidence'].title()}                           │
│                                                            │
└────────────────────────────────────────────────────────────┘""")
        
        # Show recommendations
        recommendations = data.get('recommendations', [])
        if recommendations:
            print(f"""
┌─ 💡 **PERSONALIZED RECOMMENDATIONS** ──────────────────────┐
│                                                            │""")
            
            for i, rec in enumerate(recommendations, 1):
                print(f"│  {i}. {rec:<54} │")
            
            print("""│                                                            │
└────────────────────────────────────────────────────────────┘""")
        
        print(f"""
🎯 **Dashboard powered by YOUR data!**
• All metrics calculated from your uploaded CSV files
• Forecasts based on your actual business trends  
• Recommendations tailored to your data patterns

💡 Upload more data to see even more accurate insights!
""")
    
    def _get_arrow(self, direction: str) -> str:
        """Get arrow emoji for direction"""
        arrows = {
            'up': '↗',
            'down': '↓', 
            'neutral': '→'
        }
        return arrows.get(direction, '→')
    
    def data_upload_flow(self):
        """Enhanced data upload with knowledge base integration"""
        
        print(f"""
┌─ 📤 **UPLOAD CSV TO KNOWLEDGE BASE** ──────────────────────┐
│                                                            │
│  Upload CSV files to build your personal knowledge base   │
│  Your data will be remembered and you can chat with it!   │
│                                                            │
└────────────────────────────────────────────────────────────┘
""")
        
        try:
            file_path = input("📁 Enter CSV file path (or 'sample' for demo): ").strip()
            
            if file_path.lower() == 'sample':
                # Create sample data
                sample_file = self.create_sample_csv()
                file_path = sample_file
                print(f"✅ Created sample file: {sample_file}")
            
            if not os.path.exists(file_path):
                print("❌ File not found")
                return
            
            # Check file format
            if not file_path.lower().endswith('.csv'):
                print("❌ Only CSV files are supported")
                return
            
            # Check upload permissions
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            can_upload, message = self.auth.can_upload_data(self.current_token, file_size_mb)
            
            print(f"\n{message}")
            
            if not can_upload:
                return
            
            # Upload to knowledge base
            print(f"\n🔄 Processing {os.path.basename(file_path)}...")
            
            try:
                dataset_id = self.knowledge_base.add_dataset(
                    user_id=self.current_user.user_id,
                    csv_file_path=file_path,
                    filename=os.path.basename(file_path)
                )
                
                # Update user data usage
                self.auth.update_data_usage(self.current_token, file_size_mb)
                
                # Get dataset info
                dataset_info = self.knowledge_base.get_dataset_info(dataset_id)
                
                print(f"\n✅ **DATA UPLOADED SUCCESSFULLY!**")
                print(f"📊 Dataset ID: {dataset_id}")
                print(f"📁 Filename: {dataset_info['filename']}")
                print(f"📈 Rows: {dataset_info['row_count']:,}")
                print(f"📋 Columns: {len(dataset_info['columns'])}")
                
                # Show insights
                insights = dataset_info['insights']
                print(f"\n💡 **QUICK INSIGHTS:**")
                print(f"• Data Quality Score: {insights['data_quality']['overall_score']:.1%}")
                print(f"• Missing Values: {insights['summary']['missing_values']}")
                print(f"• Duplicate Rows: {insights['summary']['duplicate_rows']}")
                
                # Show potential questions
                questions = insights.get('potential_questions', [])[:5]
                if questions:
                    print(f"\n🤔 **TRY ASKING:**")
                    for i, question in enumerate(questions, 1):
                        print(f"  {i}. {question}")
                
                print(f"\n🎉 **Ready to chat!** Type 'chat' to start asking questions about your data!")
                
            except Exception as e:
                print(f"❌ Upload failed: {e}")
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
    
    def chat_with_data(self):
        """Enhanced chat interface with RAG capabilities"""
        
        user_datasets = self.knowledge_base.get_user_datasets(self.current_user.user_id)
        
        if not user_datasets:
            print("""
❌ **NO DATA TO CHAT WITH**

You need to upload CSV files first to chat with your data.
Type 'upload' to add some data, or 'sample' to create demo data.
""")
            return
        
        print(f"""
╭─────────────────────────────────────────────────────────────╮
│  💬 **CHAT WITH YOUR DATA** - {len(user_datasets)} datasets loaded      │
╰─────────────────────────────────────────────────────────────╯

🧠 **Your Knowledge Base:**""")
        
        for i, dataset in enumerate(user_datasets[:3], 1):
            print(f"  {i}. {dataset['filename']} ({dataset['row_count']} rows)")
        
        if len(user_datasets) > 3:
            print(f"  ... and {len(user_datasets) - 3} more datasets")
        
        # Get suggestions
        suggestions = self.knowledge_base._get_user_suggestions(user_datasets)
        
        print(f"""
💡 **SUGGESTED QUESTIONS:**""")
        for i, suggestion in enumerate(suggestions[:5], 1):
            print(f"  {i}. {suggestion}")
        
        print(f"""
🎯 **Chat Commands:**
• Ask any question about your data
• Type 'suggestions' for more question ideas
• Type 'datasets' to see your data files
• Type 'back' to return to main menu

💬 **Start chatting with your data!**
""")
        
        while True:
            try:
                user_query = input("\n🤔 Ask about your data: ").strip()
                
                if not user_query:
                    continue
                
                if user_query.lower() in ['back', 'exit', 'quit']:
                    break
                
                if user_query.lower() == 'suggestions':
                    self.show_suggestions()
                    continue
                
                if user_query.lower() == 'datasets':
                    self.show_datasets()
                    continue
                
                # Process query with RAG
                print("\n🔍 Analyzing your data...")
                result = self.knowledge_base.query_data(self.current_user.user_id, user_query)
                
                if result["success"]:
                    self.display_query_result(result)
                else:
                    print(f"\n❌ {result['message']}")
                    if result.get('suggestions'):
                        print("\n💡 **Try asking:**")
                        for suggestion in result['suggestions'][:3]:
                            print(f"  • {suggestion}")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting chat...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def display_query_result(self, result: Dict):
        """Display query results in a beautiful format"""
        
        print(f"""
╭─────────────────────────────────────────────────────────────╮
│  📊 **QUERY RESULTS** 🎯                                   │
╰─────────────────────────────────────────────────────────────╯

📁 **Data Sources:** {', '.join(result['datasets_used'])}
""")
        
        # Display results for each dataset
        for dataset_id, data in result['result'].items():
            print(f"\n📋 **{data['filename']}:**")
            
            # Show insights
            if data['insights']:
                for insight in data['insights']:
                    print(f"  💡 {insight}")
            
            # Show summary data
            if data['summary']:
                print(f"  📊 **Key Findings:**")
                for key, value in data['summary'].items():
                    if isinstance(value, dict):
                        print(f"    • {key.replace('_', ' ').title()}:")
                        for k, v in list(value.items())[:5]:  # Limit to 5 items
                            if isinstance(v, float):
                                print(f"      - {k}: {v:.2f}")
                            else:
                                print(f"      - {k}: {v}")
                    elif isinstance(value, list):
                        print(f"    • {key.replace('_', ' ').title()}: {', '.join(map(str, value[:5]))}")
                    else:
                        if isinstance(value, float):
                            print(f"    • {key.replace('_', ' ').title()}: {value:.2f}")
                        else:
                            print(f"    • {key.replace('_', ' ').title()}: {value}")
            
            # Show sample data if available
            if data['data']:
                print(f"  📄 **Sample Data:**")
                for i, row in enumerate(data['data'][:3], 1):
                    print(f"    {i}. {dict(list(row.items())[:3])}...")  # Show first 3 columns
        
        # Show suggestions for follow-up questions
        if result.get('suggestions'):
            print(f"\n🤔 **FOLLOW-UP QUESTIONS:**")
            for i, suggestion in enumerate(result['suggestions'][:4], 1):
                print(f"  {i}. {suggestion}")
    
    def show_suggestions(self):
        """Show intelligent suggestions based on user's data"""
        
        user_datasets = self.knowledge_base.get_user_datasets(self.current_user.user_id)
        suggestions = self.knowledge_base._get_user_suggestions(user_datasets)
        
        print(f"""
┌─ 💡 **INTELLIGENT SUGGESTIONS** ──────────────────────────┐
│                                                            │
│  Based on your uploaded datasets:                         │
│                                                            │""")
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"│  {i:2d}. {suggestion:<50} │")
        
        print("""│                                                            │
└────────────────────────────────────────────────────────────┘""")
    
    def show_datasets(self):
        """Show user's datasets with detailed information"""
        
        user_datasets = self.knowledge_base.get_user_datasets(self.current_user.user_id)
        
        if not user_datasets:
            print("📁 No datasets found. Upload some CSV files to get started!")
            return
        
        print(f"""
┌─ 📊 **YOUR DATASETS** ({len(user_datasets)} files) ──────────────────────┐
│                                                            │""")
        
        for i, dataset in enumerate(user_datasets, 1):
            upload_date = dataset['upload_time'][:10]  # Just the date part
            print(f"│  {i:2d}. {dataset['filename']:<25} │ {dataset['row_count']:>6} rows │ {upload_date} │")
            
            # Show column info
            columns = dataset['columns'][:5]  # First 5 columns
            col_str = ', '.join(columns)
            if len(dataset['columns']) > 5:
                col_str += f" (+{len(dataset['columns']) - 5} more)"
            print(f"│      Columns: {col_str:<40} │")
        
        print("""│                                                            │
└────────────────────────────────────────────────────────────┘""")
    
    def show_query_history(self):
        """Show user's query history"""
        
        history = self.knowledge_base.get_query_history(self.current_user.user_id)
        
        if not history:
            print("📋 No query history found. Start chatting with your data!")
            return
        
        print(f"""
┌─ 📋 **QUERY HISTORY** ({len(history)} queries) ──────────────────────────┐
│                                                            │""")
        
        for i, query in enumerate(history, 1):
            query_time = query['time'][:16].replace('T', ' ')  # Format datetime
            query_text = query['query'][:45] + "..." if len(query['query']) > 45 else query['query']
            print(f"│  {i:2d}. {query_text:<45} │ {query_time} │")
        
        print("""│                                                            │
└────────────────────────────────────────────────────────────┘""")
    
    def create_sample_csv(self) -> str:
        """Create sample CSV data for demonstration"""
        
        os.makedirs("sample_data", exist_ok=True)
        
        # Create comprehensive sample sales data
        np.random.seed(42)  # For reproducible results
        
        # Date range
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Products
        products = [
            'Laptop Pro 15"', 'Wireless Mouse', 'Mechanical Keyboard', 
            'USB-C Hub', 'Webcam HD', 'Bluetooth Headphones',
            'Monitor 24"', 'Desk Lamp', 'Office Chair', 'Standing Desk'
        ]
        
        # Categories
        categories = {
            'Laptop Pro 15"': 'Computers', 'Wireless Mouse': 'Accessories',
            'Mechanical Keyboard': 'Accessories', 'USB-C Hub': 'Accessories',
            'Webcam HD': 'Electronics', 'Bluetooth Headphones': 'Electronics',
            'Monitor 24"': 'Displays', 'Desk Lamp': 'Furniture',
            'Office Chair': 'Furniture', 'Standing Desk': 'Furniture'
        }
        
        # Generate data
        data = []
        for date in dates:
            # Simulate varying daily sales
            daily_sales = np.random.randint(5, 25)
            
            for _ in range(daily_sales):
                product = np.random.choice(products)
                quantity = np.random.randint(1, 5)
                
                # Price varies by product
                base_prices = {
                    'Laptop Pro 15"': 1299, 'Wireless Mouse': 29, 'Mechanical Keyboard': 89,
                    'USB-C Hub': 49, 'Webcam HD': 79, 'Bluetooth Headphones': 159,
                    'Monitor 24"': 299, 'Desk Lamp': 39, 'Office Chair': 199, 'Standing Desk': 399
                }
                
                price = base_prices[product] * (0.9 + np.random.random() * 0.2)  # ±10% variation
                revenue = price * quantity
                
                # Add seasonal effects
                month = date.month
                if month in [11, 12]:  # Holiday season
                    revenue *= 1.3
                elif month in [6, 7]:  # Summer sale
                    revenue *= 0.8
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'product_name': product,
                    'category': categories[product],
                    'quantity': quantity,
                    'unit_price': round(price, 2),
                    'total_revenue': round(revenue, 2),
                    'customer_id': f'CUST_{np.random.randint(1000, 9999)}',
                    'sales_rep': np.random.choice(['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson']),
                    'region': np.random.choice(['North', 'South', 'East', 'West']),
                    'channel': np.random.choice(['Online', 'Retail Store', 'Phone', 'Partner'])
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        filename = "sample_data/comprehensive_sales_data.csv"
        df.to_csv(filename, index=False)
        
        print(f"✅ Created comprehensive sample dataset with {len(df)} sales records")
        return filename
    
    def show_help(self):
        """Show comprehensive help"""
        
        print(f"""
┌─ ❓ **SUPERX COMPLETE SYSTEM HELP** ──────────────────────┐
│                                                            │
│  🔐 **AUTHENTICATION:**                                   │
│    • Secure login with role-based access                 │
│    • Data usage tracking and limits                      │
│                                                            │
│  📤 **DATA UPLOAD:**                                      │
│    • Upload CSV files to build knowledge base            │
│    • Persistent storage across sessions                  │
│    • Automatic data analysis and insights                │
│                                                            │
│  💬 **CHAT WITH DATA:**                                   │
│    • Ask questions in natural language                   │
│    • Get intelligent suggestions based on your data      │
│    • RAG-powered responses from your CSV files           │
│                                                            │
│  📊 **DASHBOARD:**                                        │
│    • Live metrics from your uploaded data                │
│    • Personalized insights and forecasts                 │
│    • Visual analytics and trends                         │
│                                                            │
│  🎯 **COMMANDS:**                                         │
│    upload   - Upload CSV files                           │
│    chat     - Chat with your data                        │
│    dashboard - View live dashboard                        │
│    datasets - View uploaded files                        │
│    sample   - Create sample data                         │
│    history  - View query history                         │
│    logout   - Sign out                                    │
│                                                            │
└────────────────────────────────────────────────────────────┘

💡 **Tips:**
• Upload multiple CSV files to build a richer knowledge base
• Your data persists between sessions - no need to re-upload
• Ask specific questions like "What are my top selling products?"
• Use the dashboard to see live metrics from your data
""")
    
    def logout(self):
        """Enhanced logout with session cleanup"""
        
        if self.current_token:
            self.auth.logout(self.current_token)
            print("\n👋 Successfully logged out. Your data is safely stored!")
            print("💾 All uploaded data will be available when you login again.")
            self.current_token = None
            self.current_user = None

def main():
    """Main application entry point"""
    
    print("🚀 Starting SuperX Complete System...")
    
    # Initialize system
    system = SuperXCompleteSystem()
    
    try:
        # Display welcome screen
        system.display_welcome()
        
        # Handle login
        if system.login_flow():
            # Handle user actions
            system.handle_user_actions()
        
        print("\n🙏 Thank you for using SuperX AI Platform!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Your data is safely stored.")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        print("💾 Don't worry - your data is safely stored!")

if __name__ == "__main__":
    main()gestions                         │
│    • Query multiple datasets simultaneously              │
│                                                            │
│  🎯 **COMMANDS:**                                         │
│    • upload   - Add CSV files to knowledge base          │
│    • chat     - Start chatting with your data            │
│    • datasets - View uploaded files                      │
│    • sample   - Create demo data                         │
│    • history  - View query history                       │
│    • logout   - Sign out                                 │
│                                                            │
└────────────────────────────────────────────────────────────┘

💡 **EXAMPLE QUESTIONS TO ASK YOUR DATA:**
• "What are my sales trends over time?"
• "Which products perform best?"
• "Show me customer analysis by region"
• "What's the average revenue per customer?"
• "Compare performance across different channels"
""")
    
    def logout(self):
        """Logout user"""
        
        if self.current_token:
            self.auth.logout(self.current_token)
            print("\n👋 Successfully logged out. Your data is safely stored!")
            print("💾 All your uploaded datasets will be available when you login again.")
            self.current_token = None
            self.current_user = None

def main():
    """Main application entry point"""
    
    print("🚀 Starting SuperX Complete System...")
    
    system = SuperXCompleteSystem()
    
    # Display welcome screen
    system.display_welcome()
    
    # Handle login
    if system.login_flow():
        # Handle user actions
        system.handle_user_actions()
    
    print("""
🙏 **THANK YOU FOR USING SUPERX!**

Your data is safely stored and will be available next time you login.
The system remembers all your uploads and learns from your interactions.

🌟 Key Features You Experienced:
• Secure authentication with persistent sessions
• CSV upload with intelligent data analysis  
• RAG-powered chat with your own data
• Smart suggestions based on your datasets
• Query history and data management

Come back anytime to continue exploring your data! 🚀
""")

if __name__ == "__main__":
    main()