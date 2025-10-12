"""
SuperX Complete System - Simplified Version
Login, CSV Upload, and Chat with Data (No external dependencies)
"""

import os
import json
import csv
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import sqlite3

class SimpleAuth:
    """Simple authentication system"""
    
    def __init__(self):
        self.users = {
            "admin": {
                "password": "admin123",
                "role": "admin",
                "company": "SuperX Corp",
                "upload_limit_mb": 1000,
                "uploaded_mb": 0
            },
            "manager": {
                "password": "manager123", 
                "role": "manager",
                "company": "SuperX Retail",
                "upload_limit_mb": 500,
                "uploaded_mb": 0
            },
            "analyst": {
                "password": "analyst123",
                "role": "analyst", 
                "company": "SuperX Analytics",
                "upload_limit_mb": 200,
                "uploaded_mb": 0
            }
        }
        self.sessions = {}
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        if username in self.users and self.users[username]["password"] == password:
            token = f"token_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.sessions[token] = username
            return {
                "success": True,
                "message": f"✅ Welcome back, {username}!",
                "token": token,
                "user": self.users[username]
            }
        return {"success": False, "message": "❌ Invalid credentials", "token": None}
    
    def get_user(self, token: str) -> Optional[Dict]:
        username = self.sessions.get(token)
        return self.users.get(username) if username else None
    
    def can_upload(self, token: str, size_mb: float) -> tuple:
        user = self.get_user(token)
        if not user:
            return False, "❌ Invalid session"
        
        remaining = user["upload_limit_mb"] - user["uploaded_mb"]
        if size_mb > remaining:
            return False, f"❌ Upload limit exceeded. {remaining:.1f}MB remaining"
        
        return True, f"✅ Upload approved. {remaining - size_mb:.1f}MB will remain"
    
    def update_usage(self, token: str, size_mb: float):
        username = self.sessions.get(token)
        if username and username in self.users:
            self.users[username]["uploaded_mb"] += size_mb

class SimpleKnowledgeBase:
    """Simple CSV knowledge base"""
    
    def __init__(self):
        self.storage_dir = "data/simple_kb"
        os.makedirs(self.storage_dir, exist_ok=True)
        self.datasets = {}
        self.load_datasets()
    
    def load_datasets(self):
        """Load existing datasets"""
        metadata_file = os.path.join(self.storage_dir, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.datasets = json.load(f)
            except:
                self.datasets = {}
    
    def save_datasets(self):
        """Save datasets metadata"""
        metadata_file = os.path.join(self.storage_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.datasets, f, indent=2, default=str)
    
    def add_dataset(self, user_id: str, csv_file: str, filename: str) -> str:
        """Add CSV dataset"""
        
        # Read CSV file
        data = []
        columns = []
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                data = list(reader)
        except Exception as e:
            raise Exception(f"Failed to read CSV: {e}")
        
        # Generate dataset ID
        dataset_id = f"{user_id}_{hashlib.md5(f'{filename}_{datetime.now()}'.encode()).hexdigest()[:8]}"
        
        # Analyze data
        insights = self.analyze_data(data, columns)
        
        # Store dataset
        dataset_info = {
            "dataset_id": dataset_id,
            "user_id": user_id,
            "filename": filename,
            "upload_time": datetime.now().isoformat(),
            "columns": columns,
            "row_count": len(data),
            "data": data[:1000],  # Store first 1000 rows
            "insights": insights
        }
        
        # Save to file
        dataset_file = os.path.join(self.storage_dir, f"{dataset_id}.json")
        with open(dataset_file, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
        
        # Update metadata
        if user_id not in self.datasets:
            self.datasets[user_id] = []
        self.datasets[user_id].append(dataset_id)
        self.save_datasets()
        
        return dataset_id
    
    def analyze_data(self, data: List[Dict], columns: List[str]) -> Dict:
        """Analyze CSV data"""
        
        if not data:
            return {"questions": ["No data to analyze"]}
        
        # Generate potential questions
        questions = [
            f"What are the key insights from this dataset?",
            f"Show me a summary of the data",
            f"What are the trends in this data?"
        ]
        
        # Add column-specific questions
        for col in columns[:5]:  # First 5 columns
            questions.extend([
                f"What are the unique values in {col}?",
                f"Show me statistics for {col}",
                f"What is the distribution of {col}?"
            ])
        
        # Business-specific questions
        if any(keyword in ' '.join(columns).lower() for keyword in ['sales', 'revenue', 'price']):
            questions.extend([
                "What are the sales trends?",
                "Show me revenue analysis",
                "Which products perform best?"
            ])
        
        if any(keyword in ' '.join(columns).lower() for keyword in ['date', 'time']):
            questions.extend([
                "Show me trends over time",
                "What is the date range?",
                "Show me monthly patterns"
            ])
        
        return {
            "questions": questions[:15],  # Limit to 15 questions
            "summary": {
                "total_rows": len(data),
                "columns": len(columns),
                "sample_data": data[:3] if data else []
            }
        }
    
    def get_user_datasets(self, user_id: str) -> List[Dict]:
        """Get user's datasets"""
        
        dataset_ids = self.datasets.get(user_id, [])
        datasets = []
        
        for dataset_id in dataset_ids:
            dataset_file = os.path.join(self.storage_dir, f"{dataset_id}.json")
            if os.path.exists(dataset_file):
                try:
                    with open(dataset_file, 'r') as f:
                        dataset = json.load(f)
                        datasets.append(dataset)
                except:
                    continue
        
        return datasets
    
    def query_data(self, user_id: str, query: str) -> Dict:
        """Query user's data"""
        
        datasets = self.get_user_datasets(user_id)
        
        if not datasets:
            return {
                "success": False,
                "message": "No datasets found. Upload CSV files first!",
                "suggestions": ["Upload a CSV file to get started"]
            }
        
        # Simple query processing
        query_lower = query.lower()
        results = {}
        
        for dataset in datasets:
            dataset_results = {
                "filename": dataset["filename"],
                "insights": [],
                "data": []
            }
            
            # Basic queries
            if any(word in query_lower for word in ['summary', 'overview', 'describe']):
                dataset_results["insights"].append(f"Dataset has {dataset['row_count']} rows and {len(dataset['columns'])} columns")
                dataset_results["insights"].append(f"Columns: {', '.join(dataset['columns'][:5])}")
            
            if any(word in query_lower for word in ['show', 'display', 'sample']):
                dataset_results["data"] = dataset["data"][:5]  # Show first 5 rows
            
            if any(word in query_lower for word in ['columns', 'fields']):
                dataset_results["insights"].append(f"Available columns: {', '.join(dataset['columns'])}")
            
            # Column-specific queries
            for col in dataset["columns"]:
                if col.lower() in query_lower:
                    # Get unique values for this column
                    values = list(set([row.get(col, '') for row in dataset["data"]]))[:10]
                    dataset_results["insights"].append(f"{col} values: {', '.join(map(str, values))}")
            
            results[dataset["dataset_id"]] = dataset_results
        
        # Get suggestions
        all_questions = []
        for dataset in datasets:
            all_questions.extend(dataset.get("insights", {}).get("questions", []))
        
        return {
            "success": True,
            "results": results,
            "datasets_used": [d["filename"] for d in datasets],
            "suggestions": list(set(all_questions))[:8]
        }

class SuperXSimpleSystem:
    """Complete SuperX system - simplified version"""
    
    def __init__(self):
        self.auth = SimpleAuth()
        self.kb = SimpleKnowledgeBase()
        self.current_token = None
        self.current_user = None
    
    def display_welcome(self):
        """Display welcome screen"""
        print("""
╭─────────────────────────────────────────────────────────────╮
│  🚀 **SUPERX AI PLATFORM** - COMPLETE SYSTEM 🤖            │
╰─────────────────────────────────────────────────────────────╯

🔐 **LOGIN & UPLOAD CSV DATA**
• Three demo accounts with different access levels
• Upload CSV files and chat with your data
• Persistent storage - data survives restarts
• Smart suggestions based on your datasets

┌─ 👥 **DEMO ACCOUNTS** ─────────────────────────────────────┐
│                                                            │
│  🔑 **Admin** (admin/admin123)                             │
│     • Full access, 1GB upload limit                       │
│                                                            │
│  🔑 **Manager** (manager/manager123)                       │
│     • Management features, 500MB upload limit             │
│                                                            │
│  🔑 **Analyst** (analyst/analyst123)                       │
│     • Analytics features, 200MB upload limit              │
│                                                            │
└────────────────────────────────────────────────────────────┘

💡 **Features:**
• Upload CSV files to build your knowledge base
• Ask questions about your data in natural language
• Get intelligent suggestions based on your uploads
• Data persists between sessions
""")
    
    def login_flow(self):
        """Handle login"""
        
        print("\n🔐 **LOGIN TO SUPERX**")
        print("-" * 30)
        
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
                self.show_dashboard()
                return True
            else:
                print(f"\n{result['message']}")
                retry = input("🔄 Try again? (y/n): ").lower()
                if retry != 'y':
                    return False
    
    def show_dashboard(self):
        """Show user dashboard"""
        
        datasets = self.kb.get_user_datasets(self.current_token.split('_')[1])
        
        print(f"""
╭─────────────────────────────────────────────────────────────╮
│  👋 **WELCOME {self.current_user['role'].upper()}** - YOUR DASHBOARD 📊        │
╰─────────────────────────────────────────────────────────────╯

┌─ 👤 **ACCOUNT INFO** ──────────────────────────────────────┐
│  🏢 Company: {self.current_user['company']:<40} │
│  👤 Role: {self.current_user['role'].title():<47} │
│  📊 Data: {self.current_user['uploaded_mb']}MB / {self.current_user['upload_limit_mb']}MB                        │
└────────────────────────────────────────────────────────────┘

┌─ 📊 **YOUR DATA** ─────────────────────────────────────────┐""")
        
        if datasets:
            print(f"│  📁 Datasets: {len(datasets)} files uploaded                    │")
            for i, dataset in enumerate(datasets[:3], 1):
                filename = dataset['filename'][:35] + "..." if len(dataset['filename']) > 35 else dataset['filename']
                print(f"│  {i}. {filename:<40} ({dataset['row_count']} rows) │")
        else:
            print("│  📁 No datasets uploaded yet                           │")
            print("│  💡 Upload CSV files to start chatting with your data │")
        
        print("""└────────────────────────────────────────────────────────────┘

🎯 **COMMANDS:**
• 'upload' - Upload CSV files
• 'chat' - Chat with your data
• 'datasets' - View your files
• 'sample' - Create demo data
• 'logout' - Sign out
""")
    
    def handle_actions(self):
        """Handle user actions"""
        
        while self.current_token:
            action = input("\n🎯 What would you like to do? ").lower().strip()
            
            if action == 'upload':
                self.upload_csv()
            elif action == 'chat':
                self.chat_with_data()
            elif action == 'datasets':
                self.show_datasets()
            elif action == 'sample':
                self.create_sample()
            elif action == 'logout':
                self.logout()
                break
            elif action in ['quit', 'exit']:
                break
            else:
                print("❓ Available: upload, chat, datasets, sample, logout")
    
    def upload_csv(self):
        """Handle CSV upload"""
        
        print("\n📤 **UPLOAD CSV FILE**")
        print("-" * 25)
        
        file_path = input("📁 Enter CSV file path (or 'sample' for demo): ").strip()
        
        if file_path.lower() == 'sample':
            file_path = self.create_sample_csv()
            print(f"✅ Created sample file: {file_path}")
        
        if not os.path.exists(file_path):
            print("❌ File not found")
            return
        
        if not file_path.lower().endswith('.csv'):
            print("❌ Only CSV files supported")
            return
        
        # Check upload limit
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        can_upload, message = self.auth.can_upload(self.current_token, file_size_mb)
        
        print(f"\n{message}")
        if not can_upload:
            return
        
        try:
            print(f"\n🔄 Processing {os.path.basename(file_path)}...")
            
            user_id = self.current_token.split('_')[1]
            dataset_id = self.kb.add_dataset(user_id, file_path, os.path.basename(file_path))
            
            self.auth.update_usage(self.current_token, file_size_mb)
            
            # Get dataset info
            datasets = self.kb.get_user_datasets(user_id)
            dataset = next((d for d in datasets if d['dataset_id'] == dataset_id), None)
            
            if dataset:
                print(f"\n✅ **UPLOAD SUCCESSFUL!**")
                print(f"📊 File: {dataset['filename']}")
                print(f"📈 Rows: {dataset['row_count']:,}")
                print(f"📋 Columns: {len(dataset['columns'])}")
                
                # Show sample questions
                questions = dataset['insights']['questions'][:5]
                if questions:
                    print(f"\n🤔 **TRY ASKING:**")
                    for i, q in enumerate(questions, 1):
                        print(f"  {i}. {q}")
                
                print(f"\n🎉 Ready to chat! Type 'chat' to start asking questions!")
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
    
    def chat_with_data(self):
        """Chat with uploaded data"""
        
        user_id = self.current_token.split('_')[1]
        datasets = self.kb.get_user_datasets(user_id)
        
        if not datasets:
            print("\n❌ No data to chat with. Upload CSV files first!")
            return
        
        print(f"""
╭─────────────────────────────────────────────────────────────╮
│  💬 **CHAT WITH YOUR DATA** - {len(datasets)} files loaded         │
╰─────────────────────────────────────────────────────────────╯

📁 **Your Files:**""")
        
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset['filename']} ({dataset['row_count']} rows)")
        
        # Show suggestions
        all_questions = []
        for dataset in datasets:
            all_questions.extend(dataset['insights']['questions'][:3])
        
        print(f"\n💡 **SUGGESTED QUESTIONS:**")
        for i, q in enumerate(list(set(all_questions))[:5], 1):
            print(f"  {i}. {q}")
        
        print(f"\n💬 **Start asking questions!** (type 'back' to return)")
        
        while True:
            query = input("\n🤔 Ask about your data: ").strip()
            
            if not query or query.lower() in ['back', 'exit']:
                break
            
            print("\n🔍 Analyzing your data...")
            result = self.kb.query_data(user_id, query)
            
            if result["success"]:
                print(f"\n📊 **RESULTS FROM:** {', '.join(result['datasets_used'])}")
                
                for dataset_id, data in result["results"].items():
                    print(f"\n📋 **{data['filename']}:**")
                    
                    if data['insights']:
                        for insight in data['insights']:
                            print(f"  💡 {insight}")
                    
                    if data['data']:
                        print(f"  📄 **Sample Data:**")
                        for i, row in enumerate(data['data'][:3], 1):
                            # Show first 3 columns of each row
                            sample = {k: v for k, v in list(row.items())[:3]}
                            print(f"    {i}. {sample}")
                
                if result.get('suggestions'):
                    print(f"\n🤔 **TRY NEXT:**")
                    for i, suggestion in enumerate(result['suggestions'][:3], 1):
                        print(f"  {i}. {suggestion}")
            else:
                print(f"\n❌ {result['message']}")
    
    def show_datasets(self):
        """Show user's datasets"""
        
        user_id = self.current_token.split('_')[1]
        datasets = self.kb.get_user_datasets(user_id)
        
        if not datasets:
            print("\n📁 No datasets found. Upload CSV files to get started!")
            return
        
        print(f"""
┌─ 📊 **YOUR DATASETS** ({len(datasets)} files) ──────────────────────┐""")
        
        for i, dataset in enumerate(datasets, 1):
            upload_date = dataset['upload_time'][:10]
            print(f"│  {i}. {dataset['filename']:<30} │ {dataset['row_count']:>5} rows │ {upload_date} │")
        
        print("└────────────────────────────────────────────────────────────┘")
    
    def create_sample_csv(self) -> str:
        """Create sample CSV data"""
        
        os.makedirs("sample_data", exist_ok=True)
        
        # Create sample sales data
        filename = "sample_data/sample_sales.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['date', 'product', 'category', 'quantity', 'price', 'revenue', 'region'])
            
            # Sample data
            products = ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones']
            categories = ['Electronics', 'Accessories', 'Accessories', 'Displays', 'Audio']
            regions = ['North', 'South', 'East', 'West']
            
            import random
            random.seed(42)
            
            for i in range(100):
                date = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
                product_idx = random.randint(0, 4)
                product = products[product_idx]
                category = categories[product_idx]
                quantity = random.randint(1, 10)
                price = random.randint(50, 500)
                revenue = quantity * price
                region = random.choice(regions)
                
                writer.writerow([date, product, category, quantity, price, revenue, region])
        
        return filename
    
    def create_sample(self):
        """Create sample data"""
        
        try:
            filename = self.create_sample_csv()
            print(f"✅ Created sample CSV: {filename}")
            print("💡 Now you can upload it by typing 'upload' and entering the path!")
        except Exception as e:
            print(f"❌ Failed to create sample: {e}")
    
    def logout(self):
        """Logout user"""
        
        print("\n👋 Logged out successfully!")
        print("💾 Your data is safely stored and will be available next time.")
        self.current_token = None
        self.current_user = None

def main():
    """Main application"""
    
    system = SuperXSimpleSystem()
    
    system.display_welcome()
    
    if system.login_flow():
        system.handle_actions()
    
    print("""
🙏 **THANK YOU FOR USING SUPERX!**

✨ **What you experienced:**
• Secure login with three different user roles
• CSV upload with persistent storage
• Natural language chat with your data
• Smart suggestions based on your datasets
• Data survives system restarts

🚀 **Your data is safely stored and ready for next time!**
""")

if __name__ == "__main__":
    main()