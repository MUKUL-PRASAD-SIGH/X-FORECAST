"""
Create 3 test users with sample data
"""

import sqlite3
import hashlib
import uuid
import os

def create_test_users():
    """Create 3 test users"""
    
    # Create users database
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            company_name TEXT NOT NULL,
            business_type TEXT NOT NULL,
            subscription_tier TEXT DEFAULT 'basic',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS business_profiles (
            user_id TEXT PRIMARY KEY,
            company_name TEXT NOT NULL,
            business_type TEXT NOT NULL,
            industry TEXT,
            data_sources TEXT,
            model_config TEXT,
            storage_path TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # Test users data
    test_users = [
        {
            "email": "admin@superx.com",
            "password": "admin123",
            "company_name": "SuperX Corporation",
            "business_type": "supermarket",
            "industry": "retail"
        },
        {
            "email": "john@techcorp.com", 
            "password": "john123",
            "company_name": "TechCorp Industries",
            "business_type": "ecommerce",
            "industry": "technology"
        },
        {
            "email": "sarah@healthplus.com",
            "password": "sarah123", 
            "company_name": "HealthPlus Medical",
            "business_type": "retail",
            "industry": "healthcare"
        }
    ]
    
    for user_data in test_users:
        user_id = str(uuid.uuid4())
        password_hash = hashlib.sha256(user_data["password"].encode()).hexdigest()
        storage_path = f"data/users/{user_id}"
        
        # Create user directory
        os.makedirs(storage_path, exist_ok=True)
        
        try:
            # Insert user
            cursor.execute('''
                INSERT OR REPLACE INTO users (user_id, email, password_hash, company_name, business_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, user_data["email"], password_hash, user_data["company_name"], user_data["business_type"]))
            
            # Insert business profile
            cursor.execute('''
                INSERT OR REPLACE INTO business_profiles 
                (user_id, company_name, business_type, industry, data_sources, model_config, storage_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, user_data["company_name"], user_data["business_type"], user_data["industry"], "[]", "{}", storage_path))
            
            print(f"Created user: {user_data['email']} / {user_data['password']} ({user_data['company_name']})")
            
        except Exception as e:
            print(f"Error creating user {user_data['email']}: {e}")
    
    conn.commit()
    conn.close()
    
    print("\nTest users created successfully!")
    print("\nLogin Credentials:")
    print("1. admin@superx.com / admin123 (SuperX Corporation)")
    print("2. john@techcorp.com / john123 (TechCorp Industries)")  
    print("3. sarah@healthplus.com / sarah123 (HealthPlus Medical)")

if __name__ == "__main__":
    create_test_users()