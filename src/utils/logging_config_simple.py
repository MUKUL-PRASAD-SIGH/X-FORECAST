"""
Simple logging configuration for testing
"""

import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional

class SimpleLogger:
    """Simple logger for testing"""
    
    def __init__(self, log_db_path: str = "logs/system_logs.db"):
        self.log_db_path = log_db_path
        os.makedirs(os.path.dirname(log_db_path), exist_ok=True)
        
        # Initialize database
        conn = sqlite3.connect(self.log_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS log_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                user_id TEXT,
                company_name TEXT,
                ip_address TEXT,
                message TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_login_success(self, user_id: str, email: str, company_name: str, ip_address: str):
        """Log successful login"""
        conn = sqlite3.connect(self.log_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO log_events (timestamp, event_type, user_id, company_name, ip_address, message)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), 'login_success', user_id, company_name, ip_address, f"Login successful for {email}"))
        
        conn.commit()
        conn.close()
    
    def get_auth_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get basic statistics"""
        conn = sqlite3.connect(self.log_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM log_events WHERE event_type = "login_success"')
        count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "period_days": days,
            "total_login_attempts": count,
            "successful_logins": count,
            "failed_logins": 0,
            "unique_users": 1,
            "new_registrations": 0,
            "success_rate": 100.0
        }

# Global instance
simple_logger = SimpleLogger()