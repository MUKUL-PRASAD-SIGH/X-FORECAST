"""
Document Count Updater
Updates document counts in business_profiles table based on actual uploaded files
"""

import sqlite3
import os
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class DocumentCountUpdater:
    """Update document counts in business profiles"""
    
    def __init__(self, users_db_path: str = "users.db", rag_db_path: str = "rag_vector_db.db"):
        self.users_db_path = users_db_path
        self.rag_db_path = rag_db_path
    
    def update_document_counts(self) -> Dict[str, int]:
        """
        Update document counts for all users based on actual uploaded files
        
        Returns:
            Dictionary with update statistics
        """
        try:
            stats = {"users_updated": 0, "total_pdf_count": 0, "total_csv_count": 0}
            
            # Get all users
            users_conn = sqlite3.connect(self.users_db_path)
            users_cursor = users_conn.cursor()
            
            users_cursor.execute("SELECT user_id FROM users WHERE is_active = 1")
            users = users_cursor.fetchall()
            
            # Connect to RAG database to get document counts
            rag_conn = sqlite3.connect(self.rag_db_path)
            rag_cursor = rag_conn.cursor()
            
            for (user_id,) in users:
                # Count PDF documents
                rag_cursor.execute('''
                    SELECT COUNT(*) FROM user_documents 
                    WHERE user_id = ? AND file_type = 'pdf' AND processing_status = 'success'
                ''', (user_id,))
                pdf_count = rag_cursor.fetchone()[0]
                
                # Count CSV documents by checking file system
                csv_count = self._count_csv_files(user_id)
                
                total_documents = pdf_count + csv_count
                
                # Update business_profiles
                users_cursor.execute('''
                    UPDATE business_profiles 
                    SET pdf_count = ?, csv_count = ?, total_documents = ?
                    WHERE user_id = ?
                ''', (pdf_count, csv_count, total_documents, user_id))
                
                if users_cursor.rowcount > 0:
                    stats["users_updated"] += 1
                    stats["total_pdf_count"] += pdf_count
                    stats["total_csv_count"] += csv_count
                    
                    logger.info(f"Updated user {user_id}: {pdf_count} PDFs, {csv_count} CSVs")
            
            users_conn.commit()
            users_conn.close()
            rag_conn.close()
            
            logger.info(f"Document count update completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error updating document counts: {str(e)}")
            return {"error": str(e)}
    
    def _count_csv_files(self, user_id: str) -> int:
        """Count CSV files for a user by checking file system"""
        try:
            csv_dir = f"data/users/{user_id}/csv"
            if os.path.exists(csv_dir):
                csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]
                return len(csv_files)
            return 0
        except Exception:
            return 0
    
    def get_document_summary(self) -> Dict[str, any]:
        """Get summary of all documents in the system"""
        try:
            summary = {
                "total_users": 0,
                "users_with_documents": 0,
                "total_pdfs": 0,
                "total_csvs": 0,
                "total_documents": 0,
                "processing_status": {}
            }
            
            # Get user count
            users_conn = sqlite3.connect(self.users_db_path)
            users_cursor = users_conn.cursor()
            
            users_cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
            summary["total_users"] = users_cursor.fetchone()[0]
            
            users_cursor.execute('''
                SELECT COUNT(*) FROM business_profiles 
                WHERE total_documents > 0
            ''')
            summary["users_with_documents"] = users_cursor.fetchone()[0]
            
            users_cursor.execute("SELECT SUM(pdf_count), SUM(csv_count), SUM(total_documents) FROM business_profiles")
            pdf_sum, csv_sum, total_sum = users_cursor.fetchone()
            
            summary["total_pdfs"] = pdf_sum or 0
            summary["total_csvs"] = csv_sum or 0
            summary["total_documents"] = total_sum or 0
            
            users_conn.close()
            
            # Get processing status from RAG database
            if os.path.exists(self.rag_db_path):
                rag_conn = sqlite3.connect(self.rag_db_path)
                rag_cursor = rag_conn.cursor()
                
                rag_cursor.execute('''
                    SELECT processing_status, COUNT(*) 
                    FROM user_documents 
                    GROUP BY processing_status
                ''')
                
                for status, count in rag_cursor.fetchall():
                    summary["processing_status"][status] = count
                
                rag_conn.close()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting document summary: {str(e)}")
            return {"error": str(e)}

def update_all_document_counts():
    """Convenience function to update all document counts"""
    updater = DocumentCountUpdater()
    
    print("Updating document counts...")
    stats = updater.update_document_counts()
    
    print("\nUPDATE RESULTS:")
    print("-" * 30)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nGetting document summary...")
    summary = updater.get_document_summary()
    
    print("\nDOCUMENT SUMMARY:")
    print("-" * 30)
    for key, value in summary.items():
        if key == "processing_status":
            print(f"{key}:")
            for status, count in value.items():
                print(f"  {status}: {count}")
        else:
            print(f"{key}: {value}")
    
    return stats, summary

if __name__ == "__main__":
    update_all_document_counts()