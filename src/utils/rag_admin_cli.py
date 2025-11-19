#!/usr/bin/env python3
"""
RAG Management CLI Utility
Command-line interface for managing RAG systems across all users
"""

import argparse
import sys
import os
import json
from datetime import datetime
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.enhanced_rag_manager import enhanced_rag_manager as rag_manager
from rag.rag_manager import RAGHealthStatus
from auth.user_management import user_manager

def print_status_table(statuses: List[Dict]):
    """Print RAG status in a formatted table"""
    if not statuses:
        print("No users found.")
        return
    
    print("\n" + "="*100)
    print(f"{'User ID':<36} {'Company':<20} {'Status':<15} {'Documents':<10} {'Last Updated':<20}")
    print("="*100)
    
    for status in statuses:
        user_id = status.get('user_id', 'Unknown')[:35]
        company = status.get('company_name', 'Unknown')[:19]
        rag_status = status.get('status', 'Unknown')[:14]
        doc_counts = status.get('document_counts', {})
        total_docs = doc_counts.get('total', 0)
        last_updated = status.get('initialized_at', 'Never')[:19]
        
        print(f"{user_id:<36} {company:<20} {rag_status:<15} {total_docs:<10} {last_updated:<20}")
    
    print("="*100)

def print_health_status(health: RAGHealthStatus):
    """Print detailed health status for a user"""
    print(f"\n{'='*60}")
    print(f"RAG Health Status for {health.company_name}")
    print(f"{'='*60}")
    print(f"User ID: {health.user_id}")
    print(f"Company: {health.company_name}")
    print(f"Initialized: {'Yes' if health.is_initialized else 'No'}")
    print(f"Status: {health.status}")
    print(f"Total Documents: {health.total_documents}")
    print(f"  - CSV Files: {health.csv_count}")
    print(f"  - PDF Files: {health.pdf_count}")
    print(f"Last Updated: {health.last_updated or 'Never'}")
    print(f"Index Version: {health.index_version}")
    
    if health.error_message:
        print(f"Error: {health.error_message}")
    
    print(f"{'='*60}")

def print_diagnostics(diagnostics: Dict):
    """Print diagnostic results"""
    print(f"\n{'='*60}")
    print(f"RAG Diagnostics for User {diagnostics.get('user_id', 'Unknown')}")
    print(f"{'='*60}")
    print(f"Timestamp: {diagnostics.get('timestamp', 'Unknown')}")
    print(f"Overall Status: {diagnostics.get('overall_status', 'Unknown')}")
    
    checks = diagnostics.get('checks', {})
    for check_name, check_result in checks.items():
        status = check_result.get('status', 'unknown')
        message = check_result.get('message', 'No message')
        print(f"\n{check_name.title()}: {status.upper()}")
        print(f"  {message}")
        
        if 'recommendations' in check_result:
            for rec in check_result['recommendations']:
                print(f"  → {rec}")
    
    recommendations = diagnostics.get('recommendations', [])
    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    print(f"{'='*60}")

def list_all_users():
    """List all users and their RAG status"""
    print("Fetching RAG status for all users...")
    
    # Get system-wide status
    system_status = rag_manager.get_system_wide_rag_status()
    
    if not system_status.get('success'):
        print(f"Error getting system status: {system_status.get('error')}")
        return
    
    print(f"\nSystem Overview:")
    print(f"Total Users: {system_status.get('total_users', 0)}")
    print(f"Initialized: {system_status.get('initialized_users', 0)}")
    print(f"Uninitialized: {system_status.get('uninitialized_users', 0)}")
    print(f"Initialization Rate: {system_status.get('initialization_rate', 0)}%")
    print(f"System Health: {system_status.get('system_health', 'Unknown')}")
    
    # Get individual user statuses (simplified for CLI)
    # This would need to be implemented to iterate through users
    print("\nNote: Use 'status <user_id>' to check individual user status")

def check_user_status(user_id: str):
    """Check RAG status for a specific user"""
    print(f"Checking RAG status for user: {user_id}")
    
    status = rag_manager.check_rag_initialization_status(user_id)
    
    if not status.get('success'):
        print(f"Error: {status.get('error')}")
        return
    
    print(f"\nUser ID: {user_id}")
    print(f"Company: {status.get('company_name', 'Unknown')}")
    print(f"Initialized: {'Yes' if status.get('is_initialized') else 'No'}")
    print(f"Status: {status.get('status', 'Unknown')}")
    print(f"Requires Initialization: {'Yes' if status.get('requires_initialization') else 'No'}")
    
    if status.get('error_message'):
        print(f"Error Message: {status.get('error_message')}")
    
    doc_counts = status.get('document_counts', {})
    print(f"Documents: {doc_counts.get('total', 0)} total ({doc_counts.get('csv', 0)} CSV, {doc_counts.get('pdf', 0)} PDF)")

def get_user_health(user_id: str):
    """Get detailed health status for a user"""
    print(f"Getting health status for user: {user_id}")
    
    health = rag_manager.get_rag_health_status(user_id)
    print_health_status(health)

def run_user_diagnostics(user_id: str):
    """Run diagnostics for a user"""
    print(f"Running diagnostics for user: {user_id}")
    
    diagnostics = rag_manager.run_rag_diagnostics(user_id)
    print_diagnostics(diagnostics)

def initialize_user_rag(user_id: str, company_name: str = None, force: bool = False):
    """Initialize RAG for a user"""
    if not company_name:
        # Try to get company name from user data
        # This would need user_manager integration
        company_name = "Unknown Company"
    
    print(f"Initializing RAG for user: {user_id} (Company: {company_name})")
    
    result = rag_manager.initialize_rag_for_user(user_id, company_name, force)
    
    if result.get('success'):
        print(f"✓ {result.get('message')}")
    else:
        print(f"✗ Error: {result.get('error')}")

def reset_user_rag(user_id: str, company_name: str = None):
    """Reset RAG for a user"""
    if not company_name:
        company_name = "Unknown Company"
    
    print(f"Resetting RAG for user: {user_id} (Company: {company_name})")
    print("WARNING: This will delete all RAG data for this user!")
    
    confirm = input("Are you sure? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    result = rag_manager.reset_rag_system(user_id, company_name)
    
    if result.get('success'):
        print(f"✓ {result.get('message')}")
    else:
        print(f"✗ Error: {result.get('error')}")

def migrate_users(batch_size: int = 10):
    """Migrate existing users to RAG system"""
    print(f"Migrating users to RAG system (batch size: {batch_size})")
    
    result = rag_manager.migrate_existing_users_to_rag(batch_size)
    
    if result.get('success'):
        print(f"✓ Migration completed:")
        print(f"  Users processed: {result.get('users_processed', 0)}")
        print(f"  Users migrated: {result.get('users_migrated', 0)}")
        
        errors = result.get('errors', [])
        if errors:
            print(f"  Errors: {len(errors)}")
            for error in errors[:5]:  # Show first 5 errors
                print(f"    - {error}")
    else:
        print(f"✗ Migration failed: {result.get('error')}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="RAG Management CLI Utility")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all users and their RAG status')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check RAG status for a user')
    status_parser.add_argument('user_id', help='User ID to check')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Get health status for a user')
    health_parser.add_argument('user_id', help='User ID to check')
    
    # Diagnostics command
    diag_parser = subparsers.add_parser('diagnostics', help='Run diagnostics for a user')
    diag_parser.add_argument('user_id', help='User ID to diagnose')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize RAG for a user')
    init_parser.add_argument('user_id', help='User ID to initialize')
    init_parser.add_argument('--company', help='Company name')
    init_parser.add_argument('--force', action='store_true', help='Force reinitialization')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset RAG for a user')
    reset_parser.add_argument('user_id', help='User ID to reset')
    reset_parser.add_argument('--company', help='Company name')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate existing users to RAG')
    migrate_parser.add_argument('--batch-size', type=int, default=10, help='Batch size for migration')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_all_users()
        elif args.command == 'status':
            check_user_status(args.user_id)
        elif args.command == 'health':
            get_user_health(args.user_id)
        elif args.command == 'diagnostics':
            run_user_diagnostics(args.user_id)
        elif args.command == 'init':
            initialize_user_rag(args.user_id, args.company, args.force)
        elif args.command == 'reset':
            reset_user_rag(args.user_id, args.company)
        elif args.command == 'migrate':
            migrate_users(args.batch_size)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()