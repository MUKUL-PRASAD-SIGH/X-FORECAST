"""
Fallback RAG System
Provides basic RAG functionality when sentence_transformers or other dependencies are missing.
Uses TF-IDF and basic text matching as fallback mechanisms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from datetime import datetime
import sqlite3
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FallbackDocumentSource:
    """Source attribution for fallback RAG responses"""
    type: str  # 'csv' or 'pdf'
    filename: str
    page_number: Optional[int] = None
    relevance_score: float = 0.0
    doc_id: str = ""

@dataclass
class FallbackRAGResponse:
    """Response from fallback RAG system"""
    response_text: str
    confidence: float
    sources: List[FallbackDocumentSource]
    company_context: str
    fallback_mode: bool = True

class FallbackRAGSystem:
    """
    Fallback RAG system that works without sentence_transformers
    Uses TF-IDF and keyword matching for basic document retrieval
    """
    
    def __init__(self):
        self.user_documents = {}  # Document store per user
        self.user_metadata = {}  # Company metadata per user
        self.db_path = "rag_vector_db.db"
        self.vectorizer = None
        self.use_sklearn = self._check_sklearn_availability()
        self._init_database()
        
        logger.info("Initialized fallback RAG system (limited functionality)")
    
    def _check_sklearn_availability(self) -> bool:
        """Check if sklearn is available for TF-IDF fallback"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            return True
        except ImportError:
            logger.warning("sklearn not available - using basic keyword matching")
            return False
    
    def _init_database(self):
        """Initialize database with same structure as full RAG system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                company_name TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB,
                document_type TEXT DEFAULT 'csv',
                source_file TEXT,
                page_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, doc_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_documents (
                document_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_status TEXT DEFAULT 'pending',
                file_size INTEGER,
                page_count INTEGER DEFAULT 0,
                file_hash TEXT,
                metadata TEXT,
                error_message TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                user_id TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                total_documents INTEGER DEFAULT 0,
                last_updated TIMESTAMP,
                index_version TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def initialize_company_rag(self, user_id: str, company_name: str) -> bool:
        """
        Initialize fallback RAG system for a company
        
        Args:
            user_id: User identifier
            company_name: Company name
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create user metadata entry
            self.user_metadata[user_id] = {"company_name": company_name}
            
            # Initialize empty structures
            self.user_documents[user_id] = []
            
            # Update session info
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO user_sessions 
                (user_id, company_name, total_documents, last_updated, index_version)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, company_name, 0, datetime.now(), "fallback-1.0"))
            conn.commit()
            conn.close()
            
            logger.info(f"Initialized fallback RAG system for {company_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing fallback RAG for {company_name}: {e}")
            return False
    
    def load_company_data(self, user_id: str, company_name: str, dataset_path: str) -> bool:
        """
        Load and process company data for fallback system
        
        Args:
            user_id: User identifier
            company_name: Company name
            dataset_path: Path to CSV dataset
            
        Returns:
            Boolean indicating success
        """
        try:
            if not os.path.exists(dataset_path):
                logger.error(f"Dataset not found: {dataset_path}")
                return False
            
            # Load and preprocess data
            df = pd.read_csv(dataset_path)
            documents = self._preprocess_company_data(df, company_name)
            
            # Clear existing data for user
            self._clear_user_data(user_id)
            
            # Store documents
            for doc_id, content, metadata in documents:
                self._add_document(user_id, company_name, doc_id, content, metadata)
            
            # Build search index
            self._build_user_index(user_id, company_name)
            
            logger.info(f"Loaded {len(documents)} documents for {company_name} (fallback mode)")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data for {company_name}: {e}")
            return False
    
    def _preprocess_company_data(self, df: pd.DataFrame, company_name: str) -> List[tuple]:
        """Preprocess company data into searchable documents (same as full system)"""
        documents = []
        
        # Product-level documents
        if 'product' in df.columns:
            for product in df['product'].unique():
                product_data = df[df['product'] == product]
                
                total_revenue = product_data['revenue'].sum() if 'revenue' in df.columns else 0
                total_quantity = product_data['quantity'].sum() if 'quantity' in df.columns else 0
                category = product_data['category'].iloc[0] if 'category' in df.columns else 'Unknown'
                
                content = f"Product: {product}. Category: {category}. Total revenue: {total_revenue}. Total quantity sold: {total_quantity}. Company: {company_name}"
                
                documents.append((
                    f"product_{product.replace(' ', '_')}",
                    content,
                    {"type": "product", "product": product, "category": category, "revenue": total_revenue}
                ))
        
        # Category-level documents
        if 'category' in df.columns:
            for category in df['category'].unique():
                category_data = df[df['category'] == category]
                
                total_revenue = category_data['revenue'].sum() if 'revenue' in df.columns else 0
                product_count = category_data['product'].nunique() if 'product' in df.columns else 0
                
                content = f"Category: {category}. Total revenue: {total_revenue}. Number of products: {product_count}. Company: {company_name}"
                
                documents.append((
                    f"category_{category.replace(' ', '_')}",
                    content,
                    {"type": "category", "category": category, "revenue": total_revenue, "product_count": product_count}
                ))
        
        # Company overview document
        total_revenue = df['revenue'].sum() if 'revenue' in df.columns else 0
        total_products = df['product'].nunique() if 'product' in df.columns else 0
        total_categories = df['category'].nunique() if 'category' in df.columns else 0
        
        overview_content = f"Company overview for {company_name}: Total revenue {total_revenue}, {total_products} products across {total_categories} categories"
        
        documents.append((
            "company_overview",
            overview_content,
            {"type": "overview", "total_revenue": total_revenue, "total_products": total_products}
        ))
        
        return documents
    
    def _add_document(self, user_id: str, company_name: str, doc_id: str, content: str, metadata: Dict, 
                     document_type: str = 'csv', source_file: str = None, page_number: int = None):
        """Add document to database (without embeddings)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_vectors 
            (user_id, company_name, doc_id, content, metadata, document_type, source_file, page_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, company_name, doc_id, content, json.dumps(metadata), 
              document_type, source_file, page_number))
        
        conn.commit()
        conn.close()
    
    def _build_user_index(self, user_id: str, company_name: str):
        """Build search index for user (TF-IDF or keyword-based)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT doc_id, content, metadata, document_type, source_file, page_number
            FROM user_vectors WHERE user_id = ?
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return
        
        documents = []
        for doc_id, content, metadata, document_type, source_file, page_number in results:
            documents.append({
                'doc_id': doc_id,
                'content': content,
                'metadata': json.loads(metadata),
                'document_type': document_type or 'csv',
                'source_file': source_file,
                'page_number': page_number
            })
        
        self.user_documents[user_id] = documents
        self.user_metadata[user_id] = {"company_name": company_name}
        
        # Build TF-IDF index if sklearn is available
        if self.use_sklearn:
            self._build_tfidf_index(user_id, documents)
        
        # Update session info
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO user_sessions 
            (user_id, company_name, total_documents, last_updated, index_version)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, company_name, len(documents), datetime.now(), "fallback-1.0"))
        conn.commit()
        conn.close()
    
    def _build_tfidf_index(self, user_id: str, documents: List[Dict]):
        """Build TF-IDF index for better text matching"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            texts = [doc['content'] for doc in documents]
            
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Store in user metadata for later use
            if user_id not in self.user_metadata:
                self.user_metadata[user_id] = {}
            
            self.user_metadata[user_id]['tfidf_vectorizer'] = vectorizer
            self.user_metadata[user_id]['tfidf_matrix'] = tfidf_matrix
            
            logger.debug(f"Built TF-IDF index for user {user_id} with {len(texts)} documents")
            
        except Exception as e:
            logger.warning(f"Failed to build TF-IDF index: {e}")
    
    def _clear_user_data(self, user_id: str):
        """Clear existing user data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM user_vectors WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
        
        # Clear memory
        if user_id in self.user_documents:
            del self.user_documents[user_id]
        if user_id in self.user_metadata:
            # Keep company name but clear search indices
            company_name = self.user_metadata[user_id].get("company_name")
            self.user_metadata[user_id] = {"company_name": company_name} if company_name else {}
    
    def query_user_knowledge(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query user's knowledge base using fallback methods
        
        Args:
            user_id: User identifier
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        if user_id not in self.user_documents:
            return []
        
        documents = self.user_documents[user_id]
        
        if self.use_sklearn and 'tfidf_vectorizer' in self.user_metadata.get(user_id, {}):
            return self._query_with_tfidf(user_id, query, top_k)
        else:
            return self._query_with_keywords(user_id, query, top_k)
    
    def _query_with_tfidf(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        """Query using TF-IDF similarity"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            user_meta = self.user_metadata[user_id]
            vectorizer = user_meta['tfidf_vectorizer']
            tfidf_matrix = user_meta['tfidf_matrix']
            documents = self.user_documents[user_id]
            
            # Transform query
            query_vector = vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    doc = documents[idx]
                    results.append({
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'score': float(similarities[idx]),
                        'doc_id': doc['doc_id'],
                        'document_type': doc.get('document_type', 'csv'),
                        'source_file': doc.get('source_file'),
                        'page_number': doc.get('page_number')
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF query failed: {e}")
            return self._query_with_keywords(user_id, query, top_k)
    
    def _query_with_keywords(self, user_id: str, query: str, top_k: int) -> List[Dict]:
        """Query using simple keyword matching"""
        documents = self.user_documents[user_id]
        query_words = set(query.lower().split())
        
        results = []
        for doc in documents:
            content_words = set(doc['content'].lower().split())
            
            # Calculate simple word overlap score
            overlap = len(query_words.intersection(content_words))
            if overlap > 0:
                score = overlap / len(query_words.union(content_words))
                
                results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': score,
                    'doc_id': doc['doc_id'],
                    'document_type': doc.get('document_type', 'csv'),
                    'source_file': doc.get('source_file'),
                    'page_number': doc.get('page_number')
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def generate_personalized_response(self, user_id: str, query: str) -> FallbackRAGResponse:
        """
        Generate personalized response using fallback RAG
        
        Args:
            user_id: User identifier
            query: User query
            
        Returns:
            FallbackRAGResponse with limited functionality notice
        """
        # Get company context
        company_name = self.user_metadata.get(user_id, {}).get("company_name", "Unknown Company")
        
        # Retrieve relevant documents
        relevant_docs = self.query_user_knowledge(user_id, query, top_k=5)
        
        if not relevant_docs:
            return FallbackRAGResponse(
                response_text=f"⚠️ Limited Search Mode: I don't have specific data for {company_name} yet. Please upload your company data to get personalized insights. Note: Advanced AI features are currently unavailable.",
                confidence=0.3,
                sources=[],
                company_context=company_name,
                fallback_mode=True
            )
        
        # Generate response based on retrieved documents
        response_text = self._generate_fallback_response(query, relevant_docs, company_name)
        
        # Calculate confidence (lower than full system)
        avg_score = np.mean([doc['score'] for doc in relevant_docs])
        confidence = min(0.7, avg_score * 0.8)  # Lower confidence in fallback mode
        
        # Create source objects
        sources = []
        for doc in relevant_docs:
            source = FallbackDocumentSource(
                type=doc.get('document_type', 'csv'),
                filename=doc.get('source_file', 'Unknown'),
                page_number=doc.get('page_number'),
                relevance_score=doc['score'],
                doc_id=doc['doc_id']
            )
            sources.append(source)
        
        return FallbackRAGResponse(
            response_text=response_text,
            confidence=confidence,
            sources=sources,
            company_context=company_name,
            fallback_mode=True
        )
    
    def _generate_fallback_response(self, query: str, docs: List[Dict], company_name: str) -> str:
        """Generate response with fallback mode notice"""
        query_lower = query.lower()
        
        # Add fallback mode notice
        fallback_notice = "⚠️ **Limited Search Mode** (Advanced AI features unavailable)\n\n"
        
        # Extract information from documents (simplified version)
        products = []
        categories = []
        revenues = []
        
        for doc in docs:
            metadata = doc['metadata']
            if metadata.get('type') == 'product':
                products.append(metadata.get('product'))
                revenues.append(metadata.get('revenue', 0))
            elif metadata.get('type') == 'category':
                categories.append(metadata.get('category'))
        
        # Generate basic response
        if any(word in query_lower for word in ['product', 'catalog', 'items']):
            if products:
                product_list = '\n'.join([f"• {p}" for p in products[:5]])
                return f"{fallback_notice}📦 **{company_name} Products**:\n\n{product_list}\n\n*Note: This is a basic keyword search. Install sentence-transformers for advanced AI search.*"
        
        elif any(word in query_lower for word in ['revenue', 'sales', 'money']):
            if revenues:
                total_revenue = sum(revenues)
                return f"{fallback_notice}💰 **{company_name} Revenue**:\n\nTotal Revenue: ₹{total_revenue:,.0f}\n\n*Note: Basic calculation only. Advanced analytics require full AI system.*"
        
        # Default response
        return f"{fallback_notice}🔍 **{company_name} Search Results**:\n\nFound {len(docs)} matching items in your data. For advanced AI insights and natural language processing, please install the required dependencies.\n\n*Current mode: Basic keyword matching*"

# Global fallback instance
fallback_rag_system = FallbackRAGSystem()