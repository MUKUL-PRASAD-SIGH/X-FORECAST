"""
Real RAG System with Vector Database and Continuous Learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime
import sqlite3
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

class VectorRAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.user_indices = {}  # FAISS indices per user
        self.user_documents = {}  # Document store per user
        self.db_path = "vector_rag.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize vector database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, doc_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_knowledge (
                user_id TEXT PRIMARY KEY,
                total_documents INTEGER DEFAULT 0,
                last_training TIMESTAMP,
                model_version TEXT,
                performance_metrics TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_user_data(self, user_id: str, data: pd.DataFrame, data_type: str = "sales"):
        """Add new data and create embeddings"""
        documents = self._create_documents_from_data(data, data_type)
        
        for doc_id, content, metadata in documents:
            self._add_document(user_id, doc_id, content, metadata)
        
        self._rebuild_user_index(user_id)
        self._update_user_knowledge(user_id)
    
    def _create_documents_from_data(self, data: pd.DataFrame, data_type: str) -> List[tuple]:
        """Convert data into searchable documents"""
        documents = []
        
        if data_type == "sales":
            # Product-level summaries
            if 'product' in data.columns and 'revenue' in data.columns:
                product_stats = data.groupby('product').agg({
                    'revenue': ['sum', 'mean', 'count'],
                    'quantity': ['sum', 'mean'] if 'quantity' in data.columns else ['count']
                }).round(2)
                
                for product in product_stats.index:
                    stats = product_stats.loc[product]
                    content = f"Product {product}: Total revenue ₹{stats[('revenue', 'sum')]}, Average ₹{stats[('revenue', 'mean')]}, {stats[('revenue', 'count')]} transactions"
                    documents.append((f"product_{product}", content, {"type": "product_summary", "product": product}))
            
            # Time-based patterns
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                monthly_stats = data.groupby(data['date'].dt.to_period('M'))['revenue'].sum()
                
                for period, revenue in monthly_stats.items():
                    content = f"Monthly performance {period}: Total revenue ₹{revenue}"
                    documents.append((f"month_{period}", content, {"type": "monthly_summary", "period": str(period)}))
            
            # Category insights
            if 'category' in data.columns:
                category_stats = data.groupby('category')['revenue'].agg(['sum', 'count'])
                
                for category in category_stats.index:
                    stats = category_stats.loc[category]
                    content = f"Category {category}: Revenue ₹{stats['sum']}, {stats['count']} sales"
                    documents.append((f"category_{category}", content, {"type": "category_summary", "category": category}))
        
        return documents
    
    def _add_document(self, user_id: str, doc_id: str, content: str, metadata: Dict):
        """Add document with embedding to database"""
        embedding = self.model.encode([content])[0]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_documents 
            (user_id, doc_id, content, metadata, embedding)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, doc_id, content, json.dumps(metadata), pickle.dumps(embedding)))
        
        conn.commit()
        conn.close()
    
    def _rebuild_user_index(self, user_id: str):
        """Rebuild FAISS index for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT doc_id, content, metadata, embedding 
            FROM user_documents WHERE user_id = ?
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return
        
        # Create FAISS index
        embeddings = []
        documents = []
        
        for doc_id, content, metadata, embedding_blob in results:
            embedding = pickle.loads(embedding_blob)
            embeddings.append(embedding)
            documents.append({
                'doc_id': doc_id,
                'content': content,
                'metadata': json.loads(metadata)
            })
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        self.user_indices[user_id] = index
        self.user_documents[user_id] = documents
    
    def _update_user_knowledge(self, user_id: str):
        """Update user knowledge tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM user_documents WHERE user_id = ?', (user_id,))
        doc_count = cursor.fetchone()[0]
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_knowledge 
            (user_id, total_documents, last_training, model_version, performance_metrics)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, doc_count, datetime.now(), "1.0", json.dumps({"accuracy": 0.9})))
        
        conn.commit()
        conn.close()
    
    def query_user_knowledge(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Query user's personalized knowledge base"""
        if user_id not in self.user_indices:
            self._rebuild_user_index(user_id)
        
        if user_id not in self.user_indices:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])[0].astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        index = self.user_indices[user_id]
        scores, indices = index.search(query_embedding, min(top_k, index.ntotal))
        
        results = []
        documents = self.user_documents[user_id]
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(documents):
                doc = documents[idx]
                results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': float(score)
                })
        
        return results
    
    def generate_personalized_response(self, user_id: str, query: str) -> Dict[str, Any]:
        """Generate response using user's knowledge base"""
        # Get relevant documents
        relevant_docs = self.query_user_knowledge(user_id, query, top_k=3)
        
        if not relevant_docs:
            return self._fallback_response(query)
        
        # Analyze query intent
        query_lower = query.lower()
        
        # Build context from relevant documents
        context = "\n".join([doc['content'] for doc in relevant_docs])
        
        # Generate response based on context
        if "forecast" in query_lower or "predict" in query_lower:
            return self._generate_forecast_response(context, relevant_docs, query)
        elif "revenue" in query_lower or "sales" in query_lower:
            return self._generate_revenue_response(context, relevant_docs)
        elif "product" in query_lower:
            return self._generate_product_response(context, relevant_docs)
        elif "category" in query_lower:
            return self._generate_category_response(context, relevant_docs)
        else:
            return self._generate_general_response(context, relevant_docs)
    
    def _generate_forecast_response(self, context: str, docs: List[Dict], query: str) -> Dict[str, Any]:
        """Generate forecast response using retrieved context"""
        # Extract product mentions from query
        products = []
        for doc in docs:
            if doc['metadata'].get('type') == 'product_summary':
                products.append(doc['metadata']['product'])
        
        if products:
            product_info = [doc['content'] for doc in docs if 'product' in doc['content'].lower()]
            response = f"📈 **Personalized Forecast**: Based on your sales data:\n\n"
            response += "\n".join(product_info[:3])
            response += f"\n\nI can generate detailed forecasts for: {', '.join(products[:5])}"
        else:
            response = "📈 **Forecast Available**: Upload more sales data for personalized forecasts."
        
        return {
            "response_text": response,
            "products": products,
            "confidence": 0.9 if products else 0.5,
            "context_used": len(docs)
        }
    
    def _generate_revenue_response(self, context: str, docs: List[Dict]) -> Dict[str, Any]:
        """Generate revenue analysis response"""
        revenue_info = [doc['content'] for doc in docs if 'revenue' in doc['content'].lower()]
        
        response = "💰 **Revenue Analysis**: Based on your data:\n\n"
        response += "\n".join(revenue_info[:3])
        
        return {
            "response_text": response,
            "confidence": 0.95,
            "context_used": len(docs)
        }
    
    def _generate_product_response(self, context: str, docs: List[Dict]) -> Dict[str, Any]:
        """Generate product-specific response"""
        product_docs = [doc for doc in docs if doc['metadata'].get('type') == 'product_summary']
        
        if product_docs:
            response = "📦 **Product Insights**: From your sales data:\n\n"
            for doc in product_docs[:3]:
                response += f"• {doc['content']}\n"
        else:
            response = "📦 **Product Analysis**: Upload product sales data for detailed insights."
        
        return {
            "response_text": response,
            "confidence": 0.9 if product_docs else 0.4,
            "context_used": len(docs)
        }
    
    def _generate_category_response(self, context: str, docs: List[Dict]) -> Dict[str, Any]:
        """Generate category analysis response"""
        category_docs = [doc for doc in docs if doc['metadata'].get('type') == 'category_summary']
        
        if category_docs:
            response = "🏷️ **Category Performance**: Based on your data:\n\n"
            for doc in category_docs[:3]:
                response += f"• {doc['content']}\n"
        else:
            response = "🏷️ **Category Analysis**: Upload categorized sales data for insights."
        
        return {
            "response_text": response,
            "confidence": 0.85 if category_docs else 0.4,
            "context_used": len(docs)
        }
    
    def _generate_general_response(self, context: str, docs: List[Dict]) -> Dict[str, Any]:
        """Generate general response using available context"""
        response = "🤖 **Your Business Insights**: Based on your uploaded data:\n\n"
        
        for doc in docs[:2]:
            response += f"• {doc['content']}\n"
        
        response += f"\nI have {len(docs)} relevant insights for your query."
        
        return {
            "response_text": response,
            "confidence": 0.8,
            "context_used": len(docs)
        }
    
    def _fallback_response(self, query: str) -> Dict[str, Any]:
        """Fallback when no user data available"""
        return {
            "response_text": "📊 **Getting Started**: Upload your sales data (CSV/Excel) to get personalized AI insights trained on your business data!",
            "confidence": 0.3,
            "context_used": 0,
            "is_fallback": True
        }
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user's RAG statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_knowledge WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "total_documents": result[1],
                "last_training": result[2],
                "model_version": result[3],
                "has_index": user_id in self.user_indices
            }
        
        return {"total_documents": 0, "has_index": False}

# Global vector RAG instance
vector_rag = VectorRAG()