"""
Real Vector RAG System - Personalized for Each Company Login
Uses actual vector embeddings and preprocessing for scalable multi-tenant RAG
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
import faiss
from dataclasses import dataclass

@dataclass
class RAGResponse:
    response_text: str
    confidence: float
    sources: List[str]
    company_context: str

class RealVectorRAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.user_indices = {}  # FAISS indices per user
        self.user_documents = {}  # Document store per user
        self.user_metadata = {}  # Company metadata per user
        self.db_path = "rag_vector_db.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize vector database for multi-tenant RAG"""
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, doc_id)
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
    
    def load_company_data(self, user_id: str, company_name: str, dataset_path: str):
        """Load and process company data into vector embeddings"""
        try:
            if not os.path.exists(dataset_path):
                print(f"Dataset not found: {dataset_path}")
                return False
            
            # Load and preprocess data
            df = pd.read_csv(dataset_path)
            documents = self._preprocess_company_data(df, company_name)
            
            # Clear existing data for user
            self._clear_user_data(user_id)
            
            # Create embeddings and store
            for doc_id, content, metadata in documents:
                self._add_document(user_id, company_name, doc_id, content, metadata)
            
            # Build FAISS index
            self._build_user_index(user_id, company_name)
            
            print(f"Loaded {len(documents)} documents for {company_name}")
            return True
            
        except Exception as e:
            print(f"Error loading data for {company_name}: {e}")
            return False
    
    def _preprocess_company_data(self, df: pd.DataFrame, company_name: str) -> List[tuple]:
        """Preprocess company data into searchable documents"""
        documents = []
        
        # Product-level documents
        if 'product' in df.columns:
            for product in df['product'].unique():
                product_data = df[df['product'] == product]
                
                # Product summary
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
        
        # Time-based documents
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            monthly_data = df.groupby(df['date'].dt.to_period('M')).agg({
                'revenue': 'sum',
                'quantity': 'sum'
            }).reset_index()
            
            for _, row in monthly_data.iterrows():
                period = str(row['date'])
                revenue = row['revenue'] if 'revenue' in row else 0
                
                content = f"Monthly performance for {period}: Revenue {revenue}. Company: {company_name}"
                
                documents.append((
                    f"month_{period}",
                    content,
                    {"type": "monthly", "period": period, "revenue": revenue}
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
    
    def _add_document(self, user_id: str, company_name: str, doc_id: str, content: str, metadata: Dict):
        """Add document with embedding to database"""
        embedding = self.model.encode([content])[0]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_vectors 
            (user_id, company_name, doc_id, content, metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, company_name, doc_id, content, json.dumps(metadata), pickle.dumps(embedding)))
        
        conn.commit()
        conn.close()
    
    def _build_user_index(self, user_id: str, company_name: str):
        """Build FAISS index for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT doc_id, content, metadata, embedding 
            FROM user_vectors WHERE user_id = ?
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
        
        # Create and populate FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        self.user_indices[user_id] = index
        self.user_documents[user_id] = documents
        self.user_metadata[user_id] = {"company_name": company_name}
        
        # Update session info
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO user_sessions 
            (user_id, company_name, total_documents, last_updated, index_version)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, company_name, len(documents), datetime.now(), "1.0"))
        conn.commit()
        conn.close()
    
    def _clear_user_data(self, user_id: str):
        """Clear existing user data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM user_vectors WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
        
        # Clear memory
        if user_id in self.user_indices:
            del self.user_indices[user_id]
        if user_id in self.user_documents:
            del self.user_documents[user_id]
        if user_id in self.user_metadata:
            del self.user_metadata[user_id]
    
    def query_user_knowledge(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Query user's personalized knowledge base using vector similarity"""
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
            if idx < len(documents) and score > 0.3:  # Similarity threshold
                doc = documents[idx]
                results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': float(score),
                    'doc_id': doc['doc_id']
                })
        
        return results
    
    def generate_personalized_response(self, user_id: str, query: str) -> RAGResponse:
        """Generate personalized response using vector RAG"""
        # Get company context
        company_name = self.user_metadata.get(user_id, {}).get("company_name", "Unknown Company")
        
        # Retrieve relevant documents
        relevant_docs = self.query_user_knowledge(user_id, query, top_k=3)
        
        if not relevant_docs:
            return RAGResponse(
                response_text=f"I don't have specific data for {company_name} yet. Please upload your company data to get personalized insights.",
                confidence=0.3,
                sources=[],
                company_context=company_name
            )
        
        # Generate response based on retrieved documents
        response_text = self._generate_contextual_response(query, relevant_docs, company_name)
        
        # Calculate confidence based on similarity scores
        avg_score = np.mean([doc['score'] for doc in relevant_docs])
        confidence = min(0.95, avg_score * 1.2)
        
        sources = [doc['doc_id'] for doc in relevant_docs]
        
        return RAGResponse(
            response_text=response_text,
            confidence=confidence,
            sources=sources,
            company_context=company_name
        )
    
    def _generate_contextual_response(self, query: str, docs: List[Dict], company_name: str) -> str:
        """Generate contextual response from retrieved documents"""
        query_lower = query.lower()
        
        # Extract information from documents
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
        
        # Generate response based on query intent
        if any(word in query_lower for word in ['product', 'catalog', 'items']):
            if products:
                product_list = '\n'.join([f"• {p}" for p in products[:5]])
                return f"📦 **{company_name} Products**:\n\n{product_list}\n\nBased on your company data, these are your key products."
        
        elif any(word in query_lower for word in ['revenue', 'sales', 'money']):
            if revenues:
                total_revenue = sum(revenues)
                top_products = [(products[i], revenues[i]) for i in range(len(products))]
                top_products.sort(key=lambda x: x[1], reverse=True)
                
                top_list = '\n'.join([f"• {p}: ₹{r:,.0f}" for p, r in top_products[:3]])
                return f"💰 **{company_name} Revenue Analysis**:\n\nTotal Revenue: ₹{total_revenue:,.0f}\n\nTop Performers:\n{top_list}"
        
        elif any(word in query_lower for word in ['forecast', 'predict', 'future']):
            if categories:
                cat_list = ', '.join(categories[:3])
                return f"📈 **{company_name} Forecast**:\n\nI can generate forecasts for: {cat_list}\n\nWhich specific area would you like me to focus on?"
        
        elif any(word in query_lower for word in ['recommend', 'suggest', 'advice']):
            if products and revenues:
                best_product = products[revenues.index(max(revenues))]
                return f"🎯 **{company_name} Recommendations**:\n\n• Focus marketing on {best_product} (top performer)\n• Analyze demand patterns for better inventory\n• Consider expanding similar product lines"
        
        # Default response with context
        context_summary = f"I have data on {len(products)} products" if products else "your business data"
        return f"🤖 **{company_name} AI Assistant**:\n\nBased on {context_summary}, I can help with:\n• Product analysis\n• Revenue insights\n• Forecasting\n• Business recommendations\n\nWhat specific area interests you?"
    
    def update_user_data(self, user_id: str, new_data_path: str):
        """Update user data with new uploaded file"""
        company_name = self.user_metadata.get(user_id, {}).get("company_name", "Unknown")
        
        try:
            # Load new data
            new_df = pd.read_csv(new_data_path)
            new_documents = self._preprocess_company_data(new_df, company_name)
            
            # Add new documents to existing
            for doc_id, content, metadata in new_documents:
                # Use timestamp to make doc_id unique
                unique_doc_id = f"{doc_id}_{int(datetime.now().timestamp())}"
                self._add_document(user_id, company_name, unique_doc_id, content, metadata)
            
            # Rebuild index
            self._build_user_index(user_id, company_name)
            
            print(f"Updated {company_name} data with {len(new_documents)} new documents")
            return True
            
        except Exception as e:
            print(f"Error updating data for {company_name}: {e}")
            return False

# Global instance
real_vector_rag = RealVectorRAG()