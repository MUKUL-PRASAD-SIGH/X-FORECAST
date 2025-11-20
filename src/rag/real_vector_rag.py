"""
Real Vector RAG System - Personalized for Each Company Login
Uses actual vector embeddings and preprocessing for scalable multi-tenant RAG
Enhanced with PDF processing capabilities for document-based knowledge
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from datetime import datetime
import sqlite3
import pickle
from dataclasses import dataclass
import logging
import time

# Import comprehensive logging system
from src.utils.logging_config import comprehensive_logger

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

# Import dependencies with graceful degradation
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence_transformers not available - RAG system will use fallback mode")
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("faiss not available - will use slower similarity search")
    faiss = None
    FAISS_AVAILABLE = False

# Import PDF processor
try:
    from .pdf_processor import PDFProcessor, PDFExtractionResult, PDFMetadata
    PDF_PROCESSOR_AVAILABLE = True
except ImportError:
    logger.warning("PDF processor not available")
    PDFProcessor = None
    PDFExtractionResult = None
    PDFMetadata = None
    PDF_PROCESSOR_AVAILABLE = False

@dataclass
class DocumentSource:
    """Source attribution for RAG responses"""
    type: str  # 'csv' or 'pdf'
    filename: str
    page_number: Optional[int] = None
    relevance_score: float = 0.0
    doc_id: str = ""

@dataclass
class RAGResponse:
    response_text: str
    confidence: float
    sources: List[DocumentSource]
    company_context: str

class RealVectorRAG:
    def __init__(self):
        # Initialize with dependency checking
        self.dependencies_available = self._check_dependencies()
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = self._create_fallback_model()
        
        self.user_indices = {}  # FAISS indices per user
        self.user_documents = {}  # Document store per user
        self.user_metadata = {}  # Company metadata per user
        self.db_path = "rag_vector_db.db"
        
        if PDF_PROCESSOR_AVAILABLE:
            self.pdf_processor = PDFProcessor()
        else:
            self.pdf_processor = None
            
        self._init_database()
        
        if not self.dependencies_available:
            logger.warning("RealVectorRAG initialized with limited functionality due to missing dependencies")
    
    def _check_dependencies(self) -> bool:
        """Check if all critical dependencies are available"""
        return SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE
    
    def _create_fallback_model(self):
        """Create fallback model when sentence_transformers is not available"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            logger.info("Using TF-IDF fallback model")
            
            class TfidfFallbackModel:
                def __init__(self):
                    self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                    self.fitted = False
                
                def encode(self, texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    if not self.fitted:
                        self.vectorizer.fit(texts)
                        self.fitted = True
                    
                    return self.vectorizer.transform(texts).toarray()
            
            return TfidfFallbackModel()
            
        except ImportError:
            logger.error("sklearn also not available - RAG functionality severely limited")
            return None
    
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
                document_type TEXT DEFAULT 'csv',
                source_file TEXT,
                page_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, doc_id)
            )
        ''')
        
        # Add PDF documents tracking table
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
                error_message TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
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
    
    def _add_document(self, user_id: str, company_name: str, doc_id: str, content: str, metadata: Dict, 
                     document_type: str = 'csv', source_file: str = None, page_number: int = None):
        """Add document with embedding to database"""
        embedding = self.model.encode([content])[0]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_vectors 
            (user_id, company_name, doc_id, content, metadata, embedding, document_type, source_file, page_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, company_name, doc_id, content, json.dumps(metadata, cls=NumpyEncoder), pickle.dumps(embedding), 
              document_type, source_file, page_number))
        
        conn.commit()
        conn.close()
    
    def _build_user_index(self, user_id: str, company_name: str):
        """Build FAISS index for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT doc_id, content, metadata, embedding, document_type, source_file, page_number
            FROM user_vectors WHERE user_id = ?
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return
        
        # Create FAISS index
        embeddings = []
        documents = []
        
        for doc_id, content, metadata, embedding_blob, document_type, source_file, page_number in results:
            embedding = pickle.loads(embedding_blob)
            embeddings.append(embedding)
            documents.append({
                'doc_id': doc_id,
                'content': content,
                'metadata': json.loads(metadata),
                'document_type': document_type or 'csv',
                'source_file': source_file,
                'page_number': page_number
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
        """Query user's personalized knowledge base using vector similarity with caching"""
        start_time = time.time()
        company_name = self.user_metadata.get(user_id, {}).get("company_name", "Unknown Company")
        
        try:
            # Check cache first
            try:
                from src.utils.performance_optimizer import rag_cache
                cached_result = rag_cache.get_cached_response(user_id, query)
                if cached_result:
                    response_time_ms = (time.time() - start_time) * 1000
                    comprehensive_logger.log_rag_query_success(
                        user_id=user_id,
                        company_name=company_name,
                        query=query,
                        response_time_ms=response_time_ms,
                        relevance_score=0.9  # Cached results are considered high relevance
                    )
                    logger.debug(f"RAG cache hit for user {user_id[:8]}... ({response_time_ms:.2f}ms)")
                    return cached_result
            except ImportError:
                logger.debug("RAG cache not available")
            except Exception as e:
                logger.warning(f"RAG cache error: {str(e)}")
            
            if user_id not in self.user_indices:
                # Log query failure - no RAG initialized
                comprehensive_logger.log_rag_query_failure(
                    user_id=user_id,
                    company_name=company_name,
                    query=query,
                    error_message="RAG system not initialized for user"
                )
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
            
            # Log successful query
            response_time_ms = (time.time() - start_time) * 1000
            relevance_score = np.mean([r['score'] for r in results]) if results else 0.0
            
            comprehensive_logger.log_rag_query_success(
                user_id=user_id,
                company_name=company_name,
                query=query,
                response_time_ms=response_time_ms,
                relevance_score=relevance_score
            )
            
            # Cache results for future queries
            try:
                from src.utils.performance_optimizer import rag_cache
                if results and relevance_score > 0.5:  # Only cache good results
                    rag_cache.cache_response(user_id, query, results, response_time_ms, relevance_score)
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to cache RAG results: {str(e)}")
            
            return results
            
        except Exception as e:
            # Log query failure
            comprehensive_logger.log_rag_query_failure(
                user_id=user_id,
                company_name=company_name,
                query=query,
                error_message=str(e)
            )
            logger.error(f"Error querying RAG for user {user_id}: {str(e)}")
            return []
    
    def generate_personalized_response(self, user_id: str, query: str) -> RAGResponse:
        """Generate personalized response using enhanced vector RAG with PDF support"""
        start_time = time.time()
        
        # Get company context
        company_name = self.user_metadata.get(user_id, {}).get("company_name", "Unknown Company")
        
        try:
            # Retrieve relevant documents using enhanced query
            relevant_docs = self.query_user_knowledge_enhanced(user_id, query, top_k=5)
            
            if not relevant_docs:
                response = RAGResponse(
                    response_text=f"I don't have specific data for {company_name} yet. Please upload your company data (CSV files for analytics or PDF documents for knowledge) to get personalized insights.",
                    confidence=0.3,
                    sources=[],
                    company_context=company_name
                )
                
                # Log successful response (even if no data)
                response_time_ms = (time.time() - start_time) * 1000
                comprehensive_logger.log_rag_query_success(
                    user_id=user_id,
                    company_name=company_name,
                    query=query,
                    response_time_ms=response_time_ms,
                    relevance_score=0.3
                )
                
                return response
            
            # Generate response based on retrieved documents
            response_text = self._generate_contextual_response_enhanced(query, relevant_docs, company_name)
            
            # Calculate confidence based on similarity scores
            avg_score = np.mean([doc['score'] for doc in relevant_docs])
            confidence = min(0.95, avg_score * 1.2)
            
            # Create DocumentSource objects for better attribution
            sources = []
            for doc in relevant_docs:
                source = DocumentSource(
                    type=doc.get('document_type', 'csv'),
                    filename=doc.get('source_file', 'Unknown'),
                    page_number=doc.get('page_number'),
                    relevance_score=doc['score'],
                    doc_id=doc['doc_id']
                )
                sources.append(source)
            
            # Log successful response generation
            response_time_ms = (time.time() - start_time) * 1000
            comprehensive_logger.log_rag_query_success(
                user_id=user_id,
                company_name=company_name,
                query=query,
                response_time_ms=response_time_ms,
                relevance_score=avg_score
            )
            
            return RAGResponse(
                response_text=response_text,
                confidence=confidence,
                sources=sources,
                company_context=company_name
            )
            
        except Exception as e:
            # Log response generation failure
            comprehensive_logger.log_rag_query_failure(
                user_id=user_id,
                company_name=company_name,
                query=query,
                error_message=f"Response generation failed: {str(e)}"
            )
            
            logger.error(f"Error generating RAG response for user {user_id}: {str(e)}")
            
            return RAGResponse(
                response_text=f"I'm sorry, I encountered an error while processing your query. Please try again later.",
                confidence=0.0,
                sources=[],
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
    
    def _generate_contextual_response_enhanced(self, query: str, docs: List[Dict], company_name: str) -> str:
        """Generate enhanced contextual response from retrieved documents including PDF content"""
        query_lower = query.lower()
        
        # Separate PDF and CSV documents
        pdf_docs = [doc for doc in docs if doc.get('document_type') == 'pdf']
        csv_docs = [doc for doc in docs if doc.get('document_type') == 'csv']
        
        # Extract information from CSV documents (existing logic)
        products = []
        categories = []
        revenues = []
        
        for doc in csv_docs:
            metadata = doc['metadata']
            if metadata.get('type') == 'product':
                products.append(metadata.get('product'))
                revenues.append(metadata.get('revenue', 0))
            elif metadata.get('type') == 'category':
                categories.append(metadata.get('category'))
        
        # Handle document-specific queries for PDFs
        if pdf_docs and any(word in query_lower for word in ['document', 'policy', 'procedure', 'manual', 'guide']):
            pdf_content = []
            sources_info = []
            
            for doc in pdf_docs[:3]:  # Limit to top 3 PDF results
                content_snippet = doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
                pdf_content.append(content_snippet)
                
                source_file = doc.get('source_file', 'Unknown document')
                page_num = doc.get('page_number')
                if page_num:
                    sources_info.append(f"📄 {source_file} (Page {page_num})")
                else:
                    sources_info.append(f"📄 {source_file}")
            
            sources_text = "\n".join(sources_info)
            content_text = "\n\n".join(pdf_content)
            
            return f"📚 **{company_name} Document Search**:\n\n{content_text}\n\n**Sources:**\n{sources_text}"
        
        # Generate response based on query intent (existing logic enhanced)
        if any(word in query_lower for word in ['product', 'catalog', 'items']):
            response_parts = []
            
            if products:
                product_list = '\n'.join([f"• {p}" for p in products[:5]])
                response_parts.append(f"📦 **Products from CSV Data**:\n{product_list}")
            
            if pdf_docs:
                pdf_sources = [f"• {doc.get('source_file', 'Document')}" for doc in pdf_docs[:3]]
                response_parts.append(f"📄 **Related Documents**:\n" + '\n'.join(pdf_sources))
            
            return f"**{company_name} Product Information**:\n\n" + "\n\n".join(response_parts)
        
        elif any(word in query_lower for word in ['revenue', 'sales', 'money']):
            if revenues:
                total_revenue = sum(revenues)
                top_products = [(products[i], revenues[i]) for i in range(len(products))]
                top_products.sort(key=lambda x: x[1], reverse=True)
                
                top_list = '\n'.join([f"• {p}: ₹{r:,.0f}" for p, r in top_products[:3]])
                response = f"💰 **{company_name} Revenue Analysis**:\n\nTotal Revenue: ₹{total_revenue:,.0f}\n\nTop Performers:\n{top_list}"
                
                # Add document sources if available
                if pdf_docs:
                    doc_sources = [f"• {doc.get('source_file', 'Document')}" for doc in pdf_docs[:2]]
                    response += f"\n\n📄 **Supporting Documents**:\n" + '\n'.join(doc_sources)
                
                return response
        
        elif any(word in query_lower for word in ['forecast', 'predict', 'future']):
            response_parts = []
            
            if categories:
                cat_list = ', '.join(categories[:3])
                response_parts.append(f"📈 I can generate forecasts for: {cat_list}")
            
            if pdf_docs:
                response_parts.append("📄 I also found relevant planning documents that might inform the forecast.")
            
            response = f"**{company_name} Forecast**:\n\n" + "\n\n".join(response_parts)
            response += "\n\nWhich specific area would you like me to focus on?"
            return response
        
        # General response combining all available data types
        data_summary = []
        if products:
            data_summary.append(f"{len(products)} products")
        if pdf_docs:
            unique_pdfs = len(set(doc.get('source_file') for doc in pdf_docs))
            data_summary.append(f"{unique_pdfs} documents")
        
        if data_summary:
            summary_text = " and ".join(data_summary)
            return f"🤖 **{company_name} AI Assistant**:\n\nI found information from {summary_text} that might help answer your question. Could you be more specific about what you'd like to know?"
        
        return f"🤖 **{company_name} AI Assistant**:\n\nI have access to your company data. What specific information are you looking for?"
    
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
    
    def initialize_company_rag(self, user_id: str, company_name: str):
        """Initialize RAG system for a company (alias for load_company_data with empty setup)"""
        try:
            # Create user metadata entry
            self.user_metadata[user_id] = {"company_name": company_name}
            
            # Initialize empty structures
            self.user_indices[user_id] = None
            self.user_documents[user_id] = []
            
            # Update session info
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO user_sessions 
                (user_id, company_name, total_documents, last_updated, index_version)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, company_name, 0, datetime.now(), "1.0"))
            conn.commit()
            conn.close()
            
            print(f"Initialized empty RAG system for {company_name}")
            return True
            
        except Exception as e:
            print(f"Error initializing RAG for {company_name}: {e}")
            return False
    
    def add_pdf_document(self, user_id: str, company_name: str, pdf_path: str) -> bool:
        """
        Process and add PDF document to user's knowledge base
        
        Args:
            user_id: User identifier
            company_name: Company name for context
            pdf_path: Path to PDF file
            
        Returns:
            Boolean indicating success
        """
        try:
            # Extract text from PDF
            extraction_result = self.pdf_processor.extract_text(pdf_path, user_id, company_name)
            
            # Store document metadata
            self._store_document_metadata(extraction_result)
            
            if not extraction_result.success:
                logger.error(f"PDF extraction failed for {pdf_path}: {extraction_result.error}")
                return False
            
            # Process extracted text into RAG documents
            rag_documents = self.pdf_processor.process_for_rag(extraction_result)
            
            # Add documents to vector database
            for doc in rag_documents:
                self._add_document(
                    user_id=user_id,
                    company_name=company_name,
                    doc_id=doc['doc_id'],
                    content=doc['content'],
                    metadata=doc['metadata'],
                    document_type='pdf',
                    source_file=extraction_result.metadata.filename,
                    page_number=doc['metadata'].get('page_number')
                )
            
            # Rebuild user index to include new PDF content
            self._build_user_index(user_id, company_name)
            
            logger.info(f"Successfully added PDF {extraction_result.metadata.filename} to {company_name} knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Error adding PDF document {pdf_path}: {str(e)}")
            return False
    
    def _store_document_metadata(self, extraction_result: PDFExtractionResult):
        """Store PDF document metadata in tracking table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            metadata = extraction_result.metadata
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_documents 
                (document_id, user_id, filename, file_type, file_path, processing_status, 
                 file_size, page_count, file_hash, metadata, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"pdf_{metadata.file_hash}",
                metadata.user_id,
                metadata.filename,
                'pdf',
                f"data/users/{metadata.user_id}/pdf/{metadata.filename}",
                metadata.extraction_status,
                metadata.file_size,
                metadata.page_count,
                metadata.file_hash,
                json.dumps({
                    'upload_date': metadata.upload_date.isoformat(),
                    'company_id': metadata.company_id
                }, cls=NumpyEncoder),
                metadata.error_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing document metadata: {str(e)}")
    
    def get_document_sources(self, user_id: str, doc_ids: List[str]) -> List[DocumentSource]:
        """
        Get source attribution for documents
        
        Args:
            user_id: User identifier
            doc_ids: List of document IDs
            
        Returns:
            List of DocumentSource objects with attribution info
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get document information
            placeholders = ','.join(['?' for _ in doc_ids])
            cursor.execute(f'''
                SELECT doc_id, document_type, source_file, page_number, metadata
                FROM user_vectors 
                WHERE user_id = ? AND doc_id IN ({placeholders})
            ''', [user_id] + doc_ids)
            
            results = cursor.fetchall()
            conn.close()
            
            sources = []
            for doc_id, doc_type, source_file, page_number, metadata_json in results:
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                source = DocumentSource(
                    type=doc_type or 'csv',
                    filename=source_file or metadata.get('filename', 'Unknown'),
                    page_number=page_number,
                    relevance_score=0.0,  # Will be set by query method
                    doc_id=doc_id
                )
                sources.append(source)
            
            return sources
            
        except Exception as e:
            logger.error(f"Error getting document sources: {str(e)}")
            return []
    
    def query_user_knowledge_enhanced(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """
        Enhanced query method that includes source attribution for PDF and CSV content
        
        Args:
            user_id: User identifier
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of results with enhanced source attribution
        """
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
                
                # Create enhanced result with source attribution
                result = {
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': float(score),
                    'doc_id': doc['doc_id'],
                    'document_type': doc.get('document_type', 'csv'),
                    'source_file': doc.get('source_file'),
                    'page_number': doc.get('page_number'),
                    'source_attribution': self._create_source_attribution(doc, score)
                }
                results.append(result)
        
        return results
    
    def _create_source_attribution(self, doc: Dict, score: float) -> str:
        """Create human-readable source attribution"""
        doc_type = doc.get('document_type', 'csv')
        source_file = doc.get('source_file', 'Unknown file')
        page_number = doc.get('page_number')
        
        if doc_type == 'pdf' and page_number:
            return f"Source: {source_file}, Page {page_number} (relevance: {score:.2f})"
        elif doc_type == 'pdf':
            return f"Source: {source_file} (PDF document, relevance: {score:.2f})"
        else:
            return f"Source: {source_file} (CSV data, relevance: {score:.2f})"

# Global instance
real_vector_rag = RealVectorRAG()