"""
Performance Optimization Module for Multi-Tenant RAG System
Provides PDF processing optimization, RAG query caching, and performance monitoring
"""

import os
import time
import threading
import hashlib
import pickle
import sqlite3
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

# Import comprehensive logging system
from src.utils.logging_config import comprehensive_logger

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry for RAG queries and PDF processing results"""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    operation_type: str
    duration_ms: float
    timestamp: datetime
    user_id: Optional[str] = None
    file_size_mb: Optional[float] = None
    cache_hit: bool = False
    success: bool = True
    error_message: Optional[str] = None

class PDFProcessingOptimizer:
    """
    Optimizes PDF processing performance for large files
    """
    
    def __init__(self, max_workers: int = 4, chunk_size_mb: int = 10):
        self.max_workers = max_workers
        self.chunk_size_mb = chunk_size_mb
        self.processing_cache = {}
        self.performance_metrics = []
        
    def optimize_pdf_processing(self, pdf_processor, file_path: str, user_id: str, company_id: str) -> Any:
        """
        Optimize PDF processing with chunking, caching, and parallel processing
        
        Args:
            pdf_processor: PDF processor instance
            file_path: Path to PDF file
            user_id: User ID
            company_id: Company ID
            
        Returns:
            Optimized PDF extraction result
        """
        start_time = time.time()
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(file_path)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                processing_time = (time.time() - start_time) * 1000
                self._record_performance_metric(
                    "pdf_processing_cached",
                    processing_time,
                    user_id,
                    file_size_mb,
                    cache_hit=True
                )
                logger.info(f"PDF processing cache hit for {os.path.basename(file_path)} ({processing_time:.2f}ms)")
                return cached_result
            
            # Determine processing strategy based on file size
            if file_size_mb > 50:  # Large file - use chunked processing
                result = self._process_large_pdf_chunked(pdf_processor, file_path, user_id, company_id)
            elif file_size_mb > 20:  # Medium file - use parallel page processing
                result = self._process_medium_pdf_parallel(pdf_processor, file_path, user_id, company_id)
            else:  # Small file - use standard processing with optimizations
                result = self._process_small_pdf_optimized(pdf_processor, file_path, user_id, company_id)
            
            # Cache successful results
            if result and result.success:
                self._cache_result(cache_key, result)
            
            processing_time = (time.time() - start_time) * 1000
            self._record_performance_metric(
                "pdf_processing_optimized",
                processing_time,
                user_id,
                file_size_mb,
                success=result.success if result else False,
                error_message=result.error if result and not result.success else None
            )
            
            logger.info(f"Optimized PDF processing completed for {os.path.basename(file_path)} in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._record_performance_metric(
                "pdf_processing_error",
                processing_time,
                user_id,
                file_size_mb,
                success=False,
                error_message=str(e)
            )
            logger.error(f"Error in optimized PDF processing: {str(e)}")
            raise
    
    def _process_large_pdf_chunked(self, pdf_processor, file_path: str, user_id: str, company_id: str):
        """Process large PDFs in chunks to manage memory usage"""
        logger.info(f"Processing large PDF in chunks: {os.path.basename(file_path)}")
        
        try:
            # Use streaming approach for large files
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                # Process in chunks
                chunk_size = 50  # Pages per chunk
                all_page_texts = []
                combined_text = ""
                
                for start_page in range(0, total_pages, chunk_size):
                    end_page = min(start_page + chunk_size, total_pages)
                    chunk_text = self._process_page_chunk(pdf_reader, start_page, end_page)
                    
                    all_page_texts.extend(chunk_text)
                    combined_text += "\n".join(chunk_text) + "\n"
                    
                    # Log progress for large files
                    progress = (end_page / total_pages) * 100
                    logger.info(f"PDF processing progress: {progress:.1f}% ({end_page}/{total_pages} pages)")
                
                # Create result similar to standard PDF processor
                from src.rag.pdf_processor import PDFExtractionResult, PDFMetadata
                
                metadata = PDFMetadata(
                    filename=os.path.basename(file_path),
                    upload_date=datetime.now(),
                    page_count=len(all_page_texts),
                    file_size=os.path.getsize(file_path),
                    company_id=company_id,
                    user_id=user_id,
                    file_hash=self._calculate_file_hash(file_path),
                    extraction_status="success"
                )
                
                return PDFExtractionResult(
                    text=combined_text.strip(),
                    page_count=len(all_page_texts),
                    metadata=metadata,
                    success=True,
                    page_texts=all_page_texts
                )
                
        except Exception as e:
            logger.error(f"Error in chunked PDF processing: {str(e)}")
            raise
    
    def _process_medium_pdf_parallel(self, pdf_processor, file_path: str, user_id: str, company_id: str):
        """Process medium PDFs with parallel page processing"""
        logger.info(f"Processing medium PDF with parallel processing: {os.path.basename(file_path)}")
        
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                # Process pages in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit page processing tasks
                    future_to_page = {
                        executor.submit(self._extract_page_text, pdf_reader.pages[i], i): i
                        for i in range(total_pages)
                    }
                    
                    page_results = {}
                    for future in as_completed(future_to_page):
                        page_num = future_to_page[future]
                        try:
                            page_text = future.result()
                            page_results[page_num] = page_text
                        except Exception as e:
                            logger.warning(f"Failed to process page {page_num}: {str(e)}")
                            page_results[page_num] = ""
                
                # Combine results in order
                all_page_texts = [page_results.get(i, "") for i in range(total_pages)]
                combined_text = "\n".join([f"--- Page {i+1} ---\n{text}" for i, text in enumerate(all_page_texts) if text.strip()])
                
                # Create result
                from src.rag.pdf_processor import PDFExtractionResult, PDFMetadata
                
                metadata = PDFMetadata(
                    filename=os.path.basename(file_path),
                    upload_date=datetime.now(),
                    page_count=len([t for t in all_page_texts if t.strip()]),
                    file_size=os.path.getsize(file_path),
                    company_id=company_id,
                    user_id=user_id,
                    file_hash=self._calculate_file_hash(file_path),
                    extraction_status="success"
                )
                
                return PDFExtractionResult(
                    text=combined_text.strip(),
                    page_count=len([t for t in all_page_texts if t.strip()]),
                    metadata=metadata,
                    success=True,
                    page_texts=[t for t in all_page_texts if t.strip()]
                )
                
        except Exception as e:
            logger.error(f"Error in parallel PDF processing: {str(e)}")
            raise
    
    def _process_small_pdf_optimized(self, pdf_processor, file_path: str, user_id: str, company_id: str):
        """Process small PDFs with standard method but optimized settings"""
        logger.debug(f"Processing small PDF with optimizations: {os.path.basename(file_path)}")
        
        # Use standard processor but with optimized timeout and settings
        original_timeout = pdf_processor.processing_timeout
        pdf_processor.processing_timeout = 60  # Shorter timeout for small files
        
        try:
            result = pdf_processor.extract_text(file_path, user_id, company_id)
            return result
        finally:
            pdf_processor.processing_timeout = original_timeout
    
    def _process_page_chunk(self, pdf_reader, start_page: int, end_page: int) -> List[str]:
        """Process a chunk of pages"""
        page_texts = []
        
        for page_num in range(start_page, end_page):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                cleaned_text = self._clean_text(page_text)
                if cleaned_text.strip():
                    page_texts.append(cleaned_text)
            except Exception as e:
                logger.warning(f"Failed to process page {page_num}: {str(e)}")
                continue
        
        return page_texts
    
    def _extract_page_text(self, page, page_num: int) -> str:
        """Extract text from a single page (for parallel processing)"""
        try:
            page_text = page.extract_text()
            return self._clean_text(page_text)
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace('\x00', '')  # Remove null characters
        text = text.replace('\ufffd', '')  # Remove replacement characters
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
    
    def _generate_cache_key(self, file_path: str) -> str:
        """Generate cache key for PDF file"""
        file_hash = self._calculate_file_hash(file_path)
        file_size = os.path.getsize(file_path)
        return f"pdf_{file_hash}_{file_size}"
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _get_cached_result(self, cache_key: str):
        """Get cached PDF processing result"""
        if cache_key in self.processing_cache:
            entry = self.processing_cache[cache_key]
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            return entry.value
        return None
    
    def _cache_result(self, cache_key: str, result):
        """Cache PDF processing result"""
        try:
            # Estimate size
            size_bytes = len(pickle.dumps(result))
            
            entry = CacheEntry(
                key=cache_key,
                value=result,
                timestamp=datetime.now(),
                size_bytes=size_bytes,
                ttl_seconds=3600  # 1 hour TTL for PDF results
            )
            
            self.processing_cache[cache_key] = entry
            
            # Clean old entries if cache gets too large
            if len(self.processing_cache) > 100:
                self._cleanup_cache()
                
        except Exception as e:
            logger.warning(f"Failed to cache PDF result: {str(e)}")
    
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        now = datetime.now()
        keys_to_remove = []
        
        for key, entry in self.processing_cache.items():
            # Remove expired entries
            if entry.ttl_seconds and (now - entry.timestamp).total_seconds() > entry.ttl_seconds:
                keys_to_remove.append(key)
            # Remove least recently used entries if cache is full
            elif (now - entry.last_accessed).total_seconds() > 7200:  # 2 hours
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.processing_cache[key]
        
        logger.info(f"Cleaned {len(keys_to_remove)} entries from PDF processing cache")
    
    def _record_performance_metric(self, operation_type: str, duration_ms: float, 
                                 user_id: Optional[str] = None, file_size_mb: Optional[float] = None,
                                 cache_hit: bool = False, success: bool = True, 
                                 error_message: Optional[str] = None):
        """Record performance metric"""
        metric = PerformanceMetrics(
            operation_type=operation_type,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            user_id=user_id,
            file_size_mb=file_size_mb,
            cache_hit=cache_hit,
            success=success,
            error_message=error_message
        )
        
        self.performance_metrics.append(metric)
        
        # Keep only recent metrics
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-500:]

class RAGQueryCache:
    """
    Intelligent caching system for RAG queries
    """
    
    def __init__(self, cache_db_path: str = "rag_query_cache.db", max_cache_size_mb: int = 500):
        self.cache_db_path = cache_db_path
        self.max_cache_size_mb = max_cache_size_mb
        self.memory_cache = {}  # In-memory cache for frequently accessed queries
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_queries": 0
        }
        
        self._init_cache_database()
    
    def _init_cache_database(self):
        """Initialize cache database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    response_data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    response_time_ms REAL,
                    relevance_score REAL,
                    size_bytes INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_query_cache_user_id ON query_cache(user_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_query_cache_last_accessed ON query_cache(last_accessed)
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing query cache database: {str(e)}")
    
    def get_cached_response(self, user_id: str, query: str) -> Optional[Any]:
        """
        Get cached response for a query
        
        Args:
            user_id: User ID
            query: Query string
            
        Returns:
            Cached response or None if not found
        """
        start_time = time.time()
        
        try:
            cache_key = self._generate_query_cache_key(user_id, query)
            
            # Check memory cache first
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                self.cache_stats["hits"] += 1
                self.cache_stats["total_queries"] += 1
                
                logger.debug(f"Memory cache hit for query (user: {user_id[:8]}...)")
                return entry.value
            
            # Check database cache
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT response_data, access_count, relevance_score
                FROM query_cache 
                WHERE cache_key = ? AND user_id = ?
                AND datetime(last_accessed) > datetime('now', '-24 hours')
            ''', (cache_key, user_id))
            
            result = cursor.fetchone()
            
            if result:
                response_data, access_count, relevance_score = result
                
                # Update access statistics
                cursor.execute('''
                    UPDATE query_cache 
                    SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE cache_key = ?
                ''', (cache_key,))
                
                conn.commit()
                conn.close()
                
                # Deserialize response
                cached_response = pickle.loads(response_data)
                
                # Add to memory cache if frequently accessed
                if access_count > 3:
                    self._add_to_memory_cache(cache_key, cached_response)
                
                self.cache_stats["hits"] += 1
                self.cache_stats["total_queries"] += 1
                
                cache_time = (time.time() - start_time) * 1000
                logger.debug(f"Database cache hit for query (user: {user_id[:8]}..., {cache_time:.2f}ms)")
                
                return cached_response
            
            conn.close()
            
            self.cache_stats["misses"] += 1
            self.cache_stats["total_queries"] += 1
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached response: {str(e)}")
            return None
    
    def cache_response(self, user_id: str, query: str, response: Any, 
                      response_time_ms: float, relevance_score: float = 0.0):
        """
        Cache a query response
        
        Args:
            user_id: User ID
            query: Query string
            response: Response to cache
            response_time_ms: Response time in milliseconds
            relevance_score: Relevance score of the response
        """
        try:
            cache_key = self._generate_query_cache_key(user_id, query)
            
            # Serialize response
            response_data = pickle.dumps(response)
            size_bytes = len(response_data)
            
            # Don't cache very large responses
            if size_bytes > 10 * 1024 * 1024:  # 10MB limit
                logger.warning(f"Response too large to cache: {size_bytes} bytes")
                return
            
            # Store in database
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO query_cache 
                (cache_key, user_id, query_hash, response_data, response_time_ms, 
                 relevance_score, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                cache_key, user_id, self._hash_query(query), response_data,
                response_time_ms, relevance_score, size_bytes
            ))
            
            conn.commit()
            conn.close()
            
            # Add to memory cache for high-quality responses
            if relevance_score > 0.7 or response_time_ms > 1000:
                self._add_to_memory_cache(cache_key, response)
            
            # Cleanup old entries periodically
            if self.cache_stats["total_queries"] % 100 == 0:
                self._cleanup_cache()
            
            logger.debug(f"Cached response for query (user: {user_id[:8]}..., size: {size_bytes} bytes)")
            
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}")
    
    def _generate_query_cache_key(self, user_id: str, query: str) -> str:
        """Generate cache key for query"""
        query_normalized = query.lower().strip()
        query_hash = hashlib.md5(query_normalized.encode()).hexdigest()
        return f"{user_id}_{query_hash}"
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _add_to_memory_cache(self, cache_key: str, response: Any):
        """Add response to memory cache"""
        try:
            entry = CacheEntry(
                key=cache_key,
                value=response,
                timestamp=datetime.now(),
                size_bytes=len(pickle.dumps(response)),
                ttl_seconds=1800  # 30 minutes TTL for memory cache
            )
            
            self.memory_cache[cache_key] = entry
            
            # Limit memory cache size
            if len(self.memory_cache) > 50:
                self._cleanup_memory_cache()
                
        except Exception as e:
            logger.warning(f"Failed to add to memory cache: {str(e)}")
    
    def _cleanup_memory_cache(self):
        """Clean up memory cache"""
        now = datetime.now()
        keys_to_remove = []
        
        # Remove expired entries
        for key, entry in self.memory_cache.items():
            if entry.ttl_seconds and (now - entry.timestamp).total_seconds() > entry.ttl_seconds:
                keys_to_remove.append(key)
        
        # Remove least recently used entries if still too many
        if len(self.memory_cache) - len(keys_to_remove) > 30:
            sorted_entries = sorted(
                [(k, v) for k, v in self.memory_cache.items() if k not in keys_to_remove],
                key=lambda x: x[1].last_accessed
            )
            keys_to_remove.extend([k for k, v in sorted_entries[:20]])
        
        for key in keys_to_remove:
            if key in self.memory_cache:
                del self.memory_cache[key]
        
        self.cache_stats["evictions"] += len(keys_to_remove)
        logger.debug(f"Cleaned {len(keys_to_remove)} entries from memory cache")
    
    def _cleanup_cache(self):
        """Clean up database cache"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            # Remove entries older than 7 days
            cursor.execute('''
                DELETE FROM query_cache 
                WHERE datetime(last_accessed) < datetime('now', '-7 days')
            ''')
            
            # Remove least accessed entries if cache is too large
            cursor.execute('''
                SELECT SUM(size_bytes) FROM query_cache
            ''')
            
            total_size = cursor.fetchone()[0] or 0
            max_size_bytes = self.max_cache_size_mb * 1024 * 1024
            
            if total_size > max_size_bytes:
                # Remove least accessed entries
                cursor.execute('''
                    DELETE FROM query_cache 
                    WHERE cache_key IN (
                        SELECT cache_key FROM query_cache 
                        ORDER BY access_count ASC, last_accessed ASC 
                        LIMIT (SELECT COUNT(*) / 4 FROM query_cache)
                    )
                ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Cleaned up query cache database")
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(size_bytes) as total_size_bytes,
                    AVG(access_count) as avg_access_count,
                    AVG(response_time_ms) as avg_response_time,
                    AVG(relevance_score) as avg_relevance_score
                FROM query_cache
            ''')
            
            db_stats = cursor.fetchone()
            conn.close()
            
            hit_rate = (self.cache_stats["hits"] / self.cache_stats["total_queries"] * 100) if self.cache_stats["total_queries"] > 0 else 0
            
            return {
                "hit_rate_percent": hit_rate,
                "total_queries": self.cache_stats["total_queries"],
                "cache_hits": self.cache_stats["hits"],
                "cache_misses": self.cache_stats["misses"],
                "evictions": self.cache_stats["evictions"],
                "memory_cache_size": len(self.memory_cache),
                "database_entries": db_stats[0] if db_stats else 0,
                "total_size_mb": (db_stats[1] / (1024 * 1024)) if db_stats and db_stats[1] else 0,
                "avg_access_count": db_stats[2] if db_stats else 0,
                "avg_response_time_ms": db_stats[3] if db_stats else 0,
                "avg_relevance_score": db_stats[4] if db_stats else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {str(e)}")
            return self.cache_stats

# Global instances
pdf_optimizer = PDFProcessingOptimizer()
rag_cache = RAGQueryCache()