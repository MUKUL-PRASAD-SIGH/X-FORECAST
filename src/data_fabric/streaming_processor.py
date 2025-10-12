"""
Real-time Data Streaming Processor for Cyberpunk AI Dashboard
Handles Kafka-based streaming, WebSocket connections, and data synchronization
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import websockets
import redis
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamMessage:
    """Standard message format for streaming data"""
    message_id: str
    timestamp: datetime
    source: str
    message_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class StreamingConfig:
    """Configuration for streaming data processor"""
    kafka_bootstrap_servers: List[str]
    kafka_topics: List[str]
    redis_host: str
    redis_port: int
    redis_db: int
    websocket_port: int
    batch_size: int = 100
    flush_interval: int = 5  # seconds
    max_retries: int = 3
    retry_delay: int = 1  # seconds

class StreamProcessor(ABC):
    """Abstract base class for stream processors"""
    
    @abstractmethod
    async def process_message(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Process a single stream message"""
        pass
    
    @abstractmethod
    def get_processor_name(self) -> str:
        """Get the name of this processor"""
        pass

class SalesStreamProcessor(StreamProcessor):
    """Process sales transaction streams"""
    
    def __init__(self):
        self.daily_totals = {}
        self.hourly_patterns = {}
    
    async def process_message(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Process sales transaction message"""
        try:
            if message.message_type != 'sales_transaction':
                return None
            
            data = message.data
            transaction_amount = data.get('amount', 0)
            customer_id = data.get('customer_id')
            product_id = data.get('product_id')
            
            # Update daily totals
            date_key = message.timestamp.strftime('%Y-%m-%d')
            if date_key not in self.daily_totals:
                self.daily_totals[date_key] = {'total': 0, 'count': 0}
            
            self.daily_totals[date_key]['total'] += transaction_amount
            self.daily_totals[date_key]['count'] += 1
            
            # Update hourly patterns
            hour_key = message.timestamp.strftime('%H')
            if hour_key not in self.hourly_patterns:
                self.hourly_patterns[hour_key] = {'total': 0, 'count': 0}
            
            self.hourly_patterns[hour_key]['total'] += transaction_amount
            self.hourly_patterns[hour_key]['count'] += 1
            
            # Create processed message
            processed_data = {
                'original_transaction': data,
                'daily_total': self.daily_totals[date_key]['total'],
                'daily_count': self.daily_totals[date_key]['count'],
                'hourly_average': self.hourly_patterns[hour_key]['total'] / self.hourly_patterns[hour_key]['count'],
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return StreamMessage(
                message_id=f"processed_{message.message_id}",
                timestamp=datetime.now(),
                source='sales_processor',
                message_type='processed_sales',
                data=processed_data,
                metadata={'processor': 'SalesStreamProcessor'}
            )
            
        except Exception as e:
            logger.error(f"Error processing sales message: {e}")
            return None
    
    def get_processor_name(self) -> str:
        return "SalesStreamProcessor"

class CustomerEventProcessor(StreamProcessor):
    """Process customer behavior event streams"""
    
    def __init__(self):
        self.customer_sessions = {}
        self.event_counts = {}
    
    async def process_message(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Process customer event message"""
        try:
            if message.message_type != 'customer_event':
                return None
            
            data = message.data
            customer_id = data.get('customer_id')
            event_type = data.get('event_type')
            session_id = data.get('session_id')
            
            # Track customer sessions
            if customer_id not in self.customer_sessions:
                self.customer_sessions[customer_id] = {}
            
            if session_id not in self.customer_sessions[customer_id]:
                self.customer_sessions[customer_id][session_id] = {
                    'start_time': message.timestamp,
                    'events': [],
                    'last_activity': message.timestamp
                }
            
            # Update session
            session = self.customer_sessions[customer_id][session_id]
            session['events'].append({
                'event_type': event_type,
                'timestamp': message.timestamp.isoformat(),
                'data': data
            })
            session['last_activity'] = message.timestamp
            
            # Update event counts
            if event_type not in self.event_counts:
                self.event_counts[event_type] = 0
            self.event_counts[event_type] += 1
            
            # Calculate session metrics
            session_duration = (message.timestamp - session['start_time']).total_seconds()
            event_frequency = len(session['events']) / max(session_duration / 60, 1)  # events per minute
            
            processed_data = {
                'customer_id': customer_id,
                'event_type': event_type,
                'session_duration_minutes': session_duration / 60,
                'session_event_count': len(session['events']),
                'event_frequency_per_minute': event_frequency,
                'total_event_type_count': self.event_counts[event_type],
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return StreamMessage(
                message_id=f"processed_{message.message_id}",
                timestamp=datetime.now(),
                source='customer_processor',
                message_type='processed_customer_event',
                data=processed_data,
                metadata={'processor': 'CustomerEventProcessor'}
            )
            
        except Exception as e:
            logger.error(f"Error processing customer event: {e}")
            return None
    
    def get_processor_name(self) -> str:
        return "CustomerEventProcessor"

class InventoryStreamProcessor(StreamProcessor):
    """Process inventory level streams"""
    
    def __init__(self):
        self.inventory_levels = {}
        self.reorder_alerts = {}
    
    async def process_message(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Process inventory update message"""
        try:
            if message.message_type != 'inventory_update':
                return None
            
            data = message.data
            product_id = data.get('product_id')
            current_level = data.get('current_level', 0)
            reorder_point = data.get('reorder_point', 0)
            max_level = data.get('max_level', 1000)
            
            # Update inventory tracking
            self.inventory_levels[product_id] = {
                'current_level': current_level,
                'reorder_point': reorder_point,
                'max_level': max_level,
                'last_updated': message.timestamp,
                'stock_ratio': current_level / max_level if max_level > 0 else 0
            }
            
            # Check for reorder alerts
            needs_reorder = current_level <= reorder_point
            is_overstocked = current_level > max_level * 0.9
            is_critical = current_level <= reorder_point * 0.5
            
            if needs_reorder and product_id not in self.reorder_alerts:
                self.reorder_alerts[product_id] = {
                    'alert_time': message.timestamp,
                    'level_when_triggered': current_level,
                    'is_critical': is_critical
                }
            elif not needs_reorder and product_id in self.reorder_alerts:
                del self.reorder_alerts[product_id]
            
            processed_data = {
                'product_id': product_id,
                'current_level': current_level,
                'stock_ratio': current_level / max_level if max_level > 0 else 0,
                'needs_reorder': needs_reorder,
                'is_overstocked': is_overstocked,
                'is_critical': is_critical,
                'days_of_stock_remaining': self._estimate_days_remaining(product_id, current_level),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return StreamMessage(
                message_id=f"processed_{message.message_id}",
                timestamp=datetime.now(),
                source='inventory_processor',
                message_type='processed_inventory',
                data=processed_data,
                metadata={'processor': 'InventoryStreamProcessor'}
            )
            
        except Exception as e:
            logger.error(f"Error processing inventory message: {e}")
            return None
    
    def _estimate_days_remaining(self, product_id: str, current_level: int) -> float:
        """Estimate days of stock remaining based on historical usage"""
        # Simplified estimation - in real implementation, use historical sales data
        daily_usage = max(1, current_level * 0.05)  # Assume 5% daily usage
        return current_level / daily_usage
    
    def get_processor_name(self) -> str:
        return "InventoryStreamProcessor"

class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.message_queue = queue.Queue()
        self.running = False
    
    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            # Send welcome message
            welcome_message = {
                'type': 'connection',
                'message': 'Connected to Cyberpunk AI Dashboard Stream',
                'timestamp': datetime.now().isoformat(),
                'client_count': len(self.clients)
            }
            await websocket.send(json.dumps(welcome_message))
            
            # Keep connection alive
            async for message in websocket:
                # Handle incoming messages from client
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from client: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def handle_client_message(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle messages from WebSocket clients"""
        message_type = data.get('type')
        
        if message_type == 'ping':
            await websocket.send(json.dumps({
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }))
        elif message_type == 'subscribe':
            # Handle subscription to specific data streams
            stream = data.get('stream')
            await websocket.send(json.dumps({
                'type': 'subscription_confirmed',
                'stream': stream,
                'timestamp': datetime.now().isoformat()
            }))
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return
        
        message_json = json.dumps(message)
        disconnected_clients = set()
        
        for client in self.clients:
            try:
                await client.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.running = True
        logger.info(f"Starting WebSocket server on port {self.port}")
        
        async with websockets.serve(self.register_client, "localhost", self.port):
            while self.running:
                await asyncio.sleep(1)
    
    def stop_server(self):
        """Stop the WebSocket server"""
        self.running = False

class StreamingDataProcessor:
    """Main streaming data processor coordinating all components"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.processors: Dict[str, StreamProcessor] = {}
        self.kafka_producer = None
        self.kafka_consumer = None
        self.redis_client = None
        self.websocket_manager = WebSocketManager(config.websocket_port)
        self.running = False
        self.message_buffer = []
        self.last_flush = datetime.now()
        
        # Initialize processors
        self.processors['sales'] = SalesStreamProcessor()
        self.processors['customer_events'] = CustomerEventProcessor()
        self.processors['inventory'] = InventoryStreamProcessor()
    
    async def initialize(self):
        """Initialize all streaming components"""
        try:
            # Initialize Kafka producer
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                retries=self.config.max_retries,
                retry_backoff_ms=self.config.retry_delay * 1000
            )
            
            # Initialize Kafka consumer
            self.kafka_consumer = KafkaConsumer(
                *self.config.kafka_topics,
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                group_id='cyberpunk_dashboard_group',
                auto_offset_reset='latest'
            )
            
            # Initialize Redis client
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True
            )
            
            logger.info("Streaming data processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize streaming processor: {e}")
            raise
    
    async def start_processing(self):
        """Start processing streaming data"""
        self.running = True
        logger.info("Starting streaming data processing")
        
        # Start WebSocket server
        websocket_task = asyncio.create_task(self.websocket_manager.start_server())
        
        # Start Kafka consumer
        consumer_task = asyncio.create_task(self.consume_kafka_messages())
        
        # Start periodic flush
        flush_task = asyncio.create_task(self.periodic_flush())
        
        # Start demo data generation
        demo_task = asyncio.create_task(self.generate_demo_data())
        
        try:
            await asyncio.gather(websocket_task, consumer_task, flush_task, demo_task)
        except Exception as e:
            logger.error(f"Error in streaming processing: {e}")
        finally:
            self.running = False
    
    async def consume_kafka_messages(self):
        """Consume messages from Kafka topics"""
        while self.running:
            try:
                # Poll for messages
                message_pack = self.kafka_consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        await self.process_kafka_message(message)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error consuming Kafka messages: {e}")
                await asyncio.sleep(1)
    
    async def process_kafka_message(self, kafka_message):
        """Process a single Kafka message"""
        try:
            # Convert Kafka message to StreamMessage
            message_data = kafka_message.value
            stream_message = StreamMessage(
                message_id=message_data.get('message_id', f"kafka_{int(time.time())}"),
                timestamp=datetime.fromisoformat(message_data.get('timestamp', datetime.now().isoformat())),
                source=message_data.get('source', 'kafka'),
                message_type=message_data.get('message_type', 'unknown'),
                data=message_data.get('data', {}),
                metadata=message_data.get('metadata', {})
            )
            
            # Process with appropriate processor
            processor_name = self.get_processor_for_message_type(stream_message.message_type)
            if processor_name and processor_name in self.processors:
                processed_message = await self.processors[processor_name].process_message(stream_message)
                
                if processed_message:
                    # Add to buffer for batching
                    self.message_buffer.append(processed_message)
                    
                    # Cache in Redis
                    await self.cache_message(processed_message)
                    
                    # Broadcast to WebSocket clients
                    await self.websocket_manager.broadcast_message({
                        'type': 'stream_update',
                        'data': asdict(processed_message),
                        'timestamp': datetime.now().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"Error processing Kafka message: {e}")
    
    def get_processor_for_message_type(self, message_type: str) -> Optional[str]:
        """Get the appropriate processor for a message type"""
        type_mapping = {
            'sales_transaction': 'sales',
            'customer_event': 'customer_events',
            'inventory_update': 'inventory'
        }
        return type_mapping.get(message_type)
    
    async def cache_message(self, message: StreamMessage):
        """Cache processed message in Redis"""
        try:
            cache_key = f"stream:{message.source}:{message.message_type}:{message.message_id}"
            cache_data = asdict(message)
            
            # Store with expiration (1 hour)
            self.redis_client.setex(cache_key, 3600, json.dumps(cache_data, default=str))
            
            # Update stream statistics
            stats_key = f"stats:{message.source}:{message.message_type}"
            self.redis_client.hincrby(stats_key, 'count', 1)
            self.redis_client.hset(stats_key, 'last_update', datetime.now().isoformat())
            
        except Exception as e:
            logger.error(f"Error caching message: {e}")
    
    async def periodic_flush(self):
        """Periodically flush buffered messages"""
        while self.running:
            try:
                current_time = datetime.now()
                time_since_flush = (current_time - self.last_flush).total_seconds()
                
                if len(self.message_buffer) >= self.config.batch_size or time_since_flush >= self.config.flush_interval:
                    await self.flush_messages()
                    self.last_flush = current_time
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
                await asyncio.sleep(5)
    
    async def flush_messages(self):
        """Flush buffered messages to persistent storage"""
        if not self.message_buffer:
            return
        
        try:
            # Convert messages to DataFrame for batch processing
            message_data = []
            for message in self.message_buffer:
                message_dict = asdict(message)
                message_dict['timestamp'] = message.timestamp.isoformat()
                message_data.append(message_dict)
            
            df = pd.DataFrame(message_data)
            
            # Store aggregated statistics
            await self.update_stream_statistics(df)
            
            # Clear buffer
            self.message_buffer.clear()
            
            logger.info(f"Flushed {len(message_data)} messages")
            
        except Exception as e:
            logger.error(f"Error flushing messages: {e}")
    
    async def update_stream_statistics(self, df: pd.DataFrame):
        """Update streaming statistics"""
        try:
            # Calculate statistics by source and message type
            stats = df.groupby(['source', 'message_type']).agg({
                'message_id': 'count',
                'timestamp': ['min', 'max']
            }).reset_index()
            
            # Store in Redis
            for _, row in stats.iterrows():
                stats_key = f"batch_stats:{row['source']}:{row['message_type']}"
                self.redis_client.hset(stats_key, mapping={
                    'message_count': int(row[('message_id', 'count')]),
                    'earliest_timestamp': str(row[('timestamp', 'min')]),
                    'latest_timestamp': str(row[('timestamp', 'max')]),
                    'batch_time': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error updating stream statistics: {e}")
    
    async def generate_demo_data(self):
        """Generate demo streaming data for testing"""
        while self.running:
            try:
                # Generate sales transaction
                sales_message = {
                    'message_id': f"demo_sales_{int(time.time())}",
                    'timestamp': datetime.now().isoformat(),
                    'source': 'demo_pos',
                    'message_type': 'sales_transaction',
                    'data': {
                        'customer_id': f'customer_{np.random.randint(1, 1000)}',
                        'product_id': f'SKU_{np.random.randint(1, 100):04d}',
                        'amount': round(np.random.uniform(10, 500), 2),
                        'quantity': np.random.randint(1, 10),
                        'channel': np.random.choice(['online', 'retail', 'mobile'])
                    },
                    'metadata': {'demo': True}
                }
                
                # Generate customer event
                customer_message = {
                    'message_id': f"demo_customer_{int(time.time())}",
                    'timestamp': datetime.now().isoformat(),
                    'source': 'demo_web',
                    'message_type': 'customer_event',
                    'data': {
                        'customer_id': f'customer_{np.random.randint(1, 1000)}',
                        'session_id': f'session_{np.random.randint(1, 100)}',
                        'event_type': np.random.choice(['page_view', 'click', 'purchase', 'signup', 'logout']),
                        'page': np.random.choice(['home', 'product', 'cart', 'checkout', 'profile'])
                    },
                    'metadata': {'demo': True}
                }
                
                # Generate inventory update
                inventory_message = {
                    'message_id': f"demo_inventory_{int(time.time())}",
                    'timestamp': datetime.now().isoformat(),
                    'source': 'demo_warehouse',
                    'message_type': 'inventory_update',
                    'data': {
                        'product_id': f'SKU_{np.random.randint(1, 100):04d}',
                        'current_level': np.random.randint(0, 1000),
                        'reorder_point': 50,
                        'max_level': 1000,
                        'location': np.random.choice(['warehouse_a', 'warehouse_b', 'store_1'])
                    },
                    'metadata': {'demo': True}
                }
                
                # Send messages to Kafka (simulate)
                demo_messages = [sales_message, customer_message, inventory_message]
                for msg in demo_messages:
                    # Process directly since we're in demo mode
                    stream_msg = StreamMessage(
                        message_id=msg['message_id'],
                        timestamp=datetime.fromisoformat(msg['timestamp']),
                        source=msg['source'],
                        message_type=msg['message_type'],
                        data=msg['data'],
                        metadata=msg['metadata']
                    )
                    
                    # Process with appropriate processor
                    processor_name = self.get_processor_for_message_type(stream_msg.message_type)
                    if processor_name and processor_name in self.processors:
                        processed_message = await self.processors[processor_name].process_message(stream_msg)
                        
                        if processed_message:
                            self.message_buffer.append(processed_message)
                            await self.cache_message(processed_message)
                            
                            # Broadcast to WebSocket clients
                            await self.websocket_manager.broadcast_message({
                                'type': 'demo_stream_update',
                                'data': asdict(processed_message),
                                'timestamp': datetime.now().isoformat()
                            })
                
                # Wait before generating next batch
                await asyncio.sleep(np.random.uniform(2, 8))  # Random interval between 2-8 seconds
                
            except Exception as e:
                logger.error(f"Error generating demo data: {e}")
                await asyncio.sleep(5)
    
    def stop_processing(self):
        """Stop streaming data processing"""
        self.running = False
        self.websocket_manager.stop_server()
        
        if self.kafka_consumer:
            self.kafka_consumer.close()
        
        if self.kafka_producer:
            self.kafka_producer.close()
        
        logger.info("Streaming data processor stopped")
    
    def get_stream_statistics(self) -> Dict[str, Any]:
        """Get current streaming statistics"""
        try:
            stats = {}
            
            # Get statistics from Redis
            for processor_name in self.processors.keys():
                processor_stats = {}
                
                # Get batch statistics
                batch_keys = self.redis_client.keys(f"batch_stats:*{processor_name}*")
                for key in batch_keys:
                    key_parts = key.split(':')
                    if len(key_parts) >= 3:
                        message_type = key_parts[2]
                        batch_data = self.redis_client.hgetall(key)
                        processor_stats[message_type] = batch_data
                
                stats[processor_name] = processor_stats
            
            # Add general statistics
            stats['general'] = {
                'connected_clients': len(self.websocket_manager.clients),
                'buffer_size': len(self.message_buffer),
                'last_flush': self.last_flush.isoformat(),
                'processors_active': len(self.processors)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stream statistics: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    async def test_streaming_processor():
        config = StreamingConfig(
            kafka_bootstrap_servers=['localhost:9092'],
            kafka_topics=['sales', 'customer_events', 'inventory'],
            redis_host='localhost',
            redis_port=6379,
            redis_db=0,
            websocket_port=8765
        )
        
        processor = StreamingDataProcessor(config)
        
        try:
            await processor.initialize()
            await processor.start_processing()
        except KeyboardInterrupt:
            logger.info("Stopping streaming processor...")
            processor.stop_processing()
    
    # Run test
    asyncio.run(test_streaming_processor())