import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import pickle
import os

class PerformanceOptimizer:
    def __init__(self):
        self.cache_dir = './data/cache'
        self.model_cache = {}
        self.feature_cache = {}
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @lru_cache(maxsize=128)
    def cached_feature_engineering(self, data_hash: str, data_path: str):
        """Cache feature engineering results"""
        cache_file = f"{self.cache_dir}/features_{data_hash}.pkl"
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Generate features if not cached
        from .feature_store.feature_engineer import FeatureEngineer
        engineer = FeatureEngineer()
        
        data = pd.read_csv(data_path)
        features = engineer.create_features(data)
        
        # Cache results
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f)
        
        return features
    
    def parallel_model_training(self, data: pd.DataFrame, models: List[str]) -> Dict:
        """Train multiple models in parallel"""
        
        def train_single_model(model_config):
            model_name, model_class, data_subset = model_config
            
            try:
                model = model_class()
                start_time = time.time()
                model.fit(data_subset['demand'])
                training_time = time.time() - start_time
                
                # Generate forecast for evaluation
                forecast = model.forecast(10)
                
                return {
                    'model': model_name,
                    'training_time': training_time,
                    'forecast_sample': forecast.tolist()[:5],
                    'status': 'success'
                }
            except Exception as e:
                return {
                    'model': model_name,
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Prepare model configurations
        model_configs = []
        for model_name in models:
            if model_name == 'ARIMA':
                from .classical.arima import ARIMAForecaster
                model_configs.append((model_name, ARIMAForecaster, data))
            elif model_name == 'ETS':
                from .classical.ets import ETSForecaster
                model_configs.append((model_name, ETSForecaster, data))
            elif model_name == 'Croston':
                from .intermittent.croston import CrostonForecaster
                model_configs.append((model_name, CrostonForecaster, data))
        
        # Train models in parallel
        with ThreadPoolExecutor(max_workers=min(len(model_configs), mp.cpu_count())) as executor:
            results = list(executor.map(train_single_model, model_configs))
        
        return {result['model']: result for result in results}
    
    def batch_forecast_processing(self, products: List[str], horizon: int = 30) -> Dict:
        """Process forecasts for multiple products in batches"""
        
        def process_product_batch(product_batch):
            batch_results = {}
            
            for product_id in product_batch:
                try:
                    # Load product data
                    data = pd.read_csv('./data/raw/realistic_demand_data.csv')
                    product_data = data[data['product_id'] == product_id]
                    
                    if len(product_data) < 30:
                        continue
                    
                    # Quick ensemble forecast
                    from .ensemble import EnsembleForecaster
                    forecaster = EnsembleForecaster()
                    forecaster.fit(product_data['demand'])
                    forecast = forecaster.forecast(horizon)
                    
                    batch_results[product_id] = {
                        'forecast': forecast.tolist(),
                        'confidence': 'high' if len(product_data) > 100 else 'medium',
                        'data_points': len(product_data)
                    }
                    
                except Exception as e:
                    batch_results[product_id] = {'error': str(e)}
            
            return batch_results
        
        # Split products into batches
        batch_size = max(1, len(products) // mp.cpu_count())
        product_batches = [products[i:i + batch_size] for i in range(0, len(products), batch_size)]
        
        # Process batches in parallel
        all_results = {}
        with ProcessPoolExecutor(max_workers=min(len(product_batches), mp.cpu_count())) as executor:
            batch_results = executor.map(process_product_batch, product_batches)
            
            for batch_result in batch_results:
                all_results.update(batch_result)
        
        return all_results
    
    def optimize_memory_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        
        # Optimize numeric columns
        for col in data.select_dtypes(include=['int64']).columns:
            if data[col].min() >= 0:
                if data[col].max() < 255:
                    data[col] = data[col].astype('uint8')
                elif data[col].max() < 65535:
                    data[col] = data[col].astype('uint16')
                else:
                    data[col] = data[col].astype('uint32')
            else:
                if data[col].min() > -128 and data[col].max() < 127:
                    data[col] = data[col].astype('int8')
                elif data[col].min() > -32768 and data[col].max() < 32767:
                    data[col] = data[col].astype('int16')
                else:
                    data[col] = data[col].astype('int32')
        
        # Optimize float columns
        for col in data.select_dtypes(include=['float64']).columns:
            data[col] = data[col].astype('float32')
        
        # Optimize object columns
        for col in data.select_dtypes(include=['object']).columns:
            if col != 'date':  # Keep date as object for now
                unique_count = data[col].nunique()
                total_count = len(data)
                
                if unique_count / total_count < 0.5:  # Less than 50% unique
                    data[col] = data[col].astype('category')
        
        return data
    
    def create_performance_benchmark(self) -> Dict:
        """Create comprehensive performance benchmark"""
        
        benchmark_results = {
            'timestamp': time.time(),
            'system_info': {
                'cpu_count': mp.cpu_count(),
                'python_version': f"{mp.sys.version_info.major}.{mp.sys.version_info.minor}"
            },
            'tests': {}
        }
        
        try:
            # Load test data
            data = pd.read_csv('./data/raw/realistic_demand_data.csv')
            
            # Test 1: Data loading and preprocessing
            start_time = time.time()
            optimized_data = self.optimize_memory_usage(data.copy())
            benchmark_results['tests']['data_optimization'] = {
                'time_seconds': time.time() - start_time,
                'memory_reduction': f"{(1 - optimized_data.memory_usage(deep=True).sum() / data.memory_usage(deep=True).sum()) * 100:.1f}%",
                'original_size_mb': f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f}",
                'optimized_size_mb': f"{optimized_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f}"
            }
            
            # Test 2: Model training performance
            models_to_test = ['ARIMA', 'ETS', 'Croston']
            sample_data = data.head(200)  # Use sample for speed
            
            start_time = time.time()
            training_results = self.parallel_model_training(sample_data, models_to_test)
            benchmark_results['tests']['parallel_training'] = {
                'time_seconds': time.time() - start_time,
                'models_tested': len(models_to_test),
                'successful_models': len([r for r in training_results.values() if r['status'] == 'success']),
                'results': training_results
            }
            
            # Test 3: Batch forecasting
            unique_products = data['product_id'].unique()[:10]  # Test with 10 products
            
            start_time = time.time()
            batch_results = self.batch_forecast_processing(unique_products.tolist(), horizon=7)
            benchmark_results['tests']['batch_forecasting'] = {
                'time_seconds': time.time() - start_time,
                'products_processed': len(batch_results),
                'successful_forecasts': len([r for r in batch_results.values() if 'forecast' in r]),
                'avg_forecast_length': np.mean([len(r.get('forecast', [])) for r in batch_results.values() if 'forecast' in r])
            }
            
        except Exception as e:
            benchmark_results['error'] = str(e)
        
        # Save benchmark results
        with open('./data/processed/performance_benchmark.json', 'w') as f:
            import json
            json.dump(benchmark_results, f, indent=2, default=str)
        
        return benchmark_results
    
    def get_optimization_recommendations(self, benchmark_results: Dict) -> List[str]:
        """Generate optimization recommendations based on benchmark"""
        
        recommendations = []
        
        # Memory optimization
        if 'data_optimization' in benchmark_results['tests']:
            memory_test = benchmark_results['tests']['data_optimization']
            if float(memory_test['original_size_mb']) > 100:
                recommendations.append("Consider implementing data chunking for large datasets")
            if float(memory_test['memory_reduction'].replace('%', '')) < 20:
                recommendations.append("Explore additional data type optimizations")
        
        # Training performance
        if 'parallel_training' in benchmark_results['tests']:
            training_test = benchmark_results['tests']['parallel_training']
            if training_test['time_seconds'] > 30:
                recommendations.append("Consider model simplification or data sampling for faster training")
            if training_test['successful_models'] < training_test['models_tested']:
                recommendations.append("Review failed models and implement better error handling")
        
        # Batch processing
        if 'batch_forecasting' in benchmark_results['tests']:
            batch_test = benchmark_results['tests']['batch_forecasting']
            if batch_test['time_seconds'] / batch_test['products_processed'] > 2:
                recommendations.append("Optimize individual forecast generation for better batch performance")
        
        # System recommendations
        cpu_count = benchmark_results['system_info']['cpu_count']
        if cpu_count < 4:
            recommendations.append("Consider upgrading to a multi-core system for better parallel processing")
        
        return recommendations

if __name__ == "__main__":
    optimizer = PerformanceOptimizer()
    
    print("🚀 Running performance benchmark...")
    results = optimizer.create_performance_benchmark()
    
    print("📊 Benchmark Results:")
    for test_name, test_results in results.get('tests', {}).items():
        print(f"  {test_name}: {test_results.get('time_seconds', 'N/A'):.2f}s")
    
    print("\n💡 Optimization Recommendations:")
    recommendations = optimizer.get_optimization_recommendations(results)
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")