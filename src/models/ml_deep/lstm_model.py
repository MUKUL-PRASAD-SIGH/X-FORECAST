import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class LSTMForecaster:
    def __init__(self, sequence_length=30, hidden_size=50, num_layers=2):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Run: pip install torch")
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_sequences(self, data, seq_length):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def fit(self, data: pd.Series):
        # Simplified LSTM implementation
        # In practice, would implement full PyTorch LSTM
        self.fitted_data = data
        return self
    
    def forecast(self, steps: int) -> pd.Series:
        # Simplified forecast - would use trained LSTM
        last_values = self.fitted_data.tail(self.sequence_length)
        forecast = [last_values.mean()] * steps
        return pd.Series(forecast)
    
    def evaluate(self, actual: pd.Series, predicted: pd.Series) -> dict:
        return {
            'mae': mean_absolute_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
        }