import pandas as pd
from pathlib import Path
from typing import Dict, Any

class DataConnector:
    def __init__(self, source_path: str):
        self.source_path = Path(source_path)
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.source_path / filename)
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            'rows': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df