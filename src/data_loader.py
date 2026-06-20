import pandas as pd
import numpy as np
from pathlib import Path
import logging
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplaintDataLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self):
        try:
            if not self.data_path.exists():
                zip_path = self.data_path.with_suffix('.csv.zip')
                if zip_path.exists():
                    logger.info(f"Extracting zip file: {zip_path}")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(self.data_path.parent)
            
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path, low_memory=False, encoding='utf-8')
            logger.info(f"Successfully loaded {len(self.df):,} records")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_basic_stats(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        stats = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': self.df.dtypes.to_dict()
        }
        return stats
    
    def get_products(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.df['Product'].unique().tolist()