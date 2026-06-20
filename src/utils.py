import pandas as pd
import numpy as np
from pathlib import Path
import json
import yaml
from typing import Dict, Any, List
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_json(data: Dict[str, Any], file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {file_path}")

def load_json(file_path: Path) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return json.load(f)

def save_yaml(data: Dict[str, Any], file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.info(f"Saved YAML to {file_path}")

def load_yaml(file_path: Path) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def get_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def get_file_size(file_path: Path) -> str:
    size_bytes = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def sample_stratified(df: pd.DataFrame, 
                      column: str, 
                      n_samples: int, 
                      random_state: int = 42) -> pd.DataFrame:
    sizes = df.groupby(column).size()
    sample_size = min(n_samples // len(sizes), sizes.min())
    
    sampled = df.groupby(column, group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size), random_state=random_state)
    )
    
    if len(sampled) < n_samples:
        remaining = n_samples - len(sampled)
        remaining_df = df.drop(sampled.index)
        if len(remaining_df) > 0:
            additional = remaining_df.sample(min(remaining, len(remaining_df)), 
                                           random_state=random_state)
            sampled = pd.concat([sampled, additional])
    
    return sampled

def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    info = []
    for col in df.columns:
        info.append({
            'column': col,
            'dtype': df[col].dtype,
            'null_count': df[col].isnull().sum(),
            'null_pct': (df[col].isnull().sum() / len(df)) * 100,
            'unique_values': df[col].nunique(),
            'memory_usage': df[col].memory_usage(deep=True)
        })
    return pd.DataFrame(info)

def chunk_list(lst: List[Any], chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def progress_bar(iterable, desc: str = "Processing", total: int = None):
    from tqdm import tqdm
    return tqdm(iterable, desc=desc, total=total)

def setup_logging(log_file: str = None, level: str = 'INFO'):
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )