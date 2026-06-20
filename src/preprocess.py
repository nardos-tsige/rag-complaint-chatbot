import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import logging
from typing import List, Optional, Dict, Any
from tqdm import tqdm

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplaintPreprocessor:
    def __init__(self, remove_stopwords=False, min_length=10, max_length=5000):
        self.remove_stopwords = remove_stopwords
        self.min_length = min_length
        self.max_length = max_length
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower()
        boilerplate_patterns = [
            r'i am writing to file a complaint',
            r'i am writing to complain about',
            r'i am submitting a complaint',
            r'this is a complaint regarding',
            r'i would like to file a complaint',
            r'please find below my complaint',
            r'i am filing a complaint',
            r'complaint:',
            r'dear [a-z\s]+,\s*'
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\'\"]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def filter_products(self, df, target_products):
        logger.info(f"Filtering for products: {target_products}")
        filtered_df = df[df['Product'].isin(target_products)].copy()
        logger.info(f"Kept {len(filtered_df):,} records")
        return filtered_df
    
    def filter_empty_narratives(self, df):
        before = len(df)
        filtered_df = df[df['Consumer complaint narrative'].notna() & 
                        (df['Consumer complaint narrative'].str.strip() != '')].copy()
        after = len(filtered_df)
        logger.info(f"Removed {before - after:,} records with empty narratives")
        return filtered_df
    
    def clean_narratives(self, df):
        logger.info("Cleaning narratives...")
        tqdm.pandas(desc="Cleaning narratives")
        df['cleaned_narrative'] = df['Consumer complaint narrative'].progress_apply(self.clean_text)
        df['narrative_length'] = df['cleaned_narrative'].str.len()
        before = len(df)
        df = df[(df['narrative_length'] >= self.min_length) & 
                (df['narrative_length'] <= self.max_length)]
        after = len(df)
        logger.info(f"Filtered by length: kept {after:,} records, removed {before - after:,}")
        return df
    
    def preprocess_pipeline(self, df, target_products):
        logger.info("="*60)
        logger.info("Starting preprocessing pipeline...")
        logger.info("="*60)
        original_count = len(df)
        df = self.filter_products(df, target_products)
        df = self.filter_empty_narratives(df)
        df = self.clean_narratives(df)
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        logger.info(f"Removed {original_count - len(df):,} records ({((original_count - len(df))/original_count*100):.1f}%)")
        logger.info("="*60)
        return df

class EDA:
    def __init__(self):
        self.df = None
        
    def analyze_distribution(self, df):
        product_counts = df['Product'].value_counts()
        stats = {
            'product_counts': product_counts,
            'product_percentages': (product_counts / len(df) * 100).round(2),
            'unique_products': df['Product'].nunique()
        }
        return stats
    
    def analyze_narrative_lengths(self, df):
        lengths = df['narrative_length']
        stats = {
            'mean_length': lengths.mean(),
            'median_length': lengths.median(),
            'std_length': lengths.std(),
            'min_length': lengths.min(),
            'max_length': lengths.max(),
            'q25': lengths.quantile(0.25),
            'q75': lengths.quantile(0.75)
        }
        return stats
    
    def analyze_complaint_issues(self, df):
        issue_counts = df['Issue'].value_counts()
        return {
            'top_issues': issue_counts.head(10),
            'unique_issues': df['Issue'].nunique()
        }