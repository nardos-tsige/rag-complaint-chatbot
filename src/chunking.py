import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separator: str = " "
    min_chunk_length: int = 50
    
class TextChunker:
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        logger.info(f"Initialized chunker with config: {self.config}")
        
    def chunk_text(self, text: str) -> List[str]:
        if not text or len(text) < self.config.chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.config.chunk_size, text_length)
            
            if end < text_length:
                for delimiter in ['. ', '! ', '? ', '.', '!', '?']:
                    pos = text.rfind(delimiter, start, end)
                    if pos != -1:
                        end = pos + len(delimiter)
                        break
            
            chunk = text[start:end].strip()
            
            if len(chunk) >= self.config.min_chunk_length:
                chunks.append(chunk)
            
            start = start + self.config.chunk_size - self.config.chunk_overlap
            
        return chunks
    
    def chunk_dataframe(self, df: pd.DataFrame, 
                        text_column: str = 'cleaned_narrative',
                        id_column: str = 'complaint_id') -> List[Dict[str, Any]]:
        logger.info(f"Chunking {len(df)} complaints...")
        
        chunked_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
            text = row[text_column]
            complaint_id = row.get(id_column, idx)
            
            chunks = self.chunk_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_record = {
                    'complaint_id': complaint_id,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'chunk_text': chunk,
                    'chunk_length': len(chunk),
                    'product_category': row.get('Product', 'Unknown'),
                    'issue': row.get('Issue', 'Unknown'),
                    'sub_issue': row.get('Sub-issue', 'Unknown'),
                    'company': row.get('Company', 'Unknown'),
                    'state': row.get('State', 'Unknown'),
                    'date_received': row.get('Date received', 'Unknown'),
                }
                chunked_data.append(chunk_record)
        
        logger.info(f"Created {len(chunked_data)} chunks from {len(df)} complaints")
        return chunked_data
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {}
            
        lengths = [len(c['chunk_text']) for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'std_length': pd.Series(lengths).std() if len(lengths) > 1 else 0
        }

class LangChainChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
                add_start_index=True,
            )
            logger.info(f"Initialized LangChain chunker with chunk_size={chunk_size}")
        except ImportError:
            logger.error("LangChain not installed. Please install langchain.")
            raise
            
    def chunk_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
    
    def chunk_dataframe(self, df: pd.DataFrame, 
                        text_column: str = 'cleaned_narrative',
                        id_column: str = 'complaint_id') -> List[Dict[str, Any]]:
        logger.info(f"Chunking {len(df)} complaints using LangChain...")
        
        chunked_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
            text = row[text_column]
            complaint_id = row.get(id_column, idx)
            
            chunks = self.chunk_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_record = {
                    'complaint_id': complaint_id,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'chunk_text': chunk,
                    'chunk_length': len(chunk),
                    'product_category': row.get('Product', 'Unknown'),
                    'issue': row.get('Issue', 'Unknown'),
                    'sub_issue': row.get('Sub-issue', 'Unknown'),
                    'company': row.get('Company', 'Unknown'),
                    'state': row.get('State', 'Unknown'),
                    'date_received': row.get('Date received', 'Unknown'),
                }
                chunked_data.append(chunk_record)
        
        logger.info(f"Created {len(chunked_data)} chunks")
        return chunked_data