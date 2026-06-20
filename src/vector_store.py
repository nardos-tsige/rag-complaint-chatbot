import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import pickle
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, embedding_dim: int = 384):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            logger.error("FAISS not installed. Please install faiss-cpu or faiss-gpu")
            raise
            
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def create_index(self, index_type: str = 'FlatIP') -> None:
        if index_type == 'FlatIP':
            self.index = self.faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == 'FlatL2':
            self.index = self.faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        logger.info(f"Created FAISS index of type {index_type}")
    
    def add_embeddings(self, embeddings: np.ndarray, 
                       chunks: List[Dict[str, Any]]) -> None:
        if self.index is None:
            self.create_index()
        
        embeddings = embeddings.astype(np.float32)
        
        self.index.add(embeddings)
        
        self.chunks.extend(chunks)
        self.metadata.extend([{
            'complaint_id': c['complaint_id'],
            'product_category': c['product_category'],
            'issue': c['issue'],
            'chunk_index': c['chunk_index']
        } for c in chunks])
        
        logger.info(f"Added {len(chunks)} embeddings to index")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty")
        
        query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, k)
        
        return distances, indices
    
    def search_and_retrieve(self, query_embedding: np.ndarray, 
                           k: int = 5) -> List[Dict[str, Any]]:
        distances, indices = self.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(distances[0][i])
                results.append(chunk)
        
        return results
    
    def save_index(self, index_path: Path, chunks_path: Path = None):
        if self.index is None:
            raise ValueError("Index is empty")
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved index to {index_path}")
        
        if chunks_path:
            with open(chunks_path, 'wb') as f:
                pickle.dump({'chunks': self.chunks, 'metadata': self.metadata}, f)
            logger.info(f"Saved chunks to {chunks_path}")
    
    def load_index(self, index_path: Path, chunks_path: Path = None):
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = self.faiss.read_index(str(index_path))
        self.embedding_dim = self.index.d
        logger.info(f"Loaded index from {index_path}")
        
        if chunks_path and chunks_path.exists():
            with open(chunks_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
            logger.info(f"Loaded {len(self.chunks)} chunks from {chunks_path}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        if self.index is None:
            return {'total_vectors': 0}
        
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'total_chunks': len(self.chunks)
        }