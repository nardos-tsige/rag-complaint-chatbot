import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Union
import logging
from tqdm import tqdm
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 device: str = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
    def generate_embeddings(self, texts: List[str], 
                           batch_size: int = 32,
                           show_progress: bool = True) -> np.ndarray:
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        all_embeddings = []
        
        iterator = tqdm(range(0, len(texts), batch_size), 
                       desc="Generating embeddings") if show_progress else range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            embeddings = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.append(embeddings)
        
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])
    
    def generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]], 
                                      batch_size: int = 32) -> Dict[str, Any]:
        texts = [chunk['chunk_text'] for chunk in chunks]
        
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        
        result = {
            'embeddings': embeddings,
            'chunks': chunks,
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name
        }
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return result
    
    def save_embeddings(self, data: Dict[str, Any], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved embeddings to {output_path}")
        
    def load_embeddings(self, input_path: Path) -> Dict[str, Any]:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded embeddings from {input_path}")
        return data