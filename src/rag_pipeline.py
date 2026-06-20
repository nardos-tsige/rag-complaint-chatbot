import numpy as np
from typing import List, Dict, Any, Optional
import logging
from src.embeddings import EmbeddingGenerator
from src.vector_store import FAISSVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 vector_store_path: str = None,
                 k: int = 5):
        self.embedding_model = EmbeddingGenerator(embedding_model)
        self.vector_store = FAISSVectorStore(self.embedding_model.embedding_dim)
        self.k = k
        
        if vector_store_path:
            self.load_vector_store(vector_store_path)
        
        logger.info(f"RAG Pipeline initialized with k={k}")
    
    def load_vector_store(self, vector_store_path: str):
        from pathlib import Path
        index_path = Path(vector_store_path) / 'index.faiss'
        chunks_path = Path(vector_store_path) / 'chunks.pkl'
        
        self.vector_store.load_index(index_path, chunks_path)
        logger.info(f"Loaded vector store from {vector_store_path}")
    
    def retrieve(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        if k is None:
            k = self.k
        
        query_embedding = self.embedding_model.generate_embeddings([query])
        
        results = self.vector_store.search_and_retrieve(query_embedding, k)
        
        return results
    
    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[{i}] {chunk['chunk_text']}")
        
        return "\n\n".join(context_parts)
    
    def generate_prompt(self, query: str, context: str) -> str:
        prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def query(self, query: str, k: int = None) -> Dict[str, Any]:
        retrieved_chunks = self.retrieve(query, k)
        
        context = self.format_context(retrieved_chunks)
        
        prompt = self.generate_prompt(query, context)
        
        return {
            'query': query,
            'retrieved_chunks': retrieved_chunks,
            'context': context,
            'prompt': prompt,
            'k': k or self.k
        }
    
    def get_retrieval_stats(self, query: str, k: int = None) -> Dict[str, Any]:
        results = self.query(query, k)
        
        scores = [chunk['score'] for chunk in results['retrieved_chunks']]
        
        return {
            'query': query,
            'num_retrieved': len(results['retrieved_chunks']),
            'scores': scores,
            'avg_score': np.mean(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'products': list(set([c['product_category'] for c in results['retrieved_chunks']]))
        }