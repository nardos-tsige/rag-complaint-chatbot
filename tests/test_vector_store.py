import pytest
import numpy as np
from pathlib import Path
from src.vector_store import FAISSVectorStore

class TestFAISSVectorStore:
    
    @pytest.fixture
    def vector_store(self):
        return FAISSVectorStore(embedding_dim=384)
    
    @pytest.fixture
    def sample_embeddings(self):
        return np.random.randn(10, 384).astype(np.float32)
    
    @pytest.fixture
    def sample_chunks(self):
        return [
            {
                'complaint_id': i,
                'product_category': 'Credit card',
                'issue': 'Billing',
                'chunk_index': 0
            }
            for i in range(10)
        ]
    
    def test_create_index(self, vector_store):
        vector_store.create_index(index_type='FlatIP')
        assert vector_store.index is not None
        assert vector_store.embedding_dim == 384
    
    def test_add_embeddings(self, vector_store, sample_embeddings, sample_chunks):
        vector_store.create_index()
        vector_store.add_embeddings(sample_embeddings, sample_chunks)
        
        assert vector_store.index.ntotal == 10
        assert len(vector_store.chunks) == 10
        assert len(vector_store.metadata) == 10
    
    def test_search(self, vector_store, sample_embeddings, sample_chunks):
        vector_store.create_index()
        vector_store.add_embeddings(sample_embeddings, sample_chunks)
        
        query_embedding = sample_embeddings[0:1]
        distances, indices = vector_store.search(query_embedding, k=3)
        
        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)
        assert indices[0][0] == 0
    
    def test_search_and_retrieve(self, vector_store, sample_embeddings, sample_chunks):
        vector_store.create_index()
        vector_store.add_embeddings(sample_embeddings, sample_chunks)
        
        query_embedding = sample_embeddings[0:1]
        results = vector_store.search_and_retrieve(query_embedding, k=3)
        
        assert len(results) == 3
        assert 'score' in results[0]
        assert 'complaint_id' in results[0]
        assert 'product_category' in results[0]
    
    def test_save_and_load_index(self, vector_store, sample_embeddings, sample_chunks, tmp_path):
        vector_store.create_index()
        vector_store.add_embeddings(sample_embeddings, sample_chunks)
        
        index_path = tmp_path / "test_index.faiss"
        chunks_path = tmp_path / "test_chunks.pkl"
        
        vector_store.save_index(index_path, chunks_path)
        
        assert index_path.exists()
        assert chunks_path.exists()
        
        new_store = FAISSVectorStore(384)
        new_store.load_index(index_path, chunks_path)
        
        assert new_store.index.ntotal == 10
        assert len(new_store.chunks) == 10
        assert len(new_store.metadata) == 10
    
    def test_get_index_stats(self, vector_store, sample_embeddings, sample_chunks):
        vector_store.create_index()
        vector_store.add_embeddings(sample_embeddings, sample_chunks)
        
        stats = vector_store.get_index_stats()
        assert stats['total_vectors'] == 10
        assert stats['embedding_dim'] == 384
        assert stats['total_chunks'] == 10
    
    def test_search_empty_index(self, vector_store):
        vector_store.create_index()
        query_embedding = np.random.randn(1, 384).astype(np.float32)
        
        with pytest.raises(ValueError):
            vector_store.search(query_embedding, k=3)