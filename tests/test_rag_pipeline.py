import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.rag_pipeline import RAGPipeline

class TestRAGPipeline:
    
    @pytest.fixture
    def mock_embedding_generator(self):
        mock = Mock()
        mock.embedding_dim = 384
        return mock
    
    @pytest.fixture
    def mock_vector_store(self):
        mock = Mock()
        mock.embedding_dim = 384
        return mock
    
    @patch('src.rag_pipeline.EmbeddingGenerator')
    @patch('src.rag_pipeline.FAISSVectorStore')
    def test_init(self, mock_vector_store, mock_embedding_generator):
        pipeline = RAGPipeline(embedding_model='all-MiniLM-L6-v2', k=3)
        assert pipeline.k == 3
        assert pipeline.embedding_model is not None
    
    @patch('src.rag_pipeline.EmbeddingGenerator')
    @patch('src.rag_pipeline.FAISSVectorStore')
    def test_retrieve(self, mock_vector_store, mock_embedding_generator):
        pipeline = RAGPipeline()
        
        mock_embedding_generator.generate_embeddings.return_value = np.array([[1.0, 0.0, 0.0]])
        mock_vector_store.search_and_retrieve.return_value = [
            {'score': 0.95, 'chunk_text': 'test chunk', 'product_category': 'Credit card'}
        ]
        
        results = pipeline.retrieve("test query", k=3)
        assert len(results) > 0
        assert 'score' in results[0]
        assert 'chunk_text' in results[0]
    
    @patch('src.rag_pipeline.EmbeddingGenerator')
    @patch('src.rag_pipeline.FAISSVectorStore')
    def test_format_context(self, mock_vector_store, mock_embedding_generator):
        pipeline = RAGPipeline()
        chunks = [
            {'chunk_text': 'First chunk'},
            {'chunk_text': 'Second chunk'}
        ]
        context = pipeline.format_context(chunks)
        assert 'First chunk' in context
        assert 'Second chunk' in context
        assert '[1]' in context
        assert '[2]' in context
    
    @patch('src.rag_pipeline.EmbeddingGenerator')
    @patch('src.rag_pipeline.FAISSVectorStore')
    def test_generate_prompt(self, mock_vector_store, mock_embedding_generator):
        pipeline = RAGPipeline()
        query = "test question"
        context = "test context"
        prompt = pipeline.generate_prompt(query, context)
        assert query in prompt
        assert context in prompt
        assert "CrediTrust" in prompt
    
    @patch('src.rag_pipeline.EmbeddingGenerator')
    @patch('src.rag_pipeline.FAISSVectorStore')
    def test_query(self, mock_vector_store, mock_embedding_generator):
        pipeline = RAGPipeline(k=3)
        
        mock_embedding_generator.generate_embeddings.return_value = np.array([[1.0, 0.0, 0.0]])
        mock_vector_store.search_and_retrieve.return_value = [
            {'score': 0.95, 'chunk_text': 'test chunk', 'product_category': 'Credit card'}
        ]
        
        result = pipeline.query("test query", k=3)
        assert 'query' in result
        assert 'retrieved_chunks' in result
        assert 'context' in result
        assert 'prompt' in result
        assert result['k'] == 3
    
    @patch('src.rag_pipeline.EmbeddingGenerator')
    @patch('src.rag_pipeline.FAISSVectorStore')
    def test_get_retrieval_stats(self, mock_vector_store, mock_embedding_generator):
        pipeline = RAGPipeline(k=5)
        
        mock_embedding_generator.generate_embeddings.return_value = np.array([[1.0, 0.0, 0.0]])
        mock_vector_store.search_and_retrieve.return_value = [
            {'score': 0.95, 'chunk_text': 'chunk1', 'product_category': 'Credit card'},
            {'score': 0.85, 'chunk_text': 'chunk2', 'product_category': 'Loan'},
            {'score': 0.75, 'chunk_text': 'chunk3', 'product_category': 'Credit card'}
        ]
        
        stats = pipeline.get_retrieval_stats("test query")
        assert 'query' in stats
        assert 'num_retrieved' in stats
        assert 'scores' in stats
        assert 'avg_score' in stats
        assert stats['num_retrieved'] == 3