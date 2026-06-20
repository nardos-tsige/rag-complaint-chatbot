import pytest
import numpy as np
from pathlib import Path
from src.embeddings import EmbeddingGenerator

class TestEmbeddingGenerator:
    
    @pytest.fixture
    def generator(self):
        return EmbeddingGenerator(
            model_name='all-MiniLM-L6-v2',
            device='cpu'
        )
    
    def test_init(self, generator):
        assert generator.model_name == 'all-MiniLM-L6-v2'
        assert generator.device == 'cpu'
        assert generator.embedding_dim == 384
    
    def test_generate_embeddings(self, generator):
        texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = generator.generate_embeddings(texts, batch_size=2)
        
        assert embeddings.shape == (2, 384)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32
    
    def test_generate_embeddings_single(self, generator):
        text = "This is a test sentence."
        embeddings = generator.generate_embeddings([text])
        
        assert embeddings.shape == (1, 384)
        assert isinstance(embeddings, np.ndarray)
    
    def test_generate_embeddings_empty(self, generator):
        embeddings = generator.generate_embeddings([])
        assert embeddings.shape == (0,)
    
    @pytest.mark.parametrize("texts", [
        ["Short text"],
        ["Longer text that has multiple words"],
        ["Text with numbers 123 and symbols !@#"]
    ])
    def test_generate_embeddings_various_texts(self, generator, texts):
        embeddings = generator.generate_embeddings(texts)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 384
    
    def test_generate_embeddings_for_chunks(self, generator):
        chunks = [
            {'chunk_text': 'First chunk text'},
            {'chunk_text': 'Second chunk text'}
        ]
        result = generator.generate_embeddings_for_chunks(chunks)
        
        assert 'embeddings' in result
        assert 'chunks' in result
        assert 'embedding_dim' in result
        assert 'model_name' in result
        assert result['embeddings'].shape == (2, 384)
        assert len(result['chunks']) == 2
    
    def test_save_and_load_embeddings(self, generator, tmp_path):
        chunks = [{'chunk_text': 'Test chunk'}]
        result = generator.generate_embeddings_for_chunks(chunks)
        
        output_path = tmp_path / "test_embeddings.pkl"
        generator.save_embeddings(result, output_path)
        
        assert output_path.exists()
        
        loaded = generator.load_embeddings(output_path)
        assert 'embeddings' in loaded
        assert 'chunks' in loaded
        assert loaded['embedding_dim'] == result['embedding_dim']
        assert loaded['model_name'] == result['model_name']