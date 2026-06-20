import pytest
import pandas as pd
from src.chunking import TextChunker, ChunkConfig, LangChainChunker

class TestTextChunker:
    
    @pytest.fixture
    def chunker(self):
        return TextChunker(ChunkConfig(chunk_size=50, chunk_overlap=10, min_chunk_length=10))
    
    def test_chunk_text_basic(self, chunker):
        text = "This is a test sentence. This is another sentence. And a third one."
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 0
        assert all(len(chunk) >= 10 for chunk in chunks)
    
    def test_chunk_text_short(self, chunker):
        text = "Short text."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."
    
    def test_chunk_text_empty(self, chunker):
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text(None) == []
    
    def test_chunk_dataframe(self, chunker):
        df = pd.DataFrame({
            'complaint_id': [1, 2],
            'cleaned_narrative': [
                "This is a long narrative that needs to be chunked into multiple pieces.",
                "Another narrative that is also long enough for chunking."
            ]
        })
        chunks = chunker.chunk_dataframe(df)
        assert len(chunks) > 0
        assert 'complaint_id' in chunks[0]
        assert 'chunk_text' in chunks[0]
        assert 'chunk_index' in chunks[0]
    
    def test_get_chunk_stats(self, chunker):
        chunks = [
            {'chunk_text': 'chunk one', 'chunk_length': 9},
            {'chunk_text': 'chunk two', 'chunk_length': 9},
            {'chunk_text': 'chunk three', 'chunk_length': 11}
        ]
        stats = chunker.get_chunk_stats(chunks)
        assert stats['total_chunks'] == 3
        assert stats['avg_length'] > 0
        assert stats['min_length'] > 0
        assert stats['max_length'] > 0


class TestLangChainChunker:
    
    @pytest.fixture
    def chunker(self):
        try:
            return LangChainChunker(chunk_size=50, chunk_overlap=10)
        except ImportError:
            pytest.skip("LangChain not installed")
    
    def test_chunk_text(self, chunker):
        text = "This is a test sentence. This is another sentence. And a third one."
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 0
    
    def test_chunk_dataframe(self, chunker):
        df = pd.DataFrame({
            'complaint_id': [1],
            'cleaned_narrative': ["This is a long narrative that needs to be chunked."]
        })
        chunks = chunker.chunk_dataframe(df)
        assert len(chunks) > 0
        assert 'complaint_id' in chunks[0]
        assert 'chunk_text' in chunks[0]