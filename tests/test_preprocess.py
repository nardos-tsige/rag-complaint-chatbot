import pytest
import pandas as pd
import numpy as np
from src.preprocess import ComplaintPreprocessor, EDA

class TestComplaintPreprocessor:
    
    @pytest.fixture
    def preprocessor(self):
        return ComplaintPreprocessor(
            remove_stopwords=False,
            min_length=10,
            max_length=5000
        )
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'Product': ['Credit card', 'Personal loan', 'Savings account'],
            'Consumer complaint narrative': [
                'I am writing to file a complaint about hidden fees on my credit card. This is unacceptable.',
                'The interest rate on my personal loan increased without notice.',
                'My savings account has been charging monthly maintenance fees.'
            ],
            'Company': ['Bank A', 'Bank B', 'Bank C']
        })
    
    def test_clean_text(self, preprocessor):
        text = "I am writing to file a complaint about hidden fees. This is unacceptable!"
        cleaned = preprocessor.clean_text(text)
        assert 'writing to file a complaint' not in cleaned
        assert 'hidden fees' in cleaned
        assert cleaned.islower()
    
    def test_clean_text_empty(self, preprocessor):
        assert preprocessor.clean_text(None) == ""
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(123) == ""
    
    def test_filter_products(self, preprocessor, sample_df):
        target = ['Credit card', 'Personal loan']
        filtered = preprocessor.filter_products(sample_df, target)
        assert len(filtered) == 2
        assert 'Savings account' not in filtered['Product'].values
    
    def test_filter_empty_narratives(self, preprocessor, sample_df):
        df_with_empty = pd.concat([
            sample_df,
            pd.DataFrame({
                'Product': ['Money transfer'],
                'Consumer complaint narrative': [None],
                'Company': ['Bank D']
            })
        ])
        filtered = preprocessor.filter_empty_narratives(df_with_empty)
        assert len(filtered) == len(sample_df)
        assert len(filtered) < len(df_with_empty)
    
    def test_clean_narratives(self, preprocessor, sample_df):
        cleaned = preprocessor.clean_narratives(sample_df)
        assert 'cleaned_narrative' in cleaned.columns
        assert 'narrative_length' in cleaned.columns
        assert len(cleaned) == len(sample_df)
        assert all(cleaned['narrative_length'] > 0)
    
    def test_preprocess_pipeline(self, preprocessor, sample_df):
        target = ['Credit card', 'Personal loan']
        processed = preprocessor.preprocess_pipeline(sample_df, target)
        assert len(processed) <= len(sample_df)
        assert 'cleaned_narrative' in processed.columns
        assert 'narrative_length' in processed.columns


class TestEDA:
    
    @pytest.fixture
    def eda(self):
        return EDA()
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'Product': ['Credit card', 'Credit card', 'Personal loan', 'Savings account'],
            'Issue': ['Billing dispute', 'Late fees', 'Interest rate', 'Fees'],
            'narrative_length': [100, 200, 150, 50]
        })
    
    def test_analyze_distribution(self, eda, sample_df):
        stats = eda.analyze_distribution(sample_df)
        assert 'product_counts' in stats
        assert stats['product_counts']['Credit card'] == 2
        assert stats['product_counts']['Personal loan'] == 1
        assert stats['product_counts']['Savings account'] == 1
        assert stats['unique_products'] == 3
    
    def test_analyze_narrative_lengths(self, eda, sample_df):
        stats = eda.analyze_narrative_lengths(sample_df)
        assert stats['mean_length'] == 125
        assert stats['min_length'] == 50
        assert stats['max_length'] == 200
    
    def test_analyze_complaint_issues(self, eda, sample_df):
        stats = eda.analyze_complaint_issues(sample_df)
        assert stats['unique_issues'] == 4
        assert 'top_issues' in stats