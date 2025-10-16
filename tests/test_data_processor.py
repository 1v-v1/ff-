"""
Unit tests for data_processor module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processor import DataProcessor


class TestDataProcessor:
    """Test suite for DataProcessor class"""
    
    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance"""
        return DataProcessor()
    
    @pytest.fixture
    def sample_factor_data(self):
        """Create sample factor data"""
        dates = pd.date_range('2015-01-01', periods=12, freq='MS')
        data = {
            'Mkt-RF': [1.5, -0.5, 2.0, -1.0, 0.5, 1.0, -0.5, 1.5, 0.0, 1.0, -1.0, 2.0],
            'SMB': [0.5, 0.3, -0.2, 0.1, 0.4, -0.1, 0.2, 0.3, -0.1, 0.0, 0.1, -0.2],
            'HML': [0.2, -0.1, 0.3, 0.0, -0.2, 0.1, 0.2, -0.1, 0.3, 0.1, -0.1, 0.2]
        }
        return pd.DataFrame(data, index=dates)
    
    def test_initialization(self, processor):
        """Test DataProcessor initialization"""
        assert processor is not None
        assert hasattr(processor, 'filter_by_date_range')
        assert hasattr(processor, 'calculate_cumulative_returns')
    
    def test_filter_by_date_range(self, processor, sample_factor_data):
        """Test filtering data by date range"""
        filtered = processor.filter_by_date_range(
            sample_factor_data,
            start_date='2015-03-01',
            end_date='2015-06-01'
        )
        
        assert len(filtered) == 4  # March, April, May, June
        assert filtered.index[0] == pd.Timestamp('2015-03-01')
        assert filtered.index[-1] == pd.Timestamp('2015-06-01')
    
    def test_filter_by_date_range_string_format(self, processor, sample_factor_data):
        """Test filtering with different date string formats"""
        # Test YYYY-MM format
        filtered = processor.filter_by_date_range(
            sample_factor_data,
            start_date='2015-03',
            end_date='2015-06'
        )
        
        assert len(filtered) == 4
    
    def test_filter_by_date_range_out_of_bounds(self, processor, sample_factor_data):
        """Test filtering with dates outside data range"""
        filtered = processor.filter_by_date_range(
            sample_factor_data,
            start_date='2020-01-01',
            end_date='2020-12-01'
        )
        
        assert len(filtered) == 0  # No data in this range
    
    def test_calculate_cumulative_returns_simple(self, processor):
        """Test cumulative returns calculation with simple data"""
        dates = pd.date_range('2015-01-01', periods=3, freq='MS')
        data = pd.DataFrame({
            'Factor1': [10.0, 5.0, -5.0]  # 10%, 5%, -5%
        }, index=dates)
        
        cumulative = processor.calculate_cumulative_returns(data)
        
        # Expected: (1.10 * 1.05 * 0.95 - 1) * 100
        expected_final = (1.10 * 1.05 * 0.95 - 1) * 100
        
        assert 'Factor1' in cumulative.columns
        assert len(cumulative) == 3
        assert np.isclose(cumulative['Factor1'].iloc[-1], expected_final, rtol=0.01)
    
    def test_calculate_cumulative_returns_multiple_columns(self, processor, sample_factor_data):
        """Test cumulative returns with multiple columns"""
        cumulative = processor.calculate_cumulative_returns(sample_factor_data)
        
        assert cumulative.shape == sample_factor_data.shape
        assert all(col in cumulative.columns for col in sample_factor_data.columns)
        # Cumulative returns should start near 0
        assert all(np.isclose(cumulative.iloc[0], sample_factor_data.iloc[0], rtol=0.01))
    
    def test_calculate_cumulative_returns_empty_dataframe(self, processor):
        """Test cumulative returns with empty dataframe"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError) as exc_info:
            processor.calculate_cumulative_returns(empty_df)
        
        assert 'empty' in str(exc_info.value).lower()
    
    def test_merge_factors_and_portfolios(self, processor):
        """Test merging factors and portfolios data"""
        dates = pd.date_range('2015-01-01', periods=5, freq='MS')
        
        factors = pd.DataFrame({
            'Mkt-RF': [1.0, 2.0, 3.0, 4.0, 5.0],
            'SMB': [0.1, 0.2, 0.3, 0.4, 0.5]
        }, index=dates)
        
        portfolios = pd.DataFrame({
            'Portfolio1': [1.5, 2.5, 3.5, 4.5, 5.5],
            'Portfolio2': [0.5, 1.0, 1.5, 2.0, 2.5]
        }, index=dates)
        
        merged = processor.merge_factors_and_portfolios(factors, portfolios)
        
        assert len(merged) == 5
        assert 'Mkt-RF' in merged.columns
        assert 'SMB' in merged.columns
        assert 'Portfolio1' in merged.columns
        assert 'Portfolio2' in merged.columns
    
    def test_merge_factors_and_portfolios_different_dates(self, processor):
        """Test merging with non-overlapping dates"""
        factors = pd.DataFrame({
            'Mkt-RF': [1.0, 2.0, 3.0]
        }, index=pd.date_range('2015-01-01', periods=3, freq='MS'))
        
        portfolios = pd.DataFrame({
            'Portfolio1': [1.5, 2.5, 3.5]
        }, index=pd.date_range('2015-02-01', periods=3, freq='MS'))
        
        merged = processor.merge_factors_and_portfolios(factors, portfolios)
        
        # Should only include overlapping dates (Feb, Mar)
        assert len(merged) == 2
        assert pd.Timestamp('2015-02-01') in merged.index
        assert pd.Timestamp('2015-03-01') in merged.index
    
    def test_validate_data_quality_valid(self, processor, sample_factor_data):
        """Test data quality validation with valid data"""
        result = processor.validate_data_quality(sample_factor_data)
        
        assert result['has_missing'] == False
        assert result['missing_count'] == 0
        assert result['total_rows'] == 12
    
    def test_validate_data_quality_with_missing(self, processor):
        """Test data quality validation with missing values"""
        dates = pd.date_range('2015-01-01', periods=5, freq='MS')
        data = pd.DataFrame({
            'Factor1': [1.0, np.nan, 3.0, 4.0, np.nan],
            'Factor2': [1.0, 2.0, 3.0, 4.0, 5.0]
        }, index=dates)
        
        result = processor.validate_data_quality(data)
        
        assert result['has_missing'] == True
        assert result['missing_count'] == 2
        assert result['missing_by_column']['Factor1'] == 2
        assert result['missing_by_column']['Factor2'] == 0
    
    def test_validate_data_quality_all_missing(self, processor):
        """Test data quality validation with all missing values"""
        dates = pd.date_range('2015-01-01', periods=3, freq='MS')
        data = pd.DataFrame({
            'Factor1': [np.nan, np.nan, np.nan]
        }, index=dates)
        
        result = processor.validate_data_quality(data)
        
        assert result['has_missing'] == True
        assert result['missing_count'] == 3
        assert result['missing_by_column']['Factor1'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


