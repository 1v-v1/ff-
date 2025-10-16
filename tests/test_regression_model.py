"""
Unit tests for regression_model module
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.regression_model import FamaFrenchRegression


class TestFamaFrenchRegression:
    """Test suite for FamaFrenchRegression class"""
    
    @pytest.fixture
    def regression(self):
        """Create a FamaFrenchRegression instance"""
        return FamaFrenchRegression()
    
    @pytest.fixture
    def sample_factors(self):
        """Create sample factor data"""
        dates = pd.date_range('2015-01-01', periods=60, freq='MS')
        np.random.seed(42)
        data = {
            'Mkt-RF': np.random.randn(60) * 2 + 1.0,
            'SMB': np.random.randn(60) * 1.5,
            'HML': np.random.randn(60) * 1.2
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def sample_portfolio(self, sample_factors):
        """Create sample portfolio returns based on known factor exposures"""
        # Portfolio with known betas: market=1.2, SMB=0.5, HML=0.3, alpha=0.5
        portfolio_returns = (
            0.5 +  # Alpha
            1.2 * sample_factors['Mkt-RF'] +
            0.5 * sample_factors['SMB'] +
            0.3 * sample_factors['HML'] +
            np.random.randn(60) * 0.5  # Noise
        )
        return pd.Series(portfolio_returns, index=sample_factors.index, name='Portfolio')
    
    def test_initialization(self, regression):
        """Test FamaFrenchRegression initialization"""
        assert regression is not None
        assert hasattr(regression, 'run_regression')
        assert hasattr(regression, 'extract_results')
    
    def test_run_regression_basic(self, regression, sample_factors, sample_portfolio):
        """Test basic regression functionality"""
        result = regression.run_regression(sample_portfolio, sample_factors)
        
        assert result is not None
        assert hasattr(result, 'params')  # Regression coefficients
        assert hasattr(result, 'rsquared')  # R-squared
        assert hasattr(result, 'pvalues')  # P-values
    
    def test_run_regression_returns_correct_coefficients(self, regression, sample_factors, sample_portfolio):
        """Test that regression returns expected number of coefficients"""
        result = regression.run_regression(sample_portfolio, sample_factors)
        
        # Should have intercept + 3 factor coefficients = 4 total
        assert len(result.params) == 4
        
        # Check coefficient names
        assert 'Mkt-RF' in result.params.index
        assert 'SMB' in result.params.index
        assert 'HML' in result.params.index
    
    def test_extract_results_structure(self, regression, sample_factors, sample_portfolio):
        """Test structure of extracted results"""
        regression.run_regression(sample_portfolio, sample_factors)
        results_dict = regression.extract_results()
        
        assert 'alpha' in results_dict
        assert 'beta_mkt' in results_dict
        assert 'beta_smb' in results_dict
        assert 'beta_hml' in results_dict
        assert 'r_squared' in results_dict
        assert 'alpha_pvalue' in results_dict
    
    def test_extract_results_reasonable_values(self, regression, sample_factors, sample_portfolio):
        """Test that extracted results have reasonable values"""
        regression.run_regression(sample_portfolio, sample_factors)
        results_dict = regression.extract_results()
        
        # R-squared should be between 0 and 1
        assert 0 <= results_dict['r_squared'] <= 1
        
        # P-values should be between 0 and 1
        assert 0 <= results_dict['alpha_pvalue'] <= 1
        
        # Betas should be reasonable (not extreme)
        assert -10 < results_dict['beta_mkt'] < 10
        assert -10 < results_dict['beta_smb'] < 10
        assert -10 < results_dict['beta_hml'] < 10
    
    def test_batch_regression_multiple_portfolios(self, regression, sample_factors):
        """Test batch regression with multiple portfolios"""
        # Create multiple portfolios
        dates = sample_factors.index
        portfolios = pd.DataFrame({
            'Portfolio1': np.random.randn(60) * 2 + 1.0,
            'Portfolio2': np.random.randn(60) * 2 + 0.5,
            'Portfolio3': np.random.randn(60) * 2 + 1.5
        }, index=dates)
        
        results_df = regression.batch_regression(portfolios, sample_factors)
        
        # Should have 3 rows (one per portfolio)
        assert len(results_df) == 3
        
        # Check that all portfolios are in the index
        assert 'Portfolio1' in results_df.index
        assert 'Portfolio2' in results_df.index
        assert 'Portfolio3' in results_df.index
        
        # Check that all expected columns are present
        expected_cols = ['alpha', 'beta_mkt', 'beta_smb', 'beta_hml', 'r_squared', 'alpha_pvalue']
        for col in expected_cols:
            assert col in results_df.columns
    
    def test_batch_regression_empty_portfolios(self, regression, sample_factors):
        """Test batch regression with empty portfolios DataFrame"""
        empty_portfolios = pd.DataFrame()
        
        with pytest.raises(ValueError):
            regression.batch_regression(empty_portfolios, sample_factors)
    
    def test_run_regression_mismatched_dates(self, regression):
        """Test regression with mismatched date indices"""
        factors = pd.DataFrame({
            'Mkt-RF': [1.0, 2.0, 3.0],
            'SMB': [0.5, 0.6, 0.7],
            'HML': [0.2, 0.3, 0.4]
        }, index=pd.date_range('2015-01-01', periods=3, freq='MS'))
        
        portfolio = pd.Series(
            [1.5, 2.5, 3.5],
            index=pd.date_range('2015-02-01', periods=3, freq='MS')
        )
        
        # Should automatically align on common dates
        result = regression.run_regression(portfolio, factors)
        assert result is not None
    
    def test_format_regression_output(self, regression, sample_factors, sample_portfolio):
        """Test formatting of regression output"""
        regression.run_regression(sample_portfolio, sample_factors)
        formatted = regression.format_regression_output()
        
        # Should return a string
        assert isinstance(formatted, str)
        
        # Should contain key information
        assert 'Alpha' in formatted or 'alpha' in formatted
        assert 'R-squared' in formatted or 'R²' in formatted
    
    def test_regression_preserves_data(self, regression, sample_factors, sample_portfolio):
        """Test that regression doesn't modify input data"""
        factors_copy = sample_factors.copy()
        portfolio_copy = sample_portfolio.copy()
        
        regression.run_regression(sample_portfolio, sample_factors)
        
        # Input data should be unchanged
        pd.testing.assert_frame_equal(sample_factors, factors_copy)
        pd.testing.assert_series_equal(sample_portfolio, portfolio_copy)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


