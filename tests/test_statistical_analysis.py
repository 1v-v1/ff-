"""
Unit tests for statistical_analysis module
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.statistical_analysis import StatisticalAnalyzer


class TestStatisticalAnalyzer:
    """Test suite for StatisticalAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a StatisticalAnalyzer instance"""
        return StatisticalAnalyzer()
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample monthly returns data (in percentage)"""
        dates = pd.date_range('2015-01-01', periods=12, freq='MS')
        data = {
            'Factor1': [2.0, -1.0, 3.0, 1.5, -0.5, 2.5, -1.5, 1.0, 0.5, 2.0, -1.0, 1.5],
            'Factor2': [1.0, 0.5, -0.5, 1.5, 0.0, 1.0, -0.5, 0.5, 1.0, -0.5, 1.5, 0.5]
        }
        return pd.DataFrame(data, index=dates)
    
    def test_initialization(self, analyzer):
        """Test StatisticalAnalyzer initialization"""
        assert analyzer is not None
        assert hasattr(analyzer, 'descriptive_statistics')
        assert hasattr(analyzer, 'calculate_sharpe_ratio')
        assert hasattr(analyzer, 'correlation_matrix')
    
    def test_descriptive_statistics_basic(self, analyzer, sample_returns):
        """Test basic descriptive statistics"""
        stats = analyzer.descriptive_statistics(sample_returns)
        
        assert 'Factor1' in stats.index
        assert 'Factor2' in stats.index
        assert 'mean' in stats.columns
        assert 'std' in stats.columns
        assert 'min' in stats.columns
        assert 'max' in stats.columns
        
        # Check that statistics are reasonable
        assert stats.loc['Factor1', 'mean'] > 0  # Positive average return
        assert stats.loc['Factor1', 'std'] > 0   # Positive std dev
    
    def test_descriptive_statistics_includes_skewness_kurtosis(self, analyzer, sample_returns):
        """Test that skewness and kurtosis are calculated"""
        stats = analyzer.descriptive_statistics(sample_returns)
        
        assert 'skewness' in stats.columns
        assert 'kurtosis' in stats.columns
    
    def test_descriptive_statistics_empty_dataframe(self, analyzer):
        """Test descriptive statistics with empty dataframe"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            analyzer.descriptive_statistics(empty_df)
    
    def test_calculate_sharpe_ratio_simple(self, analyzer):
        """Test Sharpe ratio calculation with simple data"""
        # Returns: 12% annual, 10% std dev, 0% risk-free rate
        # Sharpe = 12 / 10 = 1.2
        dates = pd.date_range('2015-01-01', periods=12, freq='MS')
        monthly_return = 1.0  # 1% per month = 12% annual
        monthly_std = 2.887  # Should give ~10% annual std
        
        returns = pd.Series([monthly_return] * 12, index=dates)
        
        sharpe = analyzer.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        
        # Sharpe should be positive
        assert sharpe > 0
    
    def test_calculate_sharpe_ratio_negative_returns(self, analyzer):
        """Test Sharpe ratio with negative returns"""
        dates = pd.date_range('2015-01-01', periods=12, freq='MS')
        returns = pd.Series([-1.0] * 12, index=dates)  # Negative returns
        
        sharpe = analyzer.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        
        # Sharpe should be negative for negative returns
        assert sharpe < 0
    
    def test_calculate_sharpe_ratio_with_risk_free_rate(self, analyzer):
        """Test Sharpe ratio with non-zero risk-free rate"""
        dates = pd.date_range('2015-01-01', periods=12, freq='MS')
        returns = pd.Series([2.0] * 12, index=dates)  # 2% per month
        
        sharpe_no_rf = analyzer.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_with_rf = analyzer.calculate_sharpe_ratio(returns, risk_free_rate=1.0)
        
        # With risk-free rate, excess return is lower, so Sharpe is lower
        assert sharpe_with_rf < sharpe_no_rf
    
    def test_correlation_matrix(self, analyzer, sample_returns):
        """Test correlation matrix calculation"""
        corr = analyzer.correlation_matrix(sample_returns)
        
        assert corr.shape == (2, 2)  # 2 factors
        assert 'Factor1' in corr.index
        assert 'Factor2' in corr.index
        assert 'Factor1' in corr.columns
        assert 'Factor2' in corr.columns
        
        # Diagonal should be 1.0
        assert np.isclose(corr.loc['Factor1', 'Factor1'], 1.0)
        assert np.isclose(corr.loc['Factor2', 'Factor2'], 1.0)
        
        # Matrix should be symmetric
        assert np.isclose(corr.loc['Factor1', 'Factor2'], corr.loc['Factor2', 'Factor1'])
    
    def test_correlation_matrix_single_column(self, analyzer):
        """Test correlation matrix with single column"""
        dates = pd.date_range('2015-01-01', periods=5, freq='MS')
        data = pd.DataFrame({'Factor1': [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
        
        corr = analyzer.correlation_matrix(data)
        
        assert corr.shape == (1, 1)
        assert np.isclose(corr.iloc[0, 0], 1.0)
    
    def test_annualize_returns(self, analyzer):
        """Test annualization of monthly returns"""
        monthly_mean = 1.0  # 1% per month
        annual = analyzer.annualize_returns(monthly_mean)
        
        # Annualized should be approximately 12 * 1.0 = 12.0 (simple)
        # Or (1.01)^12 - 1 = 12.68% (compound)
        assert annual > 12.0
        assert annual < 13.0
    
    def test_annualize_volatility(self, analyzer):
        """Test annualization of monthly volatility"""
        monthly_std = 2.0  # 2% per month
        annual = analyzer.annualize_volatility(monthly_std)
        
        # Annualized should be 2.0 * sqrt(12) = 6.93
        expected = 2.0 * np.sqrt(12)
        assert np.isclose(annual, expected, rtol=0.01)
    
    def test_get_summary_statistics(self, analyzer, sample_returns):
        """Test comprehensive summary statistics"""
        summary = analyzer.get_summary_statistics(sample_returns, risk_free_rate=0.0)
        
        assert 'Factor1' in summary.index
        assert 'Factor2' in summary.index
        
        # Check for all expected statistics
        expected_columns = [
            'mean_monthly', 'std_monthly', 'mean_annual', 'std_annual',
            'sharpe_ratio', 'min', 'max', 'skewness', 'kurtosis'
        ]
        
        for col in expected_columns:
            assert col in summary.columns, f"Missing column: {col}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


