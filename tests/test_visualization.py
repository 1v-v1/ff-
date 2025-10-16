"""
Unit tests for visualization module
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visualization import Visualizer


class TestVisualizer:
    """Test suite for Visualizer class"""
    
    @pytest.fixture
    def visualizer(self):
        """Create a Visualizer instance"""
        return Visualizer(dpi=100)  # Lower DPI for faster tests
    
    @pytest.fixture
    def sample_timeseries(self):
        """Create sample time series data"""
        dates = pd.date_range('2015-01-01', periods=12, freq='MS')
        data = {
            'Factor1': np.random.randn(12),
            'Factor2': np.random.randn(12)
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def sample_correlation(self):
        """Create sample correlation matrix"""
        corr_data = np.array([[1.0, 0.5, 0.3],
                             [0.5, 1.0, 0.4],
                             [0.3, 0.4, 1.0]])
        return pd.DataFrame(corr_data, 
                          index=['Factor1', 'Factor2', 'Factor3'],
                          columns=['Factor1', 'Factor2', 'Factor3'])
    
    def test_initialization(self, visualizer):
        """Test Visualizer initialization"""
        assert visualizer is not None
        assert visualizer.dpi == 100
    
    def test_plot_time_series_creates_file(self, visualizer, sample_timeseries, tmp_path):
        """Test that time series plot creates a file"""
        filepath = tmp_path / "test_timeseries.png"
        
        visualizer.plot_time_series(
            sample_timeseries,
            title="Test Time Series",
            filepath=str(filepath)
        )
        
        assert filepath.exists()
        assert filepath.stat().st_size > 0
    
    def test_plot_cumulative_returns_creates_file(self, visualizer, sample_timeseries, tmp_path):
        """Test that cumulative returns plot creates a file"""
        filepath = tmp_path / "test_cumulative.png"
        
        visualizer.plot_cumulative_returns(
            sample_timeseries,
            title="Test Cumulative Returns",
            filepath=str(filepath)
        )
        
        assert filepath.exists()
        assert filepath.stat().st_size > 0
    
    def test_plot_correlation_heatmap_creates_file(self, visualizer, sample_correlation, tmp_path):
        """Test that correlation heatmap creates a file"""
        filepath = tmp_path / "test_heatmap.png"
        
        visualizer.plot_correlation_heatmap(
            sample_correlation,
            title="Test Correlation",
            filepath=str(filepath)
        )
        
        assert filepath.exists()
        assert filepath.stat().st_size > 0
    
    def test_plot_distribution_creates_file(self, visualizer, tmp_path):
        """Test that distribution plot creates a file"""
        data = pd.Series(np.random.randn(100))
        filepath = tmp_path / "test_distribution.png"
        
        visualizer.plot_distribution(
            data,
            title="Test Distribution",
            filepath=str(filepath)
        )
        
        assert filepath.exists()
        assert filepath.stat().st_size > 0
    
    def test_plot_regression_fit_creates_file(self, visualizer, tmp_path):
        """Test that regression fit plot creates a file"""
        y_actual = pd.Series(np.random.randn(50))
        y_fitted = y_actual + np.random.randn(50) * 0.1
        filepath = tmp_path / "test_regression.png"
        
        visualizer.plot_regression_fit(
            y_actual,
            y_fitted,
            title="Test Regression Fit",
            filepath=str(filepath)
        )
        
        assert filepath.exists()
        assert filepath.stat().st_size > 0
    
    def test_plot_factor_comparison_creates_file(self, visualizer, tmp_path):
        """Test that factor comparison plot creates a file"""
        stats = pd.DataFrame({
            'mean_annual': [8.5, 3.2, 4.1]
        }, index=['Mkt-RF', 'SMB', 'HML'])
        
        filepath = tmp_path / "test_comparison.png"
        
        visualizer.plot_factor_comparison(
            stats,
            metric='mean_annual',
            title="Test Factor Comparison",
            filepath=str(filepath)
        )
        
        assert filepath.exists()
        assert filepath.stat().st_size > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


