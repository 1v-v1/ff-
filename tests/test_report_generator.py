"""
Unit tests for report_generator module
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.report_generator import ReportGenerator


class TestReportGenerator:
    """Test suite for ReportGenerator class"""
    
    @pytest.fixture
    def generator(self):
        """Create a ReportGenerator instance"""
        return ReportGenerator()
    
    @pytest.fixture
    def sample_stats(self):
        """Create sample statistics"""
        return pd.DataFrame({
            'mean_monthly': [1.0, 0.2, 0.3],
            'std_monthly': [4.5, 2.1, 2.3],
            'mean_annual': [12.7, 2.4, 3.7],
            'std_annual': [15.6, 7.3, 8.0],
            'sharpe_ratio': [0.81, 0.33, 0.46],
            'min': [-10.5, -5.2, -6.1],
            'max': [12.3, 8.4, 9.2],
            'skewness': [-0.2, 0.1, -0.3],
            'kurtosis': [0.5, 0.2, 0.8]
        }, index=['Mkt-RF', 'SMB', 'HML'])
    
    @pytest.fixture
    def sample_correlation(self):
        """Create sample correlation matrix"""
        return pd.DataFrame({
            'Mkt-RF': [1.0, 0.2, 0.3],
            'SMB': [0.2, 1.0, 0.1],
            'HML': [0.3, 0.1, 1.0]
        }, index=['Mkt-RF', 'SMB', 'HML'])
    
    @pytest.fixture
    def sample_regression(self):
        """Create sample regression results"""
        return pd.DataFrame({
            'alpha': [0.5, -0.2, 0.3],
            'beta_mkt': [1.1, 0.9, 1.2],
            'beta_smb': [0.5, -0.3, 0.8],
            'beta_hml': [0.3, 0.1, -0.2],
            'r_squared': [0.85, 0.78, 0.82],
            'alpha_pvalue': [0.03, 0.15, 0.07]
        }, index=['Portfolio1', 'Portfolio2', 'Portfolio3'])
    
    def test_initialization(self, generator):
        """Test ReportGenerator initialization"""
        assert generator is not None
    
    def test_generate_markdown_report_creates_file(
        self, 
        generator, 
        sample_stats, 
        sample_correlation, 
        sample_regression,
        tmp_path
    ):
        """Test that report generation creates a file"""
        output_path = tmp_path / "test_report.md"
        
        figures = {
            'timeseries': 'output/figures/timeseries.png',
            'cumulative': 'output/figures/cumulative.png',
            'correlation': 'output/figures/correlation.png'
        }
        
        time_range = {'start': '2015-01', 'end': '2024-12'}
        
        result_path = generator.generate_markdown_report(
            stats_summary=sample_stats,
            correlation_matrix=sample_correlation,
            regression_results=sample_regression,
            figures=figures,
            time_range=time_range,
            output_path=str(output_path)
        )
        
        assert Path(result_path).exists()
        assert Path(result_path).stat().st_size > 0
    
    def test_generated_report_contains_expected_sections(
        self,
        generator,
        sample_stats,
        sample_correlation,
        sample_regression,
        tmp_path
    ):
        """Test that generated report contains expected sections"""
        output_path = tmp_path / "test_report.md"
        
        figures = {}
        time_range = {'start': '2015-01', 'end': '2024-12'}
        
        generator.generate_markdown_report(
            stats_summary=sample_stats,
            correlation_matrix=sample_correlation,
            regression_results=sample_regression,
            figures=figures,
            time_range=time_range,
            output_path=str(output_path)
        )
        
        # Read generated report
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for expected sections
        assert '# Fama-French 三因子模型分析报告' in content
        assert '## 执行摘要' in content
        assert '## 描述性统计' in content
        assert '## 相关性分析' in content
        assert '## 回归分析' in content
        assert '## 结论' in content
    
    def test_dataframe_to_markdown(self, generator):
        """Test DataFrame to Markdown conversion"""
        df = pd.DataFrame({
            'A': [1.234, 2.345],
            'B': [3.456, 4.567]
        }, index=['Row1', 'Row2'])
        
        markdown = generator._dataframe_to_markdown(df)
        
        assert isinstance(markdown, str)
        assert 'Row1' in markdown
        assert 'Row2' in markdown
        assert '|' in markdown  # Markdown table separator


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


