"""
Unit tests for data_loader module
"""

import pytest
import pandas as pd
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock
import zipfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import FrenchDataLoader


class TestFrenchDataLoader:
    """Test suite for FrenchDataLoader class"""
    
    @pytest.fixture
    def loader(self):
        """Create a FrenchDataLoader instance"""
        return FrenchDataLoader()
    
    @pytest.fixture
    def mock_csv_content(self):
        """Create mock CSV content in French format"""
        content = """Fama/French 3 Factors
        
        
This file was created by CMPT_ME_BEME_OP_INV_RETS using the 202401 CRSP database.

,Mkt-RF,SMB,HML,RF
201501,-3.05,0.95,2.01,0.00
201502,5.49,0.83,-1.45,0.00
201503,-1.75,-2.12,0.54,0.00
        """
        return content.encode('utf-8')
    
    @pytest.fixture
    def mock_portfolio_content(self):
        """Create mock portfolio CSV content"""
        content = """25 Portfolios Formed on Size and Book-to-Market
        
        
,SMALL LoBM,SMALL HiBM,BIG LoBM,BIG HiBM,ME3 BM3
201501,-4.12,-2.45,-3.01,-2.78,-2.95
201502,6.23,5.89,5.12,5.67,5.45
201503,-2.34,-1.23,-1.65,-1.45,-1.78
        """
        return content.encode('utf-8')
    
    def test_initialization(self, loader):
        """Test FrenchDataLoader initialization"""
        assert loader is not None
        assert hasattr(loader, 'download_three_factors')
        assert hasattr(loader, 'download_portfolios')
    
    @patch('src.data_loader.requests.get')
    def test_download_three_factors_success(self, mock_get, loader, mock_csv_content):
        """Test successful download of three factors"""
        # Create a mock ZIP file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('F-F_Research_Data_Factors.CSV', mock_csv_content)
        zip_buffer.seek(0)
        
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = zip_buffer.read()
        mock_get.return_value = mock_response
        
        # Test download
        df = loader.download_three_factors(
            url="http://test.com/data.zip",
            start_date="2015-01",
            end_date="2015-03"
        )
        
        # Assertions
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert 'Mkt-RF' in df.columns or 'MKT-RF' in df.columns
        assert len(df) > 0
    
    @patch('src.data_loader.requests.get')
    def test_download_network_error(self, mock_get, loader):
        """Test handling of network errors"""
        mock_get.side_effect = Exception("Network error")
        
        with pytest.raises(Exception) as exc_info:
            loader.download_three_factors(
                url="http://test.com/data.zip",
                start_date="2015-01",
                end_date="2015-03"
            )
        
        assert "Network error" in str(exc_info.value) or "Failed to download" in str(exc_info.value)
    
    @patch('src.data_loader.requests.get')
    def test_download_portfolios_success(self, mock_get, loader, mock_portfolio_content):
        """Test successful download of portfolios"""
        # Create a mock ZIP file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('25_Portfolios_5x5.CSV', mock_portfolio_content)
        zip_buffer.seek(0)
        
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = zip_buffer.read()
        mock_get.return_value = mock_response
        
        # Test download
        df = loader.download_portfolios(
            url="http://test.com/portfolios.zip",
            start_date="2015-01",
            end_date="2015-03"
        )
        
        # Assertions
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_parse_french_data(self, loader, mock_csv_content):
        """Test parsing of French CSV format"""
        df = loader._parse_french_data(mock_csv_content, skip_rows=4)
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Three months of data
    
    def test_validate_data_valid(self, loader):
        """Test data validation with valid data"""
        df = pd.DataFrame({
            'Mkt-RF': [1.0, 2.0, 3.0],
            'SMB': [0.5, 0.6, 0.7],
            'HML': [0.2, 0.3, 0.4]
        }, index=pd.date_range('2015-01', periods=3, freq='MS'))
        
        # Should not raise exception
        is_valid = loader._validate_data(df, required_columns=['Mkt-RF', 'SMB', 'HML'])
        assert is_valid is True
    
    def test_validate_data_missing_columns(self, loader):
        """Test data validation with missing columns"""
        df = pd.DataFrame({
            'Mkt-RF': [1.0, 2.0, 3.0],
            'SMB': [0.5, 0.6, 0.7]
        }, index=pd.date_range('2015-01', periods=3, freq='MS'))
        
        with pytest.raises(ValueError) as exc_info:
            loader._validate_data(df, required_columns=['Mkt-RF', 'SMB', 'HML'])
        
        assert 'Missing required columns' in str(exc_info.value)
    
    def test_validate_data_empty_dataframe(self, loader):
        """Test data validation with empty dataframe"""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError) as exc_info:
            loader._validate_data(df, required_columns=['Mkt-RF'])
        
        assert 'empty' in str(exc_info.value).lower()
    
    def test_save_to_csv(self, loader, tmp_path):
        """Test saving data to CSV"""
        df = pd.DataFrame({
            'Mkt-RF': [1.0, 2.0, 3.0],
            'SMB': [0.5, 0.6, 0.7],
            'HML': [0.2, 0.3, 0.4]
        }, index=pd.date_range('2015-01', periods=3, freq='MS'))
        
        filepath = tmp_path / "test_data.csv"
        loader.save_to_csv(df, str(filepath))
        
        assert filepath.exists()
        
        # Read back and verify
        df_read = pd.read_csv(filepath, index_col=0)
        assert len(df_read) == 3
        assert 'Mkt-RF' in df_read.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

