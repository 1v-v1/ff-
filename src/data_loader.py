"""
Data Loader Module for Fama-French Data

This module handles downloading and loading data from Kenneth French's Data Library.
"""

import logging
import requests
import pandas as pd
import zipfile
from io import BytesIO, StringIO
from typing import Optional
import os


# Configure logging
logger = logging.getLogger(__name__)


class FrenchDataLoader:
    """
    Handles downloading and parsing data from Kenneth French's Data Library
    """
    
    def __init__(self):
        """Initialize the FrenchDataLoader"""
        logger.info("FrenchDataLoader initialized")
    
    def download_three_factors(
        self, 
        url: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Download Fama-French three factors data
        
        Args:
            url: URL to the data ZIP file
            start_date: Start date in format 'YYYY-MM'
            end_date: End date in format 'YYYY-MM'
        
        Returns:
            DataFrame with three factors data
        
        Raises:
            Exception: If download or parsing fails
        """
        logger.info(f"Downloading three factors data from {url}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        try:
            # Download the ZIP file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Successfully downloaded data, size: {len(response.content)} bytes")
            
            # Extract and parse CSV from ZIP
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                # Find the CSV file in the ZIP
                csv_files = [f for f in zf.namelist() if f.endswith('.CSV') or f.endswith('.csv')]
                if not csv_files:
                    raise ValueError("No CSV file found in ZIP archive")
                
                csv_file = csv_files[0]
                logger.debug(f"Extracting file: {csv_file}")
                
                csv_content = zf.read(csv_file)
            
            # Parse the French format CSV
            df = self._parse_french_data(csv_content, skip_rows=4)
            
            # Validate the data (check that we have the main factor columns)
            # Column names may vary in case, so we check flexibly
            df_cols_upper = [col.upper() for col in df.columns]
            if 'MKT-RF' not in df_cols_upper and 'Mkt-RF'.upper() not in df_cols_upper:
                logger.warning(f"Expected factor columns not found. Available columns: {df.columns.tolist()}")
            
            # Filter by date range
            df = self._filter_by_date(df, start_date, end_date)
            
            logger.info(f"Successfully parsed {len(df)} rows of three factors data")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download data from {url}: {str(e)}")
            raise Exception(f"Failed to download data: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing three factors data: {str(e)}")
            raise
    
    def download_portfolios(
        self, 
        url: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Download portfolio data (e.g., 25 Size-BM portfolios)
        
        Args:
            url: URL to the data ZIP file
            start_date: Start date in format 'YYYY-MM'
            end_date: End date in format 'YYYY-MM'
        
        Returns:
            DataFrame with portfolio returns
        
        Raises:
            Exception: If download or parsing fails
        """
        logger.info(f"Downloading portfolio data from {url}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        try:
            # Download the ZIP file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Successfully downloaded data, size: {len(response.content)} bytes")
            
            # Extract and parse CSV from ZIP
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                csv_files = [f for f in zf.namelist() if f.endswith('.CSV') or f.endswith('.csv')]
                if not csv_files:
                    raise ValueError("No CSV file found in ZIP archive")
                
                csv_file = csv_files[0]
                logger.debug(f"Extracting file: {csv_file}")
                
                csv_content = zf.read(csv_file)
            
            # Parse the French format CSV
            df = self._parse_french_data(csv_content, skip_rows=3)
            
            # Filter by date range
            df = self._filter_by_date(df, start_date, end_date)
            
            logger.info(f"Successfully parsed {len(df)} rows of portfolio data with {len(df.columns)} portfolios")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download data from {url}: {str(e)}")
            raise Exception(f"Failed to download data: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing portfolio data: {str(e)}")
            raise
    
    def _parse_french_data(
        self, 
        csv_content: bytes, 
        skip_rows: int = 4
    ) -> pd.DataFrame:
        """
        Parse French Data Library CSV format
        
        The French data files have a special format with header rows that need to be skipped.
        
        Args:
            csv_content: Raw CSV content as bytes
            skip_rows: Number of header rows to skip
        
        Returns:
            Parsed DataFrame with date index
        """
        try:
            # Decode content
            content_str = csv_content.decode('utf-8', errors='ignore')
            
            # Try to find the correct header row by looking for column names
            df = None
            for skip in range(0, 10):  # Try different skip values
                try:
                    # Read CSV
                    df_temp = pd.read_csv(
                        StringIO(content_str),
                        skiprows=skip,
                        index_col=0,
                        skipinitialspace=True,
                        on_bad_lines='skip'
                    )
                    
                    # Check if columns look like factor names or numeric data
                    cols_str = ' '.join(str(c).upper() for c in df_temp.columns)
                    
                    # If we see factor-like names, this is the header
                    if any(name in cols_str for name in ['MKT-RF', 'MKT_RF', 'MKTRF', 'SMB', 'HML', 'RF']):
                        # This looks like a valid header
                        date_series = df_temp.index.astype(str).str.strip()
                        mask = date_series.str.match(r'^\d{6}$', na=False)
                        df_filtered = df_temp[mask]
                        
                        if len(df_filtered) > 0:
                            df = df_filtered
                            date_series = date_series[mask]
                            logger.debug(f"Found header at skip_rows={skip}")
                            break
                            
                except Exception as e:
                    logger.debug(f"Skip={skip} failed: {str(e)}")
                    continue
            
            if df.empty or len(df) == 0:
                raise ValueError("No valid monthly data found after parsing")
            
            # Convert date to datetime and set as index
            df.index = pd.to_datetime(date_series, format='%Y%m')
            df.index.name = 'date'
            
            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with all NaN
            df = df.dropna(how='all')
            
            logger.debug(f"Parsed {len(df)} rows and {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse French data: {str(e)}")
            raise ValueError(f"Failed to parse French data format: {str(e)}")
    
    def _validate_data(
        self, 
        df: pd.DataFrame, 
        required_columns: Optional[list] = None
    ) -> bool:
        """
        Validate the downloaded data
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If validation fails
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")
        
        if required_columns:
            # Normalize column names for comparison (case-insensitive)
            df_cols_upper = [col.upper() for col in df.columns]
            required_upper = [col.upper() for col in required_columns]
            
            missing = [col for col in required_upper if col not in df_cols_upper]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
        
        logger.debug(f"Data validation passed: {len(df)} rows, {len(df.columns)} columns")
        return True
    
    def _filter_by_date(
        self, 
        df: pd.DataFrame, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Filter DataFrame by date range
        
        Args:
            df: DataFrame with DatetimeIndex
            start_date: Start date in format 'YYYY-MM'
            end_date: End date in format 'YYYY-MM'
        
        Returns:
            Filtered DataFrame
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            original_len = len(df)
            df_filtered = df[(df.index >= start) & (df.index <= end)]
            
            logger.info(f"Filtered data from {original_len} to {len(df_filtered)} rows")
            
            if df_filtered.empty:
                logger.warning(f"No data found in date range {start_date} to {end_date}")
            
            return df_filtered
            
        except Exception as e:
            logger.error(f"Error filtering by date: {str(e)}")
            raise
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            filepath: Path to save CSV file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            df.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {str(e)}")
            raise

