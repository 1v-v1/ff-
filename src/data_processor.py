"""
Data Processor Module

This module handles data processing tasks including filtering, transformation,
and validation of Fama-French factor data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


# Configure logging
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processes and transforms Fama-French factor and portfolio data
    """
    
    def __init__(self):
        """Initialize the DataProcessor"""
        logger.info("DataProcessor initialized")
    
    def filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Filter DataFrame by date range
        
        Args:
            df: DataFrame with DatetimeIndex
            start_date: Start date (format: 'YYYY-MM-DD' or 'YYYY-MM')
            end_date: End date (format: 'YYYY-MM-DD' or 'YYYY-MM')
        
        Returns:
            Filtered DataFrame
        
        Raises:
            ValueError: If dates are invalid
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            if start > end:
                raise ValueError(f"Start date {start_date} is after end date {end_date}")
            
            original_len = len(df)
            filtered_df = df[(df.index >= start) & (df.index <= end)]
            
            logger.info(f"Filtered data from {original_len} to {len(filtered_df)} rows")
            logger.info(f"Date range: {start_date} to {end_date}")
            
            if filtered_df.empty:
                logger.warning(f"No data found in date range {start_date} to {end_date}")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering by date range: {str(e)}")
            raise ValueError(f"Failed to filter by date range: {str(e)}")
    
    def calculate_cumulative_returns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate cumulative returns from monthly returns
        
        Args:
            df: DataFrame with monthly returns (in percentage)
        
        Returns:
            DataFrame with cumulative returns (in percentage)
        
        Raises:
            ValueError: If DataFrame is empty
        """
        if df.empty:
            raise ValueError("Cannot calculate cumulative returns for empty DataFrame")
        
        try:
            logger.info("Calculating cumulative returns")
            
            # Convert percentage returns to decimal
            returns_decimal = df / 100.0
            
            # Calculate cumulative returns: (1 + r1) * (1 + r2) * ... - 1
            cumulative_decimal = (1 + returns_decimal).cumprod() - 1
            
            # Convert back to percentage
            cumulative_pct = cumulative_decimal * 100.0
            
            logger.info(f"Calculated cumulative returns for {len(df)} periods")
            logger.debug(f"Final cumulative returns: {cumulative_pct.iloc[-1].to_dict()}")
            
            return cumulative_pct
            
        except Exception as e:
            logger.error(f"Error calculating cumulative returns: {str(e)}")
            raise
    
    def merge_factors_and_portfolios(
        self,
        factors_df: pd.DataFrame,
        portfolios_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge factor data and portfolio data on date index
        
        Args:
            factors_df: DataFrame with factor returns
            portfolios_df: DataFrame with portfolio returns
        
        Returns:
            Merged DataFrame with both factors and portfolios
        """
        try:
            logger.info("Merging factors and portfolios data")
            
            # Inner join on date index (only keep dates present in both)
            merged = pd.merge(
                factors_df,
                portfolios_df,
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            logger.info(f"Merged data: {len(merged)} rows, {len(merged.columns)} columns")
            logger.debug(f"Date range: {merged.index[0]} to {merged.index[-1]}")
            
            if merged.empty:
                logger.warning("Merged DataFrame is empty - no overlapping dates")
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging data: {str(e)}")
            raise
    
    def validate_data_quality(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """
        Validate data quality and return diagnostics
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary with validation results including:
                - has_missing: Boolean indicating if there are missing values
                - missing_count: Total number of missing values
                - missing_by_column: Missing values per column
                - total_rows: Total number of rows
                - total_columns: Total number of columns
        """
        try:
            logger.info("Validating data quality")
            
            total_missing = df.isna().sum().sum()
            missing_by_column = df.isna().sum().to_dict()
            
            result = {
                'has_missing': total_missing > 0,
                'missing_count': int(total_missing),
                'missing_by_column': missing_by_column,
                'total_rows': len(df),
                'total_columns': len(df.columns)
            }
            
            if result['has_missing']:
                logger.warning(f"Data has {total_missing} missing values")
                logger.debug(f"Missing by column: {missing_by_column}")
            else:
                logger.info("Data quality validation passed - no missing values")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            raise


