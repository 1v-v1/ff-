"""
Statistical Analysis Module

This module provides statistical analysis functions for Fama-French factor data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Union


# Configure logging
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Performs statistical analysis on financial data
    """
    
    def __init__(self):
        """Initialize the StatisticalAnalyzer"""
        logger.info("StatisticalAnalyzer initialized")
    
    def descriptive_statistics(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate descriptive statistics for all columns
        
        Args:
            df: DataFrame with returns data
        
        Returns:
            DataFrame with statistics (rows=variables, cols=statistics)
        
        Raises:
            ValueError: If DataFrame is empty
        """
        if df.empty:
            raise ValueError("Cannot calculate statistics for empty DataFrame")
        
        try:
            logger.info("Calculating descriptive statistics")
            
            stats_dict = {
                'mean': df.mean(),
                'std': df.std(),
                'min': df.min(),
                'max': df.max(),
                'skewness': df.skew(),
                'kurtosis': df.kurtosis()
            }
            
            stats_df = pd.DataFrame(stats_dict)
            
            logger.info(f"Calculated statistics for {len(df.columns)} variables")
            logger.debug(f"Statistics:\n{stats_df}")
            
            return stats_df
            
        except Exception as e:
            logger.error(f"Error calculating descriptive statistics: {str(e)}")
            raise
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate annualized Sharpe ratio
        
        Args:
            returns: Series of monthly returns (in percentage)
            risk_free_rate: Annual risk-free rate (in percentage)
        
        Returns:
            Annualized Sharpe ratio
        """
        try:
            # Convert to decimal
            returns_decimal = returns / 100.0
            rf_monthly_decimal = risk_free_rate / 100.0 / 12.0
            
            # Calculate excess returns
            excess_returns = returns_decimal - rf_monthly_decimal
            
            # Annualize
            mean_annual = excess_returns.mean() * 12
            std_annual = excess_returns.std() * np.sqrt(12)
            
            # Calculate Sharpe ratio
            if std_annual == 0:
                sharpe = 0.0
            else:
                sharpe = mean_annual / std_annual
            
            logger.debug(f"Sharpe ratio: {sharpe:.4f}")
            
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            raise
    
    def correlation_matrix(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix
        
        Args:
            df: DataFrame with returns data
        
        Returns:
            Correlation matrix
        """
        try:
            logger.info("Calculating correlation matrix")
            
            corr = df.corr()
            
            logger.info(f"Calculated correlation for {len(df.columns)} variables")
            logger.debug(f"Correlation matrix:\n{corr}")
            
            return corr
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            raise
    
    def annualize_returns(
        self,
        monthly_mean: float
    ) -> float:
        """
        Annualize monthly returns using compound formula
        
        Args:
            monthly_mean: Mean monthly return (in percentage)
        
        Returns:
            Annualized return (in percentage)
        """
        # Convert to decimal
        monthly_decimal = monthly_mean / 100.0
        
        # Compound: (1 + r)^12 - 1
        annual_decimal = (1 + monthly_decimal) ** 12 - 1
        
        # Convert back to percentage
        annual_pct = annual_decimal * 100.0
        
        return annual_pct
    
    def annualize_volatility(
        self,
        monthly_std: float
    ) -> float:
        """
        Annualize monthly volatility
        
        Args:
            monthly_std: Monthly standard deviation (in percentage)
        
        Returns:
            Annualized volatility (in percentage)
        """
        # Volatility scales with sqrt of time
        annual_std = monthly_std * np.sqrt(12)
        
        return annual_std
    
    def get_summary_statistics(
        self,
        df: pd.DataFrame,
        risk_free_rate: float = 0.0
    ) -> pd.DataFrame:
        """
        Get comprehensive summary statistics
        
        Args:
            df: DataFrame with returns data (in percentage)
            risk_free_rate: Annual risk-free rate (in percentage)
        
        Returns:
            DataFrame with comprehensive statistics
        """
        try:
            logger.info("Generating comprehensive summary statistics")
            
            summary_dict = {}
            
            for col in df.columns:
                series = df[col]
                
                # Monthly statistics
                mean_monthly = series.mean()
                std_monthly = series.std()
                
                # Annualized statistics
                mean_annual = self.annualize_returns(mean_monthly)
                std_annual = self.annualize_volatility(std_monthly)
                
                # Sharpe ratio
                sharpe = self.calculate_sharpe_ratio(series, risk_free_rate)
                
                # Other statistics
                min_val = series.min()
                max_val = series.max()
                skewness = series.skew()
                kurtosis = series.kurtosis()
                
                summary_dict[col] = {
                    'mean_monthly': mean_monthly,
                    'std_monthly': std_monthly,
                    'mean_annual': mean_annual,
                    'std_annual': std_annual,
                    'sharpe_ratio': sharpe,
                    'min': min_val,
                    'max': max_val,
                    'skewness': skewness,
                    'kurtosis': kurtosis
                }
            
            summary_df = pd.DataFrame(summary_dict).T
            
            logger.info("Summary statistics generation complete")
            logger.debug(f"Summary:\n{summary_df}")
            
            return summary_df
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {str(e)}")
            raise


