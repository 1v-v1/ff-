"""
Regression Model Module

This module implements the Fama-French three-factor regression model.
"""

import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Optional


# Configure logging
logger = logging.getLogger(__name__)


class FamaFrenchRegression:
    """
    Implements Fama-French three-factor regression model
    
    The model is: R_p - R_f = α + β_mkt(R_m - R_f) + β_SMB(SMB) + β_HML(HML) + ε
    """
    
    def __init__(self):
        """Initialize the FamaFrenchRegression"""
        self.model_result = None
        logger.info("FamaFrenchRegression initialized")
    
    def run_regression(
        self,
        portfolio_returns: pd.Series,
        factors_df: pd.DataFrame
    ):
        """
        Run Fama-French three-factor regression
        
        Args:
            portfolio_returns: Series of portfolio returns
            factors_df: DataFrame with factor returns (Mkt-RF, SMB, HML)
        
        Returns:
            Statsmodels regression result object
        
        Raises:
            ValueError: If data is invalid or incompatible
        """
        try:
            logger.info(f"Running regression for portfolio: {portfolio_returns.name}")
            
            # Align data on common dates
            combined = pd.concat([portfolio_returns, factors_df], axis=1, join='inner')
            
            if combined.empty:
                raise ValueError("No overlapping dates between portfolio and factors")
            
            # Separate dependent and independent variables
            y = combined.iloc[:, 0]  # Portfolio returns (first column)
            X = combined.iloc[:, 1:]  # Factor returns (remaining columns)
            
            # Add constant term for alpha
            X = sm.add_constant(X)
            
            # Run OLS regression
            model = sm.OLS(y, X)
            self.model_result = model.fit()
            
            logger.info(f"Regression complete. R²: {self.model_result.rsquared:.4f}")
            logger.debug(f"Coefficients:\n{self.model_result.params}")
            
            return self.model_result
            
        except Exception as e:
            logger.error(f"Error running regression: {str(e)}")
            raise
    
    def extract_results(self) -> Dict:
        """
        Extract key results from the regression
        
        Returns:
            Dictionary with regression results including:
                - alpha: Intercept (excess return)
                - beta_mkt: Market beta
                - beta_smb: SMB beta
                - beta_hml: HML beta
                - r_squared: R-squared value
                - adjusted_r_squared: Adjusted R-squared
                - alpha_pvalue: P-value for alpha
                - alpha_tstat: T-statistic for alpha
        
        Raises:
            ValueError: If regression hasn't been run yet
        """
        if self.model_result is None:
            raise ValueError("No regression results available. Run regression first.")
        
        try:
            # Extract coefficients
            params = self.model_result.params
            pvalues = self.model_result.pvalues
            tvalues = self.model_result.tvalues
            
            # Build results dictionary
            results = {
                'alpha': params['const'],
                'alpha_pvalue': pvalues['const'],
                'alpha_tstat': tvalues['const'],
                'r_squared': self.model_result.rsquared,
                'adjusted_r_squared': self.model_result.rsquared_adj
            }
            
            # Extract factor betas (handle different possible column names)
            if 'Mkt-RF' in params.index:
                results['beta_mkt'] = params['Mkt-RF']
                results['beta_mkt_pvalue'] = pvalues['Mkt-RF']
                results['beta_mkt_tstat'] = tvalues['Mkt-RF']
            elif 'MKT-RF' in params.index:
                results['beta_mkt'] = params['MKT-RF']
                results['beta_mkt_pvalue'] = pvalues['MKT-RF']
                results['beta_mkt_tstat'] = tvalues['MKT-RF']
            
            if 'SMB' in params.index:
                results['beta_smb'] = params['SMB']
                results['beta_smb_pvalue'] = pvalues['SMB']
                results['beta_smb_tstat'] = tvalues['SMB']
            
            if 'HML' in params.index:
                results['beta_hml'] = params['HML']
                results['beta_hml_pvalue'] = pvalues['HML']
                results['beta_hml_tstat'] = tvalues['HML']
            
            logger.debug(f"Extracted results: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting results: {str(e)}")
            raise
    
    def batch_regression(
        self,
        portfolios_df: pd.DataFrame,
        factors_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run regression for multiple portfolios
        
        Args:
            portfolios_df: DataFrame with multiple portfolio returns (columns=portfolios)
            factors_df: DataFrame with factor returns
        
        Returns:
            DataFrame with regression results for each portfolio
        
        Raises:
            ValueError: If portfolios DataFrame is empty
        """
        if portfolios_df.empty:
            raise ValueError("Portfolios DataFrame is empty")
        
        try:
            logger.info(f"Running batch regression for {len(portfolios_df.columns)} portfolios")
            
            results_list = []
            
            for col in portfolios_df.columns:
                portfolio = portfolios_df[col]
                portfolio.name = col
                
                try:
                    # Run regression for this portfolio
                    self.run_regression(portfolio, factors_df)
                    results = self.extract_results()
                    results['portfolio'] = col
                    results_list.append(results)
                    
                except Exception as e:
                    logger.warning(f"Failed to run regression for {col}: {str(e)}")
                    # Add row with NaN values
                    results_list.append({
                        'portfolio': col,
                        'alpha': np.nan,
                        'beta_mkt': np.nan,
                        'beta_smb': np.nan,
                        'beta_hml': np.nan,
                        'r_squared': np.nan
                    })
            
            # Create DataFrame from results
            results_df = pd.DataFrame(results_list)
            results_df = results_df.set_index('portfolio')
            
            logger.info(f"Batch regression complete for {len(results_df)} portfolios")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error in batch regression: {str(e)}")
            raise
    
    def format_regression_output(self) -> str:
        """
        Format regression results as a readable string
        
        Returns:
            Formatted string with regression results
        
        Raises:
            ValueError: If regression hasn't been run yet
        """
        if self.model_result is None:
            raise ValueError("No regression results available. Run regression first.")
        
        try:
            results = self.extract_results()
            
            output = []
            output.append("=" * 60)
            output.append("Fama-French Three-Factor Regression Results")
            output.append("=" * 60)
            output.append("")
            
            # Alpha
            alpha_sig = "***" if results['alpha_pvalue'] < 0.01 else ("**" if results['alpha_pvalue'] < 0.05 else ("*" if results['alpha_pvalue'] < 0.10 else ""))
            output.append(f"Alpha:        {results['alpha']:>10.4f}  (t={results['alpha_tstat']:>6.2f}) {alpha_sig}")
            
            # Betas
            output.append(f"Beta (Mkt-RF): {results['beta_mkt']:>10.4f}  (t={results['beta_mkt_tstat']:>6.2f})")
            output.append(f"Beta (SMB):    {results['beta_smb']:>10.4f}  (t={results['beta_smb_tstat']:>6.2f})")
            output.append(f"Beta (HML):    {results['beta_hml']:>10.4f}  (t={results['beta_hml_tstat']:>6.2f})")
            
            output.append("")
            output.append(f"R²:            {results['r_squared']:>10.4f}")
            output.append(f"Adjusted R²:   {results['adjusted_r_squared']:>10.4f}")
            output.append("")
            output.append("Significance: *** p<0.01, ** p<0.05, * p<0.10")
            output.append("=" * 60)
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Error formatting output: {str(e)}")
            raise


