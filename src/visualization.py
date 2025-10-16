"""
Visualization Module

This module provides visualization functions for Fama-French factor analysis.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import os


# Configure logging
logger = logging.getLogger(__name__)


class Visualizer:
    """
    Creates visualizations for factor and portfolio data
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', dpi: int = 300):
        """
        Initialize the Visualizer
        
        Args:
            style: Matplotlib style to use
            dpi: Figure DPI for saved images
        """
        self.dpi = dpi
        self._set_plot_style(style)
        logger.info(f"Visualizer initialized with style={style}, dpi={dpi}")
    
    def _set_plot_style(self, style: str) -> None:
        """
        Set the plotting style
        
        Args:
            style: Matplotlib style name
        """
        try:
            plt.style.use(style)
            sns.set_palette("husl")
        except Exception as e:
            logger.warning(f"Could not set style '{style}': {str(e)}. Using default.")
            sns.set_theme()
    
    def plot_time_series(
        self,
        df: pd.DataFrame,
        title: str,
        filepath: str,
        figsize: Tuple[int, int] = (12, 6),
        ylabel: str = "Returns (%)"
    ) -> None:
        """
        Plot time series of multiple series
        
        Args:
            df: DataFrame with time series data
            title: Plot title
            filepath: Path to save figure
            figsize: Figure size (width, height)
            ylabel: Y-axis label
        """
        try:
            logger.info(f"Creating time series plot: {title}")
            
            fig, ax = plt.subplots(figsize=figsize)
            
            for col in df.columns:
                ax.plot(df.index, df[col], label=col, linewidth=1.5)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.legend(loc='best', frameon=True, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Time series plot saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {str(e)}")
            raise
    
    def plot_cumulative_returns(
        self,
        df: pd.DataFrame,
        title: str,
        filepath: str,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot cumulative returns over time
        
        Args:
            df: DataFrame with cumulative returns
            title: Plot title
            filepath: Path to save figure
            figsize: Figure size
        """
        try:
            logger.info(f"Creating cumulative returns plot: {title}")
            
            fig, ax = plt.subplots(figsize=figsize)
            
            for col in df.columns:
                ax.plot(df.index, df[col], label=col, linewidth=2)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Cumulative Returns (%)', fontsize=12)
            ax.legend(loc='best', frameon=True, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Cumulative returns plot saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error creating cumulative returns plot: {str(e)}")
            raise
    
    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        title: str,
        filepath: str,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot correlation matrix heatmap
        
        Args:
            corr_matrix: Correlation matrix
            title: Plot title
            filepath: Path to save figure
            figsize: Figure size
        """
        try:
            logger.info(f"Creating correlation heatmap: {title}")
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create heatmap
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt='.3f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Correlation heatmap saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            raise
    
    def plot_distribution(
        self,
        series: pd.Series,
        title: str,
        filepath: str,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot distribution (histogram + KDE)
        
        Args:
            series: Data series
            title: Plot title
            filepath: Path to save figure
            figsize: Figure size
        """
        try:
            logger.info(f"Creating distribution plot: {title}")
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Histogram with KDE
            series.hist(bins=30, alpha=0.6, ax=ax, density=True, edgecolor='black')
            series.plot(kind='kde', ax=ax, linewidth=2, color='red')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Returns (%)', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=series.mean(), color='green', linestyle='--', 
                      linewidth=2, label=f'Mean: {series.mean():.2f}%')
            ax.legend(fontsize=10)
            
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Distribution plot saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}")
            raise
    
    def plot_regression_fit(
        self,
        y_actual: pd.Series,
        y_fitted: pd.Series,
        title: str,
        filepath: str,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot actual vs fitted values from regression
        
        Args:
            y_actual: Actual values
            y_fitted: Fitted values from regression
            title: Plot title
            filepath: Path to save figure
            figsize: Figure size
        """
        try:
            logger.info(f"Creating regression fit plot: {title}")
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.scatter(y_fitted, y_actual, alpha=0.6, s=50)
            
            # Add 45-degree line
            min_val = min(y_actual.min(), y_fitted.min())
            max_val = max(y_actual.max(), y_fitted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='Perfect Fit')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Fitted Values', fontsize=12)
            ax.set_ylabel('Actual Values', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Regression fit plot saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error creating regression fit plot: {str(e)}")
            raise
    
    def plot_factor_comparison(
        self,
        stats_df: pd.DataFrame,
        metric: str,
        title: str,
        filepath: str,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot bar chart comparing factors on a specific metric
        
        Args:
            stats_df: DataFrame with statistics (index=factors, columns=metrics)
            metric: Which metric to plot
            title: Plot title
            filepath: Path to save figure
            figsize: Figure size
        """
        try:
            logger.info(f"Creating factor comparison plot: {title}")
            
            fig, ax = plt.subplots(figsize=figsize)
            
            data = stats_df[metric]
            colors = plt.cm.Set3(range(len(data)))
            
            bars = ax.bar(range(len(data)), data, color=colors, edgecolor='black', linewidth=1.2)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha='right')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, data)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}',
                       ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Factor comparison plot saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error creating factor comparison plot: {str(e)}")
            raise


