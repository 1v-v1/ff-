"""
Main Entry Point for Fama-French Three-Factor Model Analysis

This script orchestrates the entire analysis pipeline:
1. Load configuration
2. Download data from French Data Library
3. Process and analyze data
4. Run regressions
5. Generate visualizations
6. Create comprehensive report
"""

import logging
import yaml
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import FrenchDataLoader
from src.data_processor import DataProcessor
from src.statistical_analysis import StatisticalAnalyzer
from src.regression_model import FamaFrenchRegression
from src.visualization import Visualizer
from src.report_generator import ReportGenerator


def setup_logging(config: dict) -> None:
    """
    Set up logging configuration
    
    Args:
        config: Configuration dictionary
    """
    log_dir = config['paths']['logs']
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file'], encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Fama-French Three-Factor Model Analysis")
    logger.info("=" * 60)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)


def create_directories(config: dict) -> None:
    """
    Create necessary directories
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config['paths']['data_raw'],
        config['paths']['data_processed'],
        config['paths']['output_figures'],
        config['paths']['output_results'],
        config['paths']['logs']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def main():
    """Main execution function"""
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        
        # Setup logging
        setup_logging(config)
        
        # Create directories
        logger.info("Creating directories...")
        create_directories(config)
        
        # Initialize modules
        logger.info("Initializing modules...")
        data_loader = FrenchDataLoader()
        data_processor = DataProcessor()
        analyzer = StatisticalAnalyzer()
        regression = FamaFrenchRegression()
        visualizer = Visualizer(dpi=config['visualization']['figure_dpi'])
        report_gen = ReportGenerator()
        
        # ===== Step 1: Download Data =====
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Downloading Data from French Data Library")
        logger.info("=" * 60)
        
        # Download three factors
        logger.info("Downloading Fama-French three factors...")
        factors_df = data_loader.download_three_factors(
            url=config['data_sources']['three_factors'],
            start_date=config['time_range']['start_date'],
            end_date=config['time_range']['end_date']
        )
        
        # Save raw factors data
        factors_raw_path = os.path.join(config['paths']['data_raw'], 'ff_three_factors.csv')
        data_loader.save_to_csv(factors_df, factors_raw_path)
        
        # Download 25 portfolios (optional - if fails, we'll use factors themselves)
        logger.info("Downloading 25 Size-BM portfolios...")
        try:
            portfolios_df = data_loader.download_portfolios(
                url=config['data_sources']['portfolios_25'],
                start_date=config['time_range']['start_date'],
                end_date=config['time_range']['end_date']
            )
            
            # Save raw portfolios data
            portfolios_raw_path = os.path.join(config['paths']['data_raw'], 'portfolios_25_size_bm.csv')
            data_loader.save_to_csv(portfolios_df, portfolios_raw_path)
        except Exception as e:
            logger.warning(f"Failed to download portfolio data: {str(e)}")
            logger.info("Will use factor data itself for demonstration purposes")
            # Use factors as demo portfolios
            portfolios_df = factors_df.copy()
            portfolios_df.columns = [f'Factor_{col}' for col in portfolios_df.columns]
        
        # ===== Step 2: Data Processing =====
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Processing Data")
        logger.info("=" * 60)
        
        # Validate data quality
        logger.info("Validating data quality...")
        quality_report = data_processor.validate_data_quality(factors_df)
        logger.info(f"Data quality report: {quality_report}")
        
        # Calculate cumulative returns
        logger.info("Calculating cumulative returns...")
        cumulative_returns = data_processor.calculate_cumulative_returns(factors_df)
        
        # Save processed data
        processed_path = os.path.join(config['paths']['data_processed'], 
                                     f"factors_{config['time_range']['start_date'].replace('-', '')}_"
                                     f"{config['time_range']['end_date'].replace('-', '')}.csv")
        data_loader.save_to_csv(factors_df, processed_path)
        
        # ===== Step 3: Statistical Analysis =====
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Statistical Analysis")
        logger.info("=" * 60)
        
        # Descriptive statistics
        logger.info("Calculating descriptive statistics...")
        stats_summary = analyzer.get_summary_statistics(
            factors_df,
            risk_free_rate=config['analysis']['risk_free_rate']
        )
        logger.info(f"\nSummary Statistics:\n{stats_summary}")
        
        # Correlation matrix
        logger.info("Calculating correlation matrix...")
        corr_matrix = analyzer.correlation_matrix(factors_df)
        logger.info(f"\nCorrelation Matrix:\n{corr_matrix}")
        
        # Save statistics
        stats_path = os.path.join(config['paths']['output_results'], 'descriptive_statistics.csv')
        stats_summary.to_csv(stats_path)
        
        corr_path = os.path.join(config['paths']['output_results'], 'correlation_matrix.csv')
        corr_matrix.to_csv(corr_path)
        
        # ===== Step 4: Regression Analysis =====
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Regression Analysis")
        logger.info("=" * 60)
        
        # Select portfolios for regression
        logger.info("Running regressions on selected portfolios...")
        selected_portfolios = config['analysis']['selected_portfolios']
        
        # Filter available portfolios
        available_portfolios = [p for p in selected_portfolios if p in portfolios_df.columns]
        
        # If no configured portfolios available, use all available columns
        if not available_portfolios:
            logger.info("Configured portfolios not found, using available portfolio columns")
            available_portfolios = portfolios_df.columns[:min(5, len(portfolios_df.columns))].tolist()
        
        logger.info(f"Available portfolios for regression: {available_portfolios}")
        
        if available_portfolios:
            portfolios_subset = portfolios_df[available_portfolios]
            
            regression_results = regression.batch_regression(portfolios_subset, factors_df)
            logger.info(f"\nRegression Results:\n{regression_results}")
            
            # Save regression results
            regression_path = os.path.join(config['paths']['output_results'], 'regression_results.csv')
            regression_results.to_csv(regression_path)
        else:
            logger.warning("No valid portfolios found for regression. Skipping...")
            regression_results = None
        
        # ===== Step 5: Visualization =====
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Creating Visualizations")
        logger.info("=" * 60)
        
        figure_paths = {}
        
        # Time series plot
        logger.info("Creating time series plot...")
        ts_path = os.path.join(config['paths']['output_figures'], 'factors_timeseries.png')
        visualizer.plot_time_series(
            factors_df,
            title='Fama-French Three Factors - Monthly Returns',
            filepath=ts_path
        )
        figure_paths['timeseries'] = ts_path
        
        # Cumulative returns plot
        logger.info("Creating cumulative returns plot...")
        cumret_path = os.path.join(config['paths']['output_figures'], 'cumulative_returns.png')
        visualizer.plot_cumulative_returns(
            cumulative_returns,
            title='Fama-French Three Factors - Cumulative Returns',
            filepath=cumret_path
        )
        figure_paths['cumulative'] = cumret_path
        
        # Correlation heatmap
        logger.info("Creating correlation heatmap...")
        corr_path = os.path.join(config['paths']['output_figures'], 'correlation_heatmap.png')
        visualizer.plot_correlation_heatmap(
            corr_matrix,
            title='Factor Correlation Matrix',
            filepath=corr_path
        )
        figure_paths['correlation'] = corr_path
        
        # Factor comparison charts
        logger.info("Creating factor comparison charts...")
        
        # Annual return comparison
        annual_return_path = os.path.join(config['paths']['output_figures'], 'factor_annual_returns.png')
        visualizer.plot_factor_comparison(
            stats_summary,
            metric='mean_annual',
            title='Annualized Returns by Factor',
            filepath=annual_return_path
        )
        figure_paths['annual_returns'] = annual_return_path
        
        # Sharpe ratio comparison
        sharpe_path = os.path.join(config['paths']['output_figures'], 'factor_sharpe_ratios.png')
        visualizer.plot_factor_comparison(
            stats_summary,
            metric='sharpe_ratio',
            title='Sharpe Ratios by Factor',
            filepath=sharpe_path
        )
        figure_paths['sharpe_ratios'] = sharpe_path
        
        # ===== Step 6: Generate Report =====
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: Generating Analysis Report")
        logger.info("=" * 60)
        
        if regression_results is not None:
            report_path = 'analysis_report.md'
            report_gen.generate_markdown_report(
                stats_summary=stats_summary,
                correlation_matrix=corr_matrix,
                regression_results=regression_results,
                figures=figure_paths,
                time_range={
                    'start': config['time_range']['start_date'],
                    'end': config['time_range']['end_date']
                },
                output_path=report_path
            )
            
            logger.info(f"Report generated successfully: {report_path}")
        else:
            logger.warning("Skipping report generation due to missing regression results")
        
        # ===== Completion =====
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"\nResults saved to:")
        logger.info(f"  - Data: {config['paths']['data_raw']}, {config['paths']['data_processed']}")
        logger.info(f"  - Statistics: {config['paths']['output_results']}")
        logger.info(f"  - Figures: {config['paths']['output_figures']}")
        logger.info(f"  - Report: {report_path if regression_results is not None else 'N/A'}")
        logger.info("\n" + "=" * 60)
        
    except Exception as e:
        logger.error(f"\n{'=' * 60}")
        logger.error(f"ERROR: {str(e)}")
        logger.error(f"{'=' * 60}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()

