"""
Report Generator Module

This module generates comprehensive Markdown reports for Fama-French analysis.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import os


# Configure logging
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates Markdown reports for analysis results
    """
    
    def __init__(self):
        """Initialize the ReportGenerator"""
        logger.info("ReportGenerator initialized")
    
    def generate_markdown_report(
        self,
        stats_summary: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        regression_results: pd.DataFrame,
        figures: Dict[str, str],
        time_range: Dict[str, str],
        output_path: str
    ) -> str:
        """
        Generate comprehensive Markdown report
        
        Args:
            stats_summary: Summary statistics DataFrame
            correlation_matrix: Correlation matrix
            regression_results: Regression results DataFrame
            figures: Dictionary mapping figure names to file paths
            time_range: Dictionary with 'start' and 'end' dates
            output_path: Path to save the report
        
        Returns:
            Path to generated report
        """
        try:
            logger.info(f"Generating Markdown report to {output_path}")
            
            # Build report sections
            sections = []
            
            # Header
            sections.append(self._create_header(time_range))
            
            # Table of Contents
            sections.append(self._create_toc())
            
            # Executive Summary
            sections.append(self._create_executive_summary(stats_summary))
            
            # Descriptive Statistics
            sections.append(self._create_statistics_section(stats_summary))
            
            # Correlation Analysis
            sections.append(self._create_correlation_section(correlation_matrix))
            
            # Time Series Analysis
            sections.append(self._create_timeseries_section(figures))
            
            # Regression Analysis
            sections.append(self._create_regression_section(regression_results))
            
            # Visualizations
            sections.append(self._create_visualizations_section(figures))
            
            # Conclusions
            sections.append(self._create_conclusions(stats_summary, regression_results))
            
            # Footer
            sections.append(self._create_footer())
            
            # Combine all sections
            report_content = "\n\n".join(sections)
            
            # Save report
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create directory if path contains a directory
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Report successfully generated at {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    def _create_header(self, time_range: Dict[str, str]) -> str:
        """Create report header"""
        header = f"""# Fama-French 三因子模型分析报告

**分析时间范围**: {time_range['start']} 至 {time_range['end']}  
**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
"""
        return header
    
    def _create_toc(self) -> str:
        """Create table of contents"""
        toc = """## 目录

1. [执行摘要](#执行摘要)
2. [描述性统计](#描述性统计)
3. [相关性分析](#相关性分析)
4. [时间序列分析](#时间序列分析)
5. [回归分析](#回归分析)
6. [可视化图表](#可视化图表)
7. [结论](#结论)

---
"""
        return toc
    
    def _create_executive_summary(self, stats_summary: pd.DataFrame) -> str:
        """Create executive summary"""
        summary = """## 执行摘要

本报告基于 Kenneth French 数据库的三因子数据，对市场超额收益 (Mkt-RF)、规模因子 (SMB) 和价值因子 (HML) 进行了全面分析。

### 关键发现

"""
        
        # Add key findings from statistics
        for factor in stats_summary.index:
            mean_annual = stats_summary.loc[factor, 'mean_annual']
            std_annual = stats_summary.loc[factor, 'std_annual']
            sharpe = stats_summary.loc[factor, 'sharpe_ratio']
            
            summary += f"- **{factor}**: "
            summary += f"年化收益率 {mean_annual:.2f}%, "
            summary += f"年化波动率 {std_annual:.2f}%, "
            summary += f"夏普比率 {sharpe:.3f}\n"
        
        summary += "\n---\n"
        
        return summary
    
    def _create_statistics_section(self, stats_summary: pd.DataFrame) -> str:
        """Create descriptive statistics section"""
        section = """## 描述性统计

### 月度统计指标

下表展示了三因子的月度统计特征：

"""
        
        # Create monthly statistics table
        monthly_stats = stats_summary[['mean_monthly', 'std_monthly', 'min', 'max', 'skewness', 'kurtosis']]
        section += self._dataframe_to_markdown(monthly_stats)
        section += "\n"
        
        section += """
### 年化统计指标

下表展示了年化后的收益率、波动率和夏普比率：

"""
        
        # Create annual statistics table
        annual_stats = stats_summary[['mean_annual', 'std_annual', 'sharpe_ratio']]
        section += self._dataframe_to_markdown(annual_stats)
        section += "\n"
        
        section += """
### 统计解读

"""
        
        # Add interpretation
        for factor in stats_summary.index:
            skew = stats_summary.loc[factor, 'skewness']
            kurt = stats_summary.loc[factor, 'kurtosis']
            
            section += f"- **{factor}**: "
            if abs(skew) < 0.5:
                section += "分布接近对称"
            elif skew > 0:
                section += f"右偏分布（偏度={skew:.2f}）"
            else:
                section += f"左偏分布（偏度={skew:.2f}）"
            
            if abs(kurt) < 1:
                section += "，接近正态分布"
            elif kurt > 1:
                section += f"，厚尾分布（峰度={kurt:.2f}）"
            else:
                section += f"，薄尾分布（峰度={kurt:.2f}）"
            section += "\n"
        
        section += "\n---\n"
        
        return section
    
    def _create_correlation_section(self, correlation_matrix: pd.DataFrame) -> str:
        """Create correlation analysis section"""
        section = """## 相关性分析

### 因子相关性矩阵

下表展示了三因子之间的相关系数：

"""
        
        section += self._dataframe_to_markdown(correlation_matrix)
        section += "\n"
        
        section += """
### 相关性解读

"""
        
        # Analyze correlations
        factors = correlation_matrix.index.tolist()
        for i in range(len(factors)):
            for j in range(i+1, len(factors)):
                corr = correlation_matrix.iloc[i, j]
                section += f"- **{factors[i]} vs {factors[j]}**: 相关系数 {corr:.3f}"
                
                if abs(corr) < 0.3:
                    section += " (弱相关)"
                elif abs(corr) < 0.7:
                    section += " (中等相关)"
                else:
                    section += " (强相关)"
                section += "\n"
        
        section += "\n---\n"
        
        return section
    
    def _create_timeseries_section(self, figures: Dict[str, str]) -> str:
        """Create time series analysis section"""
        section = """## 时间序列分析

### 因子收益率时间序列

下图展示了三因子月度收益率的时间序列变化：

"""
        
        if 'timeseries' in figures:
            section += f"![因子时间序列]({figures['timeseries']})\n\n"
        
        section += """
### 累计收益率

下图展示了三因子的累计收益率曲线：

"""
        
        if 'cumulative' in figures:
            section += f"![累计收益率]({figures['cumulative']})\n\n"
        
        section += "---\n"
        
        return section
    
    def _create_regression_section(self, regression_results: pd.DataFrame) -> str:
        """Create regression analysis section"""
        section = """## 回归分析

### 投资组合回归结果

使用 Fama-French 三因子模型对选定投资组合进行回归分析，模型形式为：

R_p - R_f = α + β_mkt(R_m - R_f) + β_SMB(SMB) + β_HML(HML) + ε

回归结果如下：

"""
        
        # Display regression results
        display_cols = ['alpha', 'beta_mkt', 'beta_smb', 'beta_hml', 'r_squared', 'alpha_pvalue']
        available_cols = [col for col in display_cols if col in regression_results.columns]
        
        section += self._dataframe_to_markdown(regression_results[available_cols])
        section += "\n"
        
        section += """
### 回归解读

"""
        
        # Interpret regression results
        for portfolio in regression_results.index:
            row = regression_results.loc[portfolio]
            alpha = row['alpha']
            alpha_pval = row['alpha_pvalue']
            r_squared = row['r_squared']
            
            section += f"- **{portfolio}**: "
            section += f"Alpha = {alpha:.4f}"
            
            if alpha_pval < 0.01:
                section += " (***显著)"
            elif alpha_pval < 0.05:
                section += " (**显著)"
            elif alpha_pval < 0.10:
                section += " (*显著)"
            else:
                section += " (不显著)"
            
            section += f", R² = {r_squared:.4f}"
            section += "\n"
        
        section += "\n---\n"
        
        return section
    
    def _create_visualizations_section(self, figures: Dict[str, str]) -> str:
        """Create visualizations section"""
        section = """## 可视化图表

### 相关性热图

"""
        
        if 'correlation' in figures:
            section += f"![相关性热图]({figures['correlation']})\n\n"
        
        section += "---\n"
        
        return section
    
    def _create_conclusions(self, stats_summary: pd.DataFrame, regression_results: pd.DataFrame) -> str:
        """Create conclusions section"""
        section = """## 结论

基于对 Fama-French 三因子模型的分析，我们得出以下结论：

### 1. 因子表现

"""
        
        # Rank factors by Sharpe ratio
        sorted_factors = stats_summary.sort_values('sharpe_ratio', ascending=False)
        
        section += "按夏普比率排序，因子表现为：\n\n"
        for i, (factor, row) in enumerate(sorted_factors.iterrows(), 1):
            section += f"{i}. **{factor}**: 夏普比率 {row['sharpe_ratio']:.3f}\n"
        
        section += """
### 2. 投资组合分析

"""
        
        if not regression_results.empty:
            # Find portfolio with highest alpha
            best_alpha_portfolio = regression_results['alpha'].idxmax()
            best_alpha = regression_results.loc[best_alpha_portfolio, 'alpha']
            
            section += f"- Alpha 最高的组合: **{best_alpha_portfolio}** (α = {best_alpha:.4f})\n"
            
            # Find portfolio with highest R-squared
            best_r2_portfolio = regression_results['r_squared'].idxmax()
            best_r2 = regression_results.loc[best_r2_portfolio, 'r_squared']
            
            section += f"- 拟合度最高的组合: **{best_r2_portfolio}** (R² = {best_r2:.4f})\n"
        
        section += """
### 3. 研究意义

本研究验证了 Fama-French 三因子模型在解释股票收益方面的有效性，为投资组合构建和风险管理提供了实证依据。

---
"""
        
        return section
    
    def _create_footer(self) -> str:
        """Create report footer"""
        footer = """## 附录

### 数据来源

- Kenneth French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

### 参考文献

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

---

*报告结束*
"""
        return footer
    
    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to Markdown table"""
        try:
            # Round numeric columns
            df_display = df.copy()
            for col in df_display.columns:
                if pd.api.types.is_numeric_dtype(df_display[col]):
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            
            # Create header
            headers = [''] + list(df_display.columns)
            table = '| ' + ' | '.join(headers) + ' |\n'
            table += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
            
            # Add rows
            for idx, row in df_display.iterrows():
                row_data = [str(idx)] + [str(val) for val in row]
                table += '| ' + ' | '.join(row_data) + ' |\n'
            
            return table
            
        except Exception as e:
            logger.error(f"Error converting DataFrame to Markdown: {str(e)}")
            return "表格生成失败\n"

