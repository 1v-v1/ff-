"""
对25个投资组合进行整体回归分析（Pooled Regression）

将25个组合的数据堆叠在一起，作为一个整体进行三因子回归，
得出平均的因子暴露和整体效应。
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载因子数据和投资组合数据"""
    print("加载数据...")
    
    # 加载三因子数据
    factors = pd.read_csv('data/raw/ff_three_factors.csv', index_col=0, parse_dates=True)
    
    # 加载25个投资组合数据
    portfolios = pd.read_csv('data/processed/25_portfolios_synthetic.csv', index_col=0, parse_dates=True)
    
    print(f"因子数据: {factors.shape}")
    print(f"投资组合数据: {portfolios.shape}")
    
    return factors, portfolios

def prepare_pooled_data(factors, portfolios):
    """准备pooled数据（堆叠所有组合）"""
    print("\n准备Pooled数据...")
    
    pooled_data = []
    
    for portfolio_name in portfolios.columns:
        df = pd.DataFrame({
            'portfolio': portfolio_name,
            'date': portfolios.index,
            'return': portfolios[portfolio_name].values,
            'Mkt-RF': factors['Mkt-RF'].values,
            'SMB': factors['SMB'].values,
            'HML': factors['HML'].values,
            'RF': factors['RF'].values
        })
        pooled_data.append(df)
    
    pooled_df = pd.concat(pooled_data, ignore_index=True)
    
    # 计算超额收益
    pooled_df['excess_return'] = pooled_df['return'] - pooled_df['RF']
    
    print(f"Pooled数据行数: {len(pooled_df)}")
    print(f"组合数量: {pooled_df['portfolio'].nunique()}")
    print(f"时间点数: {pooled_df['date'].nunique()}")
    
    return pooled_df

def run_pooled_regression(pooled_df):
    """运行Pooled OLS回归"""
    print("\n" + "="*60)
    print("Pooled OLS回归分析")
    print("="*60)
    
    # 准备回归数据
    y = pooled_df['excess_return']
    X = pooled_df[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)
    
    # OLS回归
    model = OLS(y, X).fit()
    
    # 提取结果
    results = {
        'alpha': model.params['const'],
        'beta_mkt': model.params['Mkt-RF'],
        'beta_smb': model.params['SMB'],
        'beta_hml': model.params['HML'],
        'alpha_pvalue': model.pvalues['const'],
        'beta_mkt_pvalue': model.pvalues['Mkt-RF'],
        'beta_smb_pvalue': model.pvalues['SMB'],
        'beta_hml_pvalue': model.pvalues['HML'],
        'alpha_tstat': model.tvalues['const'],
        'beta_mkt_tstat': model.tvalues['Mkt-RF'],
        'beta_smb_tstat': model.tvalues['SMB'],
        'beta_hml_tstat': model.tvalues['HML'],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'n_obs': int(model.nobs),
        'std_errors': model.bse
    }
    
    return model, results

def run_pooled_regression_with_clustering(pooled_df):
    """运行带聚类标准误的Pooled回归（按组合聚类）"""
    print("\n" + "="*60)
    print("Pooled OLS回归（聚类标准误）")
    print("="*60)
    
    # 准备回归数据
    y = pooled_df['excess_return']
    X = pooled_df[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)
    
    # OLS回归，使用聚类标准误
    model = OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': pooled_df['portfolio']})
    
    results = {
        'alpha': model.params['const'],
        'beta_mkt': model.params['Mkt-RF'],
        'beta_smb': model.params['SMB'],
        'beta_hml': model.params['HML'],
        'alpha_pvalue': model.pvalues['const'],
        'beta_mkt_pvalue': model.pvalues['Mkt-RF'],
        'beta_smb_pvalue': model.pvalues['SMB'],
        'beta_hml_pvalue': model.pvalues['HML'],
        'alpha_tstat': model.tvalues['const'],
        'beta_mkt_tstat': model.tvalues['Mkt-RF'],
        'beta_smb_tstat': model.tvalues['SMB'],
        'beta_hml_tstat': model.tvalues['HML'],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'n_obs': int(model.nobs)
    }
    
    return model, results

def print_results(results, title="回归结果"):
    """打印回归结果"""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    print(f"\n样本量: {results['n_obs']}")
    print(f"R²: {results['r_squared']:.4f}")
    print(f"调整后R²: {results['adj_r_squared']:.4f}")
    
    if 'f_statistic' in results:
        print(f"F统计量: {results['f_statistic']:.2f} (p={results['f_pvalue']:.4f})")
    
    print("\n回归系数:")
    print(f"{'参数':<15} {'系数':>10} {'t统计量':>10} {'p值':>10} {'显著性':>10}")
    print("-" * 60)
    
    def get_sig(p):
        if p < 0.01:
            return '***'
        elif p < 0.05:
            return '**'
        elif p < 0.10:
            return '*'
        else:
            return ''
    
    print(f"{'Alpha (α)':<15} {results['alpha']:>10.4f} {results['alpha_tstat']:>10.2f} {results['alpha_pvalue']:>10.4f} {get_sig(results['alpha_pvalue']):>10}")
    print(f"{'Beta_MKT':<15} {results['beta_mkt']:>10.4f} {results['beta_mkt_tstat']:>10.2f} {results['beta_mkt_pvalue']:>10.4f} {get_sig(results['beta_mkt_pvalue']):>10}")
    print(f"{'Beta_SMB':<15} {results['beta_smb']:>10.4f} {results['beta_smb_tstat']:>10.2f} {results['beta_smb_pvalue']:>10.4f} {get_sig(results['beta_smb_pvalue']):>10}")
    print(f"{'Beta_HML':<15} {results['beta_hml']:>10.4f} {results['beta_hml_tstat']:>10.2f} {results['beta_hml_pvalue']:>10.4f} {get_sig(results['beta_hml_pvalue']):>10}")
    
    print("\n显著性水平: *** p<0.01, ** p<0.05, * p<0.10")

def compare_individual_vs_pooled(pooled_results):
    """对比个别回归的平均值和pooled回归结果"""
    print("\n" + "="*60)
    print("个别回归 vs Pooled回归对比")
    print("="*60)
    
    # 加载个别回归结果
    individual_results = pd.read_csv('output/results/25_portfolios_regression_results.csv', index_col=0)
    
    comparison = pd.DataFrame({
        '指标': ['Alpha', 'Beta_MKT', 'Beta_SMB', 'Beta_HML', 'R²'],
        '个别回归平均': [
            individual_results['alpha'].mean(),
            individual_results['beta_mkt'].mean(),
            individual_results['beta_smb'].mean(),
            individual_results['beta_hml'].mean(),
            individual_results['r_squared'].mean()
        ],
        'Pooled回归': [
            pooled_results['alpha'],
            pooled_results['beta_mkt'],
            pooled_results['beta_smb'],
            pooled_results['beta_hml'],
            pooled_results['r_squared']
        ]
    })
    
    comparison['差异'] = comparison['Pooled回归'] - comparison['个别回归平均']
    
    print("\n" + comparison.to_string(index=False))
    
    return comparison

def visualize_pooled_results(model, pooled_df):
    """可视化pooled回归结果"""
    print("\n生成可视化图表...")
    
    # 预测值vs实际值
    pooled_df['predicted'] = model.predict()
    pooled_df['residuals'] = model.resid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 实际值 vs 预测值
    axes[0, 0].scatter(pooled_df['predicted'], pooled_df['excess_return'], 
                       alpha=0.3, s=10, color='steelblue')
    axes[0, 0].plot([-30, 30], [-30, 30], 'r--', linewidth=2, label='45°线')
    axes[0, 0].set_xlabel('预测超额收益率 (%)', fontsize=11)
    axes[0, 0].set_ylabel('实际超额收益率 (%)', fontsize=11)
    axes[0, 0].set_title('实际值 vs 预测值', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 残差分布
    axes[0, 1].hist(pooled_df['residuals'], bins=50, edgecolor='black', 
                    alpha=0.7, color='orange')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('残差 (%)', fontsize=11)
    axes[0, 1].set_ylabel('频数', fontsize=11)
    axes[0, 1].set_title(f'残差分布 (均值={pooled_df["residuals"].mean():.4f})', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q图
    from scipy import stats
    stats.probplot(pooled_df['residuals'], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q图（正态性检验）', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 残差 vs 预测值
    axes[1, 1].scatter(pooled_df['predicted'], pooled_df['residuals'], 
                       alpha=0.3, s=10, color='green')
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('预测值 (%)', fontsize=11)
    axes[1, 1].set_ylabel('残差 (%)', fontsize=11)
    axes[1, 1].set_title('残差 vs 预测值（异方差检验）', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/figures/pooled_regression_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("诊断图已保存: output/figures/pooled_regression_diagnostics.png")

def save_results(results_standard, results_clustered, comparison):
    """保存回归结果"""
    print("\n保存结果...")
    
    # 保存到CSV
    results_df = pd.DataFrame({
        '回归类型': ['Pooled OLS (标准误)', 'Pooled OLS (聚类标准误)'],
        'Alpha': [results_standard['alpha'], results_clustered['alpha']],
        'Alpha_pvalue': [results_standard['alpha_pvalue'], results_clustered['alpha_pvalue']],
        'Beta_MKT': [results_standard['beta_mkt'], results_clustered['beta_mkt']],
        'Beta_MKT_pvalue': [results_standard['beta_mkt_pvalue'], results_clustered['beta_mkt_pvalue']],
        'Beta_SMB': [results_standard['beta_smb'], results_clustered['beta_smb']],
        'Beta_SMB_pvalue': [results_standard['beta_smb_pvalue'], results_clustered['beta_smb_pvalue']],
        'Beta_HML': [results_standard['beta_hml'], results_clustered['beta_hml']],
        'Beta_HML_pvalue': [results_standard['beta_hml_pvalue'], results_clustered['beta_hml_pvalue']],
        'R²': [results_standard['r_squared'], results_clustered['r_squared']],
        'Adj_R²': [results_standard['adj_r_squared'], results_clustered['adj_r_squared']],
        'N_obs': [results_standard['n_obs'], results_clustered['n_obs']]
    })
    
    results_df.to_csv('output/results/pooled_regression_results.csv', index=False)
    print("结果已保存: output/results/pooled_regression_results.csv")
    
    # 保存对比结果
    comparison.to_csv('output/results/individual_vs_pooled_comparison.csv', index=False)
    print("对比结果已保存: output/results/individual_vs_pooled_comparison.csv")
    
    return results_df

def main():
    """主函数"""
    print("="*60)
    print("25个投资组合整体回归分析（Pooled Regression）")
    print("="*60)
    
    # 1. 加载数据
    factors, portfolios = load_data()
    
    # 2. 准备pooled数据
    pooled_df = prepare_pooled_data(factors, portfolios)
    
    # 3. 标准Pooled回归
    model_standard, results_standard = run_pooled_regression(pooled_df)
    print_results(results_standard, "Pooled OLS回归结果（标准误）")
    
    # 4. 聚类标准误Pooled回归
    model_clustered, results_clustered = run_pooled_regression_with_clustering(pooled_df)
    print_results(results_clustered, "Pooled OLS回归结果（聚类标准误）")
    
    # 5. 对比个别回归和pooled回归
    comparison = compare_individual_vs_pooled(results_clustered)
    
    # 6. 可视化
    visualize_pooled_results(model_standard, pooled_df)
    
    # 7. 保存结果
    results_df = save_results(results_standard, results_clustered, comparison)
    
    # 8. 总结
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    print("\n核心发现:")
    print(f"1. 整体Alpha = {results_clustered['alpha']:.4f}% (p={results_clustered['alpha_pvalue']:.4f})")
    
    if results_clustered['alpha_pvalue'] < 0.05:
        print("   → Alpha显著异于0，存在系统性定价偏误")
    else:
        print("   → Alpha不显著，支持三因子模型有效性")
    
    print(f"\n2. 整体Beta系数:")
    print(f"   - 市场Beta = {results_clustered['beta_mkt']:.4f}")
    print(f"   - 规模Beta = {results_clustered['beta_smb']:.4f}")
    print(f"   - 价值Beta = {results_clustered['beta_hml']:.4f}")
    
    print(f"\n3. 模型拟合度:")
    print(f"   - R² = {results_clustered['r_squared']:.4f}")
    print(f"   - 样本量 = {results_clustered['n_obs']:,}")
    
    print("\n生成的文件:")
    print("- output/results/pooled_regression_results.csv")
    print("- output/results/individual_vs_pooled_comparison.csv")
    print("- output/figures/pooled_regression_diagnostics.png")
    
    return results_df, comparison

if __name__ == '__main__':
    results_df, comparison = main()

