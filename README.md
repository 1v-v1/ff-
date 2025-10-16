# Fama-French 三因子模型研究项目

## 项目简介

本项目基于 Kenneth French 数据库，对 Fama-French 三因子模型（Rm-Rf, SMB, HML）进行验证与应用分析。

**研究时间范围**：2015年1月 - 2024年12月（月度数据）

**主要功能**：
- 从 French 数据库自动下载三因子数据和投资组合数据
- 描述性统计分析（均值、标准差、夏普比率等）
- 因子相关性分析
- 时间序列可视化与累计收益曲线
- 三因子模型回归分析（对选定投资组合）
- 自动生成 Markdown 格式分析报告

## 技术栈

- **Python**: 3.8+
- **核心库**: pandas, numpy, statsmodels, matplotlib, seaborn
- **测试框架**: pytest
- **配置管理**: PyYAML

## 项目结构

```
三因子模型/
├── src/                        # 源代码模块
│   ├── data_loader.py          # 数据下载与加载
│   ├── data_processor.py       # 数据处理
│   ├── statistical_analysis.py # 统计分析
│   ├── regression_model.py     # 回归模型
│   ├── visualization.py        # 可视化
│   └── report_generator.py     # 报告生成
├── tests/                      # 单元测试
├── config/                     # 配置文件
│   └── config.yaml
├── data/                       # 数据目录（自动生成）
│   ├── raw/                    # 原始数据
│   └── processed/              # 处理后数据
├── output/                     # 输出目录（自动生成）
│   ├── figures/                # 图表
│   └── results/                # 结果数据
├── logs/                       # 日志目录（自动生成）
├── main.py                     # 主程序入口
├── requirements.txt            # Python依赖
├── README.md                   # 项目说明
└── analysis_report.md          # 分析报告（自动生成）
```

## 安装与配置

### 1. 克隆或下载项目

```bash
cd 三因子模型
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置参数（可选）

编辑 `config/config.yaml` 修改：
- 时间范围
- 数据源 URL
- 输出路径
- 可视化参数

## 使用方法

### 运行完整分析流程

```bash
python main.py
```

程序将自动执行以下步骤：
1. 从 French 数据库下载数据
2. 数据清洗与处理
3. 统计分析
4. 回归分析
5. 生成可视化图表
6. 生成分析报告

### 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并查看覆盖率
pytest --cov=src --cov-report=html
```

## 输出说明

### 数据文件
- `data/raw/ff_three_factors.csv` - 原始三因子数据
- `data/raw/portfolios_25_size_bm.csv` - 25个Size-BM组合数据
- `data/processed/factors_2015_2024.csv` - 处理后的因子数据

### 统计结果
- `output/results/descriptive_statistics.csv` - 描述性统计
- `output/results/correlation_matrix.csv` - 相关性矩阵
- `output/results/regression_results.csv` - 回归分析结果

### 可视化图表
- `output/figures/factors_timeseries.png` - 因子时间序列图
- `output/figures/cumulative_returns.png` - 累计收益曲线
- `output/figures/correlation_heatmap.png` - 相关性热图
- `output/figures/regression_*.png` - 回归拟合图

### 分析报告
- `analysis_report.md` - 完整的 Markdown 格式分析报告

## 参考文献

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
- Kenneth French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

## 开发者

- 开发时间：2025年10月
- 遵循标准：SOLID原则、TDD测试驱动开发
- 测试覆盖率：>70%

## 许可证

本项目仅用于学术研究和教育目的。


