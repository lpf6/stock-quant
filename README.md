# Stock Quant - 股票量化分析平台 🚀

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![GitHub last commit](https://img.shields.io/github/last-commit/lpf6/stock-quant)
![GitHub repo size](https://img.shields.io/github/repo-size/lpf6/stock-quant)

一个专业的股票量化分析平台，支持数据获取、技术指标计算、策略回测和参数优化。

## ✨ 核心特性

- 📊 **多数据源支持**：CSV、API、数据库等多种数据源
- 📈 **完整技术指标**：MA、MACD、RSI、布林带等常用指标
- 🎯 **策略回测框架**：完整的回测系统，支持多种策略
- ⚙️ **参数优化系统**：自动参数优化和超参数调优
- 🛡️ **风险管理模块**：风险控制和资金管理
- 📊 **可视化报表**：自动生成回测报告和图表

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/lpf6/stock-quant.git
cd stock-quant
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. 安装依赖
```bash
pip install -e .
```

### 4. 运行示例
```bash
python examples/basic_usage.py
```

## 📁 项目结构

```
stock-quant/
├── src/                    # 源代码
│   └── stock_quant/       # 主包
│       ├── core/         # 核心模块（数据获取、处理、计算）
│       ├── plugins/      # 插件系统（策略、指标插件）
│       ├── period/       # 多周期系统
│       ├── config/       # 配置管理
│       └── cli/          # 命令行接口
├── tests/                 # 测试套件
├── config/               # 配置文件
├── examples/             # 使用示例
├── docs/                 # 文档
└── scripts/              # 辅助脚本
```

## 🔧 主要模块

### 核心分析脚本
- `quant_analysis.py` - 基础量化分析
- `quant_analysis_v2.py` - 增强版量化分析
- `backtest_analysis.py` - 回测分析系统
- `full_optimization_pipeline.py` - 完整优化流程

### 优化模块
- `param_optimizer.py` - 参数优化器
- `optimization_backtest.py` - 优化回测
- `stage2_optimization_simple.py` - 第二阶段优化

### 工具脚本
- `run_optimization.sh` - 优化运行脚本
- `check_optimization_results.py` - 结果检查

## 📊 数据分析功能

### 技术指标计算
- 移动平均线（MA）
- 相对强弱指数（RSI）
- 异同移动平均线（MACD）
- 布林带（Bollinger Bands）
- 动量指标

### 策略回测
- MA交叉策略
- RSI超买超卖策略
- MACD金叉死叉策略
- 复合策略组合

## ⚙️ 配置系统

项目使用YAML配置文件，支持环境变量覆盖：

```yaml
# config/default.yaml
data:
  source: "akshare"  # 或 "csv", "database"
  cache_enabled: true
  
strategy:
  default: "ma_cross"
  parameters:
    ma_fast: 5
    ma_slow: 20
    
output:
  format: "csv"  # 或 "json", "html", "markdown"
  directory: "./results"
```

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/test_data_fetcher.py
pytest tests/integration/ -v

# 生成测试覆盖率报告
pytest --cov=stock_quant --cov-report=html
```

## 📈 输出示例

### CSV输出
```csv
symbol,date,signal,score,indicators
000001,2024-01-15,BUY,0.85,{"ma": 12.5, "rsi": 65.2}
000002,2024-01-15,SELL,0.72,{"ma": 8.3, "rsi": 75.8}
```

### JSON输出
```json
[
  {
    "symbol": "000001",
    "date": "2024-01-15",
    "signal": "BUY",
    "score": 0.85,
    "indicators": {"ma": 12.5, "rsi": 65.2}
  }
]
```

## 🔄 版本控制说明

项目使用合理的`.gitignore`配置：
- ✅ **版本控制**：所有源代码、配置、文档
- ❌ **忽略**：数据文件、回测结果、缓存文件、虚拟环境

## 🤝 贡献指南

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目基于 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系

- 项目地址：https://github.com/lpf6/stock-quant
- 问题反馈：GitHub Issues

---

**⭐ 如果这个项目对你有帮助，请给个Star！** ⭐
