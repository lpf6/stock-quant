# Stock Quant - 技术面量化选股系统

一个插件化、多周期、配置化的量化选股系统，支持批量处理和技术分析。

## 特性

- **插件化架构**: 支持策略插件和指标插件，可动态加载/卸载
- **多周期回溯**: 支持日线、周线、月线不同时间框架，数据自动对齐
- **配置管理**: YAML配置文件，环境变量支持，运行时热更新
- **批量处理**: 支持上千只股票批量处理，高效内存管理
- **多种输出**: 支持JSON/CSV/HTML/Markdown多种输出格式
- **测试驱动**: 完整的单元测试、集成测试和性能测试套件
- **纯Python实现**: 无外部依赖，跨平台兼容

## 安装

### 从源码安装

```bash
# 克隆仓库
 git clone https://github.com/yourusername/stock-quant.git
cd stock-quant

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

### 依赖

- Python >= 3.8
- pandas >= 1.5.0
- numpy >= 1.21.0
- akshare >= 1.10.0 (数据源)
- pyyaml >= 6.0

## 快速开始

### 1. 分析单只股票

```bash
# 分析单只股票
stock-quant analyze --symbol 000001 --period 1y --format json

# 指定策略和指标
stock-quant analyze --symbol 000001 --strategies MACrossStrategy RSIStrategy --indicators MovingAverageIndicator
```

### 2. 批量分析

```bash
# 分析中证1000成分股
stock-quant batch --index 000852 --period 6m --output results.csv

# 分析指定股票列表
stock-quant batch --symbols 000001 000002 000333 --output results.json --format json
```

### 3. 插件管理

```bash
# 列出所有插件
stock-quant plugins list

# 查看插件信息
stock-quant plugins info MACrossStrategy

# 加载插件
stock-quant plugins load MyCustomStrategy
```

### 4. 配置管理

```bash
# 显示配置
stock-quant config show

# 设置配置项
stock-quant config set data.source "tushare"

# 验证配置
stock-quant config validate
```

## 项目结构

```
stock-quant/
├── src/stock_quant/          # 源代码
│   ├── core/                # 核心模块
│   ├── plugins/             # 插件系统
│   ├── period/              # 多周期系统
│   ├── config/              # 配置管理
│   ├── utils/               # 工具函数
│   └── cli/                 # 命令行接口
├── config/                  # 配置文件
├── tests/                   # 测试套件
├── examples/                # 示例代码
├── docs/                    # 文档
└── scripts/                 # 辅助脚本
```

## 插件开发

### 策略插件示例

```python
# plugins/custom/my_strategy.py
from stock_quant.plugins.base import StrategyPlugin
import pandas as pd

class MyCustomStrategy(StrategyPlugin):
    def __init__(self):
        super().__init__()
        self.name = "My Custom Strategy"
        self.description = "自定义策略示例"
        self.author = "Your Name"
        self.config = {
            "param1": 10,
            "param2": 20
        }
    
    def calculate_signals(self, df: pd.DataFrame) -> dict:
        """计算交易信号"""
        # 你的策略逻辑
        return {"my_signal": 1, "score": 0.5}
    
    def get_signal_descriptions(self) -> dict:
        return {
            "my_signal": "自定义信号",
            "score": "策略评分"
        }
```

### 指标插件示例

```python
# plugins/custom/my_indicator.py
from stock_quant.plugins.base import IndicatorPlugin
import pandas as pd

class MyCustomIndicator(IndicatorPlugin):
    def __init__(self):
        super().__init__()
        self.name = "My Custom Indicator"
        self.description = "自定义指标示例"
        self.author = "Your Name"
        self.config = {"period": 14}
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指标"""
        result_df = df.copy()
        # 你的指标计算逻辑
        result_df["MY_INDICATOR"] = df["close"].rolling(self.config["period"]).mean()
        return result_df
    
    def get_indicator_names(self) -> list:
        return ["MY_INDICATOR"]
```

## 配置系统

配置文件按优先级加载：
1. 环境变量（STOCK_QUANT_*）
2. local.yaml（用户本地配置）
3. {environment}.yaml（环境配置）
4. default.yaml（默认配置）

### 环境变量示例

```bash
export STOCK_QUANT_DATA_SOURCE="akshare"
export STOCK_QUANT_LOGGING_LEVEL="DEBUG"
export STOCK_QUANT_OUTPUT_FORMAT="json"
```

## 测试

```bash
# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit/

# 运行集成测试
pytest tests/integration/

# 运行性能测试
pytest tests/performance/

# 生成测试覆盖率报告
pytest --cov=stock_quant --cov-report=html
```

## 性能优化

- 批量处理支持异步数据获取
- 数据缓存减少重复请求
- 内存高效的数据处理
- 多线程支持（可选）

## 输出格式

系统支持四种输出格式：

1. **CSV**: 适合Excel导入和数据分析
2. **JSON**: 适合程序处理和API接口
3. **HTML**: 适合网页展示和报告
4. **Markdown**: 适合文档和README

## 从 quant_analysis_v2.py 迁移

原 `quant_analysis_v2.py` 的功能已重构为模块化系统：

- 数据获取 → `DataFetcher`
- 指标计算 → `IndicatorCalculator` + 指标插件
- 信号生成 → `SignalGenerator` + 策略插件
- 结果输出 → `OutputFormatter`
- 添加了配置管理、日志系统、异常处理等基础设施

## 贡献

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目基于 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目地址: [https://github.com/yourusername/stock-quant](https://github.com/yourusername/stock-quant)
- 问题反馈: [GitHub Issues](https://github.com/yourusername/stock-quant/issues)

## 致谢

感谢以下开源项目：
- [AKShare](https://github.com/akfamily/akshare) - 免费开源财经数据接口库
- [pandas](https://pandas.pydata.org/) - 强大的数据分析工具
- [numpy](https://numpy.org/) - 科学计算基础库
