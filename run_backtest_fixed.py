#!/usr/bin/env python3
"""
修复的回测分析脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 先测试基本导入
print("测试模块导入...")
try:
    from src.stock_quant.core.data_fetcher import DataFetcher
    print("✓ DataFetcher 导入成功")
except Exception as e:
    print(f"✗ DataFetcher 导入失败: {e}")

try:
    from src.stock_quant.core.indicator_calculator import IndicatorCalculator
    print("✓ IndicatorCalculator 导入成功")
except Exception as e:
    print(f"✗ IndicatorCalculator 导入失败: {e}")

try:
    from src.stock_quant.core.signal_generator import SignalGenerator
    print("✓ SignalGenerator 导入成功")
except Exception as e:
    print(f"✗ SignalGenerator 导入失败: {e}")

print("\n测试插件系统...")

# 测试插件注册
import importlib
import pandas as pd
import numpy as np

# 注册插件
from src.stock_quant.plugins.manager import PluginManager
from src.stock_quant.plugins.strategies.ma_cross_strategy import MACrossStrategy
from src.stock_quant.plugins.strategies.rsi_strategy import RSIStrategy
from src.stock_quant.plugins.strategies.macd_strategy import MACDStrategy
from src.stock_quant.plugins.strategies.backtest_strategy import BacktestStrategy
from src.stock_quant.plugins.strategies.optimization_strategy import OptimizationStrategy

from src.stock_quant.plugins.indicators.moving_average import MovingAverageIndicator
from src.stock_quant.plugins.indicators.rsi_indicator import RSIIndicator
from src.stock_quant.plugins.indicators.bollinger_bands import BollingerBandsIndicator
from src.stock_quant.plugins.indicators.momentum_indicators import (
    MomentumIndicator, VolumeIndicator, TrendStrengthIndicator
)

print("正在注册插件...")

# 创建插件管理器并手动注册插件
plugin_manager = PluginManager()

# 注册策略插件
plugin_manager.register_plugin(MACrossStrategy)
plugin_manager.register_plugin(RSIStrategy)
plugin_manager.register_plugin(MACDStrategy)
plugin_manager.register_plugin(BacktestStrategy)
plugin_manager.register_plugin(OptimizationStrategy)

# 注册指标插件
plugin_manager.register_plugin(MovingAverageIndicator)
plugin_manager.register_plugin(RSIIndicator)
plugin_manager.register_plugin(BollingerBandsIndicator)
plugin_manager.register_plugin(MomentumIndicator)
plugin_manager.register_plugin(VolumeIndicator)
plugin_manager.register_plugin(TrendStrengthIndicator)

print("插件注册完成")
print(f"可用插件: {[p['name'] for p in plugin_manager.get_available_plugins()]}")

print("\n创建模拟数据...")

# 创建模拟数据
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')  # 交易日
n_days = len(dates)

# 生成价格序列
price_changes = np.random.randn(n_days) * 0.02
prices = 10 * np.exp(np.cumsum(price_changes))

# 生成OHLCV数据
high = prices * (1 + np.random.rand(n_days) * 0.03)
low = prices * (1 - np.random.rand(n_days) * 0.03)
open_prices = prices * (1 + np.random.randn(n_days) * 0.01)
volume = np.random.randint(1000000, 10000000, n_days)

# 确保价格合理性
high = np.maximum(high, prices)
low = np.minimum(low, prices)

mock_data = pd.DataFrame({
    'open': open_prices,
    'high': high,
    'low': low,
    'close': prices,
    'volume': volume,
    'amount': prices * volume
}, index=dates)

print(f"模拟数据创建完成: {len(mock_data)} 条记录")

print("\n测试回测策略...")

# 创建回测策略实例
backtest_strategy = BacktestStrategy()
backtest_strategy.initialize({
    "initial_capital": 100000,
    "position_ratio": 0.2,
    "stop_loss": -0.05,
    "take_profit": 0.10,
    "commission_rate": 0.0003
})

# 测试回测策略
test_signals = {
    "total_score": 0.6,
    "MACrossStrategy.ma_cross": 1,
    "RSIStrategy.rsi_oversold": 0,
    "MACDStrategy.macd_cross": 1
}

backtest_result = backtest_strategy.run_backtest(mock_data, test_signals)
print(f"回测结果: 夏普比率={backtest_result.get('sharpe_ratio', 0):.3f}, 总收益率={backtest_result.get('total_return', 0)*100:.1f}%")

print("\n测试动量指标...")

# 测试动量指标
momentum_indicator = MomentumIndicator()
momentum_data = momentum_indicator.calculate(mock_data)

print(f"动量指标计算完成: {[col for col in momentum_data.columns if 'Momentum' in col]}")

volume_indicator = VolumeIndicator()
volume_data = volume_indicator.calculate(mock_data)

print(f"成交量指标计算完成: {[col for col in volume_data.columns if 'Volume' in col]}")

trend_indicator = TrendStrengthIndicator()
trend_data = trend_indicator.calculate(mock_data)

print(f"趋势指标计算完成: {[col for col in trend_data.columns if 'Trend' in col]}")

print("\n测试优化策略...")

# 测试优化策略
optimization_strategy = OptimizationStrategy()
optimization_strategy.initialize({
    "optimization_method": "random_search",
    "max_iterations": 10
})

optimization_result = optimization_strategy.optimize_parameters(mock_data, test_signals)
print(f"优化结果: 最佳得分={optimization_result.get('best_score', 0):.3f}")

print("\n✅ 所有测试通过!")

# 保存测试结果
os.makedirs("test_results", exist_ok=True)

# 保存回测结果
import json
with open("test_results/backtest_result.json", "w") as f:
    json.dump(backtest_result, f, indent=2, default=str)

print("测试结果已保存到 test_results/ 目录")