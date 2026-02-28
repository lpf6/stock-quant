#!/usr/bin/env python3
"""简化的第二阶段测试"""

import sys
import json
sys.path.append('.')

from optimization_backtest import OptimizedBacktest, test_stocks

# 第一阶段最佳参数
stage1_params = {
    "ma_fast": 50,
    "ma_slow": 100, 
    "rsi_oversold": 28,
    "rsi_overbought": 70,
    "bb_std": 2.0,
    "score_threshold": 0.45
}

# 添加风险管理参数
risk_params = {
    "stop_loss": -0.05,
    "take_profit": 0.12,
    "position_size": 0.15,
    "dynamic_stop_loss": True,
    "trailing_stop": False,
    "min_holding_days": 3,
    "max_consecutive_trades": 5,
    "trade_cooldown": 3
}

# 合并参数
combined_params = stage1_params.copy()
combined_params.update(risk_params)

print("第一阶段参数:", json.dumps(stage1_params, indent=2))
print("\n风险管理参数:", json.dumps(risk_params, indent=2))

# 创建回测实例
print(f"\n创建回测实例...")
backtest = OptimizedBacktest(combined_params)

# 测试多只股票
print(f"\n测试{len(test_stocks)}只股票...")
performance = backtest.test_multiple_stocks(test_stocks, 7)

print("\n回测结果:")
print(f"  夏普比率: {performance.get('sharpe_ratio', 0):.4f}")
print(f"  最大回撤: {performance.get('max_drawdown', 0):.4f}")
print(f"  胜率: {performance.get('win_rate', 0):.4f}")
print(f"  总收益率: {performance.get('total_return', 0):.4f}")
print(f"  有效交易数: {performance.get('valid_trades', 0)}")

# 计算综合评分
max_drawdown_score = max(0, 1 - performance.get('max_drawdown', 1))
sharpe_score = max(0, min(1, performance.get('sharpe_ratio', 0) / 2.0))
win_rate_score = performance.get('win_rate', 0)
return_score = max(0, min(1, performance.get('total_return', 0) / 0.5))

composite_score = (
    0.4 * max_drawdown_score +
    0.3 * sharpe_score +
    0.2 * win_rate_score +
    0.1 * return_score
)

print(f"  综合评分: {composite_score:.4f}")

# 检查目标
print("\n目标检查:")
print(f"  最大回撤 < 15%: {performance.get('max_drawdown', 1) < 0.15}")
print(f"  夏普比率 > 0.8: {performance.get('sharpe_ratio', 0) > 0.8}")
print(f"  胜率 > 55%: {performance.get('win_rate', 0) > 0.55}")
print(f"  综合评分 > 0.6: {composite_score > 0.6}")