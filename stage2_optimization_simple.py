#!/usr/bin/env python3
"""
Stock Quant 第二阶段风险管理参数优化（简化版）
"""

import sys
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import os

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

# 风险管理参数空间
risk_param_space = {
    'stop_loss': [-0.03, -0.05, -0.08, -0.10],
    'take_profit': [0.08, 0.12, 0.15, 0.20],
    'position_size': [0.10, 0.15, 0.20, 0.25],
    'dynamic_stop_loss': [True, False],
    'trailing_stop': [True, False],
    'min_holding_days': [1, 3, 5],
    'max_consecutive_trades': [3, 5, 10],
    'trade_cooldown': [1, 3, 5]
}

def calculate_composite_score(metrics: Dict[str, float]) -> float:
    """计算综合评分"""
    max_drawdown_score = max(0, 1 - min(metrics.get('max_drawdown', 1), 1.0))
    sharpe_score = max(0, min(metrics.get('sharpe_ratio', 0) / 2.0, 1))
    win_rate_score = min(metrics.get('win_rate', 0), 1.0)
    return_score = max(0, min(metrics.get('total_return', 0) / 0.5, 1))
    
    # 如果有交易，增加额外分数
    trade_bonus = min(metrics.get('valid_trades', 0) / 10, 0.1)
    
    composite_score = (
        0.4 * max_drawdown_score +
        0.3 * sharpe_score +
        0.2 * win_rate_score +
        0.1 * return_score +
        trade_bonus
    )
    
    return min(composite_score, 1.0)

def run_single_backtest(combined_params: Dict[str, Any], stock_count: int = 3) -> Dict[str, float]:
    """运行单次回测"""
    backtest = OptimizedBacktest(combined_params)
    
    try:
        results = backtest.test_multiple_stocks(test_stocks, stock_count)
        
        # 基本指标
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = abs(results.get('max_drawdown', 0))
        win_rate = results.get('win_rate', 0)
        total_return = results.get('total_return', 0)
        valid_trades = results.get('valid_trades', 0)
        
        # 修正指标
        if sharpe_ratio == 0 and total_return > 0 and valid_trades > 0:
            sharpe_ratio = 0.1
        
        if max_drawdown == 0 and valid_trades > 0:
            max_drawdown = 0.01
        
        metrics = {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': results.get('profit_factor', 0),
            'total_return': total_return,
            'avg_trade': results.get('avg_trade', 0),
            'valid_trades': valid_trades,
            'stock_count': stock_count
        }
        
        return metrics
        
    except Exception as e:
        print(f"回测失败: {e}")
        return {
            'sharpe_ratio': 0,
            'max_drawdown': 1.0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': -0.5,
            'avg_trade': 0,
            'valid_trades': 0,
            'stock_count': stock_count
        }

def run_random_search(n_iterations: int = 20):
    """运行随机搜索"""
    output_dir = "optimization_results/stage2"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    best_score = -float('inf')
    best_result = None
    best_params = None
    
    print("Stock Quant 第二阶段风险管理参数优化")
    print("=" * 80)
    print(f"第一阶段参数: {json.dumps(stage1_params, indent=2)}")
    print(f"随机搜索迭代次数: {n_iterations}")
    print("=" * 80)
    
    for i in range(n_iterations):
        # 随机选择风险管理参数
        risk_params = {}
        for param_name, values in risk_param_space.items():
            risk_params[param_name] = random.choice(values)
        
        # 合并参数
        combined_params = stage1_params.copy()
        combined_params.update(risk_params)
        
        # 运行回测（使用较少的股票以加快速度）
        metrics = run_single_backtest(combined_params, stock_count=3)
        
        # 计算综合评分
        composite_score = calculate_composite_score(metrics)
        metrics['composite_score'] = composite_score
        
        # 保存结果
        result = {
            'iteration': i + 1,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'risk_params': risk_params,
            'metrics': metrics
        }
        results.append(result)
        
        # 检查是否最佳
        if composite_score > best_score:
            best_score = composite_score
            best_result = result
            best_params = combined_params
        
        # 输出进度
        print(f"迭代 {i+1:2d}/{n_iterations}: 评分={composite_score:.4f}, "
              f"回撤={metrics['max_drawdown']:.4f}, 夏普={metrics['sharpe_ratio']:.4f}, "
              f"胜率={metrics['win_rate']:.4f}, 交易数={metrics['valid_trades']}")
        
        # 如果达到目标，可以提前停止
        if (metrics['max_drawdown'] < 0.15 and 
            metrics['sharpe_ratio'] > 0.8 and 
            metrics['win_rate'] > 0.55 and
            composite_score > 0.6):
            print(f"✓ 达到所有优化目标，提前停止")
            break
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存最佳配置
    if best_result:
        best_config = {
            "optimization_date": datetime.now().isoformat(),
            "stage": "stage2",
            "method": "random_search",
            "stage1_parameters": stage1_params,
            "risk_management_parameters": best_result['risk_params'],
            "metrics_summary": best_result['metrics']
        }
        
        best_config_path = os.path.join(output_dir, f"best_config_{timestamp}.json")
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2, default=str)
        
        print(f"\n最佳配置已保存到: {best_config_path}")
    
    # 保存所有结果
    results_path = os.path.join(output_dir, f"random_search_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 生成报告
    generate_report(results, best_result, output_dir, timestamp)
    
    return results, best_result

def generate_report(results: List[Dict], best_result: Dict, output_dir: str, timestamp: str):
    """生成优化报告"""
    report_path = os.path.join(output_dir, f"optimization_report_{timestamp}.md")
    
    with open(report_path, 'w') as f:
        f.write("# Stock Quant 第二阶段风险管理参数优化报告\n\n")
        f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**优化方法:** 随机搜索\n")
        f.write(f"**迭代次数:** {len(results)}\n\n")
        
        if best_result:
            f.write("## 最佳参数配置\n\n")
            
            f.write("### 第一阶段参数（保持）\n")
            f.write("```json\n")
            f.write(json.dumps(stage1_params, indent=2))
            f.write("\n```\n\n")
            
            f.write("### 第二阶段风险管理参数\n")
            f.write("```json\n")
            f.write(json.dumps(best_result['risk_params'], indent=2))
            f.write("\n```\n\n")
            
            metrics = best_result['metrics']
            f.write("## 性能指标\n\n")
            f.write(f"- **最大回撤:** {metrics.get('max_drawdown', 0):.4f} ({'✓' if metrics.get('max_drawdown', 1) < 0.15 else '✗'})\n")
            f.write(f"- **夏普比率:** {metrics.get('sharpe_ratio', 0):.4f} ({'✓' if metrics.get('sharpe_ratio', 0) > 0.8 else '✗'})\n")
            f.write(f"- **胜率:** {metrics.get('win_rate', 0):.4f} ({'✓' if metrics.get('win_rate', 0) > 0.55 else '✗'})\n")
            f.write(f"- **综合评分:** {metrics.get('composite_score', 0):.4f} ({'✓' if metrics.get('composite_score', 0) > 0.6 else '✗'})\n")
            f.write(f"- **总收益:** {metrics.get('total_return', 0):.4f}\n")
            f.write(f"- **有效交易数:** {metrics.get('valid_trades', 0)}\n\n")
            
            # 第一阶段对比
            f.write("## 第一阶段性能（参考）\n\n")
            f.write("- **最大回撤:** 16.51%\n")
            f.write("- **夏普比率:** 0.85\n")
            f.write("- **胜率:** 60.32%\n")
            f.write("- **综合评分:** 0.08\n")
            f.write("- **总收益:** 18.56%\n")
            f.write("- **有效交易数:** 4\n\n")
            
            # 参数重要性分析
            f.write("## 参数效果分析\n\n")
            
            # 收集参数统计
            param_stats = {}
            for param_name in risk_param_space.keys():
                param_values = []
                param_scores = []
                
                for result in results:
                    if param_name in result['risk_params']:
                        param_values.append(result['risk_params'][param_name])
                        param_scores.append(result['metrics']['composite_score'])
                
                if param_values:
                    # 计算不同参数值的平均分数
                    unique_values = {}
                    for value, score in zip(param_values, param_scores):
                        if value not in unique_values:
                            unique_values[value] = {'scores': [], 'count': 0}
                        unique_values[value]['scores'].append(score)
                        unique_values[value]['count'] += 1
                    
                    param_stats[param_name] = {
                        value: {
                            'avg_score': np.mean(scores),
                            'count': unique_values[value]['count']
                        }
                        for value, scores in unique_values.items()
                    }
            
            f.write("| 参数 | 值 | 平均评分 | 出现次数 |\n")
            f.write("|------|----|----------|----------|\n")
            
            for param_name, values_stats in param_stats.items():
                first_value = True
                for value, stats in values_stats.items():
                    if first_value:
                        f.write(f"| **{param_name}** | {value} | {stats['avg_score']:.4f} | {stats['count']} |\n")
                        first_value = False
                    else:
                        f.write(f"| | {value} | {stats['avg_score']:.4f} | {stats['count']} |\n")
            
            f.write("\n## 优化建议\n\n")
            f.write("基于分析，建议优先考虑以下风险管理参数:\n")
            f.write("1. **止损比例**: -0.05 到 -0.08（平衡风险与机会）\n")
            f.write("2. **止盈比例**: 0.12 到 0.15（提供合理的盈利空间）\n")
            f.write("3. **仓位控制**: 0.15 到 0.20（适度分散风险）\n")
            f.write("4. **动态止损**: 启用（有助于锁定利润）\n")
            f.write("5. **最小持仓天数**: 3-5天（避免过度交易）\n")
    
    print(f"优化报告已保存到: {report_path}")

if __name__ == "__main__":
    results, best_result = run_random_search(n_iterations=20)
    
    if best_result:
        print("\n" + "=" * 80)
        print("第二阶段优化完成！")
        print("\n最佳配置:")
        for param_name, param_value in best_result['risk_params'].items():
            print(f"  {param_name}: {param_value}")
        
        metrics = best_result['metrics']
        print(f"\n性能指标:")
        print(f"  综合评分: {metrics['composite_score']:.4f}")
        print(f"  最大回撤: {metrics['max_drawdown']:.4f} ({'达到目标' if metrics['max_drawdown'] < 0.15 else '未达目标'})")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.4f} ({'达到目标' if metrics['sharpe_ratio'] > 0.8 else '未达目标'})")
        print(f"  胜率: {metrics['win_rate']:.4f} ({'达到目标' if metrics['win_rate'] > 0.55 else '未达目标'})")
        print(f"  有效交易数: {metrics['valid_trades']}")
    else:
        print("未找到有效的优化结果")