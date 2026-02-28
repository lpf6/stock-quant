#!/usr/bin/env python3
"""
Stock Quant 第二阶段风险管理参数优化
专注于降低最大回撤，同时保持夏普比率和其他性能指标
"""

import sys
import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
import itertools

# 添加当前目录到路径
sys.path.append('.')

from optimization_backtest import OptimizedBacktest, test_stocks
from param_optimizer import ParameterOptimizer, OptimizationResult

@dataclass
class Stage2OptimizationResult:
    """第二阶段优化结果"""
    params: Dict[str, Any]
    risk_params: Dict[str, Any]
    metrics: Dict[str, float]
    iteration: int
    timestamp: str
    stock_count: int
    valid_trades: int

def create_stage2_backtest(stage1_params: Dict[str, Any]) -> callable:
    """创建第二阶段的回测函数
    
    Args:
        stage1_params: 第一阶段的最佳参数
    """
    
    def stage2_backtest(risk_params: Dict[str, Any]) -> Dict[str, float]:
        """第二阶段的回测函数
        
        Args:
            risk_params: 风险管理参数
            
        Returns:
            性能指标字典
        """
        # 合并第一阶段参数和风险管理参数
        combined_params = stage1_params.copy()
        combined_params.update(risk_params)
        
        # 设置固定的测试股票数量
        stock_count = 7
        
        # 创建回测实例
        backtest = OptimizedBacktest(combined_params)
        
        # 运行回测
        try:
            results = backtest.test_multiple_stocks(test_stocks, stock_count)
        except Exception as e:
            print(f"回测失败: {e}")
            # 返回默认值
            return {
                'sharpe_ratio': 0,
                'max_drawdown': 1.0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': -0.5,  # 惩罚性收益
                'avg_trade': 0,
                'valid_trades': 0,
                'stock_count': stock_count,
                'composite_score': 0
            }
        
        # 提取关键指标
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = abs(results.get('max_drawdown', 1.0))  # 确保正数
        win_rate = results.get('win_rate', 0)
        profit_factor = results.get('profit_factor', 0)
        total_return = results.get('total_return', 0)
        avg_trade = results.get('avg_trade', 0)
        valid_trades = results.get('valid_trades', 0)
        
        # 如果夏普比率为0但总收益率为正，设置一个基础夏普比率
        if sharpe_ratio == 0 and total_return > 0 and valid_trades > 0:
            sharpe_ratio = 0.1  # 基础值
        
        # 如果最大回撤为0但有交易，设置一个小回撤
        if max_drawdown == 0 and valid_trades > 0:
            max_drawdown = 0.01
        
        metrics = {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'avg_trade': avg_trade,
            'valid_trades': valid_trades,
            'stock_count': stock_count
        }
        
        # 计算综合评分（第二阶段重点降低最大回撤）
        # 权重分配：最大回撤 40%，夏普比率 30%，胜率 20%，总收益 10%
        max_drawdown_score = max(0, 1 - metrics['max_drawdown'])  # 回撤越小得分越高
        sharpe_score = max(0, min(1, metrics['sharpe_ratio'] / 2.0))  # 夏普比率标准化
        win_rate_score = metrics['win_rate']  # 胜率直接作为得分
        return_score = max(0, min(1, metrics['total_return'] / 0.5))  # 总收益标准化
        
        composite_score = (
            0.4 * max_drawdown_score +
            0.3 * sharpe_score +
            0.2 * win_rate_score +
            0.1 * return_score
        )
        
        metrics['composite_score'] = composite_score
        
        return metrics
    
    return stage2_backtest

def define_risk_parameter_space() -> Dict[str, List[Any]]:
    """定义风险管理参数空间"""
    return {
        # 风险管理参数
        'stop_loss': [-0.03, -0.05, -0.08, -0.10],
        'take_profit': [0.08, 0.12, 0.15, 0.20],
        'position_size': [0.10, 0.15, 0.20, 0.25],
        'dynamic_stop_loss': [True, False],
        'trailing_stop': [True, False],
        
        # 交易规则参数
        'min_holding_days': [1, 3, 5, 10],
        'max_consecutive_trades': [3, 5, 10],
        'trade_cooldown': [1, 3, 5],
        
        # 风险管理权重（可选的附加参数）
        'risk_aversion': [0.1, 0.3, 0.5],
        'volatility_scaling': [True, False]
    }

def random_search(stage1_params: Dict[str, Any], 
                  param_space: Dict[str, List[Any]], 
                  n_iterations: int = 30,
                  output_dir: str = "optimization_results/stage2") -> List[Stage2OptimizationResult]:
    """执行随机搜索优化
    
    Args:
        stage1_params: 第一阶段最佳参数
        param_space: 参数空间定义
        n_iterations: 迭代次数
        output_dir: 输出目录
        
    Returns:
        优化结果列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建回测函数
    backtest_func = create_stage2_backtest(stage1_params)
    
    results = []
    best_composite_score = -float('inf')
    best_result = None
    no_improvement_count = 0
    
    print(f"开始随机搜索优化（{n_iterations}次迭代）")
    print("=" * 80)
    
    for i in range(n_iterations):
        # 随机选择参数
        risk_params = {}
        for param_name, values in param_space.items():
            risk_params[param_name] = random.choice(values)
        
        # 运行回测
        try:
            metrics = backtest_func(risk_params)
            
            # 创建结果对象
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result = Stage2OptimizationResult(
                params={**stage1_params, **risk_params},
                risk_params=risk_params.copy(),
                metrics=metrics,
                iteration=i + 1,
                timestamp=timestamp,
                stock_count=metrics.get('stock_count', 7),
                valid_trades=metrics.get('valid_trades', 0)
            )
            
            results.append(result)
            
            # 检查停止条件
            composite_score = metrics['composite_score']
            max_drawdown = metrics['max_drawdown']
            sharpe_ratio = metrics['sharpe_ratio']
            
            # 输出进度
            print(f"迭代 {i+1}/{n_iterations}:")
            print(f"  最大回撤: {max_drawdown:.4f}, 夏普比率: {sharpe_ratio:.4f}, 综合评分: {composite_score:.4f}")
            print(f"  参数: { {k: v for k, v in risk_params.items() if not isinstance(v, bool)} }")
            
            # 检查是否达到主要目标
            if max_drawdown < 0.15 and sharpe_ratio > 0.8:
                print(f"  ✓ 达到主要目标！最大回撤 {max_drawdown:.4f} < 15%, 夏普比率 {sharpe_ratio:.4f} > 0.8")
                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_result = result
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
                
            # 检查停止条件
            if max_drawdown < 0.15 and sharpe_ratio > 0.8 and composite_score > 0.6:
                print(f"\n✓ 达到所有优化目标，提前停止")
                break
                
            if no_improvement_count >= 5:
                print(f"\n⚠ 连续{no_improvement_count}次迭代无显著改善，继续搜索...")
                no_improvement_count = 0
                
        except Exception as e:
            print(f"迭代 {i+1} 失败: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("随机搜索完成")
    
    # 保存结果
    save_results(results, best_result, output_dir, "random_search")
    
    return results, best_result

def grid_search(stage1_params: Dict[str, Any],
                param_space: Dict[str, List[Any]],
                key_params: List[str],
                output_dir: str = "optimization_results/stage2") -> List[Stage2OptimizationResult]:
    """对关键参数执行网格搜索
    
    Args:
        stage1_params: 第一阶段最佳参数
        param_space: 参数空间定义
        key_params: 关键参数列表
        output_dir: 输出目录
        
    Returns:
        优化结果列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建回测函数
    backtest_func = create_stage2_backtest(stage1_params)
    
    # 仅对关键参数进行网格搜索
    grid_space = {}
    for param in key_params:
        if param in param_space:
            grid_space[param] = param_space[param]
    
    # 生成所有参数组合
    param_names = list(grid_space.keys())
    param_values = list(grid_space.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"开始网格搜索优化（{len(param_combinations)}种组合）")
    print(f"关键参数: {param_names}")
    print("=" * 80)
    
    results = []
    best_composite_score = -float('inf')
    best_result = None
    
    for i, combination in enumerate(param_combinations):
        # 构建参数字典
        risk_params = {param_names[j]: combination[j] for j in range(len(param_names))}
        
        # 为其他参数设置默认值
        for param_name, values in param_space.items():
            if param_name not in risk_params:
                # 使用合理的默认值
                if param_name == 'stop_loss':
                    risk_params[param_name] = -0.05
                elif param_name == 'take_profit':
                    risk_params[param_name] = 0.12
                elif param_name == 'position_size':
                    risk_params[param_name] = 0.15
                elif param_name == 'dynamic_stop_loss':
                    risk_params[param_name] = True
                elif param_name == 'trailing_stop':
                    risk_params[param_name] = True
                else:
                    risk_params[param_name] = values[0]  # 使用第一个值作为默认
        
        # 运行回测
        try:
            metrics = backtest_func(risk_params)
            
            # 创建结果对象
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result = Stage2OptimizationResult(
                params={**stage1_params, **risk_params},
                risk_params=risk_params.copy(),
                metrics=metrics,
                iteration=i + 1,
                timestamp=timestamp,
                stock_count=metrics.get('stock_count', 7),
                valid_trades=metrics.get('valid_trades', 0)
            )
            
            results.append(result)
            
            # 检查是否是最佳结果
            composite_score = metrics['composite_score']
            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_result = result
            
            # 输出进度
            if (i + 1) % 10 == 0 or (i + 1) == len(param_combinations):
                max_drawdown = metrics['max_drawdown']
                sharpe_ratio = metrics['sharpe_ratio']
                print(f"进度: {i+1}/{len(param_combinations)}")
                print(f"  当前最佳: 回撤={max_drawdown:.4f}, 夏普={sharpe_ratio:.4f}, 评分={composite_score:.4f}")
                
        except Exception as e:
            print(f"组合 {i+1} 失败: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("网格搜索完成")
    
    # 保存结果
    save_results(results, best_result, output_dir, "grid_search")
    
    return results, best_result

def save_results(results: List[Stage2OptimizationResult],
                best_result: Stage2OptimizationResult,
                output_dir: str,
                method: str):
    """保存优化结果"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存最佳配置
    if best_result:
        best_config = {
            "optimization_date": datetime.now().isoformat(),
            "stage": "stage2",
            "method": method,
            "stage1_parameters": {
                "ma_fast": best_result.params.get('ma_fast'),
                "ma_slow": best_result.params.get('ma_slow'),
                "rsi_oversold": best_result.params.get('rsi_oversold'),
                "rsi_overbought": best_result.params.get('rsi_overbought'),
                "bb_std": best_result.params.get('bb_std'),
                "score_threshold": best_result.params.get('score_threshold')
            },
            "risk_management_parameters": {
                k: v for k, v in best_result.risk_params.items()
                if k not in ['ma_fast', 'ma_slow', 'rsi_oversold', 'rsi_overbought', 'bb_std', 'score_threshold']
            },
            "metrics_summary": best_result.metrics
        }
        
        best_config_path = os.path.join(output_dir, f"best_config_{timestamp}.json")
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2, default=str)
        
        print(f"最佳配置已保存到: {best_config_path}")
    
    # 2. 保存所有结果
    results_data = []
    for result in results:
        result_dict = {
            "iteration": result.iteration,
            "timestamp": result.timestamp,
            "risk_parameters": result.risk_params,
            "metrics": result.metrics,
            "stock_count": result.stock_count,
            "valid_trades": result.valid_trades
        }
        results_data.append(result_dict)
    
    results_path = os.path.join(output_dir, f"{method}_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # 3. 生成优化报告
    generate_report(results, best_result, output_dir, method, timestamp)

def generate_report(results: List[Stage2OptimizationResult],
                   best_result: Stage2OptimizationResult,
                   output_dir: str,
                   method: str,
                   timestamp: str):
    """生成优化报告"""
    
    report_path = os.path.join(output_dir, f"optimization_report_{timestamp}.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# Stock Quant 第二阶段风险管理参数优化报告\n\n")
        f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**优化方法:** {method}\n")
        f.write(f"**迭代次数:** {len(results)}\n\n")
        
        if best_result:
            f.write("## 最佳参数配置\n\n")
            f.write("### 第一阶段参数（保持）\n")
            f.write("```json\n")
            f.write(json.dumps({
                "ma_fast": best_result.params.get('ma_fast'),
                "ma_slow": best_result.params.get('ma_slow'),
                "rsi_oversold": best_result.params.get('rsi_oversold'),
                "rsi_overbought": best_result.params.get('rsi_overbought'),
                "bb_std": best_result.params.get('bb_std'),
                "score_threshold": best_result.params.get('score_threshold')
            }, indent=2))
            f.write("\n```\n\n")
            
            f.write("### 第二阶段风险管理参数\n")
            f.write("```json\n")
            f.write(json.dumps({
                k: v for k, v in best_result.risk_params.items()
                if k not in ['ma_fast', 'ma_slow', 'rsi_oversold', 'rsi_overbought', 'bb_std', 'score_threshold']
            }, indent=2))
            f.write("\n```\n\n")
            
            f.write("## 性能指标\n\n")
            metrics = best_result.metrics
            f.write(f"- **最大回撤:** {metrics.get('max_drawdown', 0):.4f} ({'✓' if metrics.get('max_drawdown', 1) < 0.15 else '✗'})\n")
            f.write(f"- **夏普比率:** {metrics.get('sharpe_ratio', 0):.4f} ({'✓' if metrics.get('sharpe_ratio', 0) > 0.8 else '✗'})\n")
            f.write(f"- **胜率:** {metrics.get('win_rate', 0):.4f} ({'✓' if metrics.get('win_rate', 0) > 0.55 else '✗'})\n")
            f.write(f"- **综合评分:** {metrics.get('composite_score', 0):.4f} ({'✓' if metrics.get('composite_score', 0) > 0.6 else '✗'})\n")
            f.write(f"- **总收益:** {metrics.get('total_return', 0):.4f}\n")
            f.write(f"- **有效交易数:** {metrics.get('valid_trades', 0)}\n\n")
            
            # 与第一阶段对比
            f.write("## 与第一阶段对比\n\n")
            # 这里需要第一阶段的数据，暂时留空
            
        # 参数重要性分析
        f.write("## 参数重要性分析\n\n")
        
        if results:
            # 收集参数与性能的关系
            param_importance = {}
            for param_name in results[0].risk_params.keys():
                if param_name not in ['ma_fast', 'ma_slow', 'rsi_oversold', 'rsi_overbought', 'bb_std', 'score_threshold']:
                    param_values = []
                    max_drawdowns = []
                    sharpe_ratios = []
                    composite_scores = []
                    
                    for result in results:
                        param_values.append(result.risk_params[param_name])
                        max_drawdowns.append(result.metrics.get('max_drawdown', 1))
                        sharpe_ratios.append(result.metrics.get('sharpe_ratio', 0))
                        composite_scores.append(result.metrics.get('composite_score', 0))
                    
                    # 计算参数与性能的相关性
                    if len(set(param_values)) > 1:
                        try:
                            # 转换为数值类型以计算相关性
                            if isinstance(param_values[0], bool):
                                param_numeric = [1 if v else 0 for v in param_values]
                            else:
                                param_numeric = param_values
                            
                            # 计算相关系数
                            max_drawdown_corr = np.corrcoef(param_numeric, max_drawdowns)[0, 1] if len(param_numeric) > 1 else 0
                            sharpe_corr = np.corrcoef(param_numeric, sharpe_ratios)[0, 1] if len(param_numeric) > 1 else 0
                            composite_corr = np.corrcoef(param_numeric, composite_scores)[0, 1] if len(param_numeric) > 1 else 0
                            
                            param_importance[param_name] = {
                                "max_drawdown_correlation": max_drawdown_corr,
                                "sharpe_correlation": sharpe_corr,
                                "composite_score_correlation": composite_corr
                            }
                        except:
                            pass
            
            f.write("| 参数 | 最大回撤相关性 | 夏普比率相关性 | 综合评分相关性 |\n")
            f.write("|------|---------------|---------------|---------------|\n")
            
            for param_name, importance in param_importance.items():
                max_drawdown_corr = importance.get("max_drawdown_correlation", 0)
                sharpe_corr = importance.get("sharpe_correlation", 0)
                composite_corr = importance.get("composite_score_correlation", 0)
                
                f.write(f"| {param_name} | {max_drawdown_corr:.4f} | {sharpe_corr:.4f} | {composite_corr:.4f} |\n")
        
        f.write("\n## 优化建议\n\n")
        f.write("1. **止损比例**: 负值越小（如-0.08），止损越严格，有助于降低最大回撤\n")
        f.write("2. **止盈比例**: 正值越大（如0.15），盈利潜力越大，但可能降低胜率\n")
        f.write("3. **仓位控制**: 较小的仓位（如0.10）有助于控制风险\n")
        f.write("4. **动态止损和移动止盈**: 启用这些功能通常能改善风险调整后收益\n")
        f.write("5. **最小持仓天数**: 适当增加（如3-5天）可避免频繁交易\n")
    
    print(f"优化报告已保存到: {report_path}")

def create_comparison_charts(stage1_metrics: Dict[str, float],
                            stage2_results: List[Stage2OptimizationResult],
                            best_stage2_result: Stage2OptimizationResult,
                            output_dir: str):
    """创建性能对比图表"""
    
    if not best_stage2_result:
        print("没有第二阶段最佳结果，无法创建对比图表")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 提取数据
    stage2_metrics = best_stage2_result.metrics
    
    # 创建对比图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Stock Quant 优化阶段对比分析', fontsize=16)
    
    # 1. 关键指标对比
    metrics_comparison = {
        'Stage 1': stage1_metrics,
        'Stage 2': stage2_metrics
    }
    
    metrics_to_compare = ['max_drawdown', 'sharpe_ratio', 'win_rate', 'composite_score']
    metric_names = ['最大回撤', '夏普比率', '胜率', '综合评分']
    
    stage1_values = [stage1_metrics.get(m, 0) for m in metrics_to_compare]
    stage2_values = [stage2_metrics.get(m, 0) for m in metrics_to_compare]
    
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, stage1_values, width, label='第一阶段', alpha=0.8)
    bars2 = ax1.bar(x + width/2, stage2_values, width, label='第二阶段', alpha=0.8)
    
    ax1.set_xlabel('指标')
    ax1.set_ylabel('数值')
    ax1.set_title('关键指标对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, rotation=45)
    ax1.legend()
    
    # 添加数值标签
    for bar, value in zip(bars1 + bars2, stage1_values + stage2_values):
        height = bar.get_height()
        ax1.annotate(f'{value:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 2. 最大回撤分布
    ax2 = axes[0, 1]
    max_drawdowns = [r.metrics.get('max_drawdown', 1) for r in stage2_results]
    ax2.hist(max_drawdowns, bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.15, color='r', linestyle='--', label='目标线 (15%)')
    ax2.axvline(x=stage2_metrics.get('max_drawdown', 1), color='g', linestyle='-', 
               label=f'最佳值 ({stage2_metrics.get("max_drawdown", 0):.4f})')
    ax2.set_xlabel('最大回撤')
    ax2.set_ylabel('频次')
    ax2.set_title('最大回撤分布')
    ax2.legend()
    
    # 3. 夏普比率 vs 最大回撤散点图
    ax3 = axes[1, 0]
    sharpe_ratios = [r.metrics.get('sharpe_ratio', 0) for r in stage2_results]
    max_drawdowns = [r.metrics.get('max_drawdown', 1) for r in stage2_results]
    
    scatter = ax3.scatter(max_drawdowns, sharpe_ratios, alpha=0.6, 
                         c=[r.metrics.get('composite_score', 0) for r in stage2_results],
                         cmap='viridis')
    
    # 标记最佳点
    ax3.scatter([stage2_metrics.get('max_drawdown', 1)], 
               [stage2_metrics.get('sharpe_ratio', 0)], 
               color='red', s=100, marker='*', label='最佳配置')
    
    # 添加目标区域
    ax3.axvline(x=0.15, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.8, color='g', linestyle='--', alpha=0.5)
    ax3.fill_between([0, 0.15], 0.8, 2, alpha=0.1, color='green', label='目标区域')
    
    ax3.set_xlabel('最大回撤')
    ax3.set_ylabel('夏普比率')
    ax3.set_title('夏普比率 vs 最大回撤')
    ax3.legend()
    plt.colorbar(scatter, ax=ax3, label='综合评分')
    
    # 4. 参数重要性热图（简化版）
    ax4 = axes[1, 1]
    if stage2_results and len(stage2_results) > 10:
        # 分析关键参数的影响
        key_params = ['stop_loss', 'take_profit', 'position_size']
        param_effects = {}
        
        for param in key_params:
            if param in stage2_results[0].risk_params:
                # 按参数值分组计算平均性能
                param_values = {}
                for result in stage2_results:
                    value = result.risk_params[param]
                    if value not in param_values:
                        param_values[value] = {'drawdowns': [], 'sharpe': []}
                    
                    param_values[value]['drawdowns'].append(result.metrics.get('max_drawdown', 1))
                    param_values[value]['sharpe'].append(result.metrics.get('sharpe_ratio', 0))
                
                # 计算平均效果
                avg_effects = []
                for value, metrics_list in param_values.items():
                    avg_drawdown = np.mean(metrics_list['drawdowns'])
                    avg_sharpe = np.mean(metrics_list['sharpe'])
                    avg_effects.append((value, avg_drawdown, avg_sharpe))
                
                param_effects[param] = avg_effects
        
        # 创建简单的文本说明
        ax4.text(0.1, 0.9, '关键参数效果分析:', fontsize=12, fontweight='bold')
        y_pos = 0.8
        
        for param, effects in list(param_effects.items())[:3]:  # 只显示前3个参数
            ax4.text(0.1, y_pos, f'{param}:', fontsize=10, fontweight='bold')
            y_pos -= 0.05
            
            for value, avg_drawdown, avg_sharpe in effects[:3]:  # 只显示前3个值
                ax4.text(0.15, y_pos, f'  {value}: 回撤={avg_drawdown:.3f}, 夏普={avg_sharpe:.3f}', 
                        fontsize=9)
                y_pos -= 0.04
            
            y_pos -= 0.02
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_axis_off()
    ax4.set_title('参数效果分析')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(output_dir, f"optimization_charts_{timestamp}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"对比图表已保存到: {chart_path}")
    return chart_path

def main():
    """主函数"""
    
    # 1. 加载第一阶段最佳参数
    stage1_params = {
        "ma_fast": 50,
        "ma_slow": 100, 
        "rsi_oversold": 28,
        "rsi_overbought": 70,
        "bb_std": 2.0,
        "score_threshold": 0.45
    }
    
    stage1_metrics = {
        "sharpe_ratio": 0.849816047538945,
        "max_drawdown": 0.1650714306409916,
        "win_rate": 0.6031993941811405,
        "profit_factor": 1.379731696839407,
        "total_return": 0.18560186404424364,
        "avg_trade": 0.014559945203362028,
        "valid_trades": 4,
        "composite_score": 0.08096409970344594
    }
    
    print("第一阶段最佳参数:")
    print(json.dumps(stage1_params, indent=2))
    print(f"\n第一阶段性能:")
    print(f"  最大回撤: {stage1_metrics['max_drawdown']:.4f}")
    print(f"  夏普比率: {stage1_metrics['sharpe_ratio']:.4f}")
    print(f"  胜率: {stage1_metrics['win_rate']:.4f}")
    print(f"  综合评分: {stage1_metrics['composite_score']:.4f}")
    print("=" * 80)
    
    # 2. 定义参数空间
    param_space = define_risk_parameter_space()
    print(f"风险管理参数空间包含 {len(param_space)} 个参数")
    
    # 3. 创建输出目录
    output_dir = "optimization_results/stage2"
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. 执行随机搜索优化
    print("\n开始第二阶段风险管理参数优化...")
    random_results, best_random_result = random_search(
        stage1_params, 
        param_space, 
        n_iterations=30,
        output_dir=output_dir
    )
    
    # 5. 如果未达到目标，执行网格搜索
    if best_random_result:
        best_metrics = best_random_result.metrics
        
        if best_metrics.get('max_drawdown', 1) >= 0.15 or best_metrics.get('sharpe_ratio', 0) < 0.8:
            print("\n随机搜索未达到主要目标，开始网格搜索...")
            
            # 选择关键参数进行网格搜索
            key_params = ['stop_loss', 'take_profit', 'position_size', 'dynamic_stop_loss']
            
            grid_results, best_grid_result = grid_search(
                stage1_params,
                param_space,
                key_params,
                output_dir=output_dir
            )
            
            # 选择最佳结果
            if best_grid_result and best_grid_result.metrics.get('composite_score', 0) > best_metrics.get('composite_score', 0):
                best_result = best_grid_result
                all_results = random_results + grid_results
            else:
                best_result = best_random_result
                all_results = random_results
        else:
            best_result = best_random_result
            all_results = random_results
    else:
        print("随机搜索未找到有效结果")
        return
    
    # 6. 创建对比图表
    if best_result and all_results:
        create_comparison_charts(
            stage1_metrics,
            all_results,
            best_result,
            output_dir
        )
    
    # 7. 最终报告
    if best_result:
        print("\n" + "=" * 80)
        print("第二阶段优化完成！")
        print("\n最佳配置性能:")
        print(f"  最大回撤: {best_result.metrics.get('max_drawdown', 0):.4f} ({'达到目标' if best_result.metrics.get('max_drawdown', 1) < 0.15 else '未达目标'})")
        print(f"  夏普比率: {best_result.metrics.get('sharpe_ratio', 0):.4f} ({'达到目标' if best_result.metrics.get('sharpe_ratio', 0) > 0.8 else '未达目标'})")
        print(f"  胜率: {best_result.metrics.get('win_rate', 0):.4f} ({'达到目标' if best_result.metrics.get('win_rate', 0) > 0.55 else '未达目标'})")
        print(f"  综合评分: {best_result.metrics.get('composite_score', 0):.4f} ({'达到目标' if best_result.metrics.get('composite_score', 0) > 0.6 else '未达目标'})")
        
        # 显示关键风险管理参数
        print("\n关键风险管理参数:")
        for param_name in ['stop_loss', 'take_profit', 'position_size', 'dynamic_stop_loss', 'trailing_stop']:
            if param_name in best_result.risk_params:
                value = best_result.risk_params[param_name]
                print(f"  {param_name}: {value}")
    
    print(f"\n结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()