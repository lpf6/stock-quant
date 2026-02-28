#!/usr/bin/env python3
"""
Stock Quant 参数优化器
基于网格搜索、随机搜索和贝叶斯优化的参数优化框架
"""

import itertools
import numpy as np
import pandas as pd
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptimizationResult:
    """优化结果"""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    iteration: int
    timestamp: str
    stock_count: int
    valid_trades: int

class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, backtest_func: Callable, output_dir: str = "optimization_results"):
        """
        初始化优化器
        
        Args:
            backtest_func: 回测函数，接受参数字典并返回性能指标字典
            output_dir: 输出目录
        """
        self.backtest_func = backtest_func
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 优化历史记录
        self.history: List[OptimizationResult] = []
        self.best_result: OptimizationResult = None
        
        # 参数空间定义
        self.param_spaces = {
            "stage1": {  # 第一阶段：基础参数优化
                "ma_fast": [3, 5, 8, 10, 12, 15, 20, 30, 50],
                "ma_slow": [10, 12, 15, 20, 30, 50, 100],
                "rsi_oversold": [25, 28, 30, 32, 35, 38],
                "rsi_overbought": [65, 68, 70, 72, 75, 78],
                "bb_std": [1.5, 1.8, 2.0, 2.2, 2.5],
                "score_threshold": [0.3, 0.35, 0.4, 0.45, 0.5]
            },
            "stage2": {  # 第二阶段：风险管理参数
                "stop_loss": [-0.03, -0.05, -0.08],
                "take_profit": [0.08, 0.12, 0.15],
                "position_size": [0.10, 0.15, 0.20]
            },
            "stage3": {  # 第三阶段：策略权重
                "trend_weight": [0.25, 0.30, 0.35, 0.40],
                "mean_reversion_weight": [0.20, 0.25, 0.30, 0.35],
                "momentum_weight": [0.20, 0.25, 0.30],
                "volatility_weight": [0.10, 0.15, 0.20]
            }
        }
        
        # 指标权重（用于综合评分）
        self.metric_weights = {
            "sharpe_ratio": 0.40,      # 夏普比率权重
            "max_drawdown": -0.20,     # 最大回撤（负权重）
            "win_rate": 0.15,          # 胜率
            "profit_factor": 0.10,     # 盈亏比
            "total_return": 0.10,      # 总收益率
            "avg_trade": 0.05          # 平均交易盈利
        }
    
    def evaluate_parameters(self, params: Dict[str, Any], iteration: int, stock_count: int = 7) -> OptimizationResult:
        """
        评估参数组合
        
        Args:
            params: 参数字典
            iteration: 迭代次数
            stock_count: 测试股票数量
            
        Returns:
            OptimizationResult: 优化结果
        """
        try:
            # 运行回测
            metrics = self.backtest_func(params, stock_count)
            
            # 计算综合评分
            composite_score = 0
            for metric_name, weight in self.metric_weights.items():
                if metric_name in metrics:
                    metric_value = metrics[metric_name]
                    
                    # 特殊处理负权重指标
                    if metric_name == "max_drawdown":
                        # 最大回撤越小越好
                        metric_score = 1.0 - min(abs(metric_value), 0.5) / 0.5
                        composite_score += metric_score * weight
                    else:
                        # 其他指标越大越好（归一化）
                        if metric_name == "sharpe_ratio":
                            norm_value = min(max(metric_value / 3.0, 0), 1)
                        elif metric_name == "win_rate":
                            norm_value = min(max((metric_value - 0.5) / 0.5, 0), 1)
                        elif metric_name == "profit_factor":
                            norm_value = min(max((metric_value - 1.0) / 2.0, 0), 1)
                        elif metric_name == "total_return":
                            norm_value = min(max(metric_value / 0.5, 0), 1)
                        elif metric_name == "avg_trade":
                            norm_value = min(max(metric_value / 0.05, 0), 1)
                        else:
                            norm_value = min(max(metric_value, 0), 1)
                        
                        composite_score += norm_value * weight
            
            # 限制评分范围
            metrics["composite_score"] = max(0, min(composite_score, 1))
            
            result = OptimizationResult(
                params=params.copy(),
                metrics=metrics,
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                stock_count=stock_count,
                valid_trades=metrics.get("valid_trades", 0)
            )
            
            self.history.append(result)
            
            # 更新最佳结果
            if self.best_result is None or metrics["composite_score"] > self.best_result.metrics["composite_score"]:
                self.best_result = result
            
            return result
            
        except Exception as e:
            print(f"参数评估失败: {e}")
            # 返回低评分结果
            return OptimizationResult(
                params=params.copy(),
                metrics={"composite_score": 0, "sharpe_ratio": -10, "max_drawdown": 1.0},
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                stock_count=stock_count,
                valid_trades=0
            )
    
    def grid_search(self, stage: str = "stage1", max_combinations: int = 100, stock_count: int = 7) -> List[OptimizationResult]:
        """
        网格搜索优化
        
        Args:
            stage: 优化阶段（stage1, stage2, stage3）
            max_combinations: 最大组合数
            stock_count: 测试股票数量
            
        Returns:
            List[OptimizationResult]: 优化结果列表
        """
        print(f"\n开始网格搜索优化（阶段: {stage}, 股票数: {stock_count}）")
        
        param_space = self.param_spaces[stage]
        param_names = list(param_space.keys())
        param_values = [param_space[name] for name in param_names]
        
        # 生成所有组合
        all_combinations = list(itertools.product(*param_values))
        
        # 随机抽样以减少计算量
        if len(all_combinations) > max_combinations:
            print(f"参数组合总数: {len(all_combinations)} > {max_combinations}，进行随机抽样")
            all_combinations = random.sample(all_combinations, max_combinations)
        else:
            print(f"参数组合总数: {len(all_combinations)}")
        
        results = []
        
        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))
            
            print(f"测试组合 {i+1}/{len(all_combinations)}: {params}")
            
            result = self.evaluate_parameters(params, i + 1, stock_count)
            results.append(result)
            
            # 打印进度
            if (i + 1) % 10 == 0 or (i + 1) == len(all_combinations):
                print(f"  进度: {i+1}/{len(all_combinations)}, "
                      f"最佳评分: {self.best_result.metrics['composite_score']:.3f}, "
                      f"最佳夏普: {self.best_result.metrics.get('sharpe_ratio', 0):.3f}")
        
        return results
    
    def random_search(self, stage: str = "stage1", n_iterations: int = 50, stock_count: int = 7) -> List[OptimizationResult]:
        """
        随机搜索优化
        
        Args:
            stage: 优化阶段
            n_iterations: 迭代次数
            stock_count: 测试股票数量
            
        Returns:
            List[OptimizationResult]: 优化结果列表
        """
        print(f"\n开始随机搜索优化（阶段: {stage}, 迭代次数: {n_iterations}）")
        
        param_space = self.param_spaces[stage]
        results = []
        
        for i in range(n_iterations):
            # 随机选择参数
            params = {}
            for param_name, values in param_space.items():
                if isinstance(values[0], (int, float)):
                    # 连续参数：在范围内随机选择
                    min_val = min(values)
                    max_val = max(values)
                    
                    if isinstance(values[0], int):
                        params[param_name] = random.randint(min_val, max_val)
                    else:
                        params[param_name] = random.uniform(min_val, max_val)
                else:
                    # 离散参数：随机选择
                    params[param_name] = random.choice(values)
            
            print(f"随机搜索 {i+1}/{n_iterations}: {params}")
            
            result = self.evaluate_parameters(params, i + 1, stock_count)
            results.append(result)
            
            # 打印进度
            if (i + 1) % 10 == 0 or (i + 1) == n_iterations:
                print(f"  进度: {i+1}/{n_iterations}, "
                      f"最佳评分: {self.best_result.metrics['composite_score']:.3f}, "
                      f"最佳夏普: {self.best_result.metrics.get('sharpe_ratio', 0):.3f}")
        
        return results
    
    def bayesian_optimization(self, stage: str = "stage1", n_iterations: int = 30, 
                             init_points: int = 5, stock_count: int = 7) -> List[OptimizationResult]:
        """
        贝叶斯优化（简化版，使用代理模型）
        
        Args:
            stage: 优化阶段
            n_iterations: 迭代次数
            init_points: 初始随机点数
            stock_count: 测试股票数量
            
        Returns:
            List[OptimizationResult]: 优化结果列表
        """
        print(f"\n开始贝叶斯优化（阶段: {stage}, 迭代次数: {n_iterations}）")
        
        # 先进行随机搜索作为初始点
        print(f"生成 {init_points} 个初始随机点...")
        init_results = self.random_search(stage, init_points, stock_count)
        
        # 简化版贝叶斯优化：使用改进的随机搜索
        param_space = self.param_spaces[stage]
        results = init_results.copy()
        
        # 分析最佳参数的特征
        best_params = self.best_result.params if self.best_result else {}
        
        for i in range(len(init_results), n_iterations):
            # 基于最佳参数进行局部搜索
            params = {}
            for param_name, values in param_space.items():
                if param_name in best_params and i % 2 == 0:
                    # 50%的概率使用最佳参数附近的值
                    best_val = best_params[param_name]
                    
                    if isinstance(best_val, (int, float)):
                        # 在最佳值附近采样
                        if isinstance(best_val, int):
                            min_val = max(min(values), best_val - 2)
                            max_val = min(max(values), best_val + 2)
                            params[param_name] = random.randint(min_val, max_val)
                        else:
                            min_val = max(min(values), best_val * 0.8)
                            max_val = min(max(values), best_val * 1.2)
                            params[param_name] = random.uniform(min_val, max_val)
                    else:
                        params[param_name] = best_val
                else:
                    # 随机搜索
                    if isinstance(values[0], (int, float)):
                        min_val = min(values)
                        max_val = max(values)
                        
                        if isinstance(values[0], int):
                            params[param_name] = random.randint(min_val, max_val)
                        else:
                            params[param_name] = random.uniform(min_val, max_val)
                    else:
                        params[param_name] = random.choice(values)
            
            print(f"贝叶斯优化 {i+1}/{n_iterations}: {params}")
            
            result = self.evaluate_parameters(params, i + 1, stock_count)
            results.append(result)
            
            # 更新最佳参数
            if result.metrics["composite_score"] > self.best_result.metrics["composite_score"]:
                best_params = result.params.copy()
            
            # 打印进度
            if (i + 1) % 10 == 0 or (i + 1) == n_iterations:
                print(f"  进度: {i+1}/{n_iterations}, "
                      f"最佳评分: {self.best_result.metrics['composite_score']:.3f}, "
                      f"最佳夏普: {self.best_result.metrics.get('sharpe_ratio', 0):.3f}")
        
        return results
    
    def save_results(self, stage: str = "stage1", method: str = "grid_search"):
        """保存优化结果"""
        if not self.history:
            print("没有优化结果可保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存所有结果
        results_data = []
        for result in self.history:
            results_data.append({
                "iteration": result.iteration,
                "timestamp": result.timestamp,
                "params": result.params,
                "metrics": result.metrics,
                "stock_count": result.stock_count,
                "valid_trades": result.valid_trades
            })
        
        # 保存最佳结果
        best_data = {
            "stage": stage,
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "best_params": self.best_result.params,
            "best_metrics": self.best_result.metrics,
            "optimization_iterations": len(self.history)
        }
        
        # 创建阶段目录
        stage_dir = os.path.join(self.output_dir, f"stage_{stage}")
        os.makedirs(stage_dir, exist_ok=True)
        
        # 保存文件
        results_file = os.path.join(stage_dir, f"{method}_results_{timestamp}.json")
        best_file = os.path.join(stage_dir, f"{method}_best_{timestamp}.json")
        config_file = os.path.join(stage_dir, f"best_config_{timestamp}.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        with open(best_file, 'w', encoding='utf-8') as f:
            json.dump(best_data, f, indent=2, ensure_ascii=False)
        
        # 保存配置（用于生产环境）
        config = {
            "optimization_date": datetime.now().isoformat(),
            "stage": stage,
            "method": method,
            "parameters": self.best_result.params,
            "metrics_summary": self.best_result.metrics,
            "stock_count": self.best_result.stock_count
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存:")
        print(f"  所有结果: {results_file}")
        print(f"  最佳结果: {best_file}")
        print(f"  配置: {config_file}")
        
        return results_file, best_file, config_file
    
    def generate_report(self, stage: str = "stage1"):
        """生成优化报告"""
        if not self.history:
            return "没有优化数据可生成报告"
        
        # 提取指标数据
        iterations = [r.iteration for r in self.history]
        composite_scores = [r.metrics.get("composite_score", 0) for r in self.history]
        sharpe_ratios = [r.metrics.get("sharpe_ratio", 0) for r in self.history]
        max_drawdowns = [r.metrics.get("max_drawdown", 0) for r in self.history]
        
        # 创建报告
        report_lines = [
            f"# Stock Quant 参数优化报告 - 阶段 {stage}",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"优化迭代次数: {len(self.history)}",
            f"测试股票数: {self.history[0].stock_count if self.history else 0}",
            "",
            "## 最佳参数配置",
            ""
        ]
        
        if self.best_result:
            # 最佳参数
            for param_name, param_value in self.best_result.params.items():
                report_lines.append(f"- **{param_name}**: {param_value}")
            
            report_lines.extend([
                "",
                "## 最佳性能指标",
                ""
            ])
            
            # 最佳指标
            for metric_name, metric_value in self.best_result.metrics.items():
                if metric_name != "composite_score":
                    report_lines.append(f"- **{metric_name}**: {metric_value:.4f}")
            
            report_lines.append(f"- **综合评分**: {self.best_result.metrics.get('composite_score', 0):.4f}")
            
            # 评估是否达到目标
            report_lines.extend([
                "",
                "## 目标达成情况",
                ""
            ])
            
            targets = {
                "sharpe_ratio > 0.8": self.best_result.metrics.get("sharpe_ratio", 0) > 0.8,
                "max_drawdown < 15%": abs(self.best_result.metrics.get("max_drawdown", 1.0)) < 0.15,
                "composite_score > 0.6": self.best_result.metrics.get("composite_score", 0) > 0.6,
                "win_rate > 55%": self.best_result.metrics.get("win_rate", 0) > 0.55
            }
            
            for target, achieved in targets.items():
                status = "✅ 达成" if achieved else "❌ 未达成"
                report_lines.append(f"- {target}: {status}")
        
        # 收敛分析
        report_lines.extend([
            "",
            "## 收敛分析",
            ""
        ])
        
        if len(self.history) > 10:
            # 计算收敛速度
            convergence_data = []
            window_size = max(1, len(self.history) // 10)
            
            for i in range(window_size, len(self.history), window_size):
                window = self.history[max(0, i-window_size):i]
                avg_score = np.mean([r.metrics.get("composite_score", 0) for r in window])
                convergence_data.append(avg_score)
            
            if len(convergence_data) > 1:
                improvement_rate = (convergence_data[-1] - convergence_data[0]) / len(convergence_data)
                if improvement_rate > 0.01:
                    report_lines.append("- 收敛状态: **持续改进中**")
                elif improvement_rate > 0:
                    report_lines.append("- 收敛状态: **缓慢改进**")
                elif improvement_rate > -0.005:
                    report_lines.append("- 收敛状态: **基本收敛**")
                else:
                    report_lines.append("- 收敛状态: **可能发散**")
            
            # 参数重要性分析（简化）
            report_lines.extend([
                "",
                "## 参数重要性分析",
                ""
            ])
            
            # 收集参数值及其对应的评分
            param_importance = {}
            for result in self.history:
                for param_name, param_value in result.params.items():
                    if param_name not in param_importance:
                        param_importance[param_name] = []
                    
                    param_importance[param_name].append({
                        "value": param_value,
                        "score": result.metrics.get("composite_score", 0)
                    })
            
            # 计算每个参数的性能范围
            for param_name, data in param_importance.items():
                if len(data) > 5:
                    # 按参数值分组
                    param_values = {}
                    for item in data:
                        val = str(item["value"])
                        if val not in param_values:
                            param_values[val] = []
                        param_values[val].append(item["score"])
                    
                    # 计算每个值的平均评分
                    avg_scores = {val: np.mean(scores) for val, scores in param_values.items()}
                    
                    if avg_scores:
                        best_val = max(avg_scores.items(), key=lambda x: x[1])
                        worst_val = min(avg_scores.items(), key=lambda x: x[1])
                        
                        report_lines.append(
                            f"- **{param_name}**: "
                            f"最佳值={best_val[0]}({best_val[1]:.3f}), "
                            f"最差值={worst_val[0]}({worst_val[1]:.3f})"
                        )
        
        report = "\n".join(report_lines)
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"stage_{stage}", f"optimization_report_{timestamp}.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"优化报告保存到: {report_file}")
        
        return report
    
    def visualize_optimization(self, stage: str = "stage1"):
        """可视化优化过程"""
        if len(self.history) < 5:
            print("数据不足，无法生成可视化图表")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Stock Quant 参数优化可视化 - 阶段 {stage}', fontsize=16, fontweight='bold')
        
        # 提取数据
        iterations = [r.iteration for r in self.history]
        composite_scores = [r.metrics.get("composite_score", 0) for r in self.history]
        sharpe_ratios = [r.metrics.get("sharpe_ratio", 0) for r in self.history]
        max_drawdowns = [abs(r.metrics.get("max_drawdown", 0)) for r in self.history]
        
        # 1. 综合评分收敛曲线
        axes[0, 0].plot(iterations, composite_scores, 'b-', alpha=0.7, label='综合评分')
        axes[0, 0].fill_between(iterations, 
                               [max(0, s - 0.05) for s in composite_scores],
                               [min(1, s + 0.05) for s in composite_scores],
                               alpha=0.2, color='blue')
        axes[0, 0].set_title('综合评分收敛曲线')
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('综合评分')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. 夏普比率变化
        axes[0, 1].plot(iterations, sharpe_ratios, 'g-', alpha=0.7, label='夏普比率')
        axes[0, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='目标(0.8)')
        axes[0, 1].fill_between(iterations, 
                               [max(-1, s - 0.2) for s in sharpe_ratios],
                               [s + 0.2 for s in sharpe_ratios],
                               alpha=0.2, color='green')
        axes[0, 1].set_title('夏普比率变化')
        axes[0, 1].set_xlabel('迭代次数')
        axes[0, 1].set_ylabel('夏普比率')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. 最大回撤变化
        axes[1, 0].plot(iterations, max_drawdowns, 'r-', alpha=0.7, label='最大回撤')
        axes[1, 0].axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='目标(15%)')
        axes[1, 0].fill_between(iterations, 
                               [0 for _ in max_drawdowns],
                               max_drawdowns,
                               alpha=0.2, color='red')
        axes[1, 0].set_title('最大回撤变化')
        axes[1, 0].set_xlabel('迭代次数')
        axes[1, 0].set_ylabel('最大回撤')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 4. 参数相关性热图（简化）
        if len(self.history) > 10:
            # 选择最重要的几个参数
            important_params = {}
            param_sample = {}
            
            # 抽样一些参数组合进行分析
            sample_size = min(20, len(self.history))
            sample_indices = np.linspace(0, len(self.history)-1, sample_size, dtype=int)
            
            for idx in sample_indices:
                result = self.history[idx]
                for param_name, param_value in result.params.items():
                    if param_name not in param_sample:
                        param_sample[param_name] = []
                    param_sample[param_name].append(param_value)
            
            # 计算参数与评分的相关性（简化）
            if param_sample:
                param_names = list(param_sample.keys())
                n_params = min(5, len(param_names))
                
                if n_params > 1:
                    # 创建子图显示前几个参数
                    axes[1, 1].clear()
                    
                    for i, param_name in enumerate(param_names[:n_params]):
                        param_values = param_sample[param_name]
                        param_scores = [self.history[idx].metrics.get("composite_score", 0) 
                                       for idx in sample_indices]
                        
                        # 归一化参数值
                        if isinstance(param_values[0], (int, float)):
                            min_val = min(param_values)
                            max_val = max(param_values)
                            norm_values = [(v - min_val) / (max_val - min_val + 1e-8) 
                                          for v in param_values]
                            
                            axes[1, 1].scatter(norm_values, param_scores, 
                                             alpha=0.6, label=param_name, s=50)
                    
                    axes[1, 1].set_title('参数与评分关系')
                    axes[1, 1].set_xlabel('参数值（归一化）')
                    axes[1, 1].set_ylabel('综合评分')
                    axes[1, 1].grid(True, alpha=0.3)
                    axes[1, 1].legend(fontsize=8)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = os.path.join(self.output_dir, f"stage_{stage}", f"optimization_charts_{timestamp}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表保存到: {chart_file}")
        
        return chart_file

def create_backtest_function():
    """创建回测函数（适配优化框架）"""
    # 导入原始回测代码中的必要函数
    # 注意：这里需要引用backtest_simplified.py中的函数
    # 由于Python模块导入问题，我们在这里重新定义一些关键函数
    
    def simplified_backtest(params: Dict[str, Any], stock_count: int = 7) -> Dict[str, float]:
        """
        简化回测函数（用于优化框架）
        
        Args:
            params: 参数字典
            stock_count: 股票数量
            
        Returns:
            Dict[str, float]: 性能指标
        """
        # 这里使用简化的模拟回测
        # 在实际应用中，应该调用真实的回测逻辑
        
        np.random.seed(42)  # 固定随机种子
        
        # 提取参数
        ma_fast = params.get("ma_fast", 5)
        ma_slow = params.get("ma_slow", 20)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)
        bb_std = params.get("bb_std", 2.0)
        score_threshold = params.get("score_threshold", 0.4)
        
        # 模拟性能指标（基于参数质量）
        # 好的参数应该产生更好的性能
        
        # 参数质量评分
        param_score = 0
        
        # MA参数：快速MA应该小于慢速MA
        if ma_fast < ma_slow:
            param_score += 0.2
        
        # RSI参数：超买应该大于超卖
        if rsi_overbought > rsi_oversold:
            param_score += 0.2
        
        # 布林带参数：标准差在合理范围内
        if 1.5 <= bb_std <= 2.5:
            param_score += 0.2
        
        # 阈值参数：在合理范围内
        if 0.3 <= score_threshold <= 0.5:
            param_score += 0.2
        
        # 模拟回测结果
        sharpe_base = 0.5  # 基准夏普比率
        sharpe_bonus = param_score * 0.5  # 参数质量带来的提升
        
        # 添加随机性
        random_factor = np.random.uniform(-0.2, 0.2)
        
        sharpe_ratio = max(0, sharpe_base + sharpe_bonus + random_factor)
        
        # 其他指标
        max_drawdown = max(0, 0.2 - param_score * 0.1 + np.random.uniform(-0.05, 0.05))
        win_rate = max(0.4, 0.5 + param_score * 0.1 + np.random.uniform(-0.05, 0.05))
        profit_factor = max(1.0, 1.2 + param_score * 0.2 + np.random.uniform(-0.1, 0.1))
        total_return = max(0, 0.1 + param_score * 0.15 + np.random.uniform(-0.05, 0.05))
        avg_trade = max(0, 0.01 + param_score * 0.01 + np.random.uniform(-0.005, 0.005))
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_return": total_return,
            "avg_trade": avg_trade,
            "valid_trades": int(stock_count * (win_rate + 0.1))
        }
    
    return simplified_backtest

def main():
    """主函数：演示参数优化"""
    print("Stock Quant 参数优化器")
    print("=" * 60)
    
    # 创建回测函数
    backtest_func = create_backtest_function()
    
    # 创建优化器
    optimizer = ParameterOptimizer(backtest_func)
    
    # 第一阶段优化：基础参数
    print("\n第一阶段：基础参数优化")
    stage1_results = optimizer.grid_search("stage1", max_combinations=50, stock_count=7)
    
    optimizer.save_results("stage1", "grid_search")
    optimizer.generate_report("stage1")
    optimizer.visualize_optimization("stage1")
    
    # 检查是否达到目标
    if optimizer.best_result and optimizer.best_result.metrics.get("sharpe_ratio", 0) > 0.8:
        print("\n✅ 第一阶段已达成目标 (夏普比率 > 0.8)")
    else:
        print("\n⚠️ 第一阶段未达成目标，继续优化...")
        # 进行随机搜索
        optimizer.random_search("stage1", n_iterations=20, stock_count=7)
        optimizer.save_results("stage1", "random_search")
        optimizer.generate_report("stage1")
        optimizer.visualize_optimization("stage1")
    
    print("\n优化完成！")
    print(f"最佳综合评分: {optimizer.best_result.metrics['composite_score']:.3f}")
    print(f"最佳夏普比率: {optimizer.best_result.metrics.get('sharpe_ratio', 0):.3f}")
    print(f"最佳参数: {optimizer.best_result.params}")

if __name__ == "__main__":
    main()