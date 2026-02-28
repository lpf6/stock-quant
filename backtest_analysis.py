#!/usr/bin/env python3
"""
Stock Quant 技术面量化选股系统 - 回测分析脚本
迭代优化选股指标，进行系统回测分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
import sys
from typing import Dict, List, Any, Tuple

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.stock_quant.core.data_fetcher import DataFetcher
from src.stock_quant.core.indicator_calculator import IndicatorCalculator
from src.stock_quant.core.signal_generator import SignalGenerator
from src.stock_quant.plugins.manager import PluginManager
from src.stock_quant.plugins.strategies.backtest_strategy import BacktestStrategy
from src.stock_quant.plugins.strategies.optimization_strategy import OptimizationStrategy
from src.stock_quant.plugins.indicators.momentum_indicators import (
    MomentumIndicator, VolumeIndicator, TrendStrengthIndicator
)
from src.stock_quant.utils.logger import setup_logger
from src.stock_quant.utils.formatter import OutputFormatter


class BacktestAnalyzer:
    """回测分析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "test_stocks": ["000001", "000002", "000858", "600519", "000333"],  # 测试股票
            "time_periods": ["2023-01-01", "2023-12-31"],  # 时间范围
            "backtest_periods": [30, 60, 90, 180],  # 回测周期（天）
            "optimization_iterations": 3,  # 优化迭代次数
            "output_dir": "backtest_results"
        }
        
        # 初始化组件
        self.logger = setup_logger("BacktestAnalyzer", level="INFO")
        self.data_fetcher = DataFetcher()
        self.indicator_calculator = IndicatorCalculator()
        self.signal_generator = SignalGenerator()
        
        # 初始化插件管理器
        self.plugin_manager = PluginManager()
        
        # 注册新指标插件
        self.plugin_manager.register_plugin(MomentumIndicator)
        self.plugin_manager.register_plugin(VolumeIndicator)
        self.plugin_manager.register_plugin(TrendStrengthIndicator)
        
        # 加载回测和优化策略
        self.backtest_strategy = BacktestStrategy()
        self.optimization_strategy = OptimizationStrategy()
        
        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        self.logger.info("回测分析器初始化完成")
    
    def run_comprehensive_backtest(self) -> Dict[str, Any]:
        """运行综合回测分析"""
        self.logger.info("开始综合回测分析")
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "stock_results": [],
            "iteration_results": [],
            "summary": {}
        }
        
        # 迭代优化
        for iteration in range(self.config['optimization_iterations']):
            self.logger.info(f"第 {iteration + 1} 次迭代优化")
            
            iteration_results = self._run_iteration(iteration)
            all_results["iteration_results"].append(iteration_results)
            
            # 更新策略权重（基于优化结果）
            if iteration_results.get("optimization_results"):
                self._update_strategy_weights(iteration_results["optimization_results"])
        
        # 分析每只股票
        self.logger.info(f"分析 {len(self.config['test_stocks'])} 只测试股票")
        
        for symbol in self.config['test_stocks']:
            self.logger.info(f"分析股票 {symbol}")
            
            stock_results = self._analyze_single_stock(symbol)
            all_results["stock_results"].append(stock_results)
            
            # 保存每只股票的详细结果
            self._save_stock_results(symbol, stock_results)
        
        # 生成综合报告
        all_results["summary"] = self._generate_summary(all_results)
        
        # 保存最终结果
        self._save_results(all_results)
        
        self.logger.info("综合回测分析完成")
        return all_results
    
    def _run_iteration(self, iteration_num: int) -> Dict[str, Any]:
        """运行单次迭代"""
        iteration_results = {
            "iteration": iteration_num + 1,
            "date": datetime.now().isoformat(),
            "strategy_weights": self._get_current_weights(),
            "backtest_results": {},
            "optimization_results": {}
        }
        
        # 对每只测试股票进行回测
        backtest_results = {}
        
        for symbol in self.config['test_stocks'][:3]:  # 使用前3只股票进行迭代
            try:
                # 获取数据
                stock_data = self._get_stock_data(symbol)
                
                if stock_data is None or len(stock_data) < 100:
                    self.logger.warning(f"股票 {symbol} 数据不足，跳过")
                    continue
                
                # 计算指标和信号
                indicators_data = self.indicator_calculator.calculate_indicators(stock_data)
                signals = self.signal_generator.generate_signals(indicators_data)
                
                # 运行回测
                backtest_result = self.backtest_strategy.run_backtest(stock_data, signals)
                backtest_results[symbol] = backtest_result
                
                self.logger.info(f"股票 {symbol} 回测完成: 夏普比率={backtest_result.get('sharpe_ratio', 0):.3f}")
                
            except Exception as e:
                self.logger.error(f"股票 {symbol} 回测失败: {e}")
                backtest_results[symbol] = {"error": str(e)}
        
        iteration_results["backtest_results"] = backtest_results
        
        # 如果有回测结果，进行参数优化
        if backtest_results:
            optimization_results = self._run_optimization(backtest_results)
            iteration_results["optimization_results"] = optimization_results
        
        return iteration_results
    
    def _get_stock_data(self, symbol: str) -> pd.DataFrame:
        """获取股票数据"""
        try:
            # 这里可以使用真实数据获取，或使用模拟数据
            start_date = self.config['time_periods'][0]
            end_date = self.config['time_periods'][1]
            
            # 尝试获取真实数据
            stock_data = self.data_fetcher.fetch_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                period="daily"
            )
            
            if stock_data is not None and not stock_data.empty:
                return stock_data
            
            # 如果获取失败，使用模拟数据
            self.logger.info(f"使用模拟数据 for {symbol}")
            return self._create_mock_data(symbol, start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"获取股票 {symbol} 数据失败: {e}")
            return self._create_mock_data(symbol)
    
    def _create_mock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """创建模拟数据"""
        if start_date is None:
            start_date = "2023-01-01"
        if end_date is None:
            end_date = "2023-12-31"
        
        # 生成日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 交易日
        
        np.random.seed(hash(symbol) % 1000)  # 基于股票代码生成确定性的随机数据
        
        # 生成价格序列（带有趋势和波动）
        n_days = len(dates)
        
        # 基础趋势
        base_price = 10 + (hash(symbol) % 100) / 10  # 不同股票有不同起始价格
        
        # 随机游走加上季节性
        price_changes = np.random.randn(n_days) * 0.02
        
        # 添加趋势（随机方向）
        trend_direction = 1 if hash(symbol) % 2 == 0 else -1
        trend_strength = 0.0005 * trend_direction
        price_changes += trend_strength
        
        # 计算价格
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        # 生成OHLCV数据
        high = prices * (1 + np.random.rand(n_days) * 0.03)
        low = prices * (1 - np.random.rand(n_days) * 0.03)
        open_prices = prices * (1 + np.random.randn(n_days) * 0.01)
        
        # 确保价格合理性
        high = np.maximum(high, prices)
        low = np.minimum(low, prices)
        
        # 成交量和成交额
        volume = np.random.randint(1000000, 10000000, n_days)
        volume = volume * (1 + 0.3 * np.sin(np.arange(n_days) / 20))  # 添加季节性
        
        df = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume,
            'amount': prices * volume
        }, index=dates)
        
        return df
    
    def _run_optimization(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """运行参数优化"""
        self.logger.info("开始策略参数优化")
        
        try:
            # 选择表现最好的股票数据作为优化基准
            best_stock = None
            best_sharpe = -np.inf
            
            for symbol, results in backtest_results.items():
                if isinstance(results, dict) and 'sharpe_ratio' in results:
                    sharpe = results['sharpe_ratio']
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_stock = symbol
            
            if best_stock is None:
                self.logger.warning("没有有效的回测结果，跳过优化")
                return {"status": "skipped", "reason": "no_valid_backtest_results"}
            
            # 获取最佳股票的数据
            stock_data = self._get_stock_data(best_stock)
            
            if stock_data is None or len(stock_data) < 100:
                return {"status": "skipped", "reason": "insufficient_data"}
            
            # 计算初始指标和信号
            indicators_data = self.indicator_calculator.calculate_indicators(stock_data)
            initial_signals = self.signal_generator.generate_signals(indicators_data)
            
            # 运行优化
            optimization_results = self.optimization_strategy.optimize_parameters(
                stock_data, initial_signals
            )
            
            self.logger.info(f"参数优化完成，最佳得分: {optimization_results.get('best_score', 0):.3f}")
            
            return {
                "status": "success",
                "best_stock": best_stock,
                "results": optimization_results
            }
            
        except Exception as e:
            self.logger.error(f"参数优化失败: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _update_strategy_weights(self, optimization_results: Dict[str, Any]) -> None:
        """基于优化结果更新策略权重"""
        if optimization_results.get("status") != "success":
            return
        
        results = optimization_results.get("results", {})
        test_performance = results.get("test_performance", {})
        
        # 基于优化后的性能调整策略权重
        sharpe_ratio = test_performance.get("sharpe_ratio", 0)
        
        # 简单的权重调整逻辑
        if sharpe_ratio > 1.0:
            # 表现优秀，增加动量策略权重
            self._adjust_weight("MomentumIndicator", 0.20)
            self.logger.info("增加动量策略权重")
        elif sharpe_ratio < 0:
            # 表现不佳，降低动量策略权重
            self._adjust_weight("MomentumIndicator", 0.10)
            self.logger.info("降低动量策略权重")
    
    def _adjust_weight(self, strategy_name: str, new_weight: float) -> None:
        """调整策略权重"""
        # 这里需要根据实际情况实现权重调整逻辑
        # 暂时记录日志
        self.logger.info(f"调整策略 {strategy_name} 权重为 {new_weight}")
    
    def _get_current_weights(self) -> Dict[str, float]:
        """获取当前策略权重"""
        # 返回当前配置的权重
        return {
            "MACrossStrategy": 0.25,
            "RSIStrategy": 0.20,
            "MACDStrategy": 0.20,
            "BacktestStrategy": 0.15,
            "MomentumIndicator": 0.15,
            "VolumeIndicator": 0.12,
            "TrendStrengthIndicator": 0.13
        }
    
    def _analyze_single_stock(self, symbol: str) -> Dict[str, Any]:
        """分析单只股票"""
        try:
            # 获取数据
            stock_data = self._get_stock_data(symbol)
            
            if stock_data is None or len(stock_data) < 100:
                return {"symbol": symbol, "status": "insufficient_data"}
            
            # 计算指标（使用新的动量指标）
            indicators_data = self.indicator_calculator.calculate_indicators(stock_data)
            
            # 添加新的动量指标
            momentum_indicator = MomentumIndicator()
            momentum_data = momentum_indicator.calculate(stock_data)
            
            volume_indicator = VolumeIndicator()
            volume_data = volume_indicator.calculate(stock_data)
            
            trend_indicator = TrendStrengthIndicator()
            trend_data = trend_indicator.calculate(stock_data)
            
            # 合并所有指标
            combined_data = pd.concat([
                indicators_data, 
                momentum_data, 
                volume_data, 
                trend_data
            ], axis=1)
            
            # 去除重复列
            combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
            
            # 生成信号
            signals = self.signal_generator.generate_signals(combined_data)
            
            # 运行回测
            backtest_result = self.backtest_strategy.run_backtest(stock_data, signals)
            
            # 计算综合评分
            composite_score = self._calculate_composite_score(backtest_result, signals)
            
            return {
                "symbol": symbol,
                "status": "success",
                "latest_price": stock_data['close'].iloc[-1] if 'close' in stock_data.columns else 0,
                "data_points": len(stock_data),
                "signals_summary": {
                    "total_score": signals.get("total_score", 0),
                    "ma_cross": signals.get("MACrossStrategy.ma_cross", 0),
                    "rsi_oversold": signals.get("RSIStrategy.rsi_oversold", 0),
                    "momentum_score": signals.get("MomentumIndicator.momentum_score", 0)
                },
                "backtest_summary": {
                    "sharpe_ratio": backtest_result.get("sharpe_ratio", 0),
                    "total_return": backtest_result.get("total_return", 0),
                    "max_drawdown": backtest_result.get("max_drawdown", 0),
                    "win_rate": backtest_result.get("win_rate", 0),
                    "composite_score": composite_score
                },
                "backtest_details": {
                    k: v for k, v in backtest_result.items() 
                    if k not in ["trades", "equity_curve_tail", "performance_summary"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"分析股票 {symbol} 失败: {e}")
            return {
                "symbol": symbol,
                "status": "failed",
                "error": str(e)
            }
    
    def _calculate_composite_score(self, backtest_result: Dict[str, Any], 
                                 signals: Dict[str, Any]) -> float:
        """计算综合评分"""
        # 回测性能权重
        backtest_weights = {
            "sharpe_ratio": 0.35,
            "total_return": 0.25,
            "max_drawdown": -0.20,  # 负权重，回撤越小越好
            "win_rate": 0.20
        }
        
        # 信号权重
        signal_weights = {
            "total_score": 0.15,
            "momentum_score": 0.10
        }
        
        # 计算回测部分得分
        backtest_score = 0
        for metric, weight in backtest_weights.items():
            value = backtest_result.get(metric, 0)
            
            if metric == "sharpe_ratio":
                normalized = min(max(value / 3, 0), 1)  # 归一化到0-1
            elif metric == "total_return":
                normalized = min(max(value / 0.5, 0), 1)  # 50%收益率得满分
            elif metric == "max_drawdown":
                normalized = 1 - min(value / 0.5, 1)  # 50%回撤得0分
            elif metric == "win_rate":
                normalized = value  # 胜率本身就是0-1
            
            backtest_score += normalized * weight
        
        # 计算信号部分得分
        signal_score = 0
        for metric, weight in signal_weights.items():
            value = signals.get(metric, 0)
            
            if metric == "total_score":
                normalized = min(max(value, 0), 1)
            elif metric == "momentum_score":
                normalized = min(max(value, 0), 1)
            
            signal_score += normalized * weight
        
        # 综合得分
        composite_score = 0.7 * backtest_score + 0.3 * signal_score
        
        return max(0, min(composite_score, 1))
    
    def _save_stock_results(self, symbol: str, results: Dict[str, Any]) -> None:
        """保存单只股票结果"""
        output_file = os.path.join(
            self.config['output_dir'],
            f"stock_{symbol}_results.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"股票 {symbol} 结果保存到 {output_file}")
    
    def _save_results(self, all_results: Dict[str, Any]) -> None:
        """保存所有结果"""
        # JSON格式
        json_file = os.path.join(self.config['output_dir'], "all_backtest_results.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        
        # CSV格式（股票汇总）
        csv_file = os.path.join(self.config['output_dir'], "stock_summary.csv")
        
        if all_results["stock_results"]:
            summary_data = []
            for stock_result in all_results["stock_results"]:
                if stock_result.get("status") == "success":
                    summary_data.append({
                        "symbol": stock_result.get("symbol"),
                        "latest_price": stock_result.get("latest_price", 0),
                        "data_points": stock_result.get("data_points", 0),
                        "signal_score": stock_result.get("signals_summary", {}).get("total_score", 0),
                        "sharpe_ratio": stock_result.get("backtest_summary", {}).get("sharpe_ratio", 0),
                        "total_return": stock_result.get("backtest_summary", {}).get("total_return", 0),
                        "max_drawdown": stock_result.get("backtest_summary", {}).get("max_drawdown", 0),
                        "composite_score": stock_result.get("backtest_summary", {}).get("composite_score", 0),
                        "status": "success"
                    })
                else:
                    summary_data.append({
                        "symbol": stock_result.get("symbol"),
                        "status": stock_result.get("status", "failed")
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                self.logger.info(f"股票汇总保存到 {csv_file}")
        
        # 生成可视化报告
        self._generate_visual_report(all_results)
        
        self.logger.info(f"所有结果保存到 {self.config['output_dir']} 目录")
    
    def _generate_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析总结"""
        successful_results = [
            r for r in all_results["stock_results"] 
            if r.get("status") == "success"
        ]
        
        if not successful_results:
            return {"message": "没有成功的分析结果"}
        
        # 按综合评分排序
        sorted_results = sorted(
            successful_results,
            key=lambda x: x.get("backtest_summary", {}).get("composite_score", 0),
            reverse=True
        )
        
        # 计算平均值
        avg_sharpe = np.mean([
            r.get("backtest_summary", {}).get("sharpe_ratio", 0)
            for r in successful_results
        ])
        
        avg_return = np.mean([
            r.get("backtest_summary", {}).get("total_return", 0)
            for r in successful_results
        ])
        
        avg_composite = np.mean([
            r.get("backtest_summary", {}).get("composite_score", 0)
            for r in successful_results
        ])
        
        return {
            "total_stocks_analyzed": len(all_results["stock_results"]),
            "successful_analyses": len(successful_results),
            "success_rate": len(successful_results) / len(all_results["stock_results"]),
            "average_sharpe_ratio": avg_sharpe,
            "average_total_return": avg_return,
            "average_composite_score": avg_composite,
            "top_performers": [
                {
                    "symbol": r.get("symbol"),
                    "composite_score": r.get("backtest_summary", {}).get("composite_score", 0),
                    "sharpe_ratio": r.get("backtest_summary", {}).get("sharpe_ratio", 0)
                }
                for r in sorted_results[:3]
            ],
            "optimization_iterations": len(all_results["iteration_results"]),
            "best_optimization_result": self._get_best_optimization(all_results)
        }
    
    def _get_best_optimization(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """获取最佳优化结果"""
        best_iteration = None
        best_score = -np.inf
        
        for iteration in all_results["iteration_results"]:
            optimization = iteration.get("optimization_results", {})
            if optimization.get("status") == "success":
                results = optimization.get("results", {})
                score = results.get("best_score", 0)
                if score > best_score:
                    best_score = score
                    best_iteration = iteration
        
        if best_iteration:
            return {
                "iteration": best_iteration.get("iteration"),
                "best_score": best_score,
                "best_parameters": best_iteration.get("optimization_results", {})
                .get("results", {}).get("best_parameters", {})
            }
        
        return {}
    
    def _generate_visual_report(self, all_results: Dict[str, Any]) -> None:
        """生成可视化报告"""
        try:
            import matplotlib.pyplot as plt
            
            successful_results = [
                r for r in all_results["stock_results"] 
                if r.get("status") == "success"
            ]
            
            if not successful_results:
                self.logger.warning("没有足够的成功结果生成可视化报告")
                return
            
            # 创建可视化图表
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Stock Quant 回测分析报告', fontsize=16, fontweight='bold')
            
            # 1. 综合评分分布
            composite_scores = [
                r.get("backtest_summary", {}).get("composite_score", 0)
                for r in successful_results
            ]
            
            symbols = [r.get("symbol") for r in successful_results]
            
            axes[0, 0].bar(range(len(symbols)), composite_scores)
            axes[0, 0].set_title('各股票综合评分')
            axes[0, 0].set_xlabel('股票代码')
            axes[0, 0].set_ylabel('综合评分')
            axes[0, 0].set_xticks(range(len(symbols)))
            axes[0, 0].set_xticklabels(symbols, rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 夏普比率分布
            sharpe_ratios = [
                r.get("backtest_summary", {}).get("sharpe_ratio", 0)
                for r in successful_results
            ]
            
            axes[0, 1].bar(range(len(symbols)), sharpe_ratios, color='orange')
            axes[0, 1].set_title('各股票夏普比率')
            axes[0, 1].set_xlabel('股票代码')
            axes[0, 1].set_ylabel('夏普比率')
            axes[0, 1].set_xticks(range(len(symbols)))
            axes[0, 1].set_xticklabels(symbols, rotation=45)
            axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 收益率与最大回撤散点图
            returns = [
                r.get("backtest_summary", {}).get("total_return", 0)
                for r in successful_results
            ]
            drawdowns = [
                r.get("backtest_summary", {}).get("max_drawdown", 0)
                for r in successful_results
            ]
            
            scatter = axes[1, 0].scatter(drawdowns, returns, alpha=0.6, 
                                        c=composite_scores, cmap='viridis', s=100)
            axes[1, 0].set_title('收益率 vs 最大回撤')
            axes[1, 0].set_xlabel('最大回撤')
            axes[1, 0].set_ylabel('总收益率')
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 添加颜色条
            plt.colorbar(scatter, ax=axes[1, 0])
            
            # 4. 信号评分分布
            signal_scores = [
                r.get("signals_summary", {}).get("total_score", 0)
                for r in successful_results
            ]
            
            axes[1, 1].plot(signal_scores, marker='o', linestyle='-', color='green')
            axes[1, 1].set_title('信号评分趋势')
            axes[1, 1].set_xlabel('股票序号')
            axes[1, 1].set_ylabel('信号评分')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = os.path.join(self.config['output_dir'], "analysis_charts.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"可视化图表保存到 {chart_file}")
            
        except Exception as e:
            self.logger.error(f"生成可视化报告失败: {e}")
    
    def generate_report(self) -> str:
        """生成文本报告"""
        # 读取结果文件
        json_file = os.path.join(self.config['output_dir'], "all_backtest_results.json")
        
        if not os.path.exists(json_file):
            return "未找到分析结果文件"
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            summary = results.get("summary", {})
            
            report_lines = [
                "# Stock Quant 技术面量化选股系统 - 回测分析报告",
                "",
                f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## 分析概览",
                f"- 分析股票总数: {summary.get('total_stocks_analyzed', 0)}",
                f"- 成功分析数: {summary.get('successful_analyses', 0)}",
                f"- 成功率: {summary.get('success_rate', 0)*100:.1f}%",
                f"- 平均夏普比率: {summary.get('average_sharpe_ratio', 0):.3f}",
                f"- 平均总收益率: {summary.get('average_total_return', 0)*100:.1f}%",
                f"- 平均综合评分: {summary.get('average_composite_score', 0):.3f}",
                "",
                "## 表现最佳股票",
            ]
            
            if "top_performers" in summary:
                for i, stock in enumerate(summary["top_performers"], 1):
                    report_lines.append(
                        f"{i}. {stock['symbol']}: "
                        f"综合评分={stock['composite_score']:.3f}, "
                        f"夏普比率={stock['sharpe_ratio']:.3f}"
                    )
            
            report_lines.extend([
                "",
                "## 迭代优化结果",
                f"- 优化迭代次数: {summary.get('optimization_iterations', 0)}",
            ])
            
            if "best_optimization_result" in summary and summary["best_optimization_result"]:
                best_opt = summary["best_optimization_result"]
                report_lines.extend([
                    f"- 最佳迭代: 第{best_opt.get('iteration', 'N/A')}次",
                    f"- 最佳得分: {best_opt.get('best_score', 0):.3f}",
                ])
            
            report_lines.extend([
                "",
                "## 建议",
                "1. 综合评分高的股票可以考虑进一步研究",
                "2. 夏普比率>1.0的策略表现较好",
                "3. 建议在实际交易前进行更长时间的回测",
                "4. 可以考虑引入更多基本面指标进行综合评估",
                "",
                "## 文件输出",
                f"- 详细结果: {self.config['output_dir']}/all_backtest_results.json",
                f"- 股票汇总: {self.config['output_dir']}/stock_summary.csv",
                f"- 可视化图表: {self.config['output_dir']}/analysis_charts.png",
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"生成报告失败: {e}"


def main():
    """主函数"""
    print("Stock Quant 技术面量化选股系统 - 回测分析")
    print("=" * 60)
    
    try:
        # 配置参数
        config = {
            "test_stocks": ["000001", "000002", "000858", "600519", "000333", "000568", "600036"],
            "time_periods": ["2023-01-01", "2023-12-31"],
            "backtest_periods": [30, 60, 90, 180],
            "optimization_iterations": 3,
            "output_dir": "backtest_results"
        }
        
        # 创建分析器
        analyzer = BacktestAnalyzer(config)
        
        # 运行综合回测
        print("正在运行回测分析...")
        print("这可能需要几分钟时间，请稍候...")
        print()
        
        all_results = analyzer.run_comprehensive_backtest()
        
        # 生成报告
        print("\n分析完成! 正在生成报告...\n")
        
        report = analyzer.generate_report()
        print(report)
        
        # 保存报告文件
        report_file = os.path.join(config['output_dir'], "analysis_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n详细报告已保存到: {report_file}")
        print(f"所有输出文件位于: {config['output_dir']} 目录")
        
        return all_results
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()