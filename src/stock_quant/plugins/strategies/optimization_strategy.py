"""策略优化插件"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from ..base import StrategyPlugin


class OptimizationStrategy(StrategyPlugin):
    """策略优化器 - 寻找最优参数组合"""
    
    def __init__(self):
        super().__init__()
        self.name = "Optimization Strategy"
        self.description = "策略参数优化器，寻找最优参数组合"
        self.author = "Stock Quant Team"
        self.version = "1.0.0"
        self.config = {
            "optimization_method": "grid_search",  # grid_search, random_search, genetic
            "metric_to_optimize": "sharpe_ratio",  # sharpe_ratio, total_return, win_rate
            "param_ranges": {
                "ma_short": [5, 10, 20],
                "ma_long": [20, 30, 60],
                "rsi_period": [7, 14, 21],
                "rsi_oversold": [25, 30, 35],
                "rsi_overbought": [65, 70, 75],
                "macd_fast": [8, 12, 16],
                "macd_slow": [17, 26, 35],
                "bb_period": [10, 20, 30],
                "bb_std": [1.5, 2.0, 2.5]
            },
            "max_iterations": 100,
            "train_test_split": 0.7,
            "signal_name": "optimization_result",
            "weight": 0.1
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> None:
        """初始化优化器"""
        if config:
            self.config.update(config)
    
    def optimize_parameters(self, df: pd.DataFrame, 
                           initial_signals: Dict[str, Any]) -> Dict[str, Any]:
        """优化策略参数"""
        if df is None or len(df) < 200:
            return self._empty_optimization_result()
        
        # 分割训练集和测试集
        split_idx = int(len(df) * self.config['train_test_split'])
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # 执行参数优化
        if self.config['optimization_method'] == "grid_search":
            best_params, best_score = self._grid_search(train_df, initial_signals)
        elif self.config['optimization_method'] == "random_search":
            best_params, best_score = self._random_search(train_df, initial_signals)
        else:
            best_params, best_score = self._default_optimization()
        
        # 在测试集上验证
        test_performance = self._evaluate_on_test_set(test_df, best_params, initial_signals)
        
        # 计算优化得分
        optimization_score = self._calculate_optimization_score(best_score, test_performance)
        
        return {
            "best_parameters": best_params,
            "best_score": best_score,
            "test_performance": test_performance,
            "score": optimization_score * self.config['weight'],
            "optimization_method": self.config['optimization_method'],
            "param_ranges_used": self.config['param_ranges'],
            "train_samples": len(train_df),
            "test_samples": len(test_df)
        }
    
    def _grid_search(self, df: pd.DataFrame, initial_signals: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """网格搜索优化"""
        best_score = -np.inf
        best_params = {}
        
        # 生成所有参数组合
        param_combinations = self._generate_parameter_combinations()
        
        # 限制迭代次数
        max_iter = min(self.config['max_iterations'], len(param_combinations))
        
        for i, params in enumerate(param_combinations[:max_iter]):
            # 使用当前参数计算性能
            score = self._evaluate_parameters(df, params, initial_signals)
            
            if score > best_score:
                best_score = score
                best_params = params
                
            if i % 10 == 0:
                print(f"  优化进度: {i+1}/{max_iter}, 当前最佳得分: {best_score:.4f}")
        
        return best_params, best_score
    
    def _random_search(self, df: pd.DataFrame, initial_signals: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """随机搜索优化"""
        best_score = -np.inf
        best_params = {}
        
        for i in range(self.config['max_iterations']):
            # 随机生成参数
            params = self._generate_random_parameters()
            
            # 评估参数
            score = self._evaluate_parameters(df, params, initial_signals)
            
            if score > best_score:
                best_score = score
                best_params = params
                
            if i % 10 == 0:
                print(f"  优化进度: {i+1}/{self.config['max_iterations']}, 当前最佳得分: {best_score:.4f}")
        
        return best_params, best_score
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """生成参数组合"""
        combinations = []
        param_ranges = self.config['param_ranges']
        
        # 这里简化处理，实际应该生成所有组合
        # 取每个参数的第一个值作为基础组合
        base_params = {}
        for param_name, values in param_ranges.items():
            if values:
                base_params[param_name] = values[0]
        
        # 生成一些变体
        for i in range(min(50, self.config['max_iterations'])):
            params = base_params.copy()
            
            # 随机修改一些参数
            for param_name, values in param_ranges.items():
                if np.random.random() < 0.3 and len(values) > 1:  # 30%概率修改参数
                    params[param_name] = np.random.choice(values)
            
            combinations.append(params)
        
        return combinations
    
    def _generate_random_parameters(self) -> Dict[str, Any]:
        """随机生成参数"""
        params = {}
        param_ranges = self.config['param_ranges']
        
        for param_name, values in param_ranges.items():
            if values:
                params[param_name] = np.random.choice(values)
        
        return params
    
    def _evaluate_parameters(self, df: pd.DataFrame, params: Dict[str, Any], 
                            initial_signals: Dict[str, Any]) -> float:
        """评估参数性能"""
        try:
            # 这里需要根据实际策略计算性能
            # 简化处理：计算模拟收益
            prices = df['close'].values
            
            if len(prices) < 2:
                return 0
            
            # 简单的模拟策略：基于参数生成买卖信号
            ma_short = params.get('ma_short', 10)
            ma_long = params.get('ma_long', 30)
            
            # 计算移动平均线
            if len(prices) >= ma_long:
                ma_short_vals = pd.Series(prices).rolling(ma_short).mean().values
                ma_long_vals = pd.Series(prices).rolling(ma_long).mean().values
                
                # 生成信号
                signals = []
                for i in range(1, len(prices)):
                    if ma_short_vals[i-1] <= ma_long_vals[i-1] and ma_short_vals[i] > ma_long_vals[i]:
                        signals.append(1)  # 买入信号
                    elif ma_short_vals[i-1] >= ma_long_vals[i-1] and ma_short_vals[i] < ma_long_vals[i]:
                        signals.append(-1)  # 卖出信号
                    else:
                        signals.append(0)  # 持有信号
                
                # 计算收益
                returns = []
                position = 0
                for i in range(1, len(prices)):
                    if i-1 < len(signals):
                        if signals[i-1] == 1:
                            position = 1  # 买入
                        elif signals[i-1] == -1:
                            position = 0  # 卖出
                    
                    if position == 1:
                        returns.append((prices[i] - prices[i-1]) / prices[i-1])
                    else:
                        returns.append(0)
                
                if returns:
                    # 计算夏普比率（简化版）
                    avg_return = np.mean(returns)
                    std_return = np.std(returns) if len(returns) > 1 else 0
                    
                    if std_return > 0:
                        sharpe = avg_return / std_return * np.sqrt(252)  # 年化
                    else:
                        sharpe = 0
                    
                    return sharpe
            
            return 0
            
        except Exception as e:
            print(f"参数评估错误: {e}")
            return 0
    
    def _evaluate_on_test_set(self, test_df: pd.DataFrame, best_params: Dict[str, Any], 
                             initial_signals: Dict[str, Any]) -> Dict[str, Any]:
        """在测试集上评估最佳参数"""
        performance = self._evaluate_parameters(test_df, best_params, initial_signals)
        
        # 计算一些基本指标
        if len(test_df) > 0:
            prices = test_df['close'].values
            total_return = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
            volatility = np.std(np.diff(prices) / prices[:-1]) * np.sqrt(252) if len(prices) > 1 else 0
        else:
            total_return = 0
            volatility = 0
        
        return {
            "sharpe_ratio": performance,
            "total_return": total_return,
            "volatility": volatility,
            "max_drawdown": self._calculate_test_drawdown(test_df) if len(test_df) > 0 else 0
        }
    
    def _calculate_test_drawdown(self, test_df: pd.DataFrame) -> float:
        """计算测试集最大回撤"""
        if len(test_df) == 0:
            return 0
        
        prices = test_df['close'].values
        peak = prices[0]
        max_dd = 0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_optimization_score(self, train_score: float, 
                                     test_performance: Dict[str, Any]) -> float:
        """计算优化得分"""
        # 基于训练集和测试集的表现加权计算
        train_weight = 0.4
        test_weight = 0.6
        
        # 训练集得分
        train_scaled = min(max(train_score / 3, 0), 1)  # 归一化到0-1
        
        # 测试集得分
        test_sharpe = test_performance.get('sharpe_ratio', 0)
        test_scaled = min(max(test_sharpe / 3, 0), 1)  # 归一化到0-1
        
        # 过拟合惩罚（如果训练集和测试集表现差距太大）
        performance_gap = abs(train_scaled - test_scaled)
        penalty = 1 - min(performance_gap * 2, 1)  # 差距越大惩罚越大
        
        score = (train_scaled * train_weight + test_scaled * test_weight) * penalty
        
        return max(0, min(score, 1))
    
    def _default_optimization(self) -> Tuple[Dict[str, Any], float]:
        """默认优化结果"""
        return {
            "ma_short": 10,
            "ma_long": 30,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_fast": 12,
            "macd_slow": 26,
            "bb_period": 20,
            "bb_std": 2.0
        }, 0.5
    
    def _empty_optimization_result(self) -> Dict[str, Any]:
        """空优化结果"""
        return {
            "best_parameters": {},
            "best_score": 0,
            "test_performance": {},
            "score": 0,
            "optimization_method": self.config['optimization_method'],
            "param_ranges_used": {},
            "train_samples": 0,
            "test_samples": 0
        }
    
    def calculate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算优化信号"""
        # 这个方法需要在有初始信号时使用
        return {
            self.config['signal_name']: 0,
            "score": 0,
            "note": "需要先有其他策略的信号作为输入"
        }
    
    def get_signal_descriptions(self) -> Dict[str, str]:
        """获取信号描述"""
        return {
            "optimization_result": "参数优化结果",
            "best_parameters": "最优参数组合",
            "best_score": "训练集最佳得分",
            "test_performance": "测试集性能",
            "score": "优化器综合得分"
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取优化器参数"""
        return {
            "优化器配置": {
                "优化方法": self.config['optimization_method'],
                "优化指标": self.config['metric_to_optimize'],
                "最大迭代次数": self.config['max_iterations'],
                "训练集比例": self.config['train_test_split'],
                "策略权重": self.config['weight']
            },
            "参数范围": self.config['param_ranges']
        }