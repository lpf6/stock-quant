"""回测策略插件"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from ..base import StrategyPlugin


class BacktestStrategy(StrategyPlugin):
    """回测策略 - 基于历史数据进行策略性能评估"""
    
    def __init__(self):
        super().__init__()
        self.name = "Backtest Strategy"
        self.description = "回测分析策略，评估策略性能指标"
        self.author = "Stock Quant Team"
        self.version = "1.0.0"
        self.config = {
            "initial_capital": 100000,    # 初始资金
            "position_ratio": 0.2,        # 单次建仓比例
            "stop_loss": -0.05,          # 止损比例
            "take_profit": 0.10,         # 止盈比例
            "commission_rate": 0.0003,   # 手续费率
            "holding_period": 20,        # 默认持仓周期
            "signal_name": "backtest_result",
            "weight": 0.15,             # 策略权重
            "metrics": ["sharpe_ratio", "max_drawdown", "total_return"]
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> None:
        """初始化回测策略"""
        if config:
            self.config.update(config)
    
    def run_backtest(self, df: pd.DataFrame, signals: Dict[str, Any]) -> Dict[str, Any]:
        """运行回测分析"""
        if df is None or len(df) < 100:
            return self._empty_backtest_result()
        
        # 提取价格数据和信号
        prices = df['close'].values
        dates = df.index
        
        # 将信号转换为交易信号
        trade_signals = self._generate_trade_signals(signals)
        
        # 执行回测
        results = self._execute_trading(
            prices=prices,
            dates=dates,
            signals=trade_signals,
            initial_capital=self.config['initial_capital']
        )
        
        # 计算性能指标
        performance = self._calculate_performance_metrics(
            equity_curve=results['equity_curve'],
            trades=results['trades'],
            risk_free_rate=0.02  # 年化无风险利率
        )
        
        # 计算策略得分
        score = self._calculate_strategy_score(performance)
        
        return {
            "sharpe_ratio": performance['sharpe_ratio'],
            "total_return": performance['total_return'],
            "annual_return": performance['annual_return'],
            "max_drawdown": performance['max_drawdown'],
            "win_rate": performance['win_rate'],
            "profit_factor": performance['profit_factor'],
            "total_trades": performance['total_trades'],
            "score": score * self.config['weight'],
            "trades": results['trades'][-10:],  # 最近10笔交易
            "equity_curve_tail": results['equity_curve'][-20:],  # 最近20天的净值
            "performance_summary": performance
        }
    
    def _generate_trade_signals(self, signals: Dict[str, Any]) -> List[int]:
        """生成交易信号列表"""
        # 这里需要根据实际信号生成交易决策
        # 0: 空仓, 1: 买入, -1: 卖出
        
        # 示例逻辑：基于综合评分
        if signals.get('total_score', 0) > 0.5:
            return [1]  # 买入信号
        elif signals.get('total_score', 0) < -0.5:
            return [-1]  # 卖出信号
        else:
            return [0]  # 持有信号
    
    def _execute_trading(self, prices: np.ndarray, dates: pd.Index, 
                         signals: List[int], 
                         initial_capital: float) -> Dict[str, Any]:
        """执行交易模拟"""
        equity = initial_capital
        cash = initial_capital
        position = 0
        shares = 0
        
        trades = []
        equity_curve = []
        
        for i, (price, date) in enumerate(zip(prices, dates)):
            # 生成交易决策（这里简化处理，实际应该更复杂）
            if i < len(signals):
                signal = signals[i % len(signals)]
            else:
                signal = 0
            
            # 更新持仓价值
            if shares > 0:
                position_value = shares * price
            else:
                position_value = 0
            
            # 执行交易
            if signal == 1 and cash > 0:  # 买入信号
                # 计算买入数量
                invest_amount = cash * self.config['position_ratio']
                commission = invest_amount * self.config['commission_rate']
                shares_to_buy = (invest_amount - commission) / price
                
                shares += shares_to_buy
                cash -= (invest_amount + commission)
                
                trades.append({
                    'date': date,
                    'type': 'BUY',
                    'price': price,
                    'shares': shares_to_buy,
                    'amount': invest_amount,
                    'commission': commission
                })
                
            elif signal == -1 and shares > 0:  # 卖出信号
                # 卖出全部持仓
                sell_amount = shares * price
                commission = sell_amount * self.config['commission_rate']
                
                cash += (sell_amount - commission)
                
                trades.append({
                    'date': date,
                    'type': 'SELL',
                    'price': price,
                    'shares': shares,
                    'amount': sell_amount,
                    'commission': commission,
                    'profit': (sell_amount - position_value - commission)
                })
                
                shares = 0
            
            # 计算当前权益
            equity = cash + (shares * price)
            equity_curve.append(equity)
            
            # 检查止损/止盈
            if position_value > 0:
                cost_basis = sum(t['amount'] for t in trades if t['type'] == 'BUY' and t.get('shares', 0) > 0)
                if cost_basis > 0:
                    current_return = (position_value - cost_basis) / cost_basis
                    
                    if current_return <= self.config['stop_loss']:
                        # 止损
                        signal = -1  # 触发卖出信号
                    elif current_return >= self.config['take_profit']:
                        # 止盈
                        signal = -1  # 触发卖出信号
        
        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'final_equity': equity,
            'final_cash': cash,
            'final_shares': shares
        }
    
    def _calculate_performance_metrics(self, equity_curve: List[float], 
                                      trades: List[Dict], 
                                      risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """计算性能指标"""
        if len(equity_curve) < 2:
            return self._default_performance()
        
        # 计算收益率
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # 总收益率
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]
        
        # 年化收益率（假设250个交易日）
        years = len(equity_array) / 250
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 夏普比率
        excess_returns = returns - risk_free_rate / 250
        sharpe_ratio = np.sqrt(250) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown(equity_array)
        
        # 交易统计数据
        if trades:
            win_trades = [t for t in trades if t.get('profit', 0) > 0]
            loss_trades = [t for t in trades if t.get('profit', 0) < 0]
            
            win_rate = len(win_trades) / len(trades) if trades else 0
            avg_win = np.mean([t['profit'] for t in win_trades]) if win_trades else 0
            avg_loss = np.mean([abs(t['profit']) for t in loss_trades]) if loss_trades else 0
            profit_factor = (sum(t['profit'] for t in win_trades) / 
                           sum(abs(t['profit']) for t in loss_trades)) if loss_trades else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'win_trades': len(win_trades) if 'win_trades' in locals() else 0,
            'loss_trades': len(loss_trades) if 'loss_trades' in locals() else 0
        }
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """计算最大回撤"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_strategy_score(self, performance: Dict[str, Any]) -> float:
        """计算策略综合得分"""
        # 基于多个指标加权计算
        weights = {
            'sharpe_ratio': 0.3,
            'total_return': 0.25,
            'max_drawdown': -0.2,  # 负权重，因为回撤越小越好
            'win_rate': 0.15,
            'profit_factor': 0.1
        }
        
        score = 0
        
        # 夏普比率得分（大于1为优秀）
        sharpe_score = min(performance['sharpe_ratio'] / 2, 1) if performance['sharpe_ratio'] > 0 else 0
        score += sharpe_score * weights['sharpe_ratio']
        
        # 总收益率得分
        return_score = min(performance['total_return'] / 0.5, 1) if performance['total_return'] > 0 else 0
        score += return_score * weights['total_return']
        
        # 最大回撤得分（回撤越小得分越高）
        drawdown_score = 1 - min(performance['max_drawdown'] / 0.5, 1)
        score += drawdown_score * weights['max_drawdown']
        
        # 胜率得分
        win_rate_score = performance['win_rate']
        score += win_rate_score * weights['win_rate']
        
        # 盈亏比得分
        profit_factor_score = min(performance['profit_factor'] / 3, 1) if performance['profit_factor'] > 0 else 0
        score += profit_factor_score * weights['profit_factor']
        
        # 有交易记录才评分
        if performance['total_trades'] == 0:
            score = 0
        
        return max(0, min(score, 1))
    
    def _empty_backtest_result(self) -> Dict[str, Any]:
        """空回测结果"""
        return {
            "sharpe_ratio": 0,
            "total_return": 0,
            "annual_return": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "total_trades": 0,
            "score": 0,
            "trades": [],
            "equity_curve_tail": [],
            "performance_summary": {}
        }
    
    def _default_performance(self) -> Dict[str, Any]:
        """默认性能指标"""
        return {
            'sharpe_ratio': 0,
            'total_return': 0,
            'annual_return': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'win_trades': 0,
            'loss_trades': 0
        }
    
    def calculate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算回测信号（需要在有信号数据时调用）"""
        # 这个方法需要在外部有信号数据时使用
        # 这里返回默认结果
        return {
            self.config['signal_name']: 0,
            "score": 0,
            "note": "需要先运行其他策略生成信号"
        }
    
    def get_signal_descriptions(self) -> Dict[str, str]:
        """获取信号描述"""
        return {
            "backtest_result": "回测结果",
            "sharpe_ratio": "夏普比率（风险调整后收益）",
            "total_return": "总收益率",
            "max_drawdown": "最大回撤",
            "win_rate": "胜率",
            "score": "策略综合得分"
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return {
            "回测参数": {
                "初始资金": self.config['initial_capital'],
                "单次建仓比例": self.config['position_ratio'],
                "止损比例": self.config['stop_loss'],
                "止盈比例": self.config['take_profit'],
                "手续费率": self.config['commission_rate'],
                "策略权重": self.config['weight']
            }
        }