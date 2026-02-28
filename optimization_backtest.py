#!/usr/bin/env python3
"""
Stock Quant 优化回测适配器
连接参数优化框架与原始回测代码
"""

import sys
import os
sys.path.append('.')

from backtest_simplified import (
    create_mock_stock_data,
    calculate_moving_average,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_momentum_indicators,
    calculate_trend_strength,
    generate_trading_signals,
    test_stocks
)

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

class OptimizedBacktest:
    """优化后的回测系统"""
    
    def __init__(self, params: Dict[str, Any]):
        """初始化回测系统"""
        self.params = params
        
        # 从参数中提取设置
        self.ma_fast = params.get('ma_fast', 5)
        self.ma_slow = params.get('ma_slow', 20)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.bb_std = params.get('bb_std', 2.0)
        self.score_threshold = params.get('score_threshold', 0.4)
        
        # 风险管理参数
        self.stop_loss = params.get('stop_loss', -0.05)
        self.take_profit = params.get('take_profit', 0.10)
        self.position_size = params.get('position_size', 0.15)
        
        # 策略权重参数
        self.trend_weight = params.get('trend_weight', 0.30)
        self.mean_reversion_weight = params.get('mean_reversion_weight', 0.25)
        self.momentum_weight = params.get('momentum_weight', 0.25)
        self.volatility_weight = params.get('volatility_weight', 0.20)
        
        # 指标权重（基于参数优化）
        self.signal_weights = self._calculate_signal_weights()
    
    def _calculate_signal_weights(self) -> Dict[str, float]:
        """基于策略权重计算信号权重"""
        # 基础权重
        base_weights = {
            'MA_cross': 0.15,
            'RSI_oversold': 0.12,
            'MACD_golden': 0.12,
            'BB_lower_touch': 0.10,
            'Momentum_positive': 0.10,
            'Volume_spike': 0.08,
            'Trend_up': 0.08,
            'RSI_mid': 0.05,
            'MACD_dead': -0.08,
            'RSI_overbought': -0.08,
            'Momentum_negative': -0.08,
            'Trend_down': -0.08
        }
        
        # 根据策略权重调整
        adjusted_weights = {}
        
        # 趋势策略加强MA和趋势信号
        for signal, weight in base_weights.items():
            adjusted_weight = weight
            
            if signal in ['MA_cross', 'Trend_up', 'Trend_down']:
                adjusted_weight *= (1 + self.trend_weight * 0.3)
            
            # 均值回归策略加强RSI和布林带信号
            if signal in ['RSI_oversold', 'RSI_overbought', 'BB_lower_touch', 'RSI_mid']:
                adjusted_weight *= (1 + self.mean_reversion_weight * 0.3)
            
            # 动量策略加强动量信号
            if signal in ['Momentum_positive', 'Momentum_negative']:
                adjusted_weight *= (1 + self.momentum_weight * 0.3)
            
            # 波动率策略加强MACD信号
            if signal in ['MACD_golden', 'MACD_dead']:
                adjusted_weight *= (1 + self.volatility_weight * 0.3)
            
            adjusted_weights[signal] = adjusted_weight
        
        # 归一化权重（保持总权重比例）
        total_positive = sum(w for w in adjusted_weights.values() if w > 0)
        total_negative = abs(sum(w for w in adjusted_weights.values() if w < 0))
        
        if total_positive > 1.0:
            scale = 1.0 / total_positive
            for signal in adjusted_weights:
                if adjusted_weights[signal] > 0:
                    adjusted_weights[signal] *= scale
        
        return adjusted_weights
    
    def generate_trading_signals_optimized(self, df: pd.DataFrame) -> Dict[str, Any]:
        """生成优化的交易信号"""
        signals = {}
        
        # 计算技术指标（使用优化后的参数）
        # 移动平均线
        ma_periods = [self.ma_fast, self.ma_slow]
        ma_df = calculate_moving_average(df, periods=ma_periods)
        
        # RSI（使用优化的阈值）
        rsi_df = calculate_rsi(df)
        # 调整RSI阈值
        latest_rsi = rsi_df['RSI14'].iloc[-1] if 'RSI14' in rsi_df.columns else 50
        signals['RSI_oversold'] = 1 if latest_rsi < self.rsi_oversold else 0
        signals['RSI_overbought'] = 1 if latest_rsi > self.rsi_overbought else 0
        signals['RSI_mid'] = 1 if 45 <= latest_rsi <= 55 else 0
        
        # MACD
        macd_df = calculate_macd(df)
        signals['MACD_golden'] = macd_df['MACD_golden_cross'].iloc[-1] if 'MACD_golden_cross' in macd_df.columns else 0
        signals['MACD_dead'] = macd_df['MACD_dead_cross'].iloc[-1] if 'MACD_dead_cross' in macd_df.columns else 0
        
        # 布林带（使用优化的标准差）
        bb_df = calculate_bollinger_bands(df, std_dev=self.bb_std)
        signals['BB_lower_touch'] = bb_df['BB_touch_lower'].iloc[-1] if 'BB_touch_lower' in bb_df.columns else 0
        signals['BB_upper_touch'] = bb_df['BB_touch_upper'].iloc[-1] if 'BB_touch_upper' in bb_df.columns else 0
        
        # 动量指标
        momentum_df = calculate_momentum_indicators(df)
        momentum_5 = momentum_df['Momentum_5'].iloc[-1] if 'Momentum_5' in momentum_df.columns else 0
        signals['Momentum_positive'] = 1 if momentum_5 > 0.01 else 0
        signals['Momentum_negative'] = 1 if momentum_5 < -0.01 else 0
        
        # 成交量
        volume_spike = momentum_df['Volume_spike'].iloc[-1] if 'Volume_spike' in momentum_df.columns else 0
        signals['Volume_spike'] = volume_spike
        
        # 趋势
        trend_df = calculate_trend_strength(df)
        signals['Trend_up'] = trend_df['Trend_up'].iloc[-1] if 'Trend_up' in trend_df.columns else 0
        signals['Trend_down'] = trend_df['Trend_down'].iloc[-1] if 'Trend_down' in trend_df.columns else 0
        
        # MA交叉信号
        if f'MA{self.ma_fast}' in ma_df.columns and f'MA{self.ma_slow}' in ma_df.columns:
            latest = ma_df.iloc[-1]
            prev = ma_df.iloc[-2]
            
            ma_cross = (prev[f'MA{self.ma_fast}'] <= prev[f'MA{self.ma_slow}']) and \
                      (latest[f'MA{self.ma_fast}'] > latest[f'MA{self.ma_slow}'])
            signals['MA_cross'] = 1 if ma_cross else 0
        else:
            signals['MA_cross'] = 0
        
        # 计算综合评分（使用优化后的权重）
        signal_score = 0
        for signal_name, weight in self.signal_weights.items():
            if signal_name in signals:
                signal_value = signals[signal_name]
                if isinstance(signal_value, (int, float)):
                    signal_score += signal_value * weight
        
        signals['total_score'] = max(0, min(signal_score, 1))
        
        return signals
    
    def run_backtest_with_risk_management(self, df: pd.DataFrame, initial_capital: float = 100000) -> Dict[str, Any]:
        """运行带风险管理的回测"""
        if len(df) < 100:
            return {"status": "insufficient_data"}
        
        # 生成优化的交易信号
        signals_df = df.copy()
        
        # 生成交易信号（使用最后60天进行回测）
        signals = []
        signal_scores = []
        
        for i in range(len(signals_df)):
            if i < 60:
                signals.append(0)  # 初始阶段不交易
                signal_scores.append(0)
            else:
                # 使用最近60天的数据计算信号
                window_df = signals_df.iloc[max(0, i-60):i+1]
                window_signals = self.generate_trading_signals_optimized(window_df)
                score = window_signals.get('total_score', 0)
                signal_scores.append(score)
                
                # 使用优化的阈值进行交易决策
                if score > self.score_threshold:
                    signals.append(1)  # 买入
                elif score < self.score_threshold * 0.5:
                    signals.append(-1)  # 卖出
                else:
                    signals.append(0)  # 持有
        
        # 执行带风险管理的交易
        equity = initial_capital
        cash = initial_capital
        shares = 0
        trades = []
        
        # 跟踪持仓成本
        position_cost = 0
        position_entry_price = 0
        
        for i in range(len(df)):
            price = df['close'].iloc[i]
            
            if i < len(signals):
                signal = signals[i]
                signal_score = signal_scores[i]
            else:
                signal = 0
                signal_score = 0
            
            # 风险管理检查
            if shares > 0:
                current_return = (price - position_entry_price) / position_entry_price
                
                # 止损检查
                if current_return < self.stop_loss:
                    signal = -1  # 强制止损
                    print(f"触发止损: 价格={price:.2f}, 收益率={current_return:.2%}")
                
                # 止盈检查
                elif current_return > self.take_profit:
                    signal = -1  # 强制止盈
                    print(f"触发止盈: 价格={price:.2f}, 收益率={current_return:.2%}")
            
            if signal == 1 and cash > 0:  # 买入
                # 使用优化的仓位控制
                invest_amount = cash * self.position_size * signal_score  # 仓位乘以信号强度
                
                if invest_amount > cash:
                    invest_amount = cash * 0.1  # 最多使用10%的资金
                
                shares_to_buy = invest_amount / price
                
                if shares_to_buy > 0:
                    shares += shares_to_buy
                    cash -= invest_amount
                    
                    # 更新持仓成本
                    position_cost += invest_amount
                    position_entry_price = price
                    
                    trades.append({
                        'date': df.index[i],
                        'type': 'BUY',
                        'price': price,
                        'shares': shares_to_buy,
                        'amount': invest_amount,
                        'signal_score': signal_score
                    })
            
            elif signal == -1 and shares > 0:  # 卖出
                # 卖出全部持仓
                sell_amount = shares * price
                cash += sell_amount
                
                # 计算交易盈亏
                trade_profit = sell_amount - position_cost
                
                trades.append({
                    'date': df.index[i],
                    'type': 'SELL',
                    'price': price,
                    'shares': shares,
                    'amount': sell_amount,
                    'profit': trade_profit,
                    'return': (sell_amount - position_cost) / position_cost if position_cost > 0 else 0
                })
                
                # 重置持仓
                shares = 0
                position_cost = 0
                position_entry_price = 0
        
        # 计算性能指标
        if len(trades) > 0:
            # 最终权益
            final_equity = cash + (shares * df['close'].iloc[-1])
            total_return = (final_equity - initial_capital) / initial_capital
            
            # 计算年化收益率
            days = (df.index[-1] - df.index[0]).days
            annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
            
            # 计算最大回撤
            equity_curve = []
            for i in range(len(df)):
                position_value = shares * df['close'].iloc[i]
                equity_curve.append(cash + position_value)
            
            max_drawdown = 0
            peak = equity_curve[0]
            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # 计算夏普比率
            returns = []
            for i in range(1, len(equity_curve)):
                if equity_curve[i-1] > 0:
                    daily_return = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                    returns.append(daily_return)
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns) if len(returns) > 1 else 0
                sharpe_ratio = avg_return / (std_return + 1e-8) * np.sqrt(252) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # 计算胜率和盈亏比
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            if sell_trades:
                winning_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
                win_rate = len(winning_trades) / len(sell_trades)
                
                # 计算盈亏比
                total_profit = sum(t.get('profit', 0) for t in winning_trades)
                losing_trades = [t for t in sell_trades if t.get('profit', 0) <= 0]
                total_loss = abs(sum(t.get('profit', 0) for t in losing_trades))
                
                profit_factor = total_profit / (total_loss + 1e-8) if total_loss > 0 else float('inf')
                
                # 平均交易盈利
                avg_trade = total_profit / len(sell_trades) if sell_trades else 0
            else:
                win_rate = 0
                profit_factor = 0
                avg_trade = 0
            
            # 买入持有对比
            buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            
            return {
                "initial_capital": initial_capital,
                "final_equity": final_equity,
                "total_return": total_return,
                "annual_return": annual_return,
                "buy_hold_return": buy_hold_return,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_trade": avg_trade,
                "total_trades": len(trades),
                "valid_trades": len(sell_trades),
                "winning_trades": len([t for t in trades if t.get('type') == 'SELL' and t.get('profit', 0) > 0]),
                "trades": trades[-10:] if trades else []
            }
        else:
            return {
                "initial_capital": initial_capital,
                "final_equity": initial_capital,
                "total_return": 0,
                "annual_return": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_trade": 0,
                "total_trades": 0,
                "valid_trades": 0,
                "winning_trades": 0,
                "trades": []
            }
    
    def test_single_stock(self, stock_symbol: str, stock_name: str) -> Dict[str, Any]:
        """测试单只股票"""
        # 创建模拟数据
        stock_data = create_mock_stock_data(stock_symbol, n_days=252)
        
        # 运行优化后的回测
        backtest_result = self.run_backtest_with_risk_management(stock_data)
        
        # 生成最新信号
        latest_signals = self.generate_trading_signals_optimized(stock_data)
        
        return {
            "symbol": stock_symbol,
            "name": stock_name,
            "latest_price": stock_data['close'].iloc[-1],
            "latest_signals": latest_signals,
            "backtest_results": backtest_result
        }
    
    def test_multiple_stocks(self, stock_list: List[Dict[str, str]], stock_count: int = 7) -> Dict[str, Any]:
        """测试多只股票并计算平均性能"""
        if len(stock_list) < stock_count:
            stock_count = len(stock_list)
        
        print(f"测试 {stock_count} 只股票，使用参数: {self.params}")
        
        test_results = []
        
        for i in range(stock_count):
            stock_info = stock_list[i]
            print(f"  测试股票 {i+1}/{stock_count}: {stock_info['symbol']} ({stock_info['name']})")
            
            result = self.test_single_stock(stock_info['symbol'], stock_info['name'])
            test_results.append(result)
        
        # 计算平均性能指标
        metrics_list = []
        
        for result in test_results:
            metrics = result.get('backtest_results', {})
            if metrics.get('total_trades', 0) > 0:
                metrics_list.append(metrics)
        
        if metrics_list:
            avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in metrics_list])
            avg_max_dd = np.mean([abs(m.get('max_drawdown', 0)) for m in metrics_list])
            avg_win_rate = np.mean([m.get('win_rate', 0) for m in metrics_list])
            avg_profit_factor = np.mean([m.get('profit_factor', 0) for m in metrics_list])
            avg_total_return = np.mean([m.get('total_return', 0) for m in metrics_list])
            avg_avg_trade = np.mean([m.get('avg_trade', 0) for m in metrics_list])
            
            # 总交易数
            total_valid_trades = sum(m.get('valid_trades', 0) for m in metrics_list)
            
            return {
                "sharpe_ratio": avg_sharpe,
                "max_drawdown": avg_max_dd,
                "win_rate": avg_win_rate,
                "profit_factor": avg_profit_factor,
                "total_return": avg_total_return,
                "avg_trade": avg_avg_trade,
                "valid_trades": total_valid_trades,
                "tested_stocks": len(test_results),
                "successful_stocks": len(metrics_list),
                "test_results": test_results
            }
        else:
            return {
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_return": 0,
                "avg_trade": 0,
                "valid_trades": 0,
                "tested_stocks": len(test_results),
                "successful_stocks": 0
            }

def create_backtest_function():
    """创建用于优化框架的回测函数"""
    def backtest_func(params: Dict[str, Any], stock_count: int = 7) -> Dict[str, float]:
        """
        回测函数（适配优化框架）
        
        Args:
            params: 参数字典
            stock_count: 测试股票数量
            
        Returns:
            Dict[str, float]: 性能指标
        """
        # 创建优化回测实例
        backtester = OptimizedBacktest(params)
        
        # 测试多只股票
        performance = backtester.test_multiple_stocks(test_stocks, stock_count)
        
        return performance
    
    return backtest_func

def main():
    """主函数：测试优化回测系统"""
    print("Stock Quant 优化回测适配器")
    print("=" * 60)
    
    # 测试参数
    test_params = {
        'ma_fast': 5,
        'ma_slow': 20,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'bb_std': 2.0,
        'score_threshold': 0.4,
        'stop_loss': -0.05,
        'take_profit': 0.10,
        'position_size': 0.15,
        'trend_weight': 0.30,
        'mean_reversion_weight': 0.25,
        'momentum_weight': 0.25,
        'volatility_weight': 0.20
    }
    
    # 创建回测器
    backtester = OptimizedBacktest(test_params)
    
    # 测试多只股票
    print("运行优化回测...")
    performance = backtester.test_multiple_stocks(test_stocks, 7)
    
    print("\n回测结果:")
    print(f"  夏普比率: {performance.get('sharpe_ratio', 0):.3f}")
    print(f"  最大回撤: {performance.get('max_drawdown', 0):.2%}")
    print(f"  胜率: {performance.get('win_rate', 0):.2%}")
    print(f"  盈亏比: {performance.get('profit_factor', 0):.2f}")
    print(f"  总收益率: {performance.get('total_return', 0):.2%}")
    print(f"  平均交易盈利: {performance.get('avg_trade', 0):.2f}")
    print(f"  有效交易数: {performance.get('valid_trades', 0)}")
    print(f"  成功测试股票数: {performance.get('successful_stocks', 0)}/{performance.get('tested_stocks', 0)}")
    
    return performance

if __name__ == "__main__":
    main()