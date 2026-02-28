#!/usr/bin/env python3
"""
Stock Quant 简化回测分析 - 专注于迭代选股指标
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

print("Stock Quant 技术面量化选股系统 - 简化回测分析")
print("=" * 60)

# 创建输出目录
output_dir = "backtest_simplified_results"
os.makedirs(output_dir, exist_ok=True)

# 定义测试股票
test_stocks = [
    {"symbol": "000001", "name": "平安银行"},
    {"symbol": "000002", "name": "万科A"},
    {"symbol": "000858", "name": "五粮液"},
    {"symbol": "600519", "name": "贵州茅台"},
    {"symbol": "000333", "name": "美的集团"},
    {"symbol": "000568", "name": "泸州老窖"},
    {"symbol": "600036", "name": "招商银行"}
]

def create_mock_stock_data(symbol: str, n_days: int = 252) -> pd.DataFrame:
    """创建模拟股票数据"""
    np.random.seed(hash(symbol) % 10000)
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=n_days*1.2),
        periods=n_days,
        freq='B'
    )
    
    # 基础价格（不同股票有不同的起始价格）
    base_price = 10 + (hash(symbol) % 100) / 5
    
    # 生成价格变化（带有趋势）
    price_changes = np.random.randn(n_days) * 0.02
    
    # 添加趋势（随机方向）
    trend_direction = 1 if hash(symbol) % 2 == 0 else -1
    trend_strength = 0.0003 * trend_direction
    price_changes += trend_strength
    
    # 生成季节性模式
    seasonal = 0.001 * np.sin(np.arange(n_days) / 50)
    price_changes += seasonal
    
    # 生成价格序列
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # 生成OHLCV数据
    high = prices * (1 + np.random.rand(n_days) * 0.025)
    low = prices * (1 - np.random.rand(n_days) * 0.025)
    open_prices = prices * (1 + np.random.randn(n_days) * 0.008)
    volume = np.random.randint(5000000, 20000000, n_days)
    
    # 确保价格合理性
    high = np.maximum(high, prices)
    low = np.minimum(low, prices)
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume,
        'amount': prices * volume
    }, index=dates)

def calculate_moving_average(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    """计算移动平均线"""
    result = df.copy()
    for period in periods:
        result[f'MA{period}'] = result['close'].rolling(window=period).mean()
    return result

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算RSI指标"""
    result = df.copy()
    delta = result['close'].diff()
    
    # 计算上涨和下跌
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # 计算RSI
    rs = gain / (loss + 1e-8)
    result[f'RSI{period}'] = 100 - (100 / (1 + rs))
    
    # RSI超买超卖信号
    result[f'RSI_oversold'] = (result[f'RSI{period}'] < 30).astype(int)
    result[f'RSI_overbought'] = (result[f'RSI{period}'] > 70).astype(int)
    
    return result

def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """计算MACD指标"""
    result = df.copy()
    
    # 计算EMA
    ema12 = result['close'].ewm(span=12, adjust=False).mean()
    ema26 = result['close'].ewm(span=26, adjust=False).mean()
    
    # MACD线和信号线
    result['MACD'] = ema12 - ema26
    result['MACD_signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    result['MACD_histogram'] = result['MACD'] - result['MACD_signal']
    
    # MACD金叉死叉
    result['MACD_golden_cross'] = ((result['MACD'].shift(1) <= result['MACD_signal'].shift(1)) & 
                                    (result['MACD'] > result['MACD_signal'])).astype(int)
    result['MACD_dead_cross'] = ((result['MACD'].shift(1) >= result['MACD_signal'].shift(1)) & 
                                  (result['MACD'] < result['MACD_signal'])).astype(int)
    
    return result

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """计算布林带"""
    result = df.copy()
    
    # 计算中轨（移动平均线）
    result['BB_middle'] = result['close'].rolling(window=period).mean()
    
    # 计算标准差
    bb_std = result['close'].rolling(window=period).std()
    
    # 计算上下轨
    result['BB_upper'] = result['BB_middle'] + (bb_std * std_dev)
    result['BB_lower'] = result['BB_middle'] - (bb_std * std_dev)
    
    # 计算带宽和百分比
    result['BB_width'] = result['BB_upper'] - result['BB_lower']
    result['BB_percent'] = (result['close'] - result['BB_lower']) / result['BB_width']
    
    # 布林带触及信号
    result['BB_touch_lower'] = (result['close'] <= result['BB_lower'] * 1.01).astype(int)
    result['BB_touch_upper'] = (result['close'] >= result['BB_upper'] * 0.99).astype(int)
    
    return result

def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算动量指标"""
    result = df.copy()
    
    # 价格动量
    for period in [5, 10, 20, 60]:
        result[f'Momentum_{period}'] = result['close'] / result['close'].shift(period) - 1
    
    # 动量变化率（加速度）
    result['Momentum_rate'] = result['Momentum_5'].diff()
    
    # 成交量动量
    result['Volume_change'] = result['volume'].pct_change()
    result['Volume_MA20'] = result['volume'].rolling(20).mean()
    result['Volume_ratio'] = result['volume'] / result['Volume_MA20']
    result['Volume_spike'] = (result['Volume_ratio'] > 2.0).astype(int)
    
    return result

def calculate_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
    """计算趋势强度"""
    result = df.copy()
    
    # 线性回归斜率（趋势方向）
    def linear_slope(arr):
        if len(arr) < 10:
            return 0
        x = np.arange(len(arr))
        return np.polyfit(x, arr, 1)[0]
    
    result['Trend_slope_20'] = result['close'].rolling(20).apply(linear_slope, raw=True)
    result['Trend_strength'] = abs(result['Trend_slope_20']) / result['close'].rolling(20).std()
    
    # 趋势状态
    result['Trend_up'] = (result['Trend_slope_20'] > 0.001).astype(int)
    result['Trend_down'] = (result['Trend_slope_20'] < -0.001).astype(int)
    
    return result

def generate_trading_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """生成交易信号"""
    signals = {}
    
    # MA交叉信号
    if 'MA5' in df.columns and 'MA20' in df.columns:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        ma_cross = (prev['MA5'] <= prev['MA20']) and (latest['MA5'] > latest['MA20'])
        signals['MA_cross'] = 1 if ma_cross else 0
    
    # RSI信号
    if 'RSI14' in df.columns:
        latest_rsi = df['RSI14'].iloc[-1]
        signals['RSI_oversold'] = 1 if latest_rsi < 30 else 0
        signals['RSI_overbought'] = 1 if latest_rsi > 70 else 0
        signals['RSI_mid'] = 1 if 45 <= latest_rsi <= 55 else 0
    
    # MACD信号
    if 'MACD_golden_cross' in df.columns:
        signals['MACD_golden'] = df['MACD_golden_cross'].iloc[-1]
        signals['MACD_dead'] = df['MACD_dead_cross'].iloc[-1]
    
    # 布林带信号
    if 'BB_touch_lower' in df.columns:
        signals['BB_lower_touch'] = df['BB_touch_lower'].iloc[-1]
        signals['BB_upper_touch'] = df['BB_touch_upper'].iloc[-1]
    
    # 动量信号
    if 'Momentum_5' in df.columns:
        momentum_5 = df['Momentum_5'].iloc[-1]
        signals['Momentum_positive'] = 1 if momentum_5 > 0.01 else 0
        signals['Momentum_negative'] = 1 if momentum_5 < -0.01 else 0
    
    # 成交量信号
    if 'Volume_spike' in df.columns:
        signals['Volume_spike'] = df['Volume_spike'].iloc[-1]
    
    # 趋势信号
    if 'Trend_up' in df.columns:
        signals['Trend_up'] = df['Trend_up'].iloc[-1]
        signals['Trend_down'] = df['Trend_down'].iloc[-1]
    
    # 计算综合评分
    signal_score = 0
    weight_factors = {
        'MA_cross': 0.20,
        'RSI_oversold': 0.15,
        'MACD_golden': 0.15,
        'BB_lower_touch': 0.10,
        'Momentum_positive': 0.10,
        'Volume_spike': 0.10,
        'Trend_up': 0.10,
        'RSI_mid': 0.05,
        'MACD_dead': -0.10,
        'RSI_overbought': -0.10,
        'Momentum_negative': -0.10,
        'Trend_down': -0.10
    }
    
    for signal_name, weight in weight_factors.items():
        if signal_name in signals:
            signal_value = signals[signal_name]
            if isinstance(signal_value, (int, float)):
                signal_score += signal_value * weight
    
    signals['total_score'] = max(0, min(signal_score, 1))  # 归一化到0-1
    
    return signals

def run_backtest(df: pd.DataFrame, initial_capital: float = 100000) -> Dict[str, Any]:
    """运行简单回测"""
    if len(df) < 100:
        return {"status": "insufficient_data"}
    
    # 生成信号（使用最后60天进行回测）
    signals_df = df.copy()
    
    # 生成交易信号（简化版：使用综合评分）
    signals = []
    for i in range(len(signals_df)):
        if i < 60:
            signals.append(0)  # 初始阶段不交易
        else:
            # 使用最近60天的数据计算信号
            window_df = signals_df.iloc[max(0, i-60):i+1]
            window_signals = generate_trading_signals(window_df)
            # 综合评分大于0.5时买入，小于0.2时卖出
            if window_signals.get('total_score', 0) > 0.5:
                signals.append(1)  # 买入
            elif window_signals.get('total_score', 0) < 0.2:
                signals.append(-1)  # 卖出
            else:
                signals.append(0)  # 持有
    
    # 执行交易
    equity = initial_capital
    cash = initial_capital
    shares = 0
    trades = []
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        
        if i < len(signals):
            signal = signals[i]
        else:
            signal = 0
        
        if signal == 1 and cash > 0:  # 买入
            # 投资20%的资金
            invest_amount = cash * 0.2
            shares_to_buy = invest_amount / price
            
            shares += shares_to_buy
            cash -= invest_amount
            
            trades.append({
                'date': df.index[i],
                'type': 'BUY',
                'price': price,
                'shares': shares_to_buy,
                'amount': invest_amount
            })
        
        elif signal == -1 and shares > 0:  # 卖出
            # 卖出全部持仓
            sell_amount = shares * price
            cash += sell_amount
            
            trades.append({
                'date': df.index[i],
                'type': 'SELL',
                'price': price,
                'shares': shares,
                'amount': sell_amount
            })
            
            shares = 0
        
        # 计算当前权益
        equity = cash + (shares * price)
    
    # 计算性能指标
    initial_price = df['close'].iloc[0]
    final_price = df['close'].iloc[-1]
    
    buy_hold_return = (final_price - initial_price) / initial_price
    
    if len(trades) > 0:
        total_return = (equity - initial_capital) / initial_capital
        
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
        
        # 计算夏普比率（简化）
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0
            sharpe_ratio = avg_return / (std_return + 1e-8) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
    else:
        total_return = 0
        annual_return = 0
        max_drawdown = 0
        sharpe_ratio = 0
    
    return {
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return": total_return,
        "annual_return": annual_return,
        "buy_hold_return": buy_hold_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "total_trades": len(trades),
        "winning_trades": len([t for t in trades if t['type'] == 'SELL' and t.get('profit', 0) > 0]),
        "profit_factor": "N/A",  # 简化版不计算
        "trades": trades[-10:] if trades else []  # 最近10笔交易
    }

def analyze_stock(stock_info: Dict[str, str]) -> Dict[str, Any]:
    """分析单只股票"""
    symbol = stock_info['symbol']
    name = stock_info['name']
    
    print(f"分析股票: {symbol} ({name})")
    
    # 1. 创建模拟数据
    stock_data = create_mock_stock_data(symbol, n_days=252)
    
    # 2. 计算技术指标
    indicators_data = stock_data.copy()
    
    # 移动平均线
    indicators_data = calculate_moving_average(indicators_data)
    
    # RSI
    indicators_data = calculate_rsi(indicators_data)
    
    # MACD
    indicators_data = calculate_macd(indicators_data)
    
    # 布林带
    indicators_data = calculate_bollinger_bands(indicators_data)
    
    # 动量指标
    indicators_data = calculate_momentum_indicators(indicators_data)
    
    # 趋势强度
    indicators_data = calculate_trend_strength(indicators_data)
    
    # 3. 生成最新信号
    latest_signals = generate_trading_signals(indicators_data)
    
    # 4. 运行回测
    backtest_result = run_backtest(indicators_data)
    
    # 5. 计算综合评分
    if backtest_result.get('status') != 'insufficient_data':
        # 基于回测表现和信号强度计算综合评分
        signal_strength = latest_signals.get('total_score', 0)
        sharpe_ratio = max(0, min(backtest_result.get('sharpe_ratio', 0) / 2, 1))  # 归一化
        total_return = max(0, min(backtest_result.get('total_return', 0) / 0.5, 1))  # 归一化
        
        composite_score = (
            signal_strength * 0.4 + 
            sharpe_ratio * 0.3 + 
            total_return * 0.3
        )
    else:
        composite_score = 0
    
    return {
        "symbol": symbol,
        "name": name,
        "latest_price": stock_data['close'].iloc[-1],
        "price_change": (stock_data['close'].iloc[-1] - stock_data['close'].iloc[-5]) / stock_data['close'].iloc[-5] if len(stock_data) >= 5 else 0,
        "latest_signals": latest_signals,
        "backtest_results": backtest_result,
        "composite_score": composite_score,
        "analysis_date": datetime.now().isoformat(),
        "data_points": len(stock_data)
    }

def optimize_indicators(stock_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """基于分析结果优化指标权重"""
    print("\n优化指标权重...")
    
    # 分析成功的股票
    successful_results = [
        r for r in stock_results 
        if r.get('composite_score', 0) > 0.3 and 
           r.get('backtest_results', {}).get('sharpe_ratio', -10) > 0
    ]
    
    if not successful_results:
        return {"status": "no_successful_results"}
    
    # 分析哪些指标在表现好的股票中最有效
    indicator_performance = {}
    
    for result in successful_results:
        signals = result.get('latest_signals', {})
        composite_score = result.get('composite_score', 0)
        sharpe_ratio = result.get('backtest_results', {}).get('sharpe_ratio', 0)
        
        for signal_name, signal_value in signals.items():
            if signal_name != 'total_score' and isinstance(signal_value, (int, float)):
                if signal_name not in indicator_performance:
                    indicator_performance[signal_name] = {
                        'count': 0,
                        'total_score': 0,
                        'total_sharpe': 0,
                        'positive_count': 0
                    }
                
                indicator_performance[signal_name]['count'] += 1
                indicator_performance[signal_name]['total_score'] += composite_score
                indicator_performance[signal_name]['total_sharpe'] += sharpe_ratio
                
                if signal_value > 0:
                    indicator_performance[signal_name]['positive_count'] += 1
    
    # 计算指标有效性
    effective_indicators = []
    for indicator, stats in indicator_performance.items():
        if stats['count'] > 0:
            avg_score = stats['total_score'] / stats['count']
            avg_sharpe = stats['total_sharpe'] / stats['count']
            positive_rate = stats['positive_count'] / stats['count']
            
            # 有效性评分
            effectiveness = (avg_score * 0.4 + avg_sharpe * 0.3 + positive_rate * 0.3)
            
            effective_indicators.append({
                'indicator': indicator,
                'effectiveness': effectiveness,
                'avg_score': avg_score,
                'avg_sharpe': avg_sharpe,
                'positive_rate': positive_rate,
                'count': stats['count']
            })
    
    # 按有效性排序
    effective_indicators.sort(key=lambda x: x['effectiveness'], reverse=True)
    
    # 生成优化建议
    top_indicators = effective_indicators[:5]
    
    optimization_suggestions = []
    for indicator in top_indicators:
        if indicator['effectiveness'] > 0.5:
            weight_suggestion = min(0.3, 0.1 + indicator['effectiveness'] * 0.2)
            optimization_suggestions.append({
                'indicator': indicator['indicator'],
                'current_weight': "待分析",
                'suggested_weight': weight_suggestion,
                'reason': f"有效性评分: {indicator['effectiveness']:.3f}"
            })
    
    return {
        "optimization_date": datetime.now().isoformat(),
        "total_stocks_analyzed": len(stock_results),
        "successful_stocks": len(successful_results),
        "effectiveness_analysis": effective_indicators,
        "top_indicators": top_indicators,
        "optimization_suggestions": optimization_suggestions,
        "summary": f"发现 {len(top_indicators)} 个有效指标，建议调整权重"
    }

def generate_report(stock_results: List[Dict[str, Any]], optimization_results: Dict[str, Any]) -> str:
    """生成分析报告"""
    report_lines = [
        "# Stock Quant 技术面量化选股系统 - 回测分析报告",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 分析概览",
        f"- 分析股票总数: {len(stock_results)}",
        f"- 平均综合评分: {np.mean([r.get('composite_score', 0) for r in stock_results]):.3f}",
        f"- 平均夏普比率: {np.mean([r.get('backtest_results', {}).get('sharpe_ratio', 0) for r in stock_results]):.3f}",
        f"- 平均总收益率: {np.mean([r.get('backtest_results', {}).get('total_return', 0) for r in stock_results])*100:.1f}%",
        "",
        "## 股票排名（按综合评分）",
        "",
        "排名 | 代码 | 名称 | 综合评分 | 夏普比率 | 总收益率 | 最新信号",
        "--- | --- | --- | --- | --- | --- | ---"
    ]
    
    # 按综合评分排序
    sorted_results = sorted(stock_results, key=lambda x: x.get('composite_score', 0), reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        symbol = result.get('symbol', 'N/A')
        name = result.get('name', 'N/A')
        composite_score = result.get('composite_score', 0)
        sharpe_ratio = result.get('backtest_results', {}).get('sharpe_ratio', 0)
        total_return = result.get('backtest_results', {}).get('total_return', 0)
        
        # 提取主要信号
        signals = result.get('latest_signals', {})
        main_signals = []
        for signal_name in ['MA_cross', 'RSI_oversold', 'MACD_golden', 'BB_lower_touch']:
            if signal_name in signals and signals[signal_name] == 1:
                main_signals.append(signal_name.replace('_', ' '))
        
        signal_text = ', '.join(main_signals) if main_signals else "无"
        
        report_lines.append(
            f"{i} | {symbol} | {name} | {composite_score:.3f} | {sharpe_ratio:.3f} | {total_return*100:.1f}% | {signal_text}"
        )
    
    report_lines.extend([
        "",
        "## 指标优化结果",
        "",
        f"分析有效股票数: {optimization_results.get('successful_stocks', 0)}",
    ])
    
    if optimization_results.get('top_indicators'):
        report_lines.append("")
        report_lines.append("### 最有效指标")
        for indicator in optimization_results.get('top_indicators', [])[:3]:
            report_lines.append(
                f"- **{indicator['indicator']}**: 有效性 {indicator['effectiveness']:.3f}, "
                f"平均夏普 {indicator['avg_sharpe']:.3f}, 正信号率 {indicator['positive_rate']*100:.0f}%"
            )
    
    if optimization_results.get('optimization_suggestions'):
        report_lines.append("")
        report_lines.append("### 权重调整建议")
        for suggestion in optimization_results.get('optimization_suggestions', []):
            report_lines.append(
                f"- **{suggestion['indicator']}**: 建议权重 {suggestion['suggested_weight']:.2f} "
                f"({suggestion['reason']})"
            )
    
    report_lines.extend([
        "",
        "## 迭代改进建议",
        "1. **增加数据量**: 使用更长历史数据进行回测",
        "2. **优化参数**: 对MA周期、RSI阈值等参数进行优化",
        "3. **添加更多指标**: 引入成交量加权价格、ATR等指标",
        "4. **风险管理**: 添加止损、仓位控制等风险管理模块",
        "5. **多时间框架**: 结合日线、周线、月线进行多时间框架分析",
        "",
        "## 后续步骤",
        "1. 对高评分股票进行深入基本面分析",
        "2. 实际交易前进行模拟交易验证",
        "3. 定期更新和优化指标权重",
        "4. 建立自动化的监控和预警系统",
    ])
    
    return "\n".join(report_lines)

def main():
    """主函数"""
    print("正在运行回测分析...")
    
    # 1. 分析每只股票
    all_results = []
    
    for stock in test_stocks:
        try:
            result = analyze_stock(stock)
            all_results.append(result)
            print(f"  {stock['symbol']} 分析完成: 评分={result.get('composite_score', 0):.3f}")
        except Exception as e:
            print(f"  {stock['symbol']} 分析失败: {e}")
            all_results.append({
                "symbol": stock['symbol'],
                "name": stock['name'],
                "error": str(e)
            })
    
    print(f"\n完成 {len(all_results)} 只股票分析")
    
    # 2. 优化指标权重
    optimization_results = optimize_indicators(all_results)
    
    # 3. 生成报告
    report = generate_report(all_results, optimization_results)
    
    # 4. 保存结果
    # 保存股票结果
    with open(f"{output_dir}/stock_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存优化结果
    with open(f"{output_dir}/optimization_results.json", "w", encoding="utf-8") as f:
        json.dump(optimization_results, f, indent=2, ensure_ascii=False, default=str)
    
    # 保存报告
    with open(f"{output_dir}/analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    # 5. 生成可视化图表
    try:
        generate_visualizations(all_results, output_dir)
    except Exception as e:
        print(f"生成可视化图表失败: {e}")
    
    print(f"\n✅ 分析完成!")
    print(f"结果已保存到 {output_dir}/ 目录")
    print(f"报告文件: {output_dir}/analysis_report.md")
    
    # 显示简要结果
    print("\n" + "="*60)
    print("简要结果:")
    
    successful_results = [r for r in all_results if r.get('composite_score', 0) > 0]
    if successful_results:
        top_3 = sorted(successful_results, key=lambda x: x.get('composite_score', 0), reverse=True)[:3]
        
        print("\n综合评分前三名:")
        for i, result in enumerate(top_3, 1):
            print(f"{i}. {result['symbol']} ({result['name']}): "
                  f"评分={result.get('composite_score', 0):.3f}, "
                  f"夏普={result.get('backtest_results', {}).get('sharpe_ratio', 0):.3f}")
    
    return all_results

def generate_visualizations(stock_results: List[Dict[str, Any]], output_dir: str):
    """生成可视化图表"""
    successful_results = [r for r in stock_results if r.get('composite_score', 0) > 0]
    
    if len(successful_results) < 3:
        print("成功结果不足，跳过可视化")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Stock Quant 回测分析可视化', fontsize=16, fontweight='bold')
    
    # 1. 综合评分柱状图
    symbols = [r['symbol'] for r in successful_results]
    composite_scores = [r.get('composite_score', 0) for r in successful_results]
    
    axes[0, 0].bar(range(len(symbols)), composite_scores, color='steelblue')
    axes[0, 0].set_title('股票综合评分')
    axes[0, 0].set_xlabel('股票代码')
    axes[0, 0].set_ylabel('综合评分')
    axes[0, 0].set_xticks(range(len(symbols)))
    axes[0, 0].set_xticklabels(symbols, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 夏普比率柱状图
    sharpe_ratios = [r.get('backtest_results', {}).get('sharpe_ratio', 0) for r in successful_results]
    
    axes[0, 1].bar(range(len(symbols)), sharpe_ratios, color='orange')
    axes[0, 1].set_title('股票夏普比率')
    axes[0, 1].set_xlabel('股票代码')
    axes[0, 1].set_ylabel('夏普比率')
    axes[0, 1].set_xticks(range(len(symbols)))
    axes[0, 1].set_xticklabels(symbols, rotation=45)
    axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='夏普=1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 收益率散点图
    returns = [r.get('backtest_results', {}).get('total_return', 0) for r in successful_results]
    drawdowns = [r.get('backtest_results', {}).get('max_drawdown', 0) for r in successful_results]
    
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
    
    # 4. 指标有效性雷达图（简化）
    if len(successful_results) > 0:
        # 提取常见信号
        signal_counts = {}
        for result in successful_results:
            signals = result.get('latest_signals', {})
            for signal_name, signal_value in signals.items():
                if signal_name != 'total_score' and isinstance(signal_value, (int, float)):
                    if signal_name not in signal_counts:
                        signal_counts[signal_name] = 0
                    if signal_value > 0:
                        signal_counts[signal_name] += 1
        
        # 取出现频率最高的5个信号
        if signal_counts:
            top_signals = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            signal_names = [s[0] for s in top_signals]
            signal_values = [s[1] / len(successful_results) for s in top_signals]
            
            axes[1, 1].bar(range(len(signal_names)), signal_values, color='green')
            axes[1, 1].set_title('常见有效信号')
            axes[1, 1].set_xlabel('信号类型')
            axes[1, 1].set_ylabel('出现频率')
            axes[1, 1].set_xticks(range(len(signal_names)))
            axes[1, 1].set_xticklabels([s.replace('_', '\n') for s in signal_names], fontsize=9)
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = f"{output_dir}/analysis_charts.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表保存到: {chart_path}")

if __name__ == "__main__":
    main()