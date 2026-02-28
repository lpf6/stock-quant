import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 获取中证1000成分股
def get_csi1000_constituents():
    """获取中证1000指数最新成分股"""
    try:
        # 使用akshare获取中证1000成分股
        csi1000_df = ak.index_stock_cons_csindex(symbol="000852")
        print(f"成功获取 {len(csi1000_df)} 只中证1000成分股")
        return csi1000_df
    except Exception as e:
        print(f"获取成分股失败: {e}")
        # 备用方法：从本地文件获取
        return None

# 获取股票历史数据
def get_stock_data(symbol, start_date=None, end_date=None, period="daily"):
    """获取单只股票历史数据"""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    try:
        # 获取A股数据
        if symbol.startswith('sh') or symbol.startswith('sz'):
            symbol_code = symbol
        else:
            symbol_code = f"sh{symbol}" if symbol.startswith('6') else f"sz{symbol}"
        
        stock_df = ak.stock_zh_a_hist(symbol=symbol_code, period=period, 
                                     start_date=start_date, end_date=end_date, adjust="qfq")
        if stock_df.empty:
            return None
            
        stock_df['date'] = pd.to_datetime(stock_df['日期'])
        stock_df.set_index('date', inplace=True)
        stock_df.rename(columns={'开盘': 'open', '收盘': 'close', '最高': 'high', 
                               '最低': 'low', '成交量': 'volume', '成交额': 'amount'}, inplace=True)
        return stock_df[['open', 'close', 'high', 'low', 'volume', 'amount']]
    except Exception as e:
        print(f"获取股票 {symbol} 数据失败: {e}")
        return None

# 技术指标计算
def calculate_technical_indicators(df):
    """计算技术指标"""
    if df is None or len(df) < 60:
        return None
    
    # 复制数据
    data = df.copy()
    
    # 1. 移动平均线
    ma_periods = [5, 10, 20, 60]
    for period in ma_periods:
        data[f'MA{period}'] = data['close'].rolling(window=period).mean()
    
    # 2. RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # 4. 布林带
    data['BB_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + 2 * bb_std
    data['BB_lower'] = data['BB_middle'] - 2 * bb_std
    data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
    
    return data

# 策略信号生成
def generate_signals(data):
    """生成交易信号"""
    if data is None:
        return None
    
    signals = {}
    
    # 1. MA金叉策略
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    
    # MA5上穿MA20
    ma_cross = (prev['MA5'] <= prev['MA20']) and (latest['MA5'] > latest['MA20'])
    signals['ma_cross'] = 1 if ma_cross else 0
    
    # 2. RSI超卖策略
    rsi_oversold = latest['RSI'] < 30 if pd.notnull(latest['RSI']) else False
    signals['rsi_oversold'] = 1 if rsi_oversold else 0
    
    # 3. MACD金叉
    macd_cross = (prev['MACD'] <= prev['MACD_Signal']) and (latest['MACD'] > latest['MACD_Signal'])
    signals['macd_cross'] = 1 if macd_cross else 0
    
    # 4. 布林带下轨
    bb_touch = latest['close'] <= latest['BB_lower'] * 1.02  # 价格接近或低于下轨
    signals['bb_touch'] = 1 if bb_touch else 0
    
    # 综合评分
    weights = {'ma_cross': 0.3, 'rsi_oversold': 0.25, 'macd_cross': 0.25, 'bb_touch': 0.2}
    total_score = sum(signals[key] * weights[key] for key in signals)
    signals['total_score'] = total_score
    
    # 趋势确认
    signals['trend_up'] = 1 if (latest['MA5'] > latest['MA10'] > latest['MA20']) else 0
    signals['volume_spike'] = 1 if latest['volume'] > data['volume'].rolling(20).mean().iloc[-1] * 1.5 else 0
    
    return signals

# 主函数
def main():
    print("=" * 60)
    print("中证1000技术面量化选股系统")
    print("=" * 60)
    
    # 1. 获取成分股
    print("\n1. 获取中证1000成分股...")
    constituents = get_csi1000_constituents()
    
    if constituents is None:
        # 使用示例数据
        print("使用示例成分股数据...")
        constituents = pd.DataFrame({
            '成分券代码': ['000001', '000002', '000333', '000858', '002415',
                       '300059', '600036', '600519', '601318', '601888'],
            '成分券名称': ['平安银行', '万科A', '美的集团', '五粮液', '海康威视',
                       '东方财富', '招商银行', '贵州茅台', '中国平安', '中国中免']
        })
    
    print(f"成分股数量: {len(constituents)}")
    
    # 2. 分析每只股票
    print("\n2. 分析股票技术指标...")
    results = []
    
    # 限制分析数量（演示用）
    sample_size = min(50, len(constituents))
    symbols = constituents['成分券代码'].head(sample_size).tolist()
    
    for i, symbol in enumerate(symbols, 1):
        print(f"进度: {i}/{sample_size} - 分析 {symbol}", end="\r")
        
        # 获取股票数据
        stock_data = get_stock_data(symbol)
        
        if stock_data is not None and len(stock_data) >= 60:
            # 计算技术指标
            tech_data = calculate_technical_indicators(stock_data)
            
            if tech_data is not None:
                # 生成信号
                signals = generate_signals(tech_data)
                
                if signals:
                    result = {
                        '股票代码': symbol,
                        '股票名称': constituents[constituents['成分券代码'] == symbol]['成分券名称'].iloc[0] if symbol in constituents['成分券代码'].values else 'N/A',
                        '最新价格': tech_data['close'].iloc[-1],
                        'MA5': tech_data['MA5'].iloc[-1],
                        'MA20': tech_data['MA20'].iloc[-1],
                        'RSI': tech_data['RSI'].iloc[-1] if pd.notnull(tech_data['RSI'].iloc[-1]) else None,
                        'MACD': tech_data['MACD'].iloc[-1],
                        '布林下轨': tech_data['BB_lower'].iloc[-1],
                        'MA金叉': signals['ma_cross'],
                        'RSI超卖': signals['rsi_oversold'],
                        'MACD金叉': signals['macd_cross'],
                        '布林触及': signals['bb_touch'],
                        '综合评分': signals['total_score'],
                        '趋势向上': signals['trend_up'],
                        '放量': signals['volume_spike']
                    }
                    results.append(result)
    
    print("\n3. 生成分析报告...")
    
    if results:
        results_df = pd.DataFrame(results)
        
        # 按综合评分排序
        results_df = results_df.sort_values('综合评分', ascending=False)
        
        # 保存结果
        results_df.to_csv('/root/.openclaw/workspace/csi1000_technical_analysis.csv', index=False, encoding='utf-8-sig')
        
        # 生成Markdown报告
        md_report = generate_markdown_report(results_df, constituents)
        with open('/root/.openclaw/workspace/csi1000_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        print(f"\n分析完成！共分析 {len(results)} 只股票")
        print(f"结果已保存至:")
        print(f"  - /root/.openclaw/workspace/csi1000_technical_analysis.csv")
        print(f"  - /root/.openclaw/workspace/csi1000_analysis_report.md")
        
        # 显示前10名
        print("\n综合评分前10名:")
        print("-" * 100)
        print(results_df[['股票代码', '股票名称', '最新价格', '综合评分', 'MA金叉', 'RSI超卖', 'MACD金叉', '布林触及']].head(10).to_string(index=False))
        
    else:
        print("未找到符合条件的股票")
    
    return results_df if results else None

def generate_markdown_report(results_df, constituents):
    """生成Markdown格式的报告"""
    report = "# 中证1000技术面量化选股分析报告\n\n"
    
    report += f"**报告生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"**分析股票数量:** {len(results_df)} / {len(constituents)}\n\n"
    
    report += "## 选股逻辑\n\n"
    report += "1. **MA金叉策略**: MA5上穿MA20，权重30%\n"
    report += "2. **RSI超卖策略**: RSI < 30，权重25%\n"
    report += "3. **MACD金叉策略**: DIF上穿DEA，权重25%\n"
    report += "4. **布林带策略**: 价格触及下轨，权重20%\n"
    report += "5. **综合评分**: 加权求和，最高1分\n\n"
    
    report += "## 参数设置\n\n"
    report += "- MA周期: 5, 10, 20, 60日\n"
    report += "- RSI周期: 14日\n"
    report += "- MACD参数: (12, 26, 9)\n"
    report += "- 布林带: 20日周期，2倍标准差\n"
    report += "- 数据周期: 最近1年日线数据\n\n"
    
    report += "## 选股结果\n\n"
    report += "### 综合评分排名前20\n\n"
    
    # 创建Markdown表格
    top20 = results_df.head(20)
    table = "| 排名 | 股票代码 | 股票名称 | 最新价格 | 综合评分 | MA金叉 | RSI超卖 | MACD金叉 | 布林触及 | 趋势 |\n"
    table += "|---|---|---|---|---|---|---|---|---|---|\n"
    
    for i, (idx, row) in enumerate(top20.iterrows(), 1):
        trend = "↑" if row['趋势向上'] == 1 else "→"
        table += f"| {i} | {row['股票代码']} | {row['股票名称']} | {row['最新价格']:.2f} | {row['综合评分']:.2f} | {'✓' if row['MA金叉'] == 1 else ''} | {'✓' if row['RSI超卖'] == 1 else ''} | {'✓' if row['MACD金叉'] == 1 else ''} | {'✓' if row['布林触及'] == 1 else ''} | {trend} |\n"
    
    report += table + "\n"
    
    # 策略统计
    report += "## 策略信号统计\n\n"
    total = len(results_df)
    ma_cross_count = results_df['MA金叉'].sum()
    rsi_count = results_df['RSI超卖'].sum()
    macd_count = results_df['MACD金叉'].sum()
    bb_count = results_df['布林触及'].sum()
    
    report += f"- MA金叉信号: {ma_cross_count} 只 ({ma_cross_count/total*100:.1f}%)\n"
    report += f"- RSI超卖信号: {rsi_count} 只 ({rsi_count/total*100:.1f}%)\n"
    report += f"- MACD金叉信号: {macd_count} 只 ({macd_count/total*100:.1f}%)\n"
    report += f"- 布林触及信号: {bb_count} 只 ({bb_count/total*100:.1f}%)\n\n"
    
    # 评分分布
    report += "## 综合评分分布\n\n"
    score_bins = pd.cut(results_df['综合评分'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    score_dist = score_bins.value_counts().sort_index()
    
    for bin_range, count in score_dist.items():
        report += f"- {bin_range}: {count} 只 ({count/total*100:.1f}%)\n"
    
    report += "\n## 风险提示\n\n"
    report += "1. 本分析基于历史数据，不代表未来表现\n"
    report += "2. 技术指标存在滞后性\n"
    report += "3. 建议结合基本面分析和风险管理\n"
    report += "4. 投资有风险，入市需谨慎\n"
    
    return report

if __name__ == "__main__":
    main()