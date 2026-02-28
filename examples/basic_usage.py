"""基本使用示例"""

import pandas as pd
import numpy as np
from datetime import datetime

# 导入Stock Quant模块
from src.stock_quant.core.data_fetcher import DataFetcher
from src.stock_quant.core.indicator_calculator import IndicatorCalculator
from src.stock_quant.core.signal_generator import SignalGenerator
from src.stock_quant.plugins.manager import PluginManager
from src.stock_quant.utils.logger import setup_logger
from src.stock_quant.utils.formatter import OutputFormatter


def example_1_single_stock_analysis():
    """示例1：单只股票分析"""
    print("=== 示例1：单只股票分析 ===\n")
    
    # 1. 设置日志
    logger = setup_logger(level="INFO")
    
    # 2. 初始化数据获取器
    data_fetcher = DataFetcher({
        "data_source": "akshare",
        "max_retries": 3,
        "timeout": 30
    })
    
    # 3. 获取股票数据
    print("1. 获取股票数据...")
    symbol = "000001"  # 平安银行
    stock_data = data_fetcher.fetch_stock_data(
        symbol=symbol,
        start_date="20230101",
        end_date="20231231",
        period="daily"
    )
    
    if stock_data is None or stock_data.empty:
        print(f"无法获取股票 {symbol} 数据")
        return
    
    print(f"  获取到 {len(stock_data)} 条数据 (从 {stock_data.index[0].date()} 到 {stock_data.index[-1].date()})")
    print(f"  最新收盘价: {stock_data['close'].iloc[-1]:.2f}")
    
    # 4. 计算技术指标
    print("\n2. 计算技术指标...")
    indicator_calculator = IndicatorCalculator({
        "indicators": [
            {"name": "MovingAverageIndicator", "config": {"periods": [5, 10, 20, 60]}},
            {"name": "RSIIndicator", "config": {"period": 14}},
            {"name": "BollingerBandsIndicator", "config": {"period": 20, "std_dev": 2}}
        ]
    })
    
    indicators_data = indicator_calculator.calculate_indicators(stock_data)
    
    # 显示最新指标值
    latest = indicators_data.iloc[-1]
    print(f"  5日均线: {latest.get('MA5', 'N/A'):.2f}")
    print(f"  10日均线: {latest.get('MA10', 'N/A'):.2f}")
    print(f"  20日均线: {latest.get('MA20', 'N/A'):.2f}")
    print(f"  RSI: {latest.get('RSI', 'N/A'):.2f}")
    print(f"  布林带上轨: {latest.get('BB_upper', 'N/A'):.2f}")
    print(f"  布林带下轨: {latest.get('BB_lower', 'N/A'):.2f}")
    
    # 5. 生成交易信号
    print("\n3. 生成交易信号...")
    signal_generator = SignalGenerator({
        "strategies": [
            {"name": "MACrossStrategy", "config": {"short_period": 5, "long_period": 20}},
            {"name": "RSIStrategy", "config": {"oversold_threshold": 30, "overbought_threshold": 70}},
            {"name": "MACDStrategy", "config": {"fast_period": 12, "slow_period": 26}}
        ],
        "strategy_weights": {
            "MACrossStrategy": 0.4,
            "RSIStrategy": 0.3,
            "MACDStrategy": 0.3
        }
    })
    
    signals = signal_generator.generate_signals(indicators_data)
    
    # 显示信号
    print("\n交易信号:")
    for signal_name, signal_value in signals.items():
        if isinstance(signal_value, float):
            print(f"  {signal_name}: {signal_value:.4f}")
        else:
            print(f"  {signal_name}: {signal_value}")
    
    # 6. 输出结果
    print("\n4. 输出结果...")
    
    # CSV格式
    formatter_csv = OutputFormatter("csv", {"include_header": True})
    csv_output = formatter_csv.format_data([{"symbol": symbol, **signals}])
    print("\nCSV格式输出:")
    print(csv_output)
    
    # JSON格式
    formatter_json = OutputFormatter("json", {"indent": 2})
    json_output = formatter_json.format_data([{"symbol": symbol, **signals}])
    print("\nJSON格式输出（前200字符）:")
    print(json_output[:200] + "...")
    
    # 保存到文件
    formatter_csv.save_to_file([{"symbol": symbol, **signals}], "output/single_stock_analysis.csv")
    print("\n结果已保存到: output/single_stock_analysis.csv")
    
    return {"symbol": symbol, "signals": signals, "data": indicators_data}


def example_2_batch_stock_analysis():
    """示例2：批量股票分析"""
    print("\n\n=== 示例2：批量股票分析 ===\n")
    
    # 股票列表
    symbols = ["000001", "000002", "000858", "600519", "000333"]
    
    print(f"分析 {len(symbols)} 只股票: {', '.join(symbols)}")
    
    # 初始化组件
    data_fetcher = DataFetcher()
    indicator_calculator = IndicatorCalculator()
    signal_generator = SignalGenerator()
    
    # 批量分析
    results = []
    
    for symbol in symbols:
        print(f"\n分析股票 {symbol}...")
        
        # 获取数据（这里使用模拟数据，实际使用时取消注释下面的代码）
        # stock_data = data_fetcher.fetch_stock_data(symbol, start_date="20230701", end_date="20231231")
        
        # 创建模拟数据用于演示
        dates = pd.date_range(start='2023-07-01', periods=60, freq='D')
        np.random.seed(hash(symbol) % 1000)
        
        prices = 10 + np.cumsum(np.random.randn(60) * 0.1)
        stock_data = pd.DataFrame({
            'open': prices,
            'close': prices,
            'high': prices + np.random.rand(60) * 0.5,
            'low': prices - np.random.rand(60) * 0.5,
            'volume': np.random.randint(1000000, 5000000, 60),
            'amount': prices * np.random.randint(10000000, 50000000, 60)
        }, index=dates)
        
        if stock_data is None or stock_data.empty:
            print(f"  警告: 股票 {symbol} 数据获取失败")
            continue
        
        # 计算指标和信号
        indicators_data = indicator_calculator.calculate_indicators(stock_data)
        signals = signal_generator.generate_signals(indicators_data)
        
        # 记录结果
        results.append({
            "symbol": symbol,
            "latest_price": stock_data['close'].iloc[-1],
            "total_score": signals.get("total_score", 0),
            "ma_cross": signals.get("MACrossStrategy.ma_cross", 0),
            "rsi_oversold": signals.get("RSIStrategy.rsi_oversold", 0),
            "macd_cross": signals.get("MACDStrategy.macd_cross", 0),
            "trend_up": signals.get("trend_up", 0),
            "volume_spike": signals.get("volume_spike", 0)
        })
        
        print(f"  最新价: {stock_data['close'].iloc[-1]:.2f}")
        print(f"  综合评分: {signals.get('total_score', 0):.3f}")
    
    # 输出批量分析结果
    print("\n批量分析结果:")
    
    # 按评分排序
    results_sorted = sorted(results, key=lambda x: x["total_score"], reverse=True)
    
    print("\n排名  代码     最新价   综合评分  MA金叉  RSI超卖  MACD金叉  趋势  放量")
    print("-" * 70)
    
    for i, result in enumerate(results_sorted, 1):
        print(f"{i:2}   {result['symbol']}   {result['latest_price']:7.2f}   "
              f"{result['total_score']:7.3f}   {result['ma_cross']:4}   "
              f"{result['rsi_oversold']:6}   {result['macd_cross']:7}   "
              f"{result['trend_up']:3}   {result['volume_spike']:3}")
    
    # 保存结果
    formatter = OutputFormatter("csv")
    formatter.save_to_file(results_sorted, "output/batch_analysis_results.csv")
    
    # 同时保存为Markdown格式
    formatter_md = OutputFormatter("markdown", {"table_format": "github"})
    md_output = formatter_md.format_data(results_sorted)
    
    with open("output/batch_analysis_results.md", "w", encoding="utf-8") as f:
        f.write("# 批量股票分析结果\n\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析股票数: {len(results_sorted)}\n\n")
        f.write(md_output)
    
    print("\n结果已保存到:")
    print("  - output/batch_analysis_results.csv")
    print("  - output/batch_analysis_results.md")
    
    return results_sorted


def example_3_plugin_management():
    """示例3：插件管理"""
    print("\n\n=== 示例3：插件管理 ===\n")
    
    # 初始化插件管理器
    plugin_manager = PluginManager()
    
    # 1. 查看可用插件
    print("1. 可用插件列表:")
    
    # 所有插件
    all_plugins = plugin_manager.get_available_plugins()
    print(f"   所有插件 ({len(all_plugins)} 个):")
    for plugin in all_plugins:
        print(f"     - {plugin['name']} ({plugin['type']}): {plugin['description']}")
    
    # 策略插件
    strategy_plugins = plugin_manager.get_available_plugins("strategy")
    print(f"\n   策略插件 ({len(strategy_plugins)} 个):")
    for plugin in strategy_plugins:
        print(f"     - {plugin['name']}: {plugin['description']}")
    
    # 指标插件
    indicator_plugins = plugin_manager.get_available_plugins("indicator")
    print(f"\n   指标插件 ({len(indicator_plugins)} 个):")
    for plugin in indicator_plugins:
        print(f"     - {plugin['name']}: {plugin['description']}")
    
    # 2. 加载和使用插件
    print("\n2. 加载和使用插件:")
    
    # 加载MA交叉策略插件
    ma_strategy = plugin_manager.load_plugin("MACrossStrategy", {
        "short_period": 10,
        "long_period": 30,
        "weight": 0.35
    })
    
    print(f"   已加载插件: {ma_strategy.name}")
    print(f"   插件描述: {ma_strategy.description}")
    print(f"   插件参数: {ma_strategy.get_parameters()}")
    
    # 3. 创建测试数据并使用插件
    print("\n3. 使用插件计算信号:")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'close': [10.0, 10.2, 10.5, 10.8, 11.0, 11.3, 11.5, 11.2, 11.0, 10.8]
    })
    
    # 需要先计算移动平均线
    test_data['MA10'] = test_data['close'].rolling(10).mean()
    test_data['MA30'] = test_data['close'].rolling(30).mean()
    
    # 使用插件计算信号
    signals = ma_strategy.calculate_signals(test_data)
    
    print(f"   信号结果: {signals}")
    print(f"   信号描述: {ma_strategy.get_signal_descriptions()}")
    
    # 4. 插件依赖检查
    print("\n4. 插件依赖检查:")
    
    # 检查MA交叉策略的依赖
    dependencies = plugin_manager.check_dependencies("MACrossStrategy")
    if dependencies:
        print(f"   缺失依赖: {dependencies}")
    else:
        print("   所有依赖已满足")
    
    # 5. 卸载插件
    print("\n5. 插件卸载:")
    plugin_manager.unload_plugin("MACrossStrategy")
    print("   MA交叉策略插件已卸载")
    
    # 验证插件已卸载
    plugin = plugin_manager.get_plugin("MACrossStrategy")
    if plugin is None:
        print("   验证: 插件已成功卸载")
    
    return {"plugins": all_plugins, "strategy_plugins": strategy_plugins, "indicator_plugins": indicator_plugins}


def example_4_custom_plugin_development():
    """示例4：自定义插件开发"""
    print("\n\n=== 示例4：自定义插件开发 ===\n")
    
    print("1. 创建自定义策略插件:")
    
    # 定义自定义策略插件
    from src.stock_quant.plugins.base import StrategyPlugin
    import pandas as pd
    
    class VolumeSpikeStrategy(StrategyPlugin):
        """成交量异动策略"""
        
        def __init__(self):
            super().__init__()
            self.name = "Volume Spike Strategy"
            self.description = "检测成交量异常放大的策略"
            self.author = "示例开发者"
            self.version = "1.0.0"
            self.config = {
                "volume_multiplier": 2.0,  # 成交量倍数阈值
                "lookback_days": 20,       # 回顾天数
                "signal_name": "volume_spike_signal",
                "weight": 0.2
            }
        
        def initialize(self, config=None):
            if config:
                self.config.update(config)
        
        def calculate_signals(self, df):
            """计算成交量异动信号"""
            if df is None or len(df) < self.config["lookback_days"]:
                return {"volume_spike_signal": 0, "score": 0.0}
            
            latest = df.iloc[-1]
            
            # 计算成交量均线
            if 'volume' not in df.columns:
                return {"volume_spike_signal": 0, "score": 0.0}
            
            volume_ma = df['volume'].rolling(self.config["lookback_days"]).mean().iloc[-1]
            
            if pd.isnull(volume_ma) or volume_ma == 0:
                return {"volume_spike_signal": 0, "score": 0.0}
            
            # 检查成交量是否超过阈值
            volume_ratio = latest['volume'] / volume_ma
            volume_spike = volume_ratio > self.config["volume_multiplier"]
            
            signal_value = 1 if volume_spike else 0
            score = signal_value * self.config["weight"]
            
            return {
                self.config["signal_name"]: signal_value,
                "score": score,
                "volume_ratio": volume_ratio,
                "volume_ma": volume_ma
            }
        
        def get_signal_descriptions(self):
            return {
                "volume_spike_signal": f"成交量超过{self.config['volume_multiplier']}倍均量",
                "score": "策略评分",
                "volume_ratio": "成交量与均量比值",
                "volume_ma": "成交量均线值"
            }
    
    print("   已定义 VolumeSpikeStrategy 策略插件")
    print(f"   插件名称: {VolumeSpikeStrategy().name}")
    print(f"   插件描述: {VolumeSpikeStrategy().description}")
    print(f"   插件作者: {VolumeSpikeStrategy().author}")
    
    print("\n2. 使用自定义插件:")
    
    # 创建插件实例
    volume_strategy = VolumeSpikeStrategy()
    volume_strategy.initialize({"volume_multiplier": 1.5})  # 自定义参数
    
    # 创建测试数据
    test_dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    test_data = pd.DataFrame({
        'close': 10 + np.cumsum(np.random.randn(30) * 0.1),
        'volume': np.random.randint(1000000, 2000000, 30)
    }, index=test_dates)
    
    # 在最后一天制造成交量异动
    test_data.iloc[-1, test_data.columns.get_loc('volume')] = 5000000
    
    # 使用自定义策略
    signals = volume_strategy.calculate_signals(test_data)
    
    print(f"   信号结果:")
    for key, value in signals.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.4f}")
        else:
            print(f"     {key}: {value}")
    
    print(f"\n   信号描述:")
    for key, desc in volume_strategy.get_signal_descriptions().items():
        print(f"     {key}: {desc}")
    
    print("\n3. 集成自定义插件到系统:")
    
    # 创建插件管理器并注册自定义插件
    plugin_manager = PluginManager()
    plugin_manager.register_plugin(VolumeSpikeStrategy)
    
    # 查看是否已注册
    available_plugins = plugin_manager.get_available_plugins("strategy")
    custom_plugin_names = [p["name"] for p in available_plugins]
    
    if "Volume Spike Strategy" in custom_plugin_names:
        print("   自定义插件已成功注册到系统")
        print(f"   当前策略插件: {custom_plugin_names}")
    else:
        print("   自定义插件注册失败")
    
    return {"custom_strategy": volume_strategy, "signals": signals}


def main():
    """主函数"""
    print("Stock Quant 示例程序")
    print("=" * 50)
    
    try:
        # 创建输出目录
        import os
        os.makedirs("output", exist_ok=True)
        
        # 运行示例
        result1 = example_1_single_stock_analysis()
        result2 = example_2_batch_stock_analysis()
        result3 = example_3_plugin_management()
        result4 = example_4_custom_plugin_development()
        
        print("\n\n所有示例执行完成!")
        print("输出文件已保存到 'output' 目录")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()