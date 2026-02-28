"""命令行接口"""

import argparse
import sys
from typing import List, Optional

from ..core.data_fetcher import DataFetcher
from ..core.indicator_calculator import IndicatorCalculator
from ..core.signal_generator import SignalGenerator
from ..plugins.manager import PluginManager
from ..config.config_loader import ConfigLoader
from ..utils.logger import setup_logger
from ..utils.formatter import OutputFormatter


def main(args: Optional[List[str]] = None) -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Stock Quant - 技术面量化选股系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  stock-quant analyze --symbol 000001 --period 1y
  stock-quant batch --index 000852 --output results.csv
  stock-quant plugins list
"""
    )
    
    # 全局参数
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--format", default="csv", choices=["csv", "json", "html", "markdown"], help="输出格式")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # analyze命令：分析单只股票
    analyze_parser = subparsers.add_parser("analyze", help="分析单只股票")
    analyze_parser.add_argument("--symbol", required=True, help="股票代码")
    analyze_parser.add_argument("--period", default="1y", help="分析周期")
    analyze_parser.add_argument("--timeframe", default="daily", choices=["daily", "weekly", "monthly"], help="时间框架")
    analyze_parser.add_argument("--strategies", nargs="+", help="使用的策略列表")
    analyze_parser.add_argument("--indicators", nargs="+", help="使用的指标列表")
    
    # batch命令：批量分析
    batch_parser = subparsers.add_parser("batch", help="批量分析股票")
    batch_parser.add_argument("--symbols", nargs="+", help="股票代码列表")
    batch_parser.add_argument("--index", help="指数代码（如中证1000）")
    batch_parser.add_argument("--period", default="1y", help="分析周期")
    batch_parser.add_argument("--batch-size", type=int, default=50, help="批量大小")
    batch_parser.add_argument("--filter", action="store_true", help="启用过滤")
    
    # plugins命令：插件管理
    plugins_parser = subparsers.add_parser("plugins", help="插件管理")
    plugins_subparsers = plugins_parser.add_subparsers(dest="plugins_command", help="插件子命令")
    
    plugins_list_parser = plugins_subparsers.add_parser("list", help="列出所有插件")
    plugins_list_parser.add_argument("--type", choices=["strategy", "indicator"], help="插件类型")
    
    plugins_info_parser = plugins_subparsers.add_parser("info", help="查看插件信息")
    plugins_info_parser.add_argument("plugin_name", help="插件名称")
    
    plugins_load_parser = plugins_subparsers.add_parser("load", help="加载插件")
    plugins_load_parser.add_argument("plugin_name", help="插件名称")
    plugins_load_parser.add_argument("--config", help="插件配置文件")
    
    plugins_unload_parser = plugins_subparsers.add_parser("unload", help="卸载插件")
    plugins_unload_parser.add_argument("plugin_name", help="插件名称")
    
    # config命令：配置管理
    config_parser = subparsers.add_parser("config", help="配置管理")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="配置子命令")
    
    config_show_parser = config_subparsers.add_parser("show", help="显示配置")
    config_show_parser.add_argument("key", nargs="?", help="配置键")
    
    config_set_parser = config_subparsers.add_parser("set", help="设置配置")
    config_set_parser.add_argument("key", help="配置键")
    config_set_parser.add_argument("value", help="配置值")
    
    config_validate_parser = config_subparsers.add_parser("validate", help="验证配置")
    
    # 解析参数
    parsed_args = parser.parse_args(args)
    
    # 如果没有指定命令，显示帮助
    if not parsed_args.command:
        parser.print_help()
        sys.exit(1)
    
    # 执行命令
    try:
        if parsed_args.command == "analyze":
            analyze_stock(parsed_args)
        elif parsed_args.command == "batch":
            batch_analyze(parsed_args)
        elif parsed_args.command == "plugins":
            manage_plugins(parsed_args)
        elif parsed_args.command == "config":
            manage_config(parsed_args)
        else:
            print(f"未知命令: {parsed_args.command}")
            sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


def analyze_stock(args) -> None:
    """分析单只股票"""
    print(f"分析股票 {args.symbol} (周期: {args.period}, 时间框架: {args.timeframe})")
    
    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.load_all_configs()
    
    # 设置日志
    logger = setup_logger(level=args.log_level)
    
    # 初始化组件
    data_fetcher = DataFetcher(config.get("data", {}))
    indicator_calculator = IndicatorCalculator(config.get("indicators", {}))
    signal_generator = SignalGenerator(config.get("strategies", {}))
    
    # 获取数据
    logger.info(f"获取股票 {args.symbol} 数据...")
    stock_data = data_fetcher.fetch_stock_data(
        symbol=args.symbol,
        period=args.timeframe
    )
    
    if stock_data is None or stock_data.empty:
        print(f"无法获取股票 {args.symbol} 数据")
        sys.exit(1)
    
    print(f"获取到 {len(stock_data)} 条数据")
    
    # 计算指标
    logger.info("计算技术指标...")
    if args.indicators:
        indicators_data = indicator_calculator.calculate_specific_indicators(stock_data, args.indicators)
    else:
        indicators_data = indicator_calculator.calculate_indicators(stock_data)
    
    # 生成信号
    logger.info("生成交易信号...")
    signals = signal_generator.generate_signals(indicators_data)
    
    # 显示结果
    print("\n交易信号:")
    for signal_name, signal_value in signals.items():
        print(f"  {signal_name}: {signal_value}")
    
    # 保存结果
    if args.output:
        formatter = OutputFormatter(format_type=args.format, config=config.get("output", {}))
        result_data = [{"symbol": args.symbol, **signals}]
        formatter.save_to_file(result_data, args.output)
        print(f"结果已保存到: {args.output}")


def batch_analyze(args) -> None:
    """批量分析股票"""
    print("批量分析股票...")
    
    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.load_all_configs()
    
    # 设置日志
    logger = setup_logger(level=args.log_level)
    
    # 初始化组件
    data_fetcher = DataFetcher(config.get("data", {}))
    indicator_calculator = IndicatorCalculator(config.get("indicators", {}))
    signal_generator = SignalGenerator(config.get("strategies", {}))
    
    # 获取股票列表
    symbols = []
    if args.symbols:
        symbols = args.symbols
    elif args.index:
        logger.info(f"获取指数 {args.index} 成分股...")
        constituents = data_fetcher.get_index_constituents(args.index)
        if constituents is not None:
            symbols = constituents["成分券代码"].tolist()[:config.get("stock_pool", {}).get("max_stocks", 1000)]
    else:
        print("请指定股票代码列表或指数代码")
        sys.exit(1)
    
    print(f"分析 {len(symbols)} 只股票")
    
    # 批量分析
    results = []
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"分析股票 {i}/{len(symbols)}: {symbol}")
        
        try:
            # 获取数据
            stock_data = data_fetcher.fetch_stock_data(symbol, period="daily")
            if stock_data is None or stock_data.empty:
                logger.warning(f"股票 {symbol} 数据不足")
                continue
            
            # 计算指标和信号
            indicators_data = indicator_calculator.calculate_indicators(stock_data)
            signals = signal_generator.generate_signals(indicators_data)
            
            # 添加结果
            result = {"symbol": symbol, **signals}
            results.append(result)
            
            # 显示进度
            if i % 10 == 0 or i == len(symbols):
                print(f"进度: {i}/{len(symbols)} ({i/len(symbols)*100:.1f}%)")
                
        except Exception as e:
            logger.error(f"分析股票 {symbol} 失败: {e}")
    
    # 输出结果
    print(f"\n分析完成，成功分析 {len(results)} 只股票")
    
    if args.output:
        formatter = OutputFormatter(format_type=args.format, config=config.get("output", {}))
        formatter.save_to_file(results, args.output)
        print(f"结果已保存到: {args.output}")
    else:
        # 显示前10条结果
        print("\n前10条结果:")
        for result in results[:10]:
            print(f"{result['symbol']}: 总分={result.get('total_score', 0):.2f}")


def manage_plugins(args) -> None:
    """管理插件"""
    plugin_manager = PluginManager()
    
    if args.plugins_command == "list":
        plugins = plugin_manager.get_available_plugins(args.type)
        print(f"可用插件 ({args.type or '所有类型'}):")
        for plugin in plugins:
            print(f"  {plugin['name']} ({plugin['type']}) - {plugin['description']}")
            
    elif args.plugins_command == "info":
        plugin = plugin_manager.get_plugin(args.plugin_name)
        if plugin:
            print(f"插件信息: {args.plugin_name}")
            print(f"  名称: {plugin.name}")
            print(f"  类型: {plugin.plugin_type}")
            print(f"  描述: {plugin.description}")
            print(f"  作者: {plugin.author}")
            print(f"  版本: {plugin.version}")
        else:
            print(f"插件 {args.plugin_name} 未找到")
            
    elif args.plugins_command == "load":
        try:
            plugin = plugin_manager.load_plugin(args.plugin_name)
            print(f"插件 {args.plugin_name} 加载成功")
        except Exception as e:
            print(f"加载插件失败: {e}")
            
    elif args.plugins_command == "unload":
        if plugin_manager.unload_plugin(args.plugin_name):
            print(f"插件 {args.plugin_name} 卸载成功")
        else:
            print(f"插件 {args.plugin_name} 未找到")


def manage_config(args) -> None:
    """管理配置"""
    config_loader = ConfigLoader()
    
    if args.config_command == "show":
        if args.key:
            value = config_loader.get_config(args.key)
            print(f"{args.key}: {value}")
        else:
            config = config_loader.load_all_configs()
            print(json.dumps(config, indent=2, ensure_ascii=False))
            
    elif args.config_command == "set":
        config_loader.set_config(args.key, args.value)
        config_loader.save_config()
        print(f"配置已更新: {args.key} = {args.value}")
        
    elif args.config_command == "validate":
        errors = config_loader.validate_config()
        if errors:
            print("配置验证失败:")
            for error in errors:
                print(f"  • {error}")
        else:
            print("配置验证通过")


if __name__ == "__main__":
    main()