"""自定义异常"""


class StockQuantError(Exception):
    """Stock Quant基础异常"""
    pass


class DataFetchError(StockQuantError):
    """数据获取异常"""
    def __init__(self, symbol: str, message: str = "数据获取失败"):
        self.symbol = symbol
        super().__init__(f"股票 {symbol}: {message}")


class IndicatorCalculationError(StockQuantError):
    """指标计算异常"""
    def __init__(self, indicator: str, message: str = "指标计算失败"):
        self.indicator = indicator
        super().__init__(f"指标 {indicator}: {message}")


class StrategyExecutionError(StockQuantError):
    """策略执行异常"""
    def __init__(self, strategy: str, message: str = "策略执行失败"):
        self.strategy = strategy
        super().__init__(f"策略 {strategy}: {message}")


class PluginError(StockQuantError):
    """插件异常"""
    def __init__(self, plugin: str, message: str = "插件错误"):
        self.plugin = plugin
        super().__init__(f"插件 {plugin}: {message}")


class ConfigurationError(StockQuantError):
    """配置异常"""
    def __init__(self, config_key: str, message: str = "配置错误"):
        self.config_key = config_key
        super().__init__(f"配置 {config_key}: {message}")


class ValidationError(StockQuantError):
    """数据验证异常"""
    def __init__(self, field: str, value: any, message: str = "数据验证失败"):
        self.field = field
        self.value = value
        super().__init__(f"字段 {field}={value}: {message}")


class TimeFrameError(StockQuantError):
    """时间框架异常"""
    def __init__(self, timeframe: str, message: str = "时间框架错误"):
        self.timeframe = timeframe
        super().__init__(f"时间框架 {timeframe}: {message}")


class BatchProcessingError(StockQuantError):
    """批量处理异常"""
    def __init__(self, batch_id: str, message: str = "批量处理失败"):
        self.batch_id = batch_id
        super().__init__(f"批次 {batch_id}: {message}")


class InsufficientDataError(StockQuantError):
    """数据不足异常"""
    def __init__(self, min_required: int, actual: int, message: str = "数据不足"):
        self.min_required = min_required
        self.actual = actual
        super().__init__(f"需要至少 {min_required} 条数据，实际只有 {actual} 条: {message}")


class DependencyError(StockQuantError):
    """依赖异常"""
    def __init__(self, dependency: str, message: str = "依赖错误"):
        self.dependency = dependency
        super().__init__(f"依赖 {dependency}: {message}")