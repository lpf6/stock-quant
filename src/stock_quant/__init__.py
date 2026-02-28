"""Stock Quant - 技术面量化选股系统

一个插件化、多周期、配置化的量化选股系统，支持批量处理和技术分析。
"""

__version__ = "0.1.0"
__author__ = "Stock Quant Team"
__license__ = "MIT"

from .core.data_fetcher import DataFetcher
from .core.data_processor import DataProcessor
from .core.indicator_calculator import IndicatorCalculator
from .core.signal_generator import SignalGenerator
from .plugins.manager import PluginManager
from .plugins.base import StrategyPlugin, IndicatorPlugin
from .period.period_manager import PeriodManager
from .config.config_loader import ConfigLoader
from .utils.logger import setup_logger

__all__ = [
    "DataFetcher",
    "DataProcessor",
    "IndicatorCalculator",
    "SignalGenerator",
    "PluginManager",
    "StrategyPlugin",
    "IndicatorPlugin",
    "PeriodManager",
    "ConfigLoader",
    "setup_logger",
]