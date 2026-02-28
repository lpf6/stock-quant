"""基础类定义"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd


class BaseComponent(ABC):
    """基础组件类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新配置"""
        self.config.update(new_config)
        
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return self.config.copy()


class BaseDataFetcher(BaseComponent):
    """数据获取器基类"""
    
    @abstractmethod
    def fetch_stock_data(self, symbol: str, start_date: str = None, 
                         end_date: str = None, period: str = "daily") -> pd.DataFrame:
        """获取股票数据"""
        pass


class BaseDataProcessor(BaseComponent):
    """数据处理器基类"""
    
    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        pass
    
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """数据验证"""
        pass


class BaseIndicatorCalculator(BaseComponent):
    """指标计算器基类"""
    
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        pass


class BaseSignalGenerator(BaseComponent):
    """信号生成器基类"""
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        pass