"""插件基类定义"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class BasePlugin(ABC):
    """插件基类"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.description = ""
        self.author = ""
        self.enabled = True
        self.config = {}
    
    @property
    @abstractmethod
    def plugin_type(self) -> str:
        """插件类型（strategy/indicator）"""
        pass
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化插件"""
        pass
    
    def cleanup(self) -> None:
        """清理资源"""
        pass


class StrategyPlugin(BasePlugin):
    """策略插件基类"""
    
    @property
    def plugin_type(self) -> str:
        return "strategy"
    
    @abstractmethod
    def calculate_signals(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """计算交易信号"""
        pass
    
    @abstractmethod
    def get_signal_descriptions(self) -> Dict[str, str]:
        """获取信号描述"""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self.config.get("parameters", {})


class IndicatorPlugin(BasePlugin):
    """指标插件基类"""
    
    @property
    def plugin_type(self) -> str:
        return "indicator"
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指标"""
        pass
    
    @abstractmethod
    def get_indicator_names(self) -> List[str]:
        """获取指标名称"""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取指标参数"""
        return self.config.get("parameters", {})


class PluginMetadata:
    """插件元数据"""
    
    def __init__(
        self,
        name: str,
        plugin_type: str,
        version: str,
        description: str = "",
        author: str = "",
        dependencies: List[str] = None,
        config_schema: Dict[str, Any] = None
    ):
        self.name = name
        self.plugin_type = plugin_type
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies or []
        self.config_schema = config_schema or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "plugin_type": self.plugin_type,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "config_schema": self.config_schema
        }