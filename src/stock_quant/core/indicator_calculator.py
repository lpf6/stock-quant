"""技术指标计算器"""

from typing import Dict, List, Optional, Any
import pandas as pd

from .base import BaseIndicatorCalculator
from ..plugins.manager import PluginManager
from ..plugins.base import IndicatorPlugin


class IndicatorCalculator(BaseIndicatorCalculator):
    """技术指标计算器，支持插件化指标计算"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.plugin_manager = PluginManager()
        self.loaded_indicators: Dict[str, IndicatorPlugin] = {}
        self.default_indicators = ["MovingAverageIndicator", "RSIIndicator", "BollingerBandsIndicator"]
    
    def initialize(self) -> None:
        """初始化指标插件"""
        # 加载默认指标插件
        for indicator_name in self.default_indicators:
            try:
                indicator = self.plugin_manager.load_plugin(indicator_name)
                self.loaded_indicators[indicator_name] = indicator
            except Exception as e:
                print(f"加载指标插件 {indicator_name} 失败: {e}")
        
        # 加载用户配置的指标
        custom_indicators = self.config.get("indicators", [])
        for indicator_config in custom_indicators:
            self.load_indicator(indicator_config)
    
    def load_indicator(self, indicator_config: Dict[str, Any]) -> bool:
        """加载指标插件"""
        indicator_name = indicator_config.get("name")
        if not indicator_name:
            return False
        
        try:
            config = indicator_config.get("config", {})
            indicator = self.plugin_manager.load_plugin(indicator_name, config)
            self.loaded_indicators[indicator_name] = indicator
            return True
        except Exception as e:
            print(f"加载指标插件 {indicator_name} 失败: {e}")
            return False
    
    def unload_indicator(self, indicator_name: str) -> bool:
        """卸载指标插件"""
        if indicator_name in self.loaded_indicators:
            self.plugin_manager.unload_plugin(indicator_name)
            del self.loaded_indicators[indicator_name]
            return True
        return False
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有已加载指标"""
        self.initialize()
        
        if df is None or df.empty:
            return df
        
        result_df = df.copy()
        
        # 按顺序计算所有指标
        for indicator_name, indicator in self.loaded_indicators.items():
            try:
                result_df = indicator.calculate(result_df)
            except Exception as e:
                print(f"计算指标 {indicator_name} 失败: {e}")
        
        return result_df
    
    def calculate_specific_indicators(self, df: pd.DataFrame, indicator_names: List[str]) -> pd.DataFrame:
        """计算特定指标"""
        self.initialize()
        
        if df is None or df.empty:
            return df
        
        result_df = df.copy()
        
        for indicator_name in indicator_names:
            if indicator_name in self.loaded_indicators:
                try:
                    result_df = self.loaded_indicators[indicator_name].calculate(result_df)
                except Exception as e:
                    print(f"计算指标 {indicator_name} 失败: {e}")
            else:
                print(f"指标 {indicator_name} 未加载")
        
        return result_df
    
    def get_available_indicators(self) -> List[str]:
        """获取可用指标列表"""
        return list(self.loaded_indicators.keys())
    
    def get_indicator_info(self, indicator_name: str) -> Optional[Dict[str, Any]]:
        """获取指标信息"""
        if indicator_name in self.loaded_indicators:
            indicator = self.loaded_indicators[indicator_name]
            return {
                "name": indicator.name,
                "description": indicator.description,
                "author": indicator.author,
                "version": indicator.version,
                "parameters": indicator.get_parameters()
            }
        return None
    
    def update_indicator_config(self, indicator_name: str, config: Dict[str, Any]) -> bool:
        """更新指标配置"""
        if indicator_name in self.loaded_indicators:
            indicator = self.loaded_indicators[indicator_name]
            indicator.initialize(config)
            return True
        return False