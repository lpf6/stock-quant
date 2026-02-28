"""移动平均线指标插件"""

import pandas as pd
from typing import List, Dict, Any
from ..base import IndicatorPlugin


class MovingAverageIndicator(IndicatorPlugin):
    """移动平均线指标"""
    
    def __init__(self):
        super().__init__()
        self.name = "Moving Average Indicator"
        self.description = "计算多种周期的移动平均线"
        self.author = "Stock Quant Team"
        self.config = {
            "periods": [5, 10, 20, 60],  # 默认计算周期
            "column_name": "close",       # 计算基准列
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> None:
        """初始化指标"""
        if config:
            self.config.update(config)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均线"""
        result_df = df.copy()
        column = self.config['column_name']
        
        if column not in result_df.columns:
            raise ValueError(f"列 {column} 不存在于数据框中")
        
        # 计算各周期移动平均线
        for period in self.config['periods']:
            col_name = f"MA{period}"
            result_df[col_name] = result_df[column].rolling(window=period).mean()
        
        return result_df
    
    def get_indicator_names(self) -> List[str]:
        """获取指标名称列表"""
        return [f"MA{period}" for period in self.config['periods']]