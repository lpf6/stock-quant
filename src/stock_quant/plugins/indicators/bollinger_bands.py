"""布林带指标插件"""

import pandas as pd
from typing import List, Dict, Any
from ..base import IndicatorPlugin


class BollingerBandsIndicator(IndicatorPlugin):
    """布林带指标"""
    
    def __init__(self):
        super().__init__()
        self.name = "Bollinger Bands Indicator"
        self.description = "计算布林带指标（中轨、上轨、下轨、带宽）"
        self.author = "Stock Quant Team"
        self.config = {
            "period": 20,      # 布林带周期
            "std_dev": 2,      # 标准差倍数
            "column_name": "close",  # 计算基准列
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> None:
        """初始化指标"""
        if config:
            self.config.update(config)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带指标"""
        result_df = df.copy()
        column = self.config['column_name']
        
        if column not in result_df.columns:
            raise ValueError(f"列 {column} 不存在于数据框中")
        
        period = self.config['period']
        std_dev = self.config['std_dev']
        
        # 计算中轨（移动平均）
        result_df['BB_middle'] = result_df[column].rolling(window=period).mean()
        
        # 计算标准差
        bb_std = result_df[column].rolling(window=period).std()
        
        # 计算上轨和下轨
        result_df['BB_upper'] = result_df['BB_middle'] + std_dev * bb_std
        result_df['BB_lower'] = result_df['BB_middle'] - std_dev * bb_std
        
        # 计算带宽（百分比）
        result_df['BB_width'] = (result_df['BB_upper'] - result_df['BB_lower']) / result_df['BB_middle']
        
        # 计算价格相对于布林带的位置
        result_df['BB_position'] = (result_df[column] - result_df['BB_lower']) / (result_df['BB_upper'] - result_df['BB_lower'])
        
        return result_df
    
    def get_indicator_names(self) -> List[str]:
        """获取指标名称列表"""
        return ["BB_middle", "BB_upper", "BB_lower", "BB_width", "BB_position"]