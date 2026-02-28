"""RSI指标插件"""

import pandas as pd
from typing import List, Dict, Any
from ..base import IndicatorPlugin


class RSIIndicator(IndicatorPlugin):
    """RSI相对强弱指标"""
    
    def __init__(self):
        super().__init__()
        self.name = "RSI Indicator"
        self.description = "计算RSI相对强弱指标"
        self.author = "Stock Quant Team"
        self.config = {
            "period": 14,  # RSI计算周期
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> None:
        """初始化指标"""
        if config:
            self.config.update(config)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算RSI指标"""
        result_df = df.copy()
        
        if 'close' not in result_df.columns:
            raise ValueError("数据框中必须包含'close'列")
        
        # 计算价格变动
        delta = result_df['close'].diff()
        
        # 计算上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均上涨和下跌（使用简单移动平均）
        avg_gain = gain.rolling(window=self.config['period']).mean()
        avg_loss = loss.rolling(window=self.config['period']).mean()
        
        # 计算RS
        rs = avg_gain / avg_loss
        
        # 计算RSI
        result_df['RSI'] = 100 - (100 / (1 + rs))
        
        return result_df
    
    def get_indicator_names(self) -> List[str]:
        """获取指标名称列表"""
        return ["RSI"]