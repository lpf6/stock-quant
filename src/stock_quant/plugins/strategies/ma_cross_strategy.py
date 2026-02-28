"""移动平均线交叉策略插件"""

import pandas as pd
from typing import Dict, Any, Union
from ..base import StrategyPlugin


class MACrossStrategy(StrategyPlugin):
    """移动平均线交叉策略"""
    
    def __init__(self):
        super().__init__()
        self.name = "MA Cross Strategy"
        self.description = "移动平均线交叉策略，检测短期均线上穿长期均线"
        self.author = "Stock Quant Team"
        self.config = {
            "short_period": 5,    # 短期均线周期
            "long_period": 20,    # 长期均线周期
            "signal_name": "ma_cross",
            "weight": 0.3,       # 策略权重
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> None:
        """初始化策略"""
        if config:
            self.config.update(config)
    
    def calculate_signals(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """计算交易信号"""
        if df is None or len(df) < 2:
            return {"ma_cross": 0, "score": 0.0}
        
        # 获取最新数据
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 检查必要的列是否存在
        short_col = f"MA{self.config['short_period']}"
        long_col = f"MA{self.config['long_period']}"
        
        if short_col not in df.columns or long_col not in df.columns:
            # 计算移动平均线
            df[short_col] = df['close'].rolling(window=self.config['short_period']).mean()
            df[long_col] = df['close'].rolling(window=self.config['long_period']).mean()
            latest = df.iloc[-1]
            prev = df.iloc[-2]
        
        # 检查金叉信号
        ma_cross = False
        if pd.notnull(prev[short_col]) and pd.notnull(prev[long_col]) and \
           pd.notnull(latest[short_col]) and pd.notnull(latest[long_col]):
            ma_cross = (prev[short_col] <= prev[long_col]) and (latest[short_col] > latest[long_col])
        
        signal_value = 1 if ma_cross else 0
        score = signal_value * self.config['weight']
        
        return {
            self.config['signal_name']: signal_value,
            "score": score,
            "short_ma": latest[short_col] if pd.notnull(latest[short_col]) else None,
            "long_ma": latest[long_col] if pd.notnull(latest[long_col]) else None
        }
    
    def get_signal_descriptions(self) -> Dict[str, str]:
        """获取信号描述"""
        return {
            "ma_cross": f"{self.config['short_period']}日线上穿{self.config['long_period']}日线",
            "score": "策略评分（权重乘以信号值）"
        }