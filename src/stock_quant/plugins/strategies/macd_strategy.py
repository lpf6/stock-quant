"""MACD金叉死叉策略插件"""

import pandas as pd
from typing import Dict, Any, Union
from ..base import StrategyPlugin


class MACDStrategy(StrategyPlugin):
    """MACD策略"""
    
    def __init__(self):
        super().__init__()
        self.name = "MACD Strategy"
        self.description = "MACD金叉死叉策略，检测MACD指标信号"
        self.author = "Stock Quant Team"
        self.config = {
            "fast_period": 12,    # 快线周期
            "slow_period": 26,    # 慢线周期
            "signal_period": 9,   # 信号线周期
            "signal_name": "macd_cross",
            "weight": 0.25,      # 策略权重
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> None:
        """初始化策略"""
        if config:
            self.config.update(config)
    
    def calculate_signals(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """计算交易信号"""
        if df is None or len(df) < max(self.config['fast_period'], self.config['slow_period']) + 2:
            return {"macd_cross": 0, "score": 0.0}
        
        # 检查MACD相关列是否存在
        if 'MACD' not in df.columns or 'MACD_Signal' not in df.columns:
            # 计算MACD
            exp1 = df['close'].ewm(span=self.config['fast_period'], adjust=False).mean()
            exp2 = df['close'].ewm(span=self.config['slow_period'], adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=self.config['signal_period'], adjust=False).mean()
        
        # 获取最新数据
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 检查金叉信号
        macd_cross = False
        if pd.notnull(prev['MACD']) and pd.notnull(prev['MACD_Signal']) and \
           pd.notnull(latest['MACD']) and pd.notnull(latest['MACD_Signal']):
            macd_cross = (prev['MACD'] <= prev['MACD_Signal']) and (latest['MACD'] > latest['MACD_Signal'])
        
        signal_value = 1 if macd_cross else 0
        score = signal_value * self.config['weight']
        
        return {
            self.config['signal_name']: signal_value,
            "score": score,
            "macd_value": latest['MACD'] if pd.notnull(latest['MACD']) else None,
            "macd_signal": latest['MACD_Signal'] if pd.notnull(latest['MACD_Signal']) else None,
            "macd_histogram": (latest['MACD'] - latest['MACD_Signal']) if pd.notnull(latest['MACD']) and pd.notnull(latest['MACD_Signal']) else None
        }
    
    def get_signal_descriptions(self) -> Dict[str, str]:
        """获取信号描述"""
        return {
            "macd_cross": f"MACD金叉（快线上穿慢线）",
            "score": "策略评分（权重乘以信号值）"
        }