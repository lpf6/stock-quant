"""RSI超卖超买策略插件"""

import pandas as pd
from typing import Dict, Any, Union
from ..base import StrategyPlugin


class RSIStrategy(StrategyPlugin):
    """RSI策略"""
    
    def __init__(self):
        super().__init__()
        self.name = "RSI Strategy"
        self.description = "RSI超卖超买策略，检测RSI指标信号"
        self.author = "Stock Quant Team"
        self.config = {
            "rsi_period": 14,           # RSI计算周期
            "oversold_threshold": 35,   # 超卖阈值
            "overbought_threshold": 70, # 超买阈值
            "signal_name": "rsi_oversold",
            "weight": 0.25,            # 策略权重
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> None:
        """初始化策略"""
        if config:
            self.config.update(config)
    
    def calculate_signals(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """计算交易信号"""
        if df is None or len(df) < self.config['rsi_period'] + 1:
            return {"rsi_oversold": 0, "rsi_overbought": 0, "score": 0.0}
        
        # 检查RSI列是否存在
        if 'RSI' not in df.columns:
            # 计算RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.config['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        latest = df.iloc[-1]
        
        # 检查RSI值
        rsi_value = latest['RSI'] if pd.notnull(latest['RSI']) else None
        
        if rsi_value is None:
            return {"rsi_oversold": 0, "rsi_overbought": 0, "score": 0.0}
        
        # 判断超卖超买
        oversold = rsi_value < self.config['oversold_threshold']
        overbought = rsi_value > self.config['overbought_threshold']
        
        oversold_signal = 1 if oversold else 0
        overbought_signal = 1 if overbought else 0
        
        # 综合信号（超卖为买入信号，超买为卖出信号）
        signal_value = oversold_signal  # 仅使用超卖作为买入信号
        score = signal_value * self.config['weight']
        
        return {
            self.config['signal_name']: signal_value,
            "rsi_overbought": overbought_signal,
            "score": score,
            "rsi_value": rsi_value,
            "oversold_threshold": self.config['oversold_threshold'],
            "overbought_threshold": self.config['overbought_threshold']
        }
    
    def get_signal_descriptions(self) -> Dict[str, str]:
        """获取信号描述"""
        return {
            "rsi_oversold": f"RSI低于{self.config['oversold_threshold']}（超卖）",
            "rsi_overbought": f"RSI高于{self.config['overbought_threshold']}（超买）",
            "score": "策略评分（权重乘以信号值）"
        }