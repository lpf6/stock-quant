"""指标插件包"""

# 导入所有指标插件
from .moving_average import MovingAverageIndicator
from .rsi_indicator import RSIIndicator
from .bollinger_bands import BollingerBandsIndicator

__all__ = [
    "MovingAverageIndicator",
    "RSIIndicator",
    "BollingerBandsIndicator",
]