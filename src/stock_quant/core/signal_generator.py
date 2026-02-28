"""信号生成器"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd

from .base import BaseSignalGenerator
from ..plugins.manager import PluginManager
from ..plugins.base import StrategyPlugin


class SignalGenerator(BaseSignalGenerator):
    """信号生成器，支持插件化策略"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.plugin_manager = PluginManager()
        self.loaded_strategies: Dict[str, StrategyPlugin] = {}
        self.default_strategies = ["MACrossStrategy", "RSIStrategy", "MACDStrategy"]
        self.strategy_weights = self.config.get("strategy_weights", {})
    
    def initialize(self) -> None:
        """初始化策略插件"""
        # 加载默认策略插件
        for strategy_name in self.default_strategies:
            try:
                strategy = self.plugin_manager.load_plugin(strategy_name)
                self.loaded_strategies[strategy_name] = strategy
            except Exception as e:
                print(f"加载策略插件 {strategy_name} 失败: {e}")
        
        # 加载用户配置的策略
        custom_strategies = self.config.get("strategies", [])
        for strategy_config in custom_strategies:
            self.load_strategy(strategy_config)
    
    def load_strategy(self, strategy_config: Dict[str, Any]) -> bool:
        """加载策略插件"""
        strategy_name = strategy_config.get("name")
        if not strategy_name:
            return False
        
        try:
            config = strategy_config.get("config", {})
            strategy = self.plugin_manager.load_plugin(strategy_name, config)
            self.loaded_strategies[strategy_name] = strategy
            return True
        except Exception as e:
            print(f"加载策略插件 {strategy_name} 失败: {e}")
            return False
    
    def unload_strategy(self, strategy_name: str) -> bool:
        """卸载策略插件"""
        if strategy_name in self.loaded_strategies:
            self.plugin_manager.unload_plugin(strategy_name)
            del self.loaded_strategies[strategy_name]
            return True
        return False
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """生成交易信号"""
        self.initialize()
        
        if df is None or df.empty:
            return {}
        
        all_signals = {}
        total_score = 0.0
        
        # 执行所有策略
        for strategy_name, strategy in self.loaded_strategies.items():
            try:
                signals = strategy.calculate_signals(df)
                
                # 合并信号
                for key, value in signals.items():
                    if key != "score":  # 单独处理分数
                        all_signals[f"{strategy_name}.{key}"] = value
                
                # 累加分数（使用策略权重）
                strategy_weight = self.strategy_weights.get(strategy_name, 1.0)
                strategy_score = signals.get("score", 0.0) * strategy_weight
                total_score += strategy_score
                
            except Exception as e:
                print(f"执行策略 {strategy_name} 失败: {e}")
        
        # 添加总分数
        all_signals["total_score"] = total_score
        
        # 趋势确认
        if len(df) >= 20:
            trend_up = self._check_trend(df)
            volume_spike = self._check_volume(df)
            all_signals["trend_up"] = 1 if trend_up else 0
            all_signals["volume_spike"] = 1 if volume_spike else 0
        
        return all_signals
    
    def _check_trend(self, df: pd.DataFrame) -> bool:
        """检查趋势"""
        latest = df.iloc[-1]
        ma_columns = [col for col in df.columns if col.startswith("MA")]
        
        if len(ma_columns) >= 3:
            # 按数字排序MA列
            ma_columns_sorted = sorted(ma_columns, key=lambda x: int(x[2:]) if x[2:].isdigit() else 0)
            
            if len(ma_columns_sorted) >= 3:
                ma5_col = ma_columns_sorted[0]
                ma10_col = ma_columns_sorted[1]
                ma20_col = ma_columns_sorted[2]
                
                if all(pd.notnull(latest[col]) for col in [ma5_col, ma10_col, ma20_col]):
                    return (latest[ma5_col] > latest[ma10_col] > latest[ma20_col])
        
        return False
    
    def _check_volume(self, df: pd.DataFrame) -> bool:
        """检查成交量"""
        latest = df.iloc[-1]
        
        if 'volume' not in df.columns or pd.isnull(latest['volume']):
            return False
        
        if len(df) >= 20:
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            if pd.notnull(volume_ma) and volume_ma > 0:
                return latest['volume'] > volume_ma * 1.3
        
        return False
    
    def get_signal_names(self) -> List[str]:
        """获取信号名称列表"""
        signal_names = []
        for strategy_name, strategy in self.loaded_strategies.items():
            signal_descriptions = strategy.get_signal_descriptions()
            for key in signal_descriptions.keys():
                signal_names.append(f"{strategy_name}.{key}")
        
        signal_names.extend(["total_score", "trend_up", "volume_spike"])
        return signal_names
    
    def get_strategy_info(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """获取策略信息"""
        if strategy_name in self.loaded_strategies:
            strategy = self.loaded_strategies[strategy_name]
            return {
                "name": strategy.name,
                "description": strategy.description,
                "author": strategy.author,
                "version": strategy.version,
                "parameters": strategy.get_parameters()
            }
        return None
    
    def update_strategy_config(self, strategy_name: str, config: Dict[str, Any]) -> bool:
        """更新策略配置"""
        if strategy_name in self.loaded_strategies:
            strategy = self.loaded_strategies[strategy_name]
            strategy.initialize(config)
            return True
        return False