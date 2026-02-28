"""多周期管理器"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .config_manager import PeriodConfig


class TimeFrame(Enum):
    """时间框架枚举"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class PeriodManager:
    """多周期管理器，支持不同时间框架的数据处理"""
    
    def __init__(self, config: Optional[PeriodConfig] = None):
        self.config = config or PeriodConfig()
        self.time_frames = [TimeFrame.DAILY, TimeFrame.WEEKLY, TimeFrame.MONTHLY]
        self.data_cache: Dict[str, Dict[TimeFrame, pd.DataFrame]] = {}
    
    def convert_timeframe(
        self, 
        df: pd.DataFrame, 
        source_tf: TimeFrame, 
        target_tf: TimeFrame
    ) -> pd.DataFrame:
        """转换时间框架"""
        if source_tf == target_tf:
            return df.copy()
        
        if source_tf == TimeFrame.DAILY:
            if target_tf == TimeFrame.WEEKLY:
                return self._daily_to_weekly(df)
            elif target_tf == TimeFrame.MONTHLY:
                return self._daily_to_monthly(df)
        elif source_tf == TimeFrame.WEEKLY:
            if target_tf == TimeFrame.MONTHLY:
                return self._weekly_to_monthly(df)
            elif target_tf == TimeFrame.DAILY:
                # 无法从周线精确转回日线
                raise ValueError("无法从周线转换为日线")
        elif source_tf == TimeFrame.MONTHLY:
            raise ValueError("无法从月线转换为更小时间框架")
        
        raise ValueError(f"不支持的时间框架转换: {source_tf} -> {target_tf}")
    
    def _daily_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """日线转周线"""
        if df.empty:
            return df
        
        # 按周重采样
        weekly_df = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        })
        
        # 移除空值
        weekly_df = weekly_df.dropna()
        
        return weekly_df
    
    def _daily_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """日线转月线"""
        if df.empty:
            return df
        
        # 按月重采样
        monthly_df = df.resample('M').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        })
        
        # 移除空值
        monthly_df = monthly_df.dropna()
        
        return monthly_df
    
    def _weekly_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """周线转月线"""
        if df.empty:
            return df
        
        # 需要先将周线转为日线索引（近似）
        # 这里使用简单方法：每月取最后一周
        monthly_df = df.resample('M').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        })
        
        monthly_df = monthly_df.dropna()
        
        return monthly_df
    
    def align_data_frames(
        self, 
        dfs: Dict[TimeFrame, pd.DataFrame]
    ) -> Dict[TimeFrame, pd.DataFrame]:
        """对齐不同时间框架的数据"""
        if not dfs:
            return {}
        
        # 找到最小时间框架
        min_tf = min(dfs.keys(), key=lambda x: self._get_timeframe_weight(x))
        min_df = dfs[min_tf]
        
        aligned_dfs = {min_tf: min_df.copy()}
        
        # 对齐其他时间框架
        for tf, df in dfs.items():
            if tf == min_tf:
                continue
            
            try:
                aligned_df = self.convert_timeframe(min_df, min_tf, tf)
                
                # 确保时间索引匹配
                common_dates = aligned_df.index.intersection(df.index)
                if len(common_dates) > 0:
                    aligned_df = aligned_df.loc[common_dates]
                
                aligned_dfs[tf] = aligned_df
            except ValueError as e:
                print(f"对齐时间框架 {tf} 失败: {e}")
                continue
        
        return aligned_dfs
    
    def _get_timeframe_weight(self, timeframe: TimeFrame) -> int:
        """获取时间框架权重（用于排序）"""
        weights = {
            TimeFrame.DAILY: 1,
            TimeFrame.WEEKLY: 7,
            TimeFrame.MONTHLY: 30
        }
        return weights.get(timeframe, 999)
    
    def get_period_dates(
        self, 
        period_name: str, 
        end_date: Optional[datetime] = None
    ) -> Tuple[datetime, datetime]:
        """获取指定周期的起止日期"""
        if end_date is None:
            end_date = datetime.now()
        
        period_config = self.config.get_period_config(period_name)
        if not period_config:
            raise ValueError(f"未知的周期配置: {period_name}")
        
        months = period_config["months"]
        start_date = end_date - timedelta(days=months * 30)  # 近似
        
        return start_date, end_date
    
    def cache_data(self, symbol: str, timeframe: TimeFrame, data: pd.DataFrame) -> None:
        """缓存数据"""
        if symbol not in self.data_cache:
            self.data_cache[symbol] = {}
        self.data_cache[symbol][timeframe] = data.copy()
    
    def get_cached_data(self, symbol: str, timeframe: TimeFrame) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        if symbol in self.data_cache and timeframe in self.data_cache[symbol]:
            return self.data_cache[symbol][timeframe].copy()
        return None
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """清空缓存"""
        if symbol:
            if symbol in self.data_cache:
                del self.data_cache[symbol]
        else:
            self.data_cache.clear()
    
    def get_supported_timeframes(self) -> List[TimeFrame]:
        """获取支持的时间框架"""
        return self.time_frames.copy()
    
    def get_recommended_periods(self) -> List[str]:
        """获取推荐的周期配置"""
        return self.config.get_recommended_periods()