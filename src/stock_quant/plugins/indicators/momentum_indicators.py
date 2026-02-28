"""动量指标插件"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from ..base import IndicatorPlugin


class MomentumIndicator(IndicatorPlugin):
    """动量指标"""
    
    def __init__(self):
        super().__init__()
        self.name = "Momentum Indicator"
        self.description = "计算价格动量、速率和加速度指标"
        self.author = "Stock Quant Team"
        self.version = "1.0.0"
        self.config = {
            "periods": [5, 10, 20, 60],  # 动量计算周期
            "signal_name": "momentum",
            "weight": 0.15
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> None:
        if config:
            self.config.update(config)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:  
        result_df = df.copy()
        
        # 计算价格动量（当前价格与n天前价格的比率）
        for period in self.config['periods']:
            if len(df) >= period:
                col_name = f'Momentum_{period}'
                result_df[col_name] = df['close'] / df['close'].shift(period) - 1
                
                # 标准化动量值
                result_df[f'{col_name}_norm'] = self._normalize_momentum(
                    result_df[col_name], 
                    period
                )
        
        # 计算动量变化率（加速度）
        if f'Momentum_{self.config["periods"][0]}' in result_df.columns:
            mom_col = f'Momentum_{self.config["periods"][0]}'
            result_df['Momentum_Rate'] = result_df[mom_col].diff()
            result_df['Momentum_Acceleration'] = result_df['Momentum_Rate'].diff()
        
        # 计算综合动量分数
        result_df['Momentum_Score'] = self._calculate_momentum_score(result_df)
        
        return result_df
    
    def _normalize_momentum(self, momentum_series: pd.Series, period: int) -> pd.Series:
        """标准化动量值"""
        if len(momentum_series.dropna()) < 2:
            return momentum_series
        
        # 使用滚动窗口进行标准化
        return momentum_series.rolling(
            window=min(60, len(momentum_series)), 
            min_periods=10
        ).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
        )
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """计算综合动量分数"""
        scores = []
        
        for period in self.config['periods']:
            norm_col = f'Momentum_{period}_norm'
            if norm_col in df.columns:
                # 短期动量权重更高
                weight = 1.0 / period
                scores.append(df[norm_col] * weight)
        
        if scores:
            combined_score = pd.concat(scores, axis=1).mean(axis=1)
            # 缩放分数到0-1范围
            return (combined_score - combined_score.min()) / (combined_score.max() - combined_score.min() + 1e-8)
        else:
            return pd.Series(0, index=df.index)
    
    def get_indicator_names(self) -> List[str]:
        """获取指标名称"""
        names = []
        for period in self.config['periods']:
            names.extend([
                f'Momentum_{period}',
                f'Momentum_{period}_norm'
            ])
        
        names.extend([
            'Momentum_Rate',
            'Momentum_Acceleration',
            'Momentum_Score'
        ])
        
        return names


class VolumeIndicator(IndicatorPlugin):
    """成交量指标"""
    
    def __init__(self):
        super().__init__()
        self.name = "Volume Indicator"
        self.description = "计算成交量相关指标，包括成交量均线、换手率等"
        self.author = "Stock Quant Team"
        self.version = "1.0.0"
        self.config = {
            "volume_periods": [5, 10, 20, 60],  # 成交量均线周期
            "signal_name": "volume",
            "weight": 0.12
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> None:
        if config:
            self.config.update(config)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        
        if 'volume' not in df.columns:
            return result_df
        
        volume = df['volume']
        
        # 计算成交量均线
        for period in self.config['volume_periods']:
            col_name = f'Volume_MA_{period}'
            result_df[col_name] = volume.rolling(window=period).mean()
            
            # 计算成交量比率
            result_df[f'Volume_Ratio_{period}'] = volume / result_df[col_name]
        
        # 计算成交量标准差
        result_df['Volume_Std_20'] = volume.rolling(window=20).std()
        
        # 计算成交量异动（超过2个标准差）
        result_df['Volume_Spike'] = (
            (volume - result_df['Volume_MA_20']) > 
            (2 * result_df['Volume_Std_20'])
        ).astype(int)
        
        # 计算成交量趋势
        result_df['Volume_Trend'] = self._calculate_volume_trend(volume)
        
        # 计算综合成交量分数
        result_df['Volume_Score'] = self._calculate_volume_score(result_df)
        
        # 计算价量关系
        if 'close' in df.columns:
            price_change = df['close'].pct_change()
            volume_change = volume.pct_change()
            result_df['Price_Volume_Correlation'] = (
                price_change.rolling(20).corr(volume_change)
            )
            
            # 价量背离检测
            result_df['Price_Volume_Divergence'] = self._detect_divergence(
                price_change, 
                volume_change
            )
        
        return result_df
    
    def _calculate_volume_trend(self, volume: pd.Series) -> pd.Series:
        """计算成交量趋势"""
        if len(volume) < 20:
            return pd.Series(0, index=volume.index)
        
        # 计算短期和长期均线的位置关系
        ma_short = volume.rolling(5).mean()
        ma_long = volume.rolling(20).mean()
        
        # 趋势得分：短期均线上穿长期均线为正值
        trend_score = (ma_short - ma_long) / ma_long.where(ma_long != 0, 1)
        
        # 平滑处理
        return trend_score.rolling(5).mean()
    
    def _calculate_volume_score(self, df: pd.DataFrame) -> pd.Series:
        """计算综合成交量分数"""
        score_components = []
        
        # 1. 成交量比率分数（最近成交量活跃度）
        if 'Volume_Ratio_5' in df.columns:
            vol_ratio_5 = df['Volume_Ratio_5'].clip(0, 3)  # 限制在0-3倍
            score_components.append(vol_ratio_5 / 3)  # 归一化到0-1
        
        # 2. 成交量趋势分数
        if 'Volume_Trend' in df.columns:
            vol_trend = df['Volume_Trend'].clip(-1, 1)
            score_components.append((vol_trend + 1) / 2)  # 归一化到0-1
        
        # 3. 价量相关性分数（正值表示价量配合）
        if 'Price_Volume_Correlation' in df.columns:
            pv_corr = df['Price_Volume_Correlation'].clip(-1, 1)
            score_components.append((pv_corr + 1) / 2)  # 归一化到0-1
        
        if score_components:
            combined_score = pd.concat(score_components, axis=1).mean(axis=1)
            return combined_score
        else:
            return pd.Series(0, index=df.index)
    
    def _detect_divergence(self, price_change: pd.Series, 
                          volume_change: pd.Series) -> pd.Series:
        """检测价量背离"""
        divergence = pd.Series(0, index=price_change.index)
        
        for i in range(10, len(price_change)):
            # 检查最近10天的价量关系
            recent_price = price_change.iloc[i-10:i+1]
            recent_volume = volume_change.iloc[i-10:i+1]
            
            if len(recent_price.dropna()) > 5 and len(recent_volume.dropna()) > 5:
                # 计算相关性
                correlation = recent_price.corr(recent_volume)
                
                # 价量背离：价格上涨但成交量下降，或价格下跌但成交量上升
                if correlation < -0.5:
                    divergence.iloc[i] = 1  # 背离信号
                elif correlation > 0.5:
                    divergence.iloc[i] = -1  # 配合信号
        
        return divergence
    
    def get_indicator_names(self) -> List[str]:
        """获取指标名称"""
        names = []
        for period in self.config['volume_periods']:
            names.extend([
                f'Volume_MA_{period}',
                f'Volume_Ratio_{period}'
            ])
        
        names.extend([
            'Volume_Std_20',
            'Volume_Spike',
            'Volume_Trend',
            'Volume_Score',
            'Price_Volume_Correlation',
            'Price_Volume_Divergence'
        ])
        
        return names


class TrendStrengthIndicator(IndicatorPlugin):
    """趋势强度指标"""
    
    def __init__(self):
        super().__init__()
        self.name = "Trend Strength Indicator"
        self.description = "计算趋势强度、方向和持续性指标"
        self.author = "Stock Quant Team"
        self.version = "1.0.0"
        self.config = {
            "trend_periods": [10, 20, 60],  # 趋势分析周期
            "signal_name": "trend_strength",
            "weight": 0.13
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> None:
        if config:
            self.config.update(config)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        
        if 'close' not in df.columns:
            return result_df
        
        price = df['close']
        
        # 计算不同周期的趋势线
        for period in self.config['trend_periods']:
            # 线性回归斜率（趋势方向）
            result_df[f'Trend_Slope_{period}'] = self._calculate_slope(price, period)
            
            # 趋势强度（R平方）
            result_df[f'Trend_R2_{period}'] = self._calculate_r_squared(price, period)
            
            # 趋势持续性（自相关性）
            result_df[f'Trend_Persistence_{period}'] = self._calculate_persistence(price, period)
        
        # 计算多周期综合趋势分数
        result_df['Trend_Score'] = self._calculate_trend_score(result_df)
        
        # 计算趋势状态
        result_df['Trend_State'] = self._determine_trend_state(result_df)
        
        # 计算趋势转折点
        result_df['Trend_Change_Point'] = self._detect_trend_change(price)
        
        # 计算ADX（平均趋向指数）- 简化版
        result_df['ADX_14'] = self._calculate_adx(df)
        
        return result_df
    
    def _calculate_slope(self, price: pd.Series, period: int) -> pd.Series:
        """计算线性回归斜率"""
        if len(price) < period:
            return pd.Series(np.nan, index=price.index)
        
        slopes = []
        for i in range(len(price)):
            if i < period - 1:
                slopes.append(np.nan)
            else:
                x = np.arange(period)
                y = price.iloc[i-period+1:i+1].values
                if len(y) == period:
                    slope = np.polyfit(x, y, 1)[0]
                    # 标准化斜率（相对于价格）
                    normalized_slope = slope / price.iloc[i] if price.iloc[i] != 0 else 0
                    slopes.append(normalized_slope)
                else:
                    slopes.append(np.nan)
        
        return pd.Series(slopes, index=price.index)
    
    def _calculate_r_squared(self, price: pd.Series, period: int) -> pd.Series:
        """计算R平方（趋势强度）"""
        if len(price) < period:
            return pd.Series(np.nan, index=price.index)
        
        r2_values = []
        for i in range(len(price)):
            if i < period - 1:
                r2_values.append(np.nan)
            else:
                x = np.arange(period)
                y = price.iloc[i-period+1:i+1].values
                if len(y) == period:
                    slope, intercept = np.polyfit(x, y, 1)
                    y_pred = slope * x + intercept
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))
                    r2_values.append(r2)
                else:
                    r2_values.append(np.nan)
        
        return pd.Series(r2_values, index=price.index)
    
    def _calculate_persistence(self, price: pd.Series, period: int) -> pd.Series:
        """计算趋势持续性（自相关性）"""
        if len(price) < period:
            return pd.Series(np.nan, index=price.index)
        
        returns = price.pct_change()
        persistence = returns.rolling(period).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 2 else np.nan
        )
        
        return persistence
    
    def _calculate_trend_score(self, df: pd.DataFrame) -> pd.Series:
        """计算综合趋势分数"""
        score_components = []
        
        for period in self.config['trend_periods']:
            # 趋势强度（R2）
            r2_col = f'Trend_R2_{period}'
            if r2_col in df.columns:
                r2_score = df[r2_col].fillna(0).clip(0, 1)
                # 短期趋势权重更高
                weight = 1.5 / period
                score_components.append(r2_score * weight)
            
            # 趋势方向（正斜率加分）
            slope_col = f'Trend_Slope_{period}'
            if slope_col in df.columns:
                slope_score = np.tanh(df[slope_col].fillna(0) * 100)  # 压缩到-1到1
                slope_score = (slope_score + 1) / 2  # 转换到0-1
                weight = 1.0 / period
                score_components.append(slope_score * weight)
        
        if score_components:
            combined_score = pd.concat(score_components, axis=1).mean(axis=1)
            return combined_score
        else:
            return pd.Series(0, index=df.index)
    
    def _determine_trend_state(self, df: pd.DataFrame) -> pd.Series:
        """确定趋势状态：-1下跌，0震荡，1上涨"""
        trend_state = pd.Series(0, index=df.index)
        
        for i in range(len(df)):
            # 检查短期趋势
            if f'Trend_Slope_10' in df.columns and not pd.isna(df[f'Trend_Slope_10'].iloc[i]):
                slope_10 = df[f'Trend_Slope_10'].iloc[i]
                r2_10 = df.get(f'Trend_R2_10', pd.Series(0, index=df.index)).iloc[i]
                
                # 只有当趋势强度足够时才确定方向
                if r2_10 > 0.3:  # R2大于0.3表示有较强趋势
                    if slope_10 > 0.001:  # 斜率大于0.1%
                        trend_state.iloc[i] = 1  # 上涨趋势
                    elif slope_10 < -0.001:  # 斜率小于-0.1%
                        trend_state.iloc[i] = -1  # 下跌趋势
        
        return trend_state
    
    def _detect_trend_change(self, price: pd.Series) -> pd.Series:
        """检测趋势转折点"""
        if len(price) < 30:
            return pd.Series(0, index=price.index)
        
        change_points = pd.Series(0, index=price.index)
        
        # 计算不同周期的移动平均线
        ma_short = price.rolling(10).mean()
        ma_medium = price.rolling(20).mean()
        ma_long = price.rolling(60).mean()
        
        # 检测金叉死叉
        for i in range(1, len(price)):
            # 短期上穿中期（金叉）
            if (ma_short.iloc[i-1] <= ma_medium.iloc[i-1] and 
                ma_short.iloc[i] > ma_medium.iloc[i]):
                change_points.iloc[i] = 1  # 向上转折
            
            # 短期下穿中期（死叉）
            elif (ma_short.iloc[i-1] >= ma_medium.iloc[i-1] and 
                  ma_short.iloc[i] < ma_medium.iloc[i]):
                change_points.iloc[i] = -1  # 向下转折
        
        return change_points
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """计算ADX（简化版）"""
        if 'high' not in df.columns or 'low' not in df.columns:
            return pd.Series(0, index=df.index)
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 计算真实波幅（TR）
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算+DM和-DM
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # 计算平滑值（14周期）
        period = 14
        atr = tr.rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
        
        # 计算ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(period).mean()
        
        return adx.fillna(0)
    
    def get_indicator_names(self) -> List[str]:
        """获取指标名称"""
        names = []
        for period in self.config['trend_periods']:
            names.extend([
                f'Trend_Slope_{period}',
                f'Trend_R2_{period}',
                f'Trend_Persistence_{period}'
            ])
        
        names.extend([
            'Trend_Score',
            'Trend_State',
            'Trend_Change_Point',
            'ADX_14'
        ])
        
        return names