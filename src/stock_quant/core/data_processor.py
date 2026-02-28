"""数据处理器"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from .base import BaseDataProcessor


class DataProcessor(BaseDataProcessor):
    """数据处理模块 - 数据清洗、转换、标准化"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 默认配置
        self.default_config = {
            "clean_missing": True,
            "handle_outliers": True,
            "normalize_method": "zscore",  # zscore, minmax, robust
            "resample_freq": "D",  # 重采样频率
            "fill_method": "ffill",  # 填充方法
        }
        
        if config:
            self.config.update(config)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        if df is None or df.empty:
            return df
        
        df_clean = df.copy()
        
        # 1. 处理缺失值
        if self.config.get("clean_missing", True):
            # 前向填充
            if self.config.get("fill_method") == "ffill":
                df_clean = df_clean.fillna(method="ffill").fillna(method="bfill")
            # 线性插值
            elif self.config.get("fill_method") == "interpolate":
                df_clean = df_clean.interpolate(method="linear")
            # 删除包含缺失值的行
            else:
                df_clean = df_clean.dropna()
        
        # 2. 处理异常值
        if self.config.get("handle_outliers", True):
            df_clean = self._handle_outliers(df_clean)
        
        # 3. 数据标准化
        normalize_method = self.config.get("normalize_method")
        if normalize_method:
            df_clean = self._normalize_data(df_clean, normalize_method)
        
        return df_clean
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        df_out = df.copy()
        
        for column in df.columns:
            if column in ["open", "high", "low", "close", "volume", "amount"]:
                # 使用IQR方法检测异常值
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 将异常值缩放到边界内
                if IQR > 0:  # 避免除以0
                    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                    if outliers_mask.any():
                        # 对异常值进行缩尾处理
                        df_out.loc[outliers_mask & (df[column] < lower_bound), column] = lower_bound
                        df_out.loc[outliers_mask & (df[column] > upper_bound), column] = upper_bound
        
        return df_out
    
    def _normalize_data(self, df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """数据标准化"""
        if method == "zscore":
            return (df - df.mean()) / (df.std() + 1e-8)
        elif method == "minmax":
            return (df - df.min()) / (df.max() - df.min() + 1e-8)
        elif method == "robust":
            return (df - df.median()) / (df.quantile(0.75) - df.quantile(0.25) + 1e-8)
        else:
            return df
    
    def resample_data(self, df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
        """重采样数据"""
        if df is None or df.empty:
            return df
        
        # 确保索引是DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                return df
        
        # 定义重采样规则
        resample_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum"
        }
        
        # 只应用存在的列
        available_rules = {k: v for k, v in resample_rules.items() if k in df.columns}
        
        try:
            resampled = df.resample(freq).agg(available_rules)
            return resampled.dropna()
        except Exception as e:
            print(f"重采样失败: {e}")
            return df
    
    def calculate_returns(self, df: pd.DataFrame, price_col: str = "close") -> pd.Series:
        """计算收益率"""
        if df is None or price_col not in df.columns:
            return pd.Series()
        
        prices = df[price_col]
        returns = prices.pct_change().dropna()
        return returns
    
    def calculate_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """计算波动率"""
        if returns is None or len(returns) < window:
            return pd.Series()
        
        # 滚动波动率（年化）
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    def calculate_correlation(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """计算相关性矩阵"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        if columns is None:
            columns = df.columns.tolist()
        
        # 只选择数值列
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) < 2:
            return pd.DataFrame()
        
        return df[numeric_cols].corr()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建特征工程"""
        if df is None or df.empty:
            return df
        
        df_features = df.copy()
        
        # 价格特征
        if "close" in df.columns:
            # 对数收益率
            df_features["log_return"] = np.log(df["close"] / df["close"].shift(1))
            
            # 价格变化
            df_features["price_change"] = df["close"].diff()
            
            # 价格相对位置（最高/最低）
            if "high" in df.columns and "low" in df.columns:
                df_features["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
        
        # 成交量特征
        if "volume" in df.columns:
            # 成交量变化率
            df_features["volume_change"] = df["volume"].pct_change()
            
            # 成交量与价格变化的关系
            if "close" in df.columns:
                df_features["price_volume_ratio"] = df["close"].pct_change() / (df["volume"].pct_change() + 1e-8)
        
        # 时间特征
        if isinstance(df.index, pd.DatetimeIndex):
            df_features["day_of_week"] = df.index.dayofweek
            df_features["month"] = df.index.month
            df_features["quarter"] = df.index.quarter
        
        return df_features
    
    def split_train_test(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        """分割训练集和测试集"""
        if df is None or len(df) < 10:
            return {"train": df, "test": pd.DataFrame()}
        
        split_idx = int(len(df) * (1 - test_size))
        
        return {
            "train": df.iloc[:split_idx],
            "test": df.iloc[split_idx:]
        }
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """数据验证"""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "statistics": {}
        }
        
        if df is None or df.empty:
            validation_results["is_valid"] = False
            validation_results["issues"].append("数据为空")
            return validation_results
        
        # 检查缺失值
        missing_counts = df.isnull().sum()
        missing_total = missing_counts.sum()
        
        if missing_total > 0:
            validation_results["issues"].append(f"存在缺失值: {missing_total}个")
            validation_results["statistics"]["missing_counts"] = missing_counts.to_dict()
        
        # 检查重复值
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results["issues"].append(f"存在重复值: {duplicate_count}个")
        
        # 检查数据类型
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                validation_results["issues"].append(f"列 {column} 非数值类型")
        
        # 检查数据范围
        if "close" in df.columns:
            if (df["close"] <= 0).any():
                validation_results["issues"].append("收盘价包含非正值")
        
        if "volume" in df.columns:
            if (df["volume"] < 0).any():
                validation_results["issues"].append("成交量包含负值")
        
        # 统计信息
        if validation_results["is_valid"]:
            validation_results["statistics"].update({
                "row_count": len(df),
                "column_count": len(df.columns),
                "date_range": {
                    "start": df.index.min().strftime("%Y-%m-%d") if isinstance(df.index, pd.DatetimeIndex) else "N/A",
                    "end": df.index.max().strftime("%Y-%m-%d") if isinstance(df.index, pd.DatetimeIndex) else "N/A"
                }
            })
        
        return validation_results