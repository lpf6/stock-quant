"""数据源适配器基类"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class BaseDataSourceAdapter(ABC):
    """数据源适配器基类，定义统一的数据源接口"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """初始化数据源（加载依赖、认证等）"""
        pass
    
    @abstractmethod
    def fetch_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "daily"
    ) -> Optional[pd.DataFrame]:
        """获取单只股票数据"""
        pass
    
    @abstractmethod
    def fetch_batch_data(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "daily"
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """批量获取股票数据"""
        pass
    
    @abstractmethod
    def get_index_constituents(self, index_code: str = "000852") -> Optional[pd.DataFrame]:
        """获取指数成分股"""
        pass
    
    def is_initialized(self) -> bool:
        """检查数据源是否已初始化"""
        return self._initialized
