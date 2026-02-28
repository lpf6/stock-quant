"""数据获取模块"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from .base import BaseDataFetcher

warnings.filterwarnings('ignore')


class DataFetcher(BaseDataFetcher):
    """数据获取器，支持单只和批量股票数据获取"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.data_source = self.config.get("data_source", "akshare")
        self.max_retries = self.config.get("max_retries", 3)
        self.timeout = self.config.get("timeout", 30)
        
    def initialize(self) -> None:
        """初始化数据源"""
        if self.data_source == "akshare":
            try:
                import akshare as ak
                self.ak = ak
                print("AKShare数据源初始化成功")
            except ImportError:
                print("警告：AKShare未安装，部分功能可能受限")
                self.ak = None
        else:
            raise ValueError(f"不支持的数据源: {self.data_source}")
    
    def fetch_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "daily"
    ) -> Optional[pd.DataFrame]:
        """获取单只股票数据"""
        self.initialize()
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if self.data_source == "akshare" and self.ak:
            try:
                stock_df = self.ak.stock_zh_a_hist(
                    symbol=symbol, 
                    period=period,
                    start_date=start_date, 
                    end_date=end_date, 
                    adjust="qfq"
                )
                
                if stock_df.empty:
                    # 尝试另一种格式
                    if symbol.startswith('0') or symbol.startswith('3'):
                        symbol_code = f"sz{symbol}"
                    elif symbol.startswith('6'):
                        symbol_code = f"sh{symbol}"
                    else:
                        return None
                    
                    stock_df = self.ak.stock_zh_a_hist(
                        symbol=symbol_code, 
                        period=period,
                        start_date=start_date, 
                        end_date=end_date, 
                        adjust="qfq"
                    )
                
                if stock_df.empty:
                    return None
                
                # 处理数据格式
                stock_df['date'] = pd.to_datetime(stock_df['日期'])
                stock_df.set_index('date', inplace=True)
                stock_df.rename(columns={
                    '开盘': 'open', 
                    '收盘': 'close', 
                    '最高': 'high', 
                    '最低': 'low', 
                    '成交量': 'volume', 
                    '成交额': 'amount'
                }, inplace=True)
                
                return stock_df[['open', 'close', 'high', 'low', 'volume', 'amount']]
                
            except Exception as e:
                print(f"获取股票 {symbol} 数据失败: {e}")
                return None
        else:
            print(f"不支持的数据源或数据源未初始化: {self.data_source}")
            return None
    
    def fetch_batch_data(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "daily"
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """批量获取股票数据"""
        self.initialize()
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        results = {}
        total_symbols = len(symbols)
        
        print(f"开始批量获取 {total_symbols} 只股票数据...")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                data = self.fetch_stock_data(symbol, start_date, end_date, period)
                results[symbol] = data
                
                if i % 10 == 0 or i == total_symbols:
                    print(f"进度: {i}/{total_symbols} ({i/total_symbols*100:.1f}%)")
                    
            except Exception as e:
                print(f"获取股票 {symbol} 数据失败: {e}")
                results[symbol] = None
        
        # 统计结果
        success_count = sum(1 for data in results.values() if data is not None)
        print(f"批量获取完成: 成功 {success_count}/{total_symbols} 只股票")
        
        return results
    
    def get_index_constituents(self, index_code: str = "000852") -> Optional[pd.DataFrame]:
        """获取指数成分股"""
        self.initialize()
        
        if self.data_source == "akshare" and self.ak:
            try:
                constituents_df = self.ak.index_stock_cons_csindex(symbol=index_code)
                print(f"成功获取 {len(constituents_df)} 只指数成分股")
                return constituents_df
            except Exception as e:
                print(f"获取指数成分股失败: {e}")
                return None
        return None