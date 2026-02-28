"""数据获取器单元测试"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.stock_quant.core.data_fetcher import DataFetcher
from src.stock_quant.utils.exceptions import DataFetchError


class TestDataFetcher:
    """数据获取器测试类"""
    
    def setup_method(self):
        """测试设置"""
        self.config = {
            "data_source": "akshare",
            "max_retries": 3,
            "timeout": 30
        }
        self.fetcher = DataFetcher(self.config)
    
    @patch('src.stock_quant.core.data_fetcher.ak')
    def test_fetch_stock_data_success(self, mock_ak):
        """测试成功获取股票数据"""
        # 模拟AKShare返回数据
        mock_df = pd.DataFrame({
            '日期': ['2023-01-01', '2023-01-02'],
            '开盘': [10.0, 10.5],
            '收盘': [10.2, 10.8],
            '最高': [10.3, 11.0],
            '最低': [9.8, 10.4],
            '成交量': [1000000, 1200000],
            '成交额': [10000000, 12000000]
        })
        mock_ak.stock_zh_a_hist.return_value = mock_df
        
        # 调用方法
        result = self.fetcher.fetch_stock_data("000001")
        
        # 验证结果
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'open' in result.columns
        assert 'close' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'volume' in result.columns
        assert 'amount' in result.columns
    
    @patch('src.stock_quant.core.data_fetcher.ak')
    def test_fetch_stock_data_empty(self, mock_ak):
        """测试获取空数据"""
        mock_ak.stock_zh_a_hist.return_value = pd.DataFrame()
        
        result = self.fetcher.fetch_stock_data("000001")
        
        assert result is None
    
    @patch('src.stock_quant.core.data_fetcher.ak')
    def test_fetch_stock_data_exception(self, mock_ak):
        """测试获取数据异常"""
        mock_ak.stock_zh_a_hist.side_effect = Exception("API错误")
        
        result = self.fetcher.fetch_stock_data("000001")
        
        assert result is None
    
    def test_fetch_batch_data(self):
        """测试批量获取数据"""
        symbols = ["000001", "000002", "000003"]
        
        # 模拟fetch_stock_data方法
        mock_data = pd.DataFrame({
            'open': [10.0, 10.5],
            'close': [10.2, 10.8],
            'high': [10.3, 11.0],
            'low': [9.8, 10.4],
            'volume': [1000000, 1200000],
            'amount': [10000000, 12000000]
        })
        
        with patch.object(self.fetcher, 'fetch_stock_data', return_value=mock_data):
            results = self.fetcher.fetch_batch_data(symbols)
            
            assert isinstance(results, dict)
            assert len(results) == 3
            for symbol in symbols:
                assert symbol in results
                assert results[symbol] is not None
    
    def test_fetch_batch_data_partial_failure(self):
        """测试批量获取数据部分失败"""
        symbols = ["000001", "000002", "000003"]
        
        # 模拟部分成功，部分失败
        def mock_fetch_stock_data(symbol):
            if symbol == "000002":
                return None
            return pd.DataFrame({
                'open': [10.0],
                'close': [10.2],
                'high': [10.3],
                'low': [9.8],
                'volume': [1000000],
                'amount': [10000000]
            })
        
        with patch.object(self.fetcher, 'fetch_stock_data', side_effect=mock_fetch_stock_data):
            results = self.fetcher.fetch_batch_data(symbols)
            
            assert len(results) == 3
            assert results["000001"] is not None
            assert results["000002"] is None
            assert results["000003"] is not None
    
    @patch('src.stock_quant.core.data_fetcher.ak')
    def test_get_index_constituents(self, mock_ak):
        """测试获取指数成分股"""
        mock_constituents = pd.DataFrame({
            '成分券代码': ['000001', '000002', '000003'],
            '成分券名称': ['股票A', '股票B', '股票C'],
            '交易所': ['SZ', 'SZ', 'SH']
        })
        mock_ak.index_stock_cons_csindex.return_value = mock_constituents
        
        result = self.fetcher.get_index_constituents("000852")
        
        assert result is not None
        assert len(result) == 3
        assert '成分券代码' in result.columns
    
    def test_invalid_data_source(self):
        """测试无效数据源"""
        invalid_config = {"data_source": "invalid_source"}
        fetcher = DataFetcher(invalid_config)
        
        # 应该抛出异常
        with pytest.raises(ValueError):
            fetcher.initialize()
    
    def test_config_update(self):
        """测试配置更新"""
        new_config = {"max_retries": 5, "timeout": 60}
        self.fetcher.update_config(new_config)
        
        assert self.fetcher.config["max_retries"] == 5
        assert self.fetcher.config["timeout"] == 60