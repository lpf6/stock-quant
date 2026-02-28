"""数据管道集成测试"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from src.stock_quant.core.data_fetcher import DataFetcher
from src.stock_quant.core.indicator_calculator import IndicatorCalculator
from src.stock_quant.core.signal_generator import SignalGenerator
from src.stock_quant.utils.logger import setup_logger


class TestDataPipeline:
    """数据管道集成测试类"""
    
    def setup_method(self):
        """测试设置"""
        # 设置测试配置
        self.config = {
            "data": {
                "source": "akshare",
                "max_retries": 1,
                "timeout": 5
            },
            "indicators": {
                "default_indicators": ["MovingAverageIndicator", "RSIIndicator", "BollingerBandsIndicator"]
            },
            "strategies": {
                "default_strategies": ["MACrossStrategy", "RSIStrategy", "MACDStrategy"],
                "strategy_weights": {
                    "MACrossStrategy": 0.3,
                    "RSIStrategy": 0.25,
                    "MACDStrategy": 0.25
                }
            }
        }
        
        # 设置日志
        self.logger = setup_logger(level="WARNING")
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        # 创建60天的测试数据
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        
        # 生成价格数据（有趋势和波动）
        base_price = 10.0
        trend = 0.02  # 每天上涨2分
        volatility = 0.5  # 波动幅度
        
        prices = []
        for i in range(60):
            price = base_price + trend * i + volatility * (i % 10 - 5) / 5
            prices.append(price)
        
        # 创建DataFrame
        self.test_data = pd.DataFrame({
            'open': prices,
            'close': prices,
            'high': [p + 0.1 for p in prices],
            'low': [p - 0.1 for p in prices],
            'volume': [1000000 + i * 10000 for i in range(60)],
            'amount': [p * (1000000 + i * 10000) for i, p in enumerate(prices)]
        }, index=dates)
    
    @patch('src.stock_quant.core.data_fetcher.ak')
    def test_end_to_end_pipeline(self, mock_ak):
        """测试端到端数据管道"""
        # 模拟AKShare返回测试数据
        mock_hist_data = pd.DataFrame({
            '日期': self.test_data.index.strftime('%Y-%m-%d'),
            '开盘': self.test_data['open'],
            '收盘': self.test_data['close'],
            '最高': self.test_data['high'],
            '最低': self.test_data['low'],
            '成交量': self.test_data['volume'],
            '成交额': self.test_data['amount']
        })
        mock_ak.stock_zh_a_hist.return_value = mock_hist_data
        
        # 初始化组件
        data_fetcher = DataFetcher(self.config["data"])
        indicator_calculator = IndicatorCalculator(self.config["indicators"])
        signal_generator = SignalGenerator(self.config["strategies"])
        
        # 1. 数据获取
        stock_data = data_fetcher.fetch_stock_data("000001")
        
        assert stock_data is not None
        assert isinstance(stock_data, pd.DataFrame)
        assert len(stock_data) == 60
        assert 'open' in stock_data.columns
        assert 'close' in stock_data.columns
        
        # 2. 指标计算
        indicators_data = indicator_calculator.calculate_indicators(stock_data)
        
        assert indicators_data is not None
        assert isinstance(indicators_data, pd.DataFrame)
        assert len(indicators_data) == 60
        
        # 验证常见指标列
        expected_columns = [
            'open', 'close', 'high', 'low', 'volume', 'amount',
            'MA5', 'MA10', 'MA20', 'MA60',  # 移动平均线
            'RSI',                          # RSI
            'BB_middle', 'BB_upper', 'BB_lower', 'BB_width'  # 布林带
        ]
        
        for col in expected_columns:
            assert col in indicators_data.columns
        
        # 3. 信号生成
        signals = signal_generator.generate_signals(indicators_data)
        
        assert signals is not None
        assert isinstance(signals, dict)
        
        # 验证信号包含预期键
        expected_signals = [
            'MACrossStrategy.ma_cross',
            'MACrossStrategy.score',
            'RSIStrategy.rsi_oversold',
            'RSIStrategy.score',
            'MACDStrategy.macd_cross',
            'MACDStrategy.score',
            'total_score',
            'trend_up',
            'volume_spike'
        ]
        
        for signal in expected_signals:
            assert signal in signals
        
        # 验证信号值类型
        assert isinstance(signals['total_score'], (int, float))
        assert signals['trend_up'] in [0, 1]
        assert signals['volume_spike'] in [0, 1]
        
        # 验证总分在合理范围内
        assert 0 <= signals['total_score'] <= 1
    
    def test_indicator_calculator_with_specific_indicators(self):
        """测试指定指标计算"""
        indicator_calculator = IndicatorCalculator(self.config["indicators"])
        
        # 计算特定指标
        specific_indicators = ["MovingAverageIndicator", "RSIIndicator"]
        result = indicator_calculator.calculate_specific_indicators(
            self.test_data, 
            specific_indicators
        )
        
        assert 'MA5' in result.columns
        assert 'MA10' in result.columns
        assert 'MA20' in result.columns
        assert 'MA60' in result.columns
        assert 'RSI' in result.columns
        assert 'BB_middle' not in result.columns  # 布林带不应该被计算
    
    def test_signal_generator_with_custom_strategies(self):
        """测试自定义策略信号生成"""
        # 创建自定义策略配置
        custom_config = {
            "strategies": [
                {
                    "name": "MACrossStrategy",
                    "config": {
                        "short_period": 5,
                        "long_period": 20,
                        "weight": 0.5
                    }
                }
            ],
            "strategy_weights": {
                "MACrossStrategy": 0.5
            }
        }
        
        signal_generator = SignalGenerator(custom_config)
        
        # 先计算指标
        indicator_calculator = IndicatorCalculator(self.config["indicators"])
        indicators_data = indicator_calculator.calculate_indicators(self.test_data)
        
        # 生成信号
        signals = signal_generator.generate_signals(indicators_data)
        
        assert 'MACrossStrategy.ma_cross' in signals
        assert 'MACrossStrategy.score' in signals
        
        # 验证权重应用
        if signals['MACrossStrategy.ma_cross'] == 1:
            assert signals['MACrossStrategy.score'] == 0.5
        else:
            assert signals['MACrossStrategy.score'] == 0.0
    
    def test_batch_processing_simulation(self):
        """测试批量处理模拟"""
        symbols = ["000001", "000002", "000003"]
        
        # 模拟数据获取
        mock_data_fetcher = Mock(spec=DataFetcher)
        mock_data_fetcher.fetch_stock_data.return_value = self.test_data
        
        # 初始化其他组件
        indicator_calculator = IndicatorCalculator(self.config["indicators"])
        signal_generator = SignalGenerator(self.config["strategies"])
        
        # 批量处理
        results = []
        for symbol in symbols:
            # 获取数据
            stock_data = mock_data_fetcher.fetch_stock_data(symbol)
            
            if stock_data is not None:
                # 计算指标
                indicators_data = indicator_calculator.calculate_indicators(stock_data)
                
                # 生成信号
                signals = signal_generator.generate_signals(indicators_data)
                
                # 记录结果
                results.append({
                    "symbol": symbol,
                    **signals
                })
        
        # 验证结果
        assert len(results) == 3
        
        for result in results:
            assert "symbol" in result
            assert "total_score" in result
            assert 0 <= result["total_score"] <= 1
            
            # 验证股票代码
            assert result["symbol"] in symbols
    
    def test_error_handling_in_pipeline(self):
        """测试管道中的错误处理"""
        # 创建会失败的数据
        invalid_data = pd.DataFrame({
            'open': [10.0],
            'close': [10.2]
            # 缺少其他必需列
        })
        
        indicator_calculator = IndicatorCalculator(self.config["indicators"])
        signal_generator = SignalGenerator(self.config["strategies"])
        
        # 指标计算应该处理错误
        try:
            indicators_data = indicator_calculator.calculate_indicators(invalid_data)
            # 如果成功，可能只计算了部分指标
            if indicators_data is not None:
                # 信号生成应该处理错误
                try:
                    signals = signal_generator.generate_signals(indicators_data)
                    # 如果成功，结果可能为空或部分
                    assert signals is not None
                except Exception as e:
                    # 错误应该被捕获并处理
                    assert "错误" in str(e) or isinstance(e, (KeyError, ValueError))
        except Exception as e:
            # 错误应该被捕获并处理
            assert "错误" in str(e) or isinstance(e, (KeyError, ValueError))
    
    def test_data_consistency(self):
        """测试数据一致性"""
        indicator_calculator = IndicatorCalculator(self.config["indicators"])
        
        # 计算指标
        indicators_data = indicator_calculator.calculate_indicators(self.test_data)
        
        # 验证原始数据没有被修改
        assert 'MA5' not in self.test_data.columns
        assert 'RSI' not in self.test_data.columns
        
        # 验证指标数据是原始数据的副本
        assert indicators_data is not self.test_data
        
        # 验证原始列仍然存在
        for col in ['open', 'close', 'high', 'low', 'volume', 'amount']:
            assert col in indicators_data.columns
            # 值应该相同（浮点数容差）
            pd.testing.assert_series_equal(
                indicators_data[col], 
                self.test_data[col],
                check_names=False
            )
    
    def test_signal_consistency(self):
        """测试信号一致性"""
        # 多次运行应该得到相同的结果
        indicator_calculator = IndicatorCalculator(self.config["indicators"])
        signal_generator = SignalGenerator(self.config["strategies"])
        
        indicators_data = indicator_calculator.calculate_indicators(self.test_data)
        
        # 第一次运行
        signals1 = signal_generator.generate_signals(indicators_data)
        
        # 第二次运行
        signals2 = signal_generator.generate_signals(indicators_data)
        
        # 验证信号一致
        for key in signals1.keys():
            assert key in signals2
            
            # 浮点数比较使用容差
            if isinstance(signals1[key], float):
                assert abs(signals1[key] - signals2[key]) < 0.0001
            else:
                assert signals1[key] == signals2[key]
    
    def test_performance_of_pipeline(self):
        """测试管道性能"""
        import time
        
        # 创建更大的测试数据集
        large_test_data = pd.concat([self.test_data] * 10, ignore_index=True)
        
        indicator_calculator = IndicatorCalculator(self.config["indicators"])
        signal_generator = SignalGenerator(self.config["strategies"])
        
        # 测量指标计算时间
        start_time = time.time()
        indicators_data = indicator_calculator.calculate_indicators(large_test_data)
        indicator_time = time.time() - start_time
        
        # 测量信号生成时间
        start_time = time.time()
        signals = signal_generator.generate_signals(indicators_data)
        signal_time = time.time() - start_time
        
        # 验证结果
        assert indicators_data is not None
        assert signals is not None
        
        # 输出性能信息（测试中只记录，不断言）
        print(f"指标计算时间: {indicator_time:.3f}秒 (数据量: {len(large_test_data)})")
        print(f"信号生成时间: {signal_time:.3f}秒")
        
        # 验证性能在合理范围内（600条数据应该很快）
        assert indicator_time < 1.0  # 1秒内完成
        assert signal_time < 0.5     # 0.5秒内完成