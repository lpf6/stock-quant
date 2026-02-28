"""批量处理性能测试"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from unittest.mock import Mock, patch

from src.stock_quant.core.data_fetcher import DataFetcher
from src.stock_quant.core.indicator_calculator import IndicatorCalculator
from src.stock_quant.core.signal_generator import SignalGenerator


class TestBatchProcessingPerformance:
    """批量处理性能测试类"""
    
    def setup_method(self):
        """测试设置"""
        # 配置
        self.config = {
            "data": {"source": "akshare", "max_retries": 1, "timeout": 5},
            "indicators": {"default_indicators": ["MovingAverageIndicator", "RSIIndicator", "BollingerBandsIndicator"]},
            "strategies": {
                "default_strategies": ["MACrossStrategy", "RSIStrategy", "MACDStrategy"],
                "strategy_weights": {
                    "MACrossStrategy": 0.3,
                    "RSIStrategy": 0.25,
                    "MACDStrategy": 0.25
                }
            },
            "performance": {"batch_size": 50, "max_workers": 4}
        }
        
        # 创建模拟数据
        self.create_mock_data()
        
        # 内存跟踪
        self.process = psutil.Process(os.getpid())
    
    def create_mock_data(self, days=60):
        """创建模拟数据"""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # 生成价格数据
        np.random.seed(42)
        base_price = 10.0
        trend = 0.02
        volatility = 0.5
        
        prices = []
        for i in range(days):
            price = base_price + trend * i + volatility * np.random.randn()
            prices.append(max(price, 0.01))  # 确保价格为正
        
        self.mock_stock_data = pd.DataFrame({
            'open': prices,
            'close': prices,
            'high': [p + abs(np.random.randn() * 0.1) for p in prices],
            'low': [p - abs(np.random.randn() * 0.1) for p in prices],
            'volume': [int(1000000 + np.random.randn() * 100000) for _ in range(days)],
            'amount': [p * v for p, v in zip(prices, [1000000 + np.random.randn() * 100000 for _ in range(days)])]
        }, index=dates)
    
    def get_memory_usage(self):
        """获取内存使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.performance
    def test_single_stock_processing_time(self):
        """测试单只股票处理时间"""
        # 初始化组件
        indicator_calculator = IndicatorCalculator(self.config["indicators"])
        signal_generator = SignalGenerator(self.config["strategies"])
        
        # 测量处理时间
        start_time = time.time()
        
        # 计算指标
        indicators_data = indicator_calculator.calculate_indicators(self.mock_stock_data)
        
        # 生成信号
        signals = signal_generator.generate_signals(indicators_data)
        
        processing_time = time.time() - start_time
        
        # 验证结果
        assert indicators_data is not None
        assert signals is not None
        assert 'total_score' in signals
        
        # 输出性能信息
        print(f"单只股票处理时间: {processing_time:.3f}秒")
        print(f"数据量: {len(self.mock_stock_data)} 条")
        print(f"生成的指标数: {len([c for c in indicators_data.columns if c not in ['open', 'close', 'high', 'low', 'volume', 'amount']])}")
        print(f"生成的信号数: {len(signals)}")
        
        # 性能要求：60条数据应该在0.1秒内完成
        assert processing_time < 0.1, f"处理时间过长: {processing_time:.3f}秒"
    
    @pytest.mark.performance
    def test_batch_processing_time(self):
        """测试批量处理时间"""
        batch_sizes = [10, 50, 100, 200]
        results = []
        
        for batch_size in batch_sizes:
            # 模拟批量数据
            symbols = [f"{i:06d}" for i in range(batch_size)]
            
            # 模拟数据获取器
            mock_fetcher = Mock(spec=DataFetcher)
            mock_fetcher.fetch_stock_data.return_value = self.mock_stock_data
            
            # 初始化其他组件
            indicator_calculator = IndicatorCalculator(self.config["indicators"])
            signal_generator = SignalGenerator(self.config["strategies"])
            
            # 测量批量处理时间
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            batch_results = []
            for symbol in symbols:
                # 获取数据
                stock_data = mock_fetcher.fetch_stock_data(symbol)
                
                if stock_data is not None:
                    # 计算指标
                    indicators_data = indicator_calculator.calculate_indicators(stock_data)
                    
                    # 生成信号
                    signals = signal_generator.generate_signals(indicators_data)
                    
                    # 记录结果
                    batch_results.append({
                        "symbol": symbol,
                        "score": signals.get("total_score", 0)
                    })
            
            processing_time = time.time() - start_time
            end_memory = self.get_memory_usage()
            memory_increase = end_memory - start_memory
            
            # 记录结果
            results.append({
                "batch_size": batch_size,
                "time_seconds": processing_time,
                "memory_increase_mb": memory_increase,
                "time_per_stock": processing_time / batch_size if batch_size > 0 else 0
            })
            
            # 验证
            assert len(batch_results) == batch_size
            
            print(f"\n批量大小: {batch_size}")
            print(f"总处理时间: {processing_time:.3f}秒")
            print(f"每只股票平均时间: {processing_time/batch_size:.3f}秒")
            print(f"内存增加: {memory_increase:.2f} MB")
        
        # 分析性能趋势
        print("\n性能趋势分析:")
        for i in range(1, len(results)):
            time_ratio = results[i]["time_seconds"] / results[i-1]["time_seconds"]
            size_ratio = results[i]["batch_size"] / results[i-1]["batch_size"]
            scalability = time_ratio / size_ratio
            
            print(f"从 {results[i-1]['batch_size']} 到 {results[i]['batch_size']}: "
                  f"时间增长 {time_ratio:.2f}x, 规模增长 {size_ratio:.2f}x, 可扩展性 {scalability:.2f}")
            
            # 验证可扩展性：处理时间应该大致线性增长
            # 允许一定的非线性（如缓存效应）
            assert 0.8 <= scalability <= 1.2, f"可扩展性不佳: {scalability:.2f}"
        
        # 验证200只股票的处理时间
        last_result = results[-1]
        assert last_result["time_seconds"] < 10.0, f"200只股票处理时间过长: {last_result['time_seconds']:.1f}秒"
    
    @pytest.mark.performance
    def test_memory_efficiency(self):
        """测试内存效率"""
        batch_size = 100
        symbols = [f"{i:06d}" for i in range(batch_size)]
        
        # 模拟数据获取器
        mock_fetcher = Mock(spec=DataFetcher)
        mock_fetcher.fetch_stock_data.return_value = self.mock_stock_data
        
        # 初始化组件
        indicator_calculator = IndicatorCalculator(self.config["indicators"])
        signal_generator = SignalGenerator(self.config["strategies"])
        
        # 跟踪内存使用
        memory_samples = []
        
        batch_results = []
        for i, symbol in enumerate(symbols):
            # 定期采样内存
            if i % 10 == 0:
                memory_samples.append(self.get_memory_usage())
            
            # 处理股票
            stock_data = mock_fetcher.fetch_stock_data(symbol)
            
            if stock_data is not None:
                indicators_data = indicator_calculator.calculate_indicators(stock_data)
                signals = signal_generator.generate_signals(indicators_data)
                
                batch_results.append({
                    "symbol": symbol,
                    "score": signals.get("total_score", 0)
                })
                
                # 及时释放引用（模拟垃圾回收）
                del indicators_data
                del signals
            
            # 模拟垃圾回收
            if i % 20 == 0:
                import gc
                gc.collect()
        
        # 最终内存采样
        memory_samples.append(self.get_memory_usage())
        
        # 分析内存使用
        print(f"\n内存使用分析:")
        print(f"初始内存: {memory_samples[0]:.2f} MB")
        print(f"峰值内存: {max(memory_samples):.2f} MB")
        print(f"最终内存: {memory_samples[-1]:.2f} MB")
        print(f"内存增长: {memory_samples[-1] - memory_samples[0]:.2f} MB")
        
        # 验证内存效率
        max_memory = max(memory_samples)
        memory_per_stock = (max_memory - memory_samples[0]) / batch_size
        
        print(f"每只股票内存占用: {memory_per_stock:.2f} MB")
        
        # 内存要求：每只股票应该小于5MB
        assert memory_per_stock < 5.0, f"每只股票内存占用过高: {memory_per_stock:.2f} MB"
        
        # 总内存增长应该有限
        total_memory_increase = memory_samples[-1] - memory_samples[0]
        assert total_memory_increase < 100.0, f"总内存增长过高: {total_memory_increase:.2f} MB"
    
    @pytest.mark.performance
    def test_large_dataset_processing(self):
        """测试大数据集处理"""
        # 创建大数据集（2年数据，约500条）
        large_mock_data = self.mock_stock_data
        for _ in range(7):  # 7 * 60 ≈ 420 + 原60 ≈ 480条
            large_mock_data = pd.concat([large_mock_data, self.mock_stock_data], ignore_index=True)
        
        print(f"大数据集大小: {len(large_mock_data)} 条记录")
        print(f"内存占用: {large_mock_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # 初始化组件
        indicator_calculator = IndicatorCalculator(self.config["indicators"])
        signal_generator = SignalGenerator(self.config["strategies"])
        
        # 测量处理时间
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # 计算指标
        indicators_data = indicator_calculator.calculate_indicators(large_mock_data)
        
        # 生成信号
        signals = signal_generator.generate_signals(indicators_data)
        
        processing_time = time.time() - start_time
        end_memory = self.get_memory_usage()
        
        # 验证结果
        assert indicators_data is not None
        assert signals is not None
        
        print(f"\n大数据集处理:")
        print(f"处理时间: {processing_time:.3f}秒")
        print(f"内存增加: {end_memory - start_memory:.2f} MB")
        print(f"生成的指标列数: {len(indicators_data.columns)}")
        
        # 性能要求：500条数据应该在2秒内完成
        assert processing_time < 2.0, f"大数据集处理时间过长: {processing_time:.3f}秒"
    
    @pytest.mark.performance
    def test_concurrent_processing_simulation(self):
        """测试并发处理模拟"""
        import concurrent.futures
        
        batch_size = 100
        symbols = [f"{i:06d}" for i in range(batch_size)]
        
        # 模拟数据
        mock_fetcher = Mock(spec=DataFetcher)
        mock_fetcher.fetch_stock_data.return_value = self.mock_stock_data
        
        def process_stock(symbol):
            """处理单只股票的函数"""
            # 每个线程创建自己的组件实例（避免线程安全问题）
            indicator_calculator = IndicatorCalculator(self.config["indicators"])
            signal_generator = SignalGenerator(self.config["strategies"])
            
            stock_data = mock_fetcher.fetch_stock_data(symbol)
            
            if stock_data is not None:
                indicators_data = indicator_calculator.calculate_indicators(stock_data)
                signals = signal_generator.generate_signals(indicators_data)
                
                return {
                    "symbol": symbol,
                    "score": signals.get("total_score", 0)
                }
            return None
        
        # 测试不同并发级别
        worker_counts = [1, 2, 4, 8]
        results_by_workers = {}
        
        for workers in worker_counts:
            print(f"\n测试并发级别: {workers} 个工作线程")
            
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                # 提交任务
                future_to_symbol = {
                    executor.submit(process_stock, symbol): symbol 
                    for symbol in symbols
                }
                
                # 收集结果
                results = []
                for future in concurrent.futures.as_completed(future_to_symbol):
                    result = future.result()
                    if result:
                        results.append(result)
            
            processing_time = time.time() - start_time
            
            results_by_workers[workers] = {
                "time_seconds": processing_time,
                "results_count": len(results),
                "throughput": batch_size / processing_time
            }
            
            print(f"处理时间: {processing_time:.3f}秒")
            print(f"吞吐量: {batch_size/processing_time:.2f} 股票/秒")
            
            # 验证所有股票都处理完成
            assert len(results) == batch_size
        
        # 分析并发性能
        print("\n并发性能分析:")
        baseline = results_by_workers[1]["throughput"]
        
        for workers in [2, 4, 8]:
            throughput = results_by_workers[workers]["throughput"]
            speedup = throughput / baseline
            efficiency = speedup / workers
            
            print(f"{workers}个工作线程: 加速比 {speedup:.2f}x, 效率 {efficiency:.2%}")
            
            # 验证并发有效性（允许一定的开销）
            # 2线程至少1.5倍加速，4线程至少2.5倍加速
            if workers == 2:
                assert speedup >= 1.5, f"2线程加速不足: {speedup:.2f}x"
            elif workers == 4:
                assert speedup >= 2.5, f"4线程加速不足: {speedup:.2f}x"
    
    @pytest.mark.performance
    def test_stress_test_1000_stocks(self):
        """压力测试：1000只股票"""
        batch_size = 1000
        symbols = [f"{i:06d}" for i in range(batch_size)]
        
        # 模拟数据获取器
        mock_fetcher = Mock(spec=DataFetcher)
        mock_fetcher.fetch_stock_data.return_value = self.mock_stock_data
        
        # 初始化组件
        indicator_calculator = IndicatorCalculator(self.config["indicators"])
        signal_generator = SignalGenerator(self.config["strategies"])
        
        print(f"\n压力测试: {batch_size} 只股票")
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        processed_count = 0
        batch_results = []
        
        # 分批处理（模拟实际批量处理）
        batch_config_size = self.config["performance"].get("batch_size", 50)
        
        for i in range(0, len(symbols), batch_config_size):
            batch_symbols = symbols[i:i + batch_config_size]
            
            for symbol in batch_symbols:
                stock_data = mock_fetcher.fetch_stock_data(symbol)
                
                if stock_data is not None:
                    indicators_data = indicator_calculator.calculate_indicators(stock_data)
                    signals = signal_generator.generate_signals(indicators_data)
                    
                    batch_results.append({
                        "symbol": symbol,
                        "score": signals.get("total_score", 0)
                    })
                    
                    processed_count += 1
                    
                    # 显示进度
                    if processed_count % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"进度: {processed_count}/{batch_size} ({processed_count/batch_size*100:.1f}%), "
                              f"耗时: {elapsed:.1f}秒, "
                              f"速率: {processed_count/elapsed:.1f} 股票/秒")
        
        total_time = time.time() - start_time
        end_memory = self.get_memory_usage()
        
        print(f"\n压力测试结果:")
        print(f"总处理时间: {total_time:.1f}秒")
        print(f"平均每只股票: {total_time/batch_size:.3f}秒")
        print(f"吞吐量: {batch_size/total_time:.1f} 股票/秒")
        print(f"内存使用: {end_memory - start_memory:.1f} MB 增长")
        print(f"峰值内存: {max(self.process.memory_info().rss for _ in range(10)) / 1024 / 1024:.1f} MB")
        
        # 验证结果
        assert len(batch_results) == batch_size
        
        # 性能要求：1000只股票应该在60秒内完成
        assert total_time < 60.0, f"1000只股票处理时间过长: {total_time:.1f}秒"
        
        # 内存要求：1000只股票内存增长应该小于200MB
        memory_increase = end_memory - start_memory
        assert memory_increase < 200.0, f"内存使用过高: {memory_increase:.1f} MB"