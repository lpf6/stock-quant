"""插件系统单元测试"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from src.stock_quant.plugins.base import (
    BasePlugin, StrategyPlugin, IndicatorPlugin, PluginMetadata
)
from src.stock_quant.plugins.manager import PluginManager
from src.stock_quant.plugins.registry import PluginRegistry
from src.stock_quant.plugins.strategies.ma_cross_strategy import MACrossStrategy
from src.stock_quant.plugins.indicators.moving_average import MovingAverageIndicator


class TestBasePlugin:
    """插件基类测试"""
    
    def test_base_plugin_initialization(self):
        """测试插件初始化"""
        # 创建抽象基类的模拟子类
        class MockPlugin(BasePlugin):
            @property
            def plugin_type(self):
                return "test"
            
            def initialize(self, config=None):
                self.config = config or {}
            
            def cleanup(self):
                pass
        
        plugin = MockPlugin()
        
        assert plugin.name == "MockPlugin"
        assert plugin.version == "1.0.0"
        assert plugin.description == ""
        assert plugin.author == ""
        assert plugin.enabled == True
        assert plugin.config == {}
        
    def test_strategy_plugin_interface(self):
        """测试策略插件接口"""
        class TestStrategy(StrategyPlugin):
            def calculate_signals(self, df):
                return {"signal": 1}
            
            def get_signal_descriptions(self):
                return {"signal": "测试信号"}
        
        strategy = TestStrategy()
        
        assert strategy.plugin_type == "strategy"
        assert callable(strategy.calculate_signals)
        assert callable(strategy.get_signal_descriptions)
        
    def test_indicator_plugin_interface(self):
        """测试指标插件接口"""
        class TestIndicator(IndicatorPlugin):
            def calculate(self, df):
                return df
            
            def get_indicator_names(self):
                return ["test_indicator"]
        
        indicator = TestIndicator()
        
        assert indicator.plugin_type == "indicator"
        assert callable(indicator.calculate)
        assert callable(indicator.get_indicator_names)


class TestPluginRegistry:
    """插件注册表测试"""
    
    def setup_method(self):
        """测试设置"""
        self.registry = PluginRegistry()
        
        # 创建测试插件类
        class TestPlugin(BasePlugin):
            @property
            def plugin_type(self):
                return "test"
            
            def initialize(self, config=None):
                pass
            
            def cleanup(self):
                pass
        
        self.test_plugin_class = TestPlugin
        
        self.metadata = PluginMetadata(
            name="TestPlugin",
            plugin_type="test",
            version="1.0.0",
            description="测试插件",
            author="测试作者",
            dependencies=["dep1", "dep2"],
            config_schema={"param1": "int"}
        )
    
    def test_register_plugin(self):
        """测试注册插件"""
        self.registry.register(self.test_plugin_class, self.metadata)
        
        assert "TestPlugin" in self.registry._plugins
        assert "TestPlugin" in self.registry._metadata
        assert self.registry.plugin_exists("TestPlugin") == True
    
    def test_unregister_plugin(self):
        """测试取消注册插件"""
        self.registry.register(self.test_plugin_class, self.metadata)
        self.registry.unregister("TestPlugin")
        
        assert "TestPlugin" not in self.registry._plugins
        assert "TestPlugin" not in self.registry._metadata
        assert self.registry.plugin_exists("TestPlugin") == False
    
    def test_get_plugin_class(self):
        """测试获取插件类"""
        self.registry.register(self.test_plugin_class, self.metadata)
        
        plugin_class = self.registry.get_plugin_class("TestPlugin")
        assert plugin_class == self.test_plugin_class
        
        # 测试不存在的插件
        assert self.registry.get_plugin_class("NonExistent") is None
    
    def test_get_plugin_metadata(self):
        """测试获取插件元数据"""
        self.registry.register(self.test_plugin_class, self.metadata)
        
        metadata = self.registry.get_plugin_metadata("TestPlugin")
        assert metadata.name == "TestPlugin"
        assert metadata.plugin_type == "test"
        assert metadata.version == "1.0.0"
        
        # 测试不存在的插件
        assert self.registry.get_plugin_metadata("NonExistent") is None
    
    def test_get_available_plugins(self):
        """测试获取可用插件列表"""
        self.registry.register(self.test_plugin_class, self.metadata)
        
        plugins = self.registry.get_available_plugins()
        assert len(plugins) == 1
        assert plugins[0]["name"] == "TestPlugin"
        
        # 测试按类型过滤
        plugins = self.registry.get_available_plugins("test")
        assert len(plugins) == 1
        
        plugins = self.registry.get_available_plugins("other")
        assert len(plugins) == 0
    
    def test_count_plugins(self):
        """测试统计插件数量"""
        assert self.registry.count_plugins() == 0
        
        self.registry.register(self.test_plugin_class, self.metadata)
        assert self.registry.count_plugins() == 1
        
        # 测试按类型统计
        assert self.registry.count_plugins("test") == 1
        assert self.registry.count_plugins("other") == 0
    
    def test_clear(self):
        """测试清空注册表"""
        self.registry.register(self.test_plugin_class, self.metadata)
        assert self.registry.count_plugins() == 1
        
        self.registry.clear()
        assert self.registry.count_plugins() == 0


class TestPluginManager:
    """插件管理器测试"""
    
    def setup_method(self):
        """测试设置"""
        self.manager = PluginManager(plugin_dirs=[])
    
    def test_initialization(self):
        """测试初始化"""
        assert isinstance(self.manager.registry, PluginRegistry)
        assert isinstance(self.manager.plugin_dirs, list)
        assert isinstance(self.manager.loaded_plugins, dict)
    
    @patch('src.stock_quant.plugins.manager.pkgutil')
    @patch('src.stock_quant.plugins.manager.importlib')
    def test_discover_plugins(self, mock_importlib, mock_pkgutil):
        """测试发现插件"""
        # 模拟模块迭代器
        mock_module_info = Mock()
        mock_module_info.name = "test_module"
        mock_pkgutil.iter_modules.return_value = [mock_module_info]
        
        # 模拟导入
        mock_module = Mock()
        mock_importlib.import_module.return_value = mock_module
        
        # 模拟插件类
        class TestPlugin(BasePlugin):
            @property
            def plugin_type(self):
                return "test"
            
            def initialize(self, config=None):
                pass
            
            def cleanup(self):
                pass
        
        mock_module.TestPlugin = TestPlugin
        
        # 获取模块成员
        mock_importlib.inspect.getmembers.return_value = [
            ("TestPlugin", TestPlugin),
            ("BasePlugin", BasePlugin),
            ("StrategyPlugin", StrategyPlugin),
            ("IndicatorPlugin", IndicatorPlugin)
        ]
        
        # 调用方法
        self.manager._discover_plugins()
        
        # 验证插件已注册
        available_plugins = self.manager.get_available_plugins()
        assert len(available_plugins) > 0
    
    def test_register_and_load_plugin(self):
        """测试注册和加载插件"""
        # 创建测试插件类
        class TestStrategy(StrategyPlugin):
            def calculate_signals(self, df):
                return {"test_signal": 1}
            
            def get_signal_descriptions(self):
                return {"test_signal": "测试信号"}
        
        # 注册插件
        self.manager.register_plugin(TestStrategy)
        
        # 验证插件已注册
        available_plugins = self.manager.get_available_plugins("strategy")
        assert len(available_plugins) == 1
        assert available_plugins[0]["name"] == "TestStrategy"
        
        # 加载插件
        plugin = self.manager.load_plugin("TestStrategy", {"param": "value"})
        
        assert plugin is not None
        assert plugin.name == "TestStrategy"
        assert "TestStrategy" in self.manager.loaded_plugins
    
    def test_load_nonexistent_plugin(self):
        """测试加载不存在的插件"""
        with pytest.raises(ValueError):
            self.manager.load_plugin("NonExistentPlugin")
    
    def test_unload_plugin(self):
        """测试卸载插件"""
        # 先加载一个插件
        class TestStrategy(StrategyPlugin):
            def calculate_signals(self, df):
                return {}
            
            def get_signal_descriptions(self):
                return {}
        
        self.manager.register_plugin(TestStrategy)
        self.manager.load_plugin("TestStrategy")
        
        assert "TestStrategy" in self.manager.loaded_plugins
        
        # 卸载插件
        self.manager.unload_plugin("TestStrategy")
        
        assert "TestStrategy" not in self.manager.loaded_plugins
    
    def test_get_plugin(self):
        """测试获取插件"""
        # 先加载一个插件
        class TestStrategy(StrategyPlugin):
            def calculate_signals(self, df):
                return {}
            
            def get_signal_descriptions(self):
                return {}
        
        self.manager.register_plugin(TestStrategy)
        self.manager.load_plugin("TestStrategy")
        
        plugin = self.manager.get_plugin("TestStrategy")
        assert plugin is not None
        assert plugin.name == "TestStrategy"
        
        # 测试获取不存在的插件
        assert self.manager.get_plugin("NonExistent") is None
    
    def test_check_dependencies(self):
        """测试检查依赖"""
        # 创建有依赖的插件
        class TestStrategy(StrategyPlugin):
            def calculate_signals(self, df):
                return {}
            
            def get_signal_descriptions(self):
                return {}
        
        # 模拟元数据
        from src.stock_quant.plugins.base import PluginMetadata
        
        metadata = PluginMetadata(
            name="TestStrategy",
            plugin_type="strategy",
            version="1.0.0",
            dependencies=["Indicator1", "Indicator2"]
        )
        
        # 手动注册插件和元数据
        self.manager.registry.register(TestStrategy, metadata)
        
        # 检查依赖（应该返回缺失的依赖）
        missing_deps = self.manager.check_dependencies("TestStrategy")
        assert len(missing_deps) == 2
        assert "Indicator1" in missing_deps
        assert "Indicator2" in missing_deps
        
        # 加载依赖插件后再检查
        class TestIndicator(IndicatorPlugin):
            def calculate(self, df):
                return df
            
            def get_indicator_names(self):
                return ["test"]
        
        self.manager.register_plugin(TestIndicator)
        self.manager.load_plugin("TestIndicator")
        
        # 更新元数据，只依赖一个已加载的插件
        metadata.dependencies = ["TestIndicator"]
        
        missing_deps = self.manager.check_dependencies("TestStrategy")
        assert len(missing_deps) == 0
    
    def test_reload_plugin(self):
        """测试重新加载插件"""
        # 创建测试插件
        class TestStrategy(StrategyPlugin):
            def __init__(self):
                super().__init__()
                self.counter = 0
            
            def calculate_signals(self, df):
                self.counter += 1
                return {"counter": self.counter}
            
            def get_signal_descriptions(self):
                return {}
        
        self.manager.register_plugin(TestStrategy)
        
        # 第一次加载
        plugin1 = self.manager.load_plugin("TestStrategy")
        result1 = plugin1.calculate_signals(None)
        
        # 重新加载
        self.manager.reload_plugin("TestStrategy")
        plugin2 = self.manager.get_plugin("TestStrategy")
        result2 = plugin2.calculate_signals(None)
        
        # 验证计数器重置
        assert result1["counter"] == 1
        assert result2["counter"] == 1  # 重新加载后应该重置
    
    def test_ma_cross_strategy_integration(self):
        """测试MA交叉策略集成"""
        strategy = MACrossStrategy()
        strategy.initialize({"short_period": 5, "long_period": 10})
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'close': [10.0, 10.2, 10.5, 10.3, 10.8, 11.0, 11.2, 11.5, 11.3, 11.8,
                      12.0, 12.2, 12.5, 12.3, 12.8, 13.0, 13.2, 13.5, 13.3, 13.8]
        })
        
        # 计算信号
        signals = strategy.calculate_signals(test_data)
        
        assert isinstance(signals, dict)
        assert "ma_cross" in signals
        assert "score" in signals
        
    def test_moving_average_indicator_integration(self):
        """测试移动平均线指标集成"""
        indicator = MovingAverageIndicator()
        indicator.initialize({"periods": [5, 10]})
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'close': [10.0, 10.2, 10.5, 10.3, 10.8, 11.0, 11.2, 11.5, 11.3, 11.8]
        })
        
        # 计算指标
        result = indicator.calculate(test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert "MA5" in result.columns
        assert "MA10" in result.columns
        
        indicator_names = indicator.get_indicator_names()
        assert "MA5" in indicator_names
        assert "MA10" in indicator_names