"""插件管理器"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Union

from .base import BasePlugin, StrategyPlugin, IndicatorPlugin, PluginMetadata
from .registry import PluginRegistry


class PluginManager:
    """插件管理器，支持动态加载/卸载插件"""
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self.registry = PluginRegistry()
        self.plugin_dirs = plugin_dirs or []
        self.loaded_plugins: Dict[str, BasePlugin] = {}
        self._discover_plugins()
    
    def _discover_plugins(self) -> None:
        """自动发现插件"""
        # 扫描插件目录
        for plugin_dir in self.plugin_dirs:
            if Path(plugin_dir).exists():
                self._scan_directory(plugin_dir)
        
        # 扫描内置插件
        self._scan_builtin_plugins()
    
    def _scan_directory(self, directory: str) -> None:
        """扫描目录中的插件"""
        try:
            for module_info in pkgutil.iter_modules([directory]):
                module_name = f"{Path(directory).name}.{module_info.name}"
                self._load_module_plugins(module_name)
        except Exception as e:
            print(f"扫描插件目录 {directory} 失败: {e}")
    
    def _scan_builtin_plugins(self) -> None:
        """扫描内置插件"""
        try:
            # 扫描strategies目录
            strategies_module = "stock_quant.plugins.strategies"
            self._load_module_plugins(strategies_module)
            
            # 扫描indicators目录
            indicators_module = "stock_quant.plugins.indicators"
            self._load_module_plugins(indicators_module)
        except ImportError:
            pass  # 模块可能还不存在
    
    def _load_module_plugins(self, module_name: str) -> None:
        """加载模块中的插件"""
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj) 
                    and issubclass(obj, BasePlugin) 
                    and obj not in [BasePlugin, StrategyPlugin, IndicatorPlugin]
                ):
                    self.register_plugin(obj)
        except Exception as e:
            print(f"加载模块 {module_name} 失败: {e}")
    
    def register_plugin(self, plugin_class: Type[BasePlugin]) -> None:
        """注册插件类"""
        plugin_instance = plugin_class()
        metadata = PluginMetadata(
            name=plugin_instance.name,
            plugin_type=plugin_instance.plugin_type,
            version=plugin_instance.version,
            description=plugin_instance.description,
            author=plugin_instance.author
        )
        self.registry.register(plugin_class, metadata)
    
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> BasePlugin:
        """加载插件实例"""
        plugin_class = self.registry.get_plugin_class(plugin_name)
        if not plugin_class:
            raise ValueError(f"插件 {plugin_name} 未注册")
        
        plugin_instance = plugin_class()
        plugin_instance.initialize(config or {})
        self.loaded_plugins[plugin_name] = plugin_instance
        return plugin_instance
    
    def unload_plugin(self, plugin_name: str) -> None:
        """卸载插件"""
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            plugin.cleanup()
            del self.loaded_plugins[plugin_name]
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """获取已加载的插件实例"""
        return self.loaded_plugins.get(plugin_name)
    
    def get_all_plugins(self) -> Dict[str, BasePlugin]:
        """获取所有已加载插件"""
        return self.loaded_plugins.copy()
    
    def get_available_plugins(self, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取可用插件列表"""
        return self.registry.get_available_plugins(plugin_type)
    
    def reload_plugin(self, plugin_name: str) -> None:
        """重新加载插件"""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            config = plugin.config.copy()
            self.unload_plugin(plugin_name)
            self.load_plugin(plugin_name, config)
    
    def check_dependencies(self, plugin_name: str) -> List[str]:
        """检查插件依赖"""
        metadata = self.registry.get_plugin_metadata(plugin_name)
        if not metadata:
            return []
        
        missing_deps = []
        for dep in metadata.dependencies:
            if dep not in self.loaded_plugins:
                missing_deps.append(dep)
        return missing_deps