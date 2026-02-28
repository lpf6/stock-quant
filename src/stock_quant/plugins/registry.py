"""插件注册表"""

from typing import Dict, List, Optional, Type, Any
from .base import BasePlugin, PluginMetadata


class PluginRegistry:
    """插件注册表，管理插件元数据和类"""
    
    def __init__(self):
        self._plugins: Dict[str, Type[BasePlugin]] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
    
    def register(self, plugin_class: Type[BasePlugin], metadata: PluginMetadata) -> None:
        """注册插件"""
        plugin_name = metadata.name
        self._plugins[plugin_name] = plugin_class
        self._metadata[plugin_name] = metadata
    
    def unregister(self, plugin_name: str) -> None:
        """取消注册插件"""
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
        if plugin_name in self._metadata:
            del self._metadata[plugin_name]
    
    def get_plugin_class(self, plugin_name: str) -> Optional[Type[BasePlugin]]:
        """获取插件类"""
        return self._plugins.get(plugin_name)
    
    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """获取插件元数据"""
        return self._metadata.get(plugin_name)
    
    def get_available_plugins(self, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取可用插件列表"""
        result = []
        for name, metadata in self._metadata.items():
            if plugin_type and metadata.plugin_type != plugin_type:
                continue
            result.append({
                "name": name,
                "type": metadata.plugin_type,
                "version": metadata.version,
                "description": metadata.description,
                "author": metadata.author,
                "dependencies": metadata.dependencies
            })
        return result
    
    def clear(self) -> None:
        """清空注册表"""
        self._plugins.clear()
        self._metadata.clear()
    
    def plugin_exists(self, plugin_name: str) -> bool:
        """检查插件是否存在"""
        return plugin_name in self._plugins
    
    def count_plugins(self, plugin_type: Optional[str] = None) -> int:
        """统计插件数量"""
        if plugin_type:
            return sum(1 for m in self._metadata.values() if m.plugin_type == plugin_type)
        return len(self._plugins)