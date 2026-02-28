"""配置加载器"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigLoader:
    """配置加载器，支持YAML文件和环境变量"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            "config"
        )
        self.configs: Dict[str, Any] = {}
        self.env_prefix = "STOCK_QUANT_"
    
    def load_all_configs(self) -> Dict[str, Any]:
        """加载所有配置文件"""
        config_files = [
            "default.yaml",
            f"{self._get_environment()}.yaml",
            "local.yaml"  # 用户本地配置
        ]
        
        merged_config = {}
        
        for config_file in config_files:
            config_path = os.path.join(self.config_dir, config_file)
            if os.path.exists(config_path):
                config = self.load_yaml_config(config_path)
                merged_config = self._merge_configs(merged_config, config)
        
        # 加载环境变量配置
        env_config = self.load_env_config()
        merged_config = self._merge_configs(merged_config, env_config)
        
        self.configs = merged_config
        return merged_config
    
    def load_yaml_config(self, file_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            return config
        except Exception as e:
            print(f"加载配置文件 {file_path} 失败: {e}")
            return {}
    
    def load_env_config(self) -> Dict[str, Any]:
        """加载环境变量配置"""
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                config_key = key[len(self.env_prefix):].lower()
                env_config[config_key] = self._parse_env_value(value)
        
        return env_config
    
    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值"""
        # 尝试解析为不同类型
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        try:
            # 尝试解析为整数
            return int(value)
        except ValueError:
            try:
                # 尝试解析为浮点数
                return float(value)
            except ValueError:
                # 保持为字符串
                return value
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置（深度合并）"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _get_environment(self) -> str:
        """获取当前环境"""
        return os.environ.get("STOCK_QUANT_ENV", "development")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        if not self.configs:
            self.load_all_configs()
        
        keys = key.split(".")
        value = self.configs
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config(self, key: str, value: Any) -> None:
        """设置配置项"""
        if not self.configs:
            self.configs = {}
        
        keys = key.split(".")
        config = self.configs
        
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, file_name: str = "local.yaml") -> None:
        """保存配置到文件"""
        if not self.configs:
            return
        
        config_path = os.path.join(self.config_dir, file_name)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.configs, f, default_flow_style=False, allow_unicode=True)
            print(f"配置已保存到: {config_path}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def reload_config(self) -> Dict[str, Any]:
        """重新加载配置"""
        return self.load_all_configs()
    
    def watch_config_changes(self, callback) -> None:
        """监听配置变化（基础实现）"""
        # 这里可以集成watchdog等库实现文件变化监听
        print("配置变化监听功能待实现")
    
    def validate_config(self) -> List[str]:
        """验证配置"""
        errors = []
        
        # 检查必需配置
        required_configs = [
            "data.source",
            "plugins.dirs",
            "period.default"
        ]
        
        for config_key in required_configs:
            if self.get_config(config_key) is None:
                errors.append(f"必需配置项缺失: {config_key}")
        
        return errors