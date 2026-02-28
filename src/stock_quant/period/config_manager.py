"""周期配置管理器"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class PeriodUnit(Enum):
    """周期单位"""
    MONTH = "month"
    YEAR = "year"
    DAY = "day"
    WEEK = "week"


@dataclass
class PeriodConfigItem:
    """周期配置项"""
    name: str
    months: int
    description: str
    recommended: bool = False


class PeriodConfig:
    """周期配置管理"""
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        self.periods: Dict[str, PeriodConfigItem] = {}
        self._initialize_defaults()
        
        if config_data:
            self.load_config(config_data)
    
    def _initialize_defaults(self) -> None:
        """初始化默认周期配置"""
        default_periods = [
            PeriodConfigItem("1m", 1, "1个月", recommended=True),
            PeriodConfigItem("3m", 3, "3个月", recommended=True),
            PeriodConfigItem("6m", 6, "6个月", recommended=True),
            PeriodConfigItem("1y", 12, "1年", recommended=True),
            PeriodConfigItem("2y", 24, "2年"),
            PeriodConfigItem("3y", 36, "3年", recommended=True),
            PeriodConfigItem("5y", 60, "5年"),
        ]
        
        for period in default_periods:
            self.add_period(period)
    
    def add_period(self, period: PeriodConfigItem) -> None:
        """添加周期配置"""
        self.periods[period.name] = period
    
    def remove_period(self, period_name: str) -> bool:
        """移除周期配置"""
        if period_name in self.periods:
            del self.periods[period_name]
            return True
        return False
    
    def get_period_config(self, period_name: str) -> Optional[Dict[str, Any]]:
        """获取周期配置"""
        if period_name in self.periods:
            period = self.periods[period_name]
            return {
                "name": period.name,
                "months": period.months,
                "description": period.description,
                "recommended": period.recommended
            }
        return None
    
    def get_all_periods(self) -> List[Dict[str, Any]]:
        """获取所有周期配置"""
        return [
            {
                "name": period.name,
                "months": period.months,
                "description": period.description,
                "recommended": period.recommended
            }
            for period in self.periods.values()
        ]
    
    def get_recommended_periods(self) -> List[str]:
        """获取推荐的周期"""
        return [
            period.name
            for period in self.periods.values()
            if period.recommended
        ]
    
    def load_config(self, config_data: Dict[str, Any]) -> None:
        """加载配置数据"""
        periods_data = config_data.get("periods", [])
        
        for period_data in periods_data:
            period = PeriodConfigItem(
                name=period_data.get("name"),
                months=period_data.get("months"),
                description=period_data.get("description", ""),
                recommended=period_data.get("recommended", False)
            )
            self.add_period(period)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "periods": [
                {
                    "name": period.name,
                    "months": period.months,
                    "description": period.description,
                    "recommended": period.recommended
                }
                for period in self.periods.values()
            ]
        }
    
    def validate_period_name(self, period_name: str) -> bool:
        """验证周期名称格式"""
        if not period_name:
            return False
        
        # 简单验证：字母数字组合
        return all(c.isalnum() for c in period_name)
    
    def get_period_by_months(self, months: int) -> Optional[str]:
        """根据月份数获取周期名称"""
        for period in self.periods.values():
            if period.months == months:
                return period.name
        return None
    
    def get_closest_period(self, months: int) -> str:
        """获取最接近的周期"""
        closest_period = None
        min_diff = float("inf")
        
        for period in self.periods.values():
            diff = abs(period.months - months)
            if diff < min_diff:
                min_diff = diff
                closest_period = period.name
        
        return closest_period or "1y"  # 默认返回1年