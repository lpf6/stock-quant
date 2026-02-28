"""日志系统"""

import logging
import logging.handlers
import os
from typing import Optional, Dict, Any
from pathlib import Path


def setup_logger(
    name: str = "stock_quant",
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    file_path: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """设置日志记录器"""
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 移除现有的处理器
    logger.handlers.clear()
    
    # 设置日志格式
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    formatter = logging.Formatter(log_format, date_format)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了文件路径）
    if file_path:
        # 确保日志目录存在
        log_dir = os.path.dirname(file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 使用RotatingFileHandler自动轮转
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "stock_quant") -> logging.Logger:
    """获取日志记录器"""
    return logging.getLogger(name)


class ProgressLogger:
    """进度日志记录器"""
    
    def __init__(self, total: int, name: str = "进度", logger: Optional[logging.Logger] = None):
        self.total = total
        self.current = 0
        self.name = name
        self.logger = logger or get_logger()
        self.start_time = None
        
    def start(self) -> None:
        """开始记录"""
        import time
        self.start_time = time.time()
        self.logger.info(f"{self.name}开始: 共 {self.total} 项")
        
    def update(self, increment: int = 1, message: Optional[str] = None) -> None:
        """更新进度"""
        self.current += increment
        percentage = (self.current / self.total) * 100
        
        if message:
            log_message = f"{self.name}: {message} ({self.current}/{self.total}, {percentage:.1f}%)"
        else:
            log_message = f"{self.name}: {self.current}/{self.total} ({percentage:.1f}%)"
        
        self.logger.info(log_message)
        
        # 每10%或最后一条记录时显示预估时间
        if percentage % 10 < 1 or self.current == self.total:
            self._log_eta()
    
    def _log_eta(self) -> None:
        """记录预估完成时间"""
        if self.start_time and self.current > 0:
            import time
            elapsed = time.time() - self.start_time
            if self.current < self.total:
                remaining = (elapsed / self.current) * (self.total - self.current)
                eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
                self.logger.info(f"预计剩余时间: {eta}")
    
    def complete(self, message: Optional[str] = None) -> None:
        """完成记录"""
        import time
        if self.start_time:
            elapsed = time.time() - self.start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            
            if message:
                complete_message = f"{self.name}完成: {message} (用时: {elapsed_str})"
            else:
                complete_message = f"{self.name}完成: 共 {self.total} 项，用时: {elapsed_str}"
            
            self.logger.info(complete_message)
        
    def error(self, message: str) -> None:
        """记录错误"""
        self.logger.error(f"{self.name}错误: {message}")


def log_execution_time(func):
    """记录函数执行时间的装饰器"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.debug(f"函数 {func.__name__} 执行时间: {execution_time:.4f}秒")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {e}")
            raise
    
    return wrapper