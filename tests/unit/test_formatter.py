"""输出格式化器单元测试"""

import pytest
import pandas as pd
import json
import csv
from io import StringIO
from src.stock_quant.utils.formatter import OutputFormatter


class TestOutputFormatter:
    """输出格式化器测试类"""
    
    def setup_method(self):
        """测试设置"""
        self.test_data = [
            {"symbol": "000001", "score": 0.85, "trend": "up"},
            {"symbol": "000002", "score": 0.72, "trend": "down"},
            {"symbol": "000003", "score": 0.91, "trend": "up"}
        ]
        
        self.test_df = pd.DataFrame(self.test_data)
        
        self.default_config = {
            "encoding": "utf-8",
            "include_header": True,
            "csv": {"delimiter": ",", "quote_char": '"'},
            "json": {"indent": 2, "sort_keys": True},
            "markdown": {"table_format": "github"}
        }
    
    def test_initialization(self):
        """测试初始化"""
        formatter = OutputFormatter("csv", self.default_config)
        
        assert formatter.format_type == "csv"
        assert formatter.config == self.default_config
    
    def test_invalid_format(self):
        """测试无效格式"""
        with pytest.raises(ValueError):
            OutputFormatter("invalid_format")
    
    def test_format_csv(self):
        """测试CSV格式化"""
        formatter = OutputFormatter("csv", self.default_config)
        result = formatter.format_data(self.test_data)
        
        # 验证CSV格式
        assert "symbol" in result
        assert "score" in result
        assert "trend" in result
        assert "000001" in result
        assert "0.85" in result
        
        # 验证CSV可以被正确解析
        reader = csv.DictReader(StringIO(result))
        rows = list(reader)
        
        assert len(rows) == 3
        assert rows[0]["symbol"] == "000001"
        assert rows[0]["score"] == "0.85"
        assert rows[0]["trend"] == "up"
    
    def test_format_csv_no_header(self):
        """测试CSV格式化（无表头）"""
        config = self.default_config.copy()
        config["include_header"] = False
        
        formatter = OutputFormatter("csv", config)
        result = formatter.format_data(self.test_data)
        
        # 验证没有表头
        reader = csv.reader(StringIO(result))
        rows = list(reader)
        
        assert len(rows) == 3  # 只有数据行，没有表头行
    
    def test_format_csv_custom_delimiter(self):
        """测试CSV格式化（自定义分隔符）"""
        config = self.default_config.copy()
        config["csv"]["delimiter"] = "\t"  # Tab分隔
        
        formatter = OutputFormatter("csv", config)
        result = formatter.format_data(self.test_data)
        
        # 验证Tab分隔符
        lines = result.strip().split("\n")
        assert len(lines) == 4  # 表头 + 3行数据
        assert "\t" in lines[0]  # 表头使用Tab分隔
    
    def test_format_json(self):
        """测试JSON格式化"""
        formatter = OutputFormatter("json", self.default_config)
        result = formatter.format_data(self.test_data)
        
        # 验证JSON格式
        parsed = json.loads(result)
        
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert parsed[0]["symbol"] == "000001"
        assert parsed[0]["score"] == 0.85
        assert parsed[0]["trend"] == "up"
        
        # 验证缩进
        assert "\n" in result  # 有缩进应该有换行
    
    def test_format_json_no_indent(self):
        """测试JSON格式化（无缩进）"""
        config = self.default_config.copy()
        config["json"]["indent"] = None
        
        formatter = OutputFormatter("json", config)
        result = formatter.format_data(self.test_data)
        
        # 验证没有缩进
        assert "\n" not in result  # 无缩进应该没有换行
        
        # 仍然可以正确解析
        parsed = json.loads(result)
        assert len(parsed) == 3
    
    def test_format_html(self):
        """测试HTML格式化"""
        formatter = OutputFormatter("html", self.default_config)
        result = formatter.format_data(self.test_data)
        
        # 验证HTML格式
        assert "<table" in result
        assert "<thead" in result
        assert "<tbody" in result
        assert "<tr>" in result
        assert "<th>" in result
        assert "<td>" in result
        
        # 验证内容
        assert "000001" in result
        assert "0.85" in result
        assert "up" in result
        
        # 验证样式
        assert "<style>" in result
        assert "background-color" in result
    
    def test_format_markdown(self):
        """测试Markdown格式化"""
        formatter = OutputFormatter("markdown", self.default_config)
        result = formatter.format_data(self.test_data)
        
        # 验证Markdown格式
        lines = result.strip().split("\n")
        
        assert len(lines) == 5  # 表头 + 分隔线 + 3行数据
        assert lines[0].startswith("|")  # 表头行
        assert lines[1].startswith("|")  # 分隔线行
        assert "---" in lines[1]  # 分隔线
        
        # 验证内容
        assert "000001" in result
        assert "0.85" in result
        assert "up" in result
    
    def test_format_markdown_simple(self):
        """测试Markdown格式化（简单格式）"""
        config = self.default_config.copy()
        config["markdown"]["table_format"] = "simple"
        
        formatter = OutputFormatter("markdown", config)
        result = formatter.format_data(self.test_data)
        
        # 验证简单格式
        lines = result.strip().split("\n")
        
        # 简单格式：分隔线 + 数据行
        assert len(lines) == 4  # 分隔线 + 3行数据
        assert not lines[0].startswith("|")  # 不是表格格式
        assert "000001" in lines[1]  # 数据行
    
    def test_format_with_dataframe(self):
        """测试用DataFrame格式化"""
        formatter = OutputFormatter("csv", self.default_config)
        result = formatter.format_data(self.test_df)
        
        # 验证结果
        reader = csv.DictReader(StringIO(result))
        rows = list(reader)
        
        assert len(rows) == 3
        assert rows[0]["symbol"] == "000001"
    
    def test_format_empty_data(self):
        """测试格式化空数据"""
        formatter = OutputFormatter("csv", self.default_config)
        
        # 空列表
        result = formatter.format_data([])
        assert result == ""
        
        # 空DataFrame
        empty_df = pd.DataFrame()
        result = formatter.format_data(empty_df)
        assert result == ""
        
        # HTML格式的空数据
        formatter_html = OutputFormatter("html", self.default_config)
        result = formatter_html.format_data([])
        assert "<table></table>" in result
    
    def test_convert_format(self):
        """测试格式转换"""
        formatter = OutputFormatter("csv", self.default_config)
        
        # CSV转JSON
        json_result = formatter.convert_format(self.test_data, "json")
        parsed_json = json.loads(json_result)
        assert isinstance(parsed_json, list)
        assert len(parsed_json) == 3
        
        # JSON转Markdown（通过中间转换）
        formatter_json = OutputFormatter("json", self.default_config)
        md_result = formatter_json.convert_format(self.test_data, "markdown")
        assert "|" in md_result  # 表格格式
        
        # 验证原始格式不变
        assert formatter.format_type == "csv"
        assert formatter_json.format_type == "json"
    
    def test_get_supported_formats(self):
        """测试获取支持的格式"""
        formats = OutputFormatter.get_supported_formats()
        
        expected_formats = ["csv", "json", "html", "markdown"]
        for fmt in expected_formats:
            assert fmt in formats
        
        assert len(formats) == 4
    
    def test_get_format_description(self):
        """测试获取格式描述"""
        # 测试已知格式
        csv_desc = OutputFormatter.get_format_description("csv")
        assert "逗号分隔值" in csv_desc
        
        json_desc = OutputFormatter.get_format_description("json")
        assert "JSON格式" in json_desc
        
        html_desc = OutputFormatter.get_format_description("html")
        assert "HTML表格" in html_desc
        
        md_desc = OutputFormatter.get_format_description("markdown")
        assert "Markdown表格" in md_desc
        
        # 测试未知格式
        unknown_desc = OutputFormatter.get_format_description("unknown")
        assert unknown_desc == "未知格式"
    
    def test_save_to_file(self, tmp_path):
        """测试保存到文件"""
        # 创建临时文件路径
        file_path = tmp_path / "test_output.csv"
        
        formatter = OutputFormatter("csv", self.default_config)
        formatter.save_to_file(self.test_data, str(file_path))
        
        # 验证文件存在
        assert file_path.exists()
        
        # 验证文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        assert "symbol" in content
        assert "000001" in content
        
        # 测试JSON格式保存
        json_path = tmp_path / "test_output.json"
        formatter_json = OutputFormatter("json", self.default_config)
        formatter_json.save_to_file(self.test_data, str(json_path))
        
        assert json_path.exists()
        
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()
            parsed = json.loads(content)
            
        assert len(parsed) == 3
    
    def test_save_to_file_create_dirs(self, tmp_path):
        """测试保存到文件时创建目录"""
        # 多层目录路径
        file_path = tmp_path / "deep" / "nested" / "dir" / "output.csv"
        
        formatter = OutputFormatter("csv", self.default_config)
        formatter.save_to_file(self.test_data, str(file_path))
        
        # 验证目录已创建且文件存在
        assert file_path.parent.exists()
        assert file_path.exists()
    
    def test_save_to_file_different_encoding(self, tmp_path):
        """测试保存到文件（不同编码）"""
        file_path = tmp_path / "test_output.csv"
        
        config = self.default_config.copy()
        config["encoding"] = "gbk"  # 测试GBK编码
        
        formatter = OutputFormatter("csv", config)
        formatter.save_to_file(self.test_data, str(file_path))
        
        assert file_path.exists()