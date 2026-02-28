"""输出格式化器"""

import json
import csv
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from io import StringIO


class OutputFormatter:
    """输出格式化器，支持多种输出格式"""
    
    def __init__(self, format_type: str = "csv", config: Optional[Dict[str, Any]] = None):
        self.format_type = format_type.lower()
        self.config = config or {}
        self._validate_format()
    
    def _validate_format(self) -> None:
        """验证输出格式"""
        valid_formats = ["csv", "json", "html", "markdown"]
        if self.format_type not in valid_formats:
            raise ValueError(f"不支持的输出格式: {self.format_type}，支持的格式: {valid_formats}")
    
    def format_data(self, data: Union[List[Dict], pd.DataFrame], **kwargs) -> str:
        """格式化数据"""
        if isinstance(data, pd.DataFrame):
            data = data.to_dict("records")
        
        format_methods = {
            "csv": self._format_csv,
            "json": self._format_json,
            "html": self._format_html,
            "markdown": self._format_markdown
        }
        
        if self.format_type in format_methods:
            return format_methods[self.format_type](data, **kwargs)
        else:
            raise ValueError(f"未实现的格式: {self.format_type}")
    
    def _format_csv(self, data: List[Dict], **kwargs) -> str:
        """格式化为CSV"""
        if not data:
            return ""
        
        csv_config = self.config.get("csv", {})
        delimiter = csv_config.get("delimiter", ",")
        quote_char = csv_config.get("quote_char", '"')
        
        output = StringIO()
        fieldnames = list(data[0].keys())
        
        writer = csv.DictWriter(
            output, 
            fieldnames=fieldnames,
            delimiter=delimiter,
            quoting=csv.QUOTE_MINIMAL if quote_char else csv.QUOTE_NONE,
            quotechar=quote_char if quote_char else '"'
        )
        
        if self.config.get("include_header", True):
            writer.writeheader()
        
        for row in data:
            writer.writerow(row)
        
        return output.getvalue()
    
    def _format_json(self, data: List[Dict], **kwargs) -> str:
        """格式化为JSON"""
        json_config = self.config.get("json", {})
        indent = json_config.get("indent", 2)
        sort_keys = json_config.get("sort_keys", True)
        
        return json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    
    def _format_html(self, data: List[Dict], **kwargs) -> str:
        """格式化为HTML表格"""
        if not data:
            return "<table></table>"
        
        html_config = self.config.get("html", {})
        template = html_config.get("template", "default")
        css_style = html_config.get("css_style", "minimal")
        
        # 生成HTML表格
        html = []
        html.append("<table border=\"1\" style=\"border-collapse: collapse;\">")
        
        # 表头
        html.append("<thead><tr>")
        for key in data[0].keys():
            html.append(f"<th>{key}</th>")
        html.append("</tr></thead>")
        
        # 表格内容
        html.append("<tbody>")
        for row in data:
            html.append("<tr>")
            for value in row.values():
                html.append(f"<td>{value}</td>")
            html.append("</tr>")
        html.append("</tbody>")
        
        html.append("</table>")
        
        # 应用CSS样式
        if css_style == "minimal":
            style = """
            <style>
            table { width: 100%; }
            th { background-color: #f2f2f2; padding: 8px; text-align: left; }
            td { padding: 8px; border-bottom: 1px solid #ddd; }
            tr:hover { background-color: #f5f5f5; }
            </style>
            """
            html.insert(0, style)
        
        return "\n".join(html)
    
    def _format_markdown(self, data: List[Dict], **kwargs) -> str:
        """格式化为Markdown表格"""
        if not data:
            return ""
        
        md_config = self.config.get("markdown", {})
        table_format = md_config.get("table_format", "github")
        
        # 生成Markdown表格
        md = []
        
        # 表头
        headers = list(data[0].keys())
        md.append("| " + " | ".join(headers) + " |")
        
        # 分隔线
        if table_format == "github":
            md.append("| " + " | ".join(["---"] * len(headers)) + " |")
        elif table_format == "grid":
            md.append("| " + " | ".join([":---:"] * len(headers)) + " |")
        elif table_format == "simple":
            md.append("-" * (sum(len(str(h)) for h in headers) + 3 * len(headers) + 1))
        
        # 表格内容
        for row in data:
            if table_format == "simple":
                md.append("  ".join(str(v) for v in row.values()))
            else:
                md.append("| " + " | ".join(str(v) for v in row.values()) + " |")
        
        return "\n".join(md)
    
    def save_to_file(self, data: Union[List[Dict], pd.DataFrame], file_path: str, **kwargs) -> None:
        """保存数据到文件"""
        formatted_data = self.format_data(data, **kwargs)
        
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        encoding = self.config.get("encoding", "utf-8")
        with open(file_path, "w", encoding=encoding) as f:
            f.write(formatted_data)
    
    def convert_format(self, data: Union[List[Dict], pd.DataFrame], target_format: str, **kwargs) -> str:
        """转换数据格式"""
        original_format = self.format_type
        self.format_type = target_format.lower()
        
        try:
            result = self.format_data(data, **kwargs)
        finally:
            self.format_type = original_format
        
        return result
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """获取支持的格式列表"""
        return ["csv", "json", "html", "markdown"]
    
    @classmethod
    def get_format_description(cls, format_type: str) -> str:
        """获取格式描述"""
        descriptions = {
            "csv": "逗号分隔值，适合Excel导入",
            "json": "JSON格式，适合程序处理",
            "html": "HTML表格，适合网页展示",
            "markdown": "Markdown表格，适合文档"
        }
        return descriptions.get(format_type.lower(), "未知格式")