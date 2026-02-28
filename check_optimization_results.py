#!/usr/bin/env python3
"""
检查参数优化结果
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

print("Stock Quant 参数优化结果检查")
print("=" * 50)

# 检查优化结果目录
optimization_dir = "/root/.openclaw/workspace/optimization_results"
if not os.path.exists(optimization_dir):
    print("❌ 优化结果目录不存在")
    print("可能原因：")
    print("  1. 优化任务仍在运行")
    print("  2. 优化任务使用了不同路径")
    print("  3. 优化任务尚未生成结果")
    
    # 搜索可能的文件
    print("\n搜索可能的优化文件...")
    workspace_files = []
    for root, dirs, files in os.walk("/root/.openclaw/workspace"):
        for file in files:
            if 'optimization' in file.lower() or 'parameter' in file.lower() or 'backtest' in file.lower():
                workspace_files.append(os.path.join(root, file))
    
    if workspace_files:
        print(f"找到 {len(workspace_files)} 个相关文件：")
        for file in workspace_files[:10]:  # 显示前10个
            print(f"  - {os.path.basename(file)}")
        if len(workspace_files) > 10:
            print(f"  ... 还有 {len(workspace_files)-10} 个文件")
    else:
        print("未找到优化相关文件")
    exit()

# 列出目录内容
print(f"优化结果目录: {optimization_dir}")
files = os.listdir(optimization_dir)
print(f"找到 {len(files)} 个文件：")

# 分类显示文件
json_files = [f for f in files if f.endswith('.json')]
csv_files = [f for f in files if f.endswith('.csv')]
md_files = [f for f in files if f.endswith('.md')]
png_files = [f for f in files if f.endswith('.png')]

total_iterations = 0
best_sharpe = -10
best_params = {}

if json_files:
    print(f"\nJSON 文件 ({len(json_files)}个)：")
    for file in json_files:
        file_path = os.path.join(optimization_dir, file)
        file_size = os.path.getsize(file_path)
        
        try:
            if file == 'optimization_history.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                total_iterations = len(data.get('iterations', []))
                print(f"  ✓ {file} ({file_size:,}字节, {total_iterations}次迭代)")
                
                # 提取最佳结果
                if data.get('best_iteration'):
                    best = data['best_iteration']
                    best_sharpe = best.get('sharpe_ratio', -10)
                    best_params = best.get('parameters', {})
                    print(f"    最佳夏普比率: {best_sharpe:.3f}")
                    
            elif file == 'best_parameters.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"  ✓ {file} ({file_size:,}字节)")
                print(f"    参数数量: {len(data)}")
                
                # 显示关键参数
                for key in ['ma_fast', 'ma_slow', 'rsi_oversold', 'rsi_overbought', 'score_threshold']:
                    if key in data:
                        print(f"    {key}: {data[key]}")
                
            elif 'iteration' in file:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                sharpe = data.get('sharpe_ratio', 0)
                print(f"  • {file}: 夏普={sharpe:.3f}")
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = data.get('parameters', {})
                    
            else:
                print(f"  • {file} ({file_size:,}字节)")
                
        except Exception as e:
            print(f"  ✗ {file}: 读取失败 - {e}")

if csv_files:
    print(f"\nCSV 文件 ({len(csv_files)}个)：")
    for file in csv_files:
        file_path = os.path.join(optimization_dir, file)
        file_size = os.path.getsize(file_path)
        
        try:
            df = pd.read_csv(file_path)
            print(f"  ✓ {file} ({file_size:,}字节, {len(df)}行, {len(df.columns)}列)")
            
            if 'sharpe_ratio' in df.columns:
                max_sharpe = df['sharpe_ratio'].max()
                avg_sharpe = df['sharpe_ratio'].mean()
                print(f"    夏普比率: 最大={max_sharpe:.3f}, 平均={avg_sharpe:.3f}")
            
        except Exception as e:
            print(f"  ✗ {file}: 读取失败 - {e}")

if md_files:
    print(f"\nMarkdown 文件 ({len(md_files)}个)：")
    for file in md_files:
        file_path = os.path.join(optimization_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"  • {file} ({file_size:,}字节)")
        
        # 显示部分内容
        if file == 'optimization_report.md':
            with open(file_path, 'r') as f:
                lines = f.readlines()[:10]
            print(f"    预览: {''.join(lines).strip()[:80]}...")

if png_files:
    print(f"\n图片文件 ({len(png_files)}个)：")
    for file in png_files:
        file_path = os.path.join(optimization_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"  • {file} ({file_size:,}字节)")

# 总结
print(f"\n📊 优化结果摘要")
print(f"总迭代次数: {total_iterations}")
print(f"最佳夏普比率: {best_sharpe:.3f}")
print(f"目标夏普比率: > 0.8")

if best_sharpe >= 0.8:
    print("✅ 达成优化目标!")
elif best_sharpe > 0:
    progress = (best_sharpe / 0.8) * 100
    print(f"📈 进度: {progress:.1f}% (还需优化)")
else:
    print("❌ 未找到有效优化结果")

# 显示最佳参数
if best_params:
    print(f"\n🎯 最佳参数配置:")
    for key, value in sorted(best_params.items()):
        if key != 'iteration_id':
            print(f"  {key}: {value}")

# 检查目标达成情况
if best_sharpe >= 0.8:
    print("\n✅ 优化任务已完成，达成目标!")
    print("下一步建议: 进行扩大样本测试和风险管理模块集成")
else:
    print("\n🔧 优化仍在进行中或未达目标")
    print("建议:")
    print("  1. 继续运行优化迭代")
    print("  2. 调整参数搜索范围")
    print("  3. 添加更多技术指标")
    print("  4. 集成风险管理模块")

# 检查是否有运行中的任务
print("\n📈 建议后续步骤:")
print("1. 查看详细报告: optimization_results/optimization_report.md")
print("2. 应用最佳参数: 更新backtest_simplified.py")
print("3. 扩大样本测试: 增加到50-100只股票")
print("4. 集成风险控制: 添加止损止盈机制")

print("\n检查完成!")
