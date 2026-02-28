#!/bin/bash
# Stock Quant 参数优化启动脚本

echo "启动 Stock Quant 参数优化系统"
echo "=================================================="

# 清理旧的优化结果
echo "清理旧的优化结果..."
rm -rf optimization_results/ full_optimization_results/ 2>/dev/null

# 创建必要的目录
mkdir -p optimization_results
mkdir -p full_optimization_results

# 运行第一阶段优化测试
echo "运行第一阶段优化测试..."
python3 param_optimizer.py

echo ""
echo "第一阶段优化完成，查看结果:"
echo "- 配置: optimization_results/stage_stage1/best_config_*.json"
echo "- 报告: optimization_results/stage_stage1/optimization_report_*.md"
echo "- 图表: optimization_results/stage_stage1/optimization_charts_*.png"

# 运行全流程优化（如果第一阶段的夏普比率未达目标）
echo ""
echo "运行全流程优化管道..."
python3 full_optimization_pipeline.py

echo ""
echo "全流程优化完成，查看结果:"
echo "- 最终报告: full_optimization_results/final_optimization_report.md"
echo "- 验证结果: full_optimization_results/validation_results.json"
echo "- 详细日志: full_optimization_results/optimization_log.md"

# 生成摘要报告
echo ""
echo "生成优化摘要..."
python3 -c "
import json
import glob
import os

# 查找最佳配置
config_files = glob.glob('full_optimization_results/stage*/best_config_*.json')
if not config_files:
    config_files = glob.glob('optimization_results/stage*/best_config_*.json')

if config_files:
    latest_config = max(config_files, key=os.path.getctime)
    with open(latest_config, 'r') as f:
        config = json.load(f)
    
    print('优化摘要:')
    print(f'  优化日期: {config.get(\"optimization_date\", \"N/A\")}')
    print(f'  阶段: {config.get(\"stage\", \"N/A\")}')
    print(f'  方法: {config.get(\"method\", \"N/A\")}')
    
    metrics = config.get('metrics_summary', {})
    print(f'  夏普比率: {metrics.get(\"sharpe_ratio\", 0):.3f}')
    print(f'  最大回撤: {metrics.get(\"max_drawdown\", 0):.2%}')
    print(f'  胜率: {metrics.get(\"win_rate\", 0):.2%}')
    print(f'  盈亏比: {metrics.get(\"profit_factor\", 0):.2f}')
    print(f'  总收益率: {metrics.get(\"total_return\", 0):.2%}')
    
    params = config.get('parameters', {})
    print(f'  测试股票数: {config.get(\"stock_count\", 0)}')
    print('')
    
    # 检查目标达成情况
    sharpe = metrics.get('sharpe_ratio', 0)
    max_dd = abs(metrics.get('max_drawdown', 0))
    
    print('目标达成情况:')
    if sharpe > 0.8:
        print('  ✅ 夏普比率 > 0.8 (达成)')
    else:
        print(f'  ❌ 夏普比率 > 0.8 (未达成: {sharpe:.3f})')
    
    if max_dd < 0.15:
        print('  ✅ 最大回撤 < 15% (达成)')
    else:
        print(f'  ❌ 最大回撤 < 15% (未达成: {max_dd:.2%})')
    
    composite_score = metrics.get('composite_score', 0)
    if composite_score > 0.6:
        print('  ✅ 综合评分 > 0.6 (达成)')
    else:
        print(f'  ❌ 综合评分 > 0.6 (未达成: {composite_score:.3f})')
else:
    print('未找到优化配置文件')
"

echo ""
echo "优化系统运行完成！"