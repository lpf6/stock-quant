#!/usr/bin/env python3
"""
Stock Quant 全流程参数优化管道
自动化执行三个阶段优化，持续迭代直到达到目标
"""

import json
import os
import sys
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Tuple

sys.path.append('.')

from param_optimizer import ParameterOptimizer, create_backtest_function
from optimization_backtest import create_backtest_function as create_real_backtest_function

class FullOptimizationPipeline:
    """全流程优化管道"""
    
    def __init__(self, max_iterations: int = 50, target_sharpe: float = 0.8, 
                 target_max_dd: float = 0.15, target_score: float = 0.6):
        """
        初始化优化管道
        
        Args:
            max_iterations: 最大迭代次数
            target_sharpe: 目标夏普比率
            target_max_dd: 目标最大回撤
            target_score: 目标综合评分
        """
        self.max_iterations = max_iterations
        self.target_sharpe = target_sharpe
        self.target_max_dd = target_max_dd
        self.target_score = target_score
        
        # 输出目录
        self.output_dir = "full_optimization_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建日志文件
        self.log_file = os.path.join(self.output_dir, "optimization_log.md")
        
        # 优化状态
        self.current_stage = 1
        self.current_iteration = 0
        self.best_sharpe = 0
        self.best_params = None
        self.convergence_count = 0
        
        # 阶段完成标志
        self.stage1_completed = False
        self.stage2_completed = False
        self.stage3_completed = False
        
        # 创建回测函数（使用真实的回测逻辑）
        self.backtest_func = create_real_backtest_function()
    
    def log_message(self, message: str):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        
        # 写入日志文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
    
    def check_stopping_conditions(self, sharpe_ratio: float, max_drawdown: float, 
                                 composite_score: float, iteration: int) -> Tuple[bool, str]:
        """
        检查停止条件
        
        Returns:
            Tuple[bool, str]: (是否停止, 停止原因)
        """
        # 条件1: 达到目标
        if (sharpe_ratio > self.target_sharpe and 
            abs(max_drawdown) < self.target_max_dd and
            composite_score > self.target_score):
            return True, f"达到目标: 夏普={sharpe_ratio:.3f}>0.8, 回撤={abs(max_drawdown):.2%}<15%, 评分={composite_score:.3f}>0.6"
        
        # 条件2: 达到最大迭代次数
        if iteration >= self.max_iterations:
            return True, f"达到最大迭代次数: {iteration}/{self.max_iterations}"
        
        # 条件3: 连续10次优化无显著改善
        if self.convergence_count >= 10:
            return True, f"连续{self.convergence_count}次优化无显著改善"
        
        return False, "继续优化"
    
    def update_convergence_count(self, new_sharpe: float, threshold: float = 0.01):
        """更新收敛计数"""
        if abs(new_sharpe - self.best_sharpe) < threshold:
            self.convergence_count += 1
        else:
            self.convergence_count = 0
        
        if new_sharpe > self.best_sharpe:
            self.best_sharpe = new_sharpe
    
    def run_stage1(self) -> Dict[str, Any]:
        """运行第一阶段：基础参数优化"""
        self.log_message(f"开始第一阶段优化 (基础参数)")
        
        # 创建优化器
        optimizer = ParameterOptimizer(self.backtest_func, 
                                      output_dir=os.path.join(self.output_dir, "stage1"))
        
        # 网格搜索
        stage1_results = optimizer.grid_search("stage1", max_combinations=100, stock_count=7)
        
        # 检查是否达到目标
        if optimizer.best_result and optimizer.best_result.metrics.get("sharpe_ratio", 0) > self.target_sharpe:
            self.stage1_completed = True
            self.log_message(f"第一阶段完成，夏普比率={optimizer.best_result.metrics.get('sharpe_ratio', 0):.3f}")
        else:
            # 进行随机搜索
            self.log_message("网格搜索未达目标，进行随机搜索...")
            stage1_random = optimizer.random_search("stage1", n_iterations=30, stock_count=7)
            
            if optimizer.best_result and optimizer.best_result.metrics.get("sharpe_ratio", 0) > self.target_sharpe:
                self.stage1_completed = True
                self.log_message(f"随机搜索完成，夏普比率={optimizer.best_result.metrics.get('sharpe_ratio', 0):.3f}")
            else:
                # 进行贝叶斯优化
                self.log_message("随机搜索未达目标，进行贝叶斯优化...")
                stage1_bayesian = optimizer.bayesian_optimization("stage1", n_iterations=20, 
                                                                 init_points=5, stock_count=7)
        
        # 保存结果
        optimizer.save_results("stage1", "combined")
        optimizer.generate_report("stage1")
        optimizer.visualize_optimization("stage1")
        
        # 更新最佳参数
        if optimizer.best_result:
            self.best_params = optimizer.best_result.params.copy()
            self.best_sharpe = optimizer.best_result.metrics.get("sharpe_ratio", 0)
            
            # 检查停止条件
            composite_score = optimizer.best_result.metrics.get("composite_score", 0)
            max_drawdown = optimizer.best_result.metrics.get("max_drawdown", 0)
            
            stop, reason = self.check_stopping_conditions(
                self.best_sharpe, max_drawdown, composite_score, 
                len(optimizer.history)
            )
            
            if stop:
                self.log_message(f"优化停止: {reason}")
                return {"status": "stopped", "reason": reason}
        
        self.stage1_completed = True
        return {"status": "completed", "best_params": self.best_params, 
                "best_sharpe": self.best_sharpe}
    
    def run_stage2(self) -> Dict[str, Any]:
        """运行第二阶段：风险管理参数优化"""
        self.log_message(f"开始第二阶段优化 (风险管理)")
        
        if not self.best_params:
            self.log_message("错误：没有第一阶段的最佳参数")
            return {"status": "error", "message": "需要第一阶段的最佳参数"}
        
        # 合并第一阶段参数
        combined_params = self.best_params.copy()
        
        # 创建优化器
        optimizer = ParameterOptimizer(self.backtest_func, 
                                      output_dir=os.path.join(self.output_dir, "stage2"))
        
        # 使用第一阶段最佳参数作为基础
        base_params = combined_params
        
        # 随机搜索风险管理参数
        best_stage2_result = None
        stage2_iterations = 0
        
        for i in range(20):  # 最多20次迭代
            stage2_iterations += 1
            self.current_iteration += 1
            
            # 生成新的风险管理参数
            risk_params = {
                "stop_loss": np.random.choice([-0.03, -0.05, -0.08]),
                "take_profit": np.random.choice([0.08, 0.12, 0.15]),
                "position_size": np.random.choice([0.10, 0.15, 0.20])
            }
            
            # 合并参数
            test_params = {**base_params, **risk_params}
            
            self.log_message(f"测试风险管理参数 {i+1}/20: {risk_params}")
            
            # 评估参数
            result = optimizer.evaluate_parameters(test_params, self.current_iteration, 7)
            
            # 更新最佳结果
            if result.metrics.get("sharpe_ratio", 0) > self.best_sharpe:
                self.best_sharpe = result.metrics.get("sharpe_ratio", 0)
                self.best_params = test_params.copy()
                best_stage2_result = result
                
                self.log_message(f"  发现改进: 夏普={self.best_sharpe:.3f}")
            
            # 检查停止条件
            stop, reason = self.check_stopping_conditions(
                result.metrics.get("sharpe_ratio", 0),
                result.metrics.get("max_drawdown", 0),
                result.metrics.get("composite_score", 0),
                self.current_iteration
            )
            
            if stop:
                self.log_message(f"优化停止: {reason}")
                break
        
        # 保存结果
        if optimizer.history:
            optimizer.save_results("stage2", "random_search")
            optimizer.generate_report("stage2")
            optimizer.visualize_optimization("stage2")
        
        self.stage2_completed = True
        return {"status": "completed", "best_params": self.best_params, 
                "best_sharpe": self.best_sharpe, "iterations": stage2_iterations}
    
    def run_stage3(self) -> Dict[str, Any]:
        """运行第三阶段：策略权重优化"""
        self.log_message(f"开始第三阶段优化 (策略权重)")
        
        if not self.best_params:
            self.log_message("错误：没有第二阶段的最佳参数")
            return {"status": "error", "message": "需要第二阶段的最佳参数"}
        
        # 合并前两个阶段参数
        combined_params = self.best_params.copy()
        
        # 创建优化器
        optimizer = ParameterOptimizer(self.backtest_func, 
                                      output_dir=os.path.join(self.output_dir, "stage3"))
        
        # 使用第二阶段最佳参数作为基础
        base_params = combined_params
        
        # 随机搜索策略权重参数
        best_stage3_result = None
        stage3_iterations = 0
        
        for i in range(30):  # 最多30次迭代
            stage3_iterations += 1
            self.current_iteration += 1
            
            # 生成新的策略权重参数
            weight_params = {
                "trend_weight": np.random.choice([0.25, 0.30, 0.35, 0.40]),
                "mean_reversion_weight": np.random.choice([0.20, 0.25, 0.30, 0.35]),
                "momentum_weight": np.random.choice([0.20, 0.25, 0.30]),
                "volatility_weight": np.random.choice([0.10, 0.15, 0.20])
            }
            
            # 确保权重总和为1
            total_weight = sum(weight_params.values())
            if total_weight > 1.0:
                # 按比例缩放
                scale = 1.0 / total_weight
                for key in weight_params:
                    weight_params[key] *= scale
            
            # 合并参数
            test_params = {**base_params, **weight_params}
            
            self.log_message(f"测试策略权重 {i+1}/30: {weight_params}")
            
            # 评估参数
            result = optimizer.evaluate_parameters(test_params, self.current_iteration, 7)
            
            # 更新收敛计数
            self.update_convergence_count(result.metrics.get("sharpe_ratio", 0))
            
            # 更新最佳结果
            if result.metrics.get("sharpe_ratio", 0) > self.best_sharpe:
                self.best_sharpe = result.metrics.get("sharpe_ratio", 0)
                self.best_params = test_params.copy()
                best_stage3_result = result
                
                self.log_message(f"  发现改进: 夏普={self.best_sharpe:.3f}")
                self.convergence_count = 0  # 重置收敛计数
            
            # 检查停止条件
            stop, reason = self.check_stopping_conditions(
                result.metrics.get("sharpe_ratio", 0),
                result.metrics.get("max_drawdown", 0),
                result.metrics.get("composite_score", 0),
                self.current_iteration
            )
            
            if stop:
                self.log_message(f"优化停止: {reason}")
                break
        
        # 保存结果
        if optimizer.history:
            optimizer.save_results("stage3", "random_search")
            optimizer.generate_report("stage3")
            optimizer.visualize_optimization("stage3")
        
        self.stage3_completed = True
        return {"status": "completed", "best_params": self.best_params, 
                "best_sharpe": self.best_sharpe, "iterations": stage3_iterations}
    
    def final_validation(self, stock_count: int = 50) -> Dict[str, Any]:
        """最终验证：在更大的样本上测试最佳参数"""
        self.log_message(f"开始最终验证 (测试{stock_count}只股票)")
        
        if not self.best_params:
            self.log_message("错误：没有最佳参数")
            return {"status": "error", "message": "没有最佳参数"}
        
        # 创建回测实例
        from optimization_backtest import OptimizedBacktest
        backtester = OptimizedBacktest(self.best_params)
        
        # 扩展股票列表进行测试
        expanded_stocks = []
        
        # 基础股票列表（来自原始代码）
        base_stocks = [
            {"symbol": "000001", "name": "平安银行"},
            {"symbol": "000002", "name": "万科A"},
            {"symbol": "000858", "name": "五粮液"},
            {"symbol": "600519", "name": "贵州茅台"},
            {"symbol": "000333", "name": "美的集团"},
            {"symbol": "000568", "name": "泸州老窖"},
            {"symbol": "600036", "name": "招商银行"}
        ]
        
        # 扩展股票列表
        for i in range(stock_count):
            if i < len(base_stocks):
                expanded_stocks.append(base_stocks[i])
            else:
                # 生成模拟股票代码
                symbol = f"600{i+100:03d}"
                name = f"模拟股票{i+1}"
                expanded_stocks.append({"symbol": symbol, "name": name})
        
        # 运行扩展测试
        validation_results = []
        
        for i, stock in enumerate(expanded_stocks):
            if i % 10 == 0:
                self.log_message(f"  测试股票 {i+1}/{len(expanded_stocks)}...")
            
            result = backtester.test_single_stock(stock["symbol"], stock["name"])
            validation_results.append(result)
        
        # 计算统计指标
        valid_results = [r for r in validation_results if r.get("backtest_results", {}).get("total_trades", 0) > 0]
        
        if valid_results:
            sharpe_list = [r.get("backtest_results", {}).get("sharpe_ratio", 0) for r in valid_results]
            max_dd_list = [abs(r.get("backtest_results", {}).get("max_drawdown", 0)) for r in valid_results]
            win_rate_list = [r.get("backtest_results", {}).get("win_rate", 0) for r in valid_results]
            total_return_list = [r.get("backtest_results", {}).get("total_return", 0) for r in valid_results]
            
            avg_sharpe = np.mean(sharpe_list)
            avg_max_dd = np.mean(max_dd_list)
            avg_win_rate = np.mean(win_rate_list)
            avg_total_return = np.mean(total_return_list)
            sharpe_std = np.std(sharpe_list)
            
            # 成功率（夏普>0的股票比例）
            success_rate = sum(1 for s in sharpe_list if s > 0) / len(sharpe_list)
            
            validation_summary = {
                "avg_sharpe": avg_sharpe,
                "avg_max_drawdown": avg_max_dd,
                "avg_win_rate": avg_win_rate,
                "avg_total_return": avg_total_return,
                "sharpe_std": sharpe_std,
                "success_rate": success_rate,
                "tested_stocks": len(expanded_stocks),
                "valid_stocks": len(valid_results),
                "parameters": self.best_params
            }
            
            self.log_message(f"验证结果:")
            self.log_message(f"  平均夏普比率: {avg_sharpe:.3f}")
            self.log_message(f"  平均最大回撤: {avg_max_dd:.2%}")
            self.log_message(f"  平均胜率: {avg_win_rate:.2%}")
            self.log_message(f"  平均总收益率: {avg_total_return:.2%}")
            self.log_message(f"  夏普比率标准差: {sharpe_std:.3f}")
            self.log_message(f"  成功率: {success_rate:.1%}")
            
            # 保存验证结果
            validation_file = os.path.join(self.output_dir, "validation_results.json")
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": validation_summary,
                    "detailed_results": validation_results[:20]  # 只保存前20个详细结果
                }, f, indent=2, ensure_ascii=False)
            
            return {"status": "completed", "summary": validation_summary}
        else:
            self.log_message("警告：没有有效的回测结果")
            return {"status": "error", "message": "没有有效的回测结果"}
    
    def generate_final_report(self):
        """生成最终优化报告"""
        self.log_message("生成最终优化报告...")
        
        report_lines = [
            "# Stock Quant 全流程参数优化报告",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"总迭代次数: {self.current_iteration}",
            f"最大迭代次数: {self.max_iterations}",
            "",
            "## 优化状态",
            ""
        ]
        
        # 阶段完成状态
        stage_status = {
            "第一阶段 (基础参数)": "✅ 完成" if self.stage1_completed else "❌ 未完成",
            "第二阶段 (风险管理)": "✅ 完成" if self.stage2_completed else "❌ 未完成",
            "第三阶段 (策略权重)": "✅ 完成" if self.stage3_completed else "❌ 未完成"
        }
        
        for stage, status in stage_status.items():
            report_lines.append(f"- {stage}: {status}")
        
        # 最佳参数和性能
        if self.best_params:
            report_lines.extend([
                "",
                "## 最佳参数配置",
                ""
            ])
            
            # 分类显示参数
            param_categories = {
                "技术指标参数": ["ma_fast", "ma_slow", "rsi_oversold", "rsi_overbought", "bb_std", "score_threshold"],
                "风险管理参数": ["stop_loss", "take_profit", "position_size"],
                "策略权重参数": ["trend_weight", "mean_reversion_weight", "momentum_weight", "volatility_weight"]
            }
            
            for category, param_names in param_categories.items():
                report_lines.append(f"### {category}")
                for param_name in param_names:
                    if param_name in self.best_params:
                        value = self.best_params[param_name]
                        if isinstance(value, float):
                            if param_name in ["stop_loss", "take_profit", "total_return"]:
                                report_lines.append(f"- **{param_name}**: {value:.2%}")
                            elif param_name in ["position_size", "trend_weight", "mean_reversion_weight", 
                                              "momentum_weight", "volatility_weight"]:
                                report_lines.append(f"- **{param_name}**: {value:.1%}")
                            else:
                                report_lines.append(f"- **{param_name}**: {value:.3f}")
                        else:
                            report_lines.append(f"- **{param_name}**: {value}")
                report_lines.append("")
        
        # 目标达成情况
        report_lines.extend([
            "## 目标达成评估",
            ""
        ])
        
        # 这里需要从验证结果中获取实际指标
        # 简化版本：假设从日志中提取
        targets = [
            f"夏普比率 > {self.target_sharpe}: {'✅ 达成' if self.best_sharpe > self.target_sharpe else '❌ 未达成'} (当前: {self.best_sharpe:.3f})",
            f"最大回撤 < {self.target_max_dd*100:.0f}%: 需验证",
            f"综合评分 > {self.target_score}: 需验证",
            f"胜率 > 55%: 需验证",
            f"年化收益率 > 20%: 需验证",
            f"盈亏比 > 1.5: 需验证"
        ]
        
        for target in targets:
            report_lines.append(f"- {target}")
        
        # 优化建议
        report_lines.extend([
            "",
            "## 优化建议",
            "",
            "### 继续改进方向",
            "1. **增加数据量**: 使用更长历史数据进行回测（至少3年）",
            "2. **添加更多指标**: 引入ATR、成交量指标、市场情绪指标",
            "3. **优化交易成本**: 考虑佣金、滑点等实际交易成本",
            "4. **多时间框架**: 结合日线、周线、月线进行多时间框架分析",
            "5. **市场状态识别**: 根据牛熊市调整策略参数",
            "",
            "### 风险管理建议",
            "1. **动态仓位调整**: 根据市场波动率调整仓位大小",
            "2. **相关性控制**: 限制相关度过高的股票同时持仓",
            "3. **分散投资**: 确保投资组合足够分散",
            "4. **定期再平衡**: 定期调整投资组合以维持风险敞口",
            "",
            "### 实施建议",
            "1. **模拟交易验证**: 在实际交易前进行至少3个月的模拟交易",
            "2. **逐步实施**: 从小资金开始，逐步增加投资规模",
            "3. **持续监控**: 建立自动化监控和预警系统",
            "4. **定期优化**: 每季度重新优化一次参数",
        ])
        
        # 保存报告
        report_file = os.path.join(self.output_dir, "final_optimization_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        self.log_message(f"最终报告保存到: {report_file}")
        
        return report_file
    
    def run_pipeline(self):
        """运行完整的优化管道"""
        self.log_message("开始Stock Quant全流程参数优化")
        self.log_message("=" * 60)
        
        try:
            # 第一阶段：基础参数优化
            stage1_result = self.run_stage1()
            if stage1_result.get("status") == "stopped":
                self.log_message("第一阶段优化提前停止")
                return stage1_result
            
            # 第二阶段：风险管理参数优化
            stage2_result = self.run_stage2()
            if stage2_result.get("status") == "stopped":
                self.log_message("第二阶段优化提前停止")
                return stage2_result
            
            # 第三阶段：策略权重优化
            stage3_result = self.run_stage3()
            if stage3_result.get("status") == "stopped":
                self.log_message("第三阶段优化提前停止")
                return stage3_result
            
            # 最终验证
            validation_result = self.final_validation(stock_count=50)
            
            # 生成最终报告
            report_file = self.generate_final_report()
            
            # 总结
            self.log_message("\n" + "=" * 60)
            self.log_message("全流程优化完成！")
            self.log_message(f"最佳夏普比率: {self.best_sharpe:.3f}")
            self.log_message(f"总迭代次数: {self.current_iteration}")
            self.log_message(f"日志文件: {self.log_file}")
            self.log_message(f"最终报告: {report_file}")
            
            return {
                "status": "completed",
                "best_sharpe": self.best_sharpe,
                "best_params": self.best_params,
                "total_iterations": self.current_iteration,
                "validation_result": validation_result
            }
            
        except Exception as e:
            self.log_message(f"优化过程中出现错误: {e}")
            return {"status": "error", "message": str(e)}

def main():
    """主函数"""
    print("Stock Quant 全流程参数优化管道")
    print("=" * 60)
    
    # 创建优化管道
    pipeline = FullOptimizationPipeline(
        max_iterations=50,
        target_sharpe=0.8,
        target_max_dd=0.15,
        target_score=0.6
    )
    
    # 运行管道
    result = pipeline.run_pipeline()
    
    print("\n优化结果摘要:")
    print(f"  状态: {result.get('status', 'unknown')}")
    print(f"  最佳夏普比率: {result.get('best_sharpe', 0):.3f}")
    print(f"  总迭代次数: {result.get('total_iterations', 0)}")
    
    if result.get('best_params'):
        print("\n最佳参数:")
        for param, value in result['best_params'].items():
            print(f"  {param}: {value}")
    
    return result

if __name__ == "__main__":
    main()