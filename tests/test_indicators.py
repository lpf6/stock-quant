#!/usr/bin/env python3
"""
指标计算单元测试
使用虚拟数据进行测试
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.stock_quant.technical_analysis.indicator_calculator import IndicatorCalculator

class TestIndicatorCalculator(unittest.TestCase):
    def setUp(self):
        """测试前设置"""
        self.calculator = IndicatorCalculator()
    
    def test_generate_virtual_price_data(self):
        """测试生成虚拟价格数据"""
        data = self.calculator.generate_virtual_price_data(days=100)
        
        self.assertEqual(len(data), 100)
        self.assertIn('open', data.columns)
        self.assertIn('high', data.columns)
        self.assertIn('low', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        
        # 检查价格是否合理
        self.assertTrue(all(data['close'] > 0))
        self.assertTrue(all(data['high'] >= data['close']))
        self.assertTrue(all(data['low'] <= data['close']))
        self.assertTrue(all(data['volume'] > 0))
    
    def test_generate_virtual_financial_data(self):
        """测试生成虚拟财务数据"""
        data = self.calculator.generate_virtual_financial_data()
        
        # 检查关键财务指标是否存在
        self.assertIn('revenue', data)
        self.assertIn('net_profit', data)
        self.assertIn('total_assets', data)
        self.assertIn('total_liabilities', data)
        self.assertIn('roe', data)
        self.assertIn('roa', data)
        self.assertIn('pe_ratio', data)
        
        # 检查指标值是否合理
        self.assertTrue(data['revenue'] > 0)
        self.assertTrue(data['net_profit'] > 0)
        self.assertTrue(0 <= data['roe'] <= 1)
        self.assertTrue(0 <= data['roa'] <= 1)
    
    def test_calculate_moving_averages(self):
        """测试计算移动平均线"""
        indicators = self.calculator.calculate_moving_averages()
        
        self.assertIn('sma_5', indicators.columns)
        self.assertIn('sma_20', indicators.columns)
        self.assertIn('ema_5', indicators.columns)
        self.assertIn('ema_20', indicators.columns)
        self.assertIn('wma_5', indicators.columns)
        
        # 检查值是否不为空
        self.assertFalse(indicators['sma_5'].dropna().empty)
        self.assertFalse(indicators['ema_20'].dropna().empty)
    
    def test_calculate_volatility_indicators(self):
        """测试计算波动率指标"""
        indicators = self.calculator.calculate_volatility_indicators()
        
        self.assertIn('BBL_20_2.0', indicators.columns)
        self.assertIn('BBM_20_2.0', indicators.columns)
        self.assertIn('BBU_20_2.0', indicators.columns)
        self.assertIn('atr', indicators.columns)
        self.assertIn('hist_vol_20', indicators.columns)
        
        self.assertFalse(indicators['atr'].dropna().empty)
        self.assertTrue(all(indicators['hist_vol_20'].dropna() > 0))
    
    def test_calculate_momentum_indicators(self):
        """测试计算动量指标"""
        indicators = self.calculator.calculate_momentum_indicators()
        
        self.assertIn('rsi_14', indicators.columns)
        self.assertIn('MACD_12_26_9', indicators.columns)
        self.assertIn('MACDh_12_26_9', indicators.columns)
        self.assertIn('MACDs_12_26_9', indicators.columns)
        self.assertIn('STOCHk_14_3_3', indicators.columns)
        self.assertIn('STOCHd_14_3_3', indicators.columns)
        
        # RSI 应该在 0-100 之间
        rsi_values = indicators['rsi_14'].dropna()
        self.assertTrue(all(rsi_values >= 0) and all(rsi_values <= 100))
    
    def test_calculate_volume_indicators(self):
        """测试计算成交量指标"""
        indicators = self.calculator.calculate_volume_indicators()
        
        self.assertIn('vol_sma_5', indicators.columns)
        self.assertIn('vol_sma_20', indicators.columns)
        self.assertIn('obv', indicators.columns)
        self.assertIn('vwap', indicators.columns)
        self.assertIn('mfi', indicators.columns)
        
        self.assertFalse(indicators['obv'].dropna().empty)
    
    def test_calculate_pattern_recognition(self):
        """测试计算形态识别指标"""
        indicators = self.calculator.calculate_pattern_recognition()
        
        self.assertIn('doji', indicators.columns)
        self.assertIn('engulfing', indicators.columns)
        self.assertIn('hammer', indicators.columns)
    
    def test_calculate_profitability_ratios(self):
        """测试计算盈利能力指标"""
        ratios = self.calculator.calculate_profitability_ratios()
        
        self.assertIn('gross_margin', ratios)
        self.assertIn('net_margin', ratios)
        self.assertIn('roe', ratios)
        self.assertIn('roa', ratios)
        self.assertIn('eps', ratios)
        
        self.assertTrue(0 <= ratios['gross_margin'] <= 1)
        self.assertTrue(0 <= ratios['net_margin'] <= 1)
    
    def test_calculate_liquidity_ratios(self):
        """测试计算流动性指标"""
        ratios = self.calculator.calculate_liquidity_ratios()
        
        self.assertIn('current_ratio', ratios)
        self.assertIn('quick_ratio', ratios)
        self.assertIn('cash_ratio', ratios)
        
        self.assertTrue(ratios['current_ratio'] > 0)
        self.assertTrue(ratios['quick_ratio'] > 0)
    
    def test_calculate_solvency_ratios(self):
        """测试计算偿债能力指标"""
        ratios = self.calculator.calculate_solvency_ratios()
        
        self.assertIn('debt_to_assets', ratios)
        self.assertIn('debt_to_equity', ratios)
        self.assertIn('interest_coverage', ratios)
        
        self.assertTrue(0 <= ratios['debt_to_assets'] <= 1)
    
    def test_calculate_efficiency_ratios(self):
        """测试计算运营效率指标"""
        ratios = self.calculator.calculate_efficiency_ratios()
        
        self.assertIn('asset_turnover', ratios)
        self.assertIn('inventory_turnover', ratios)
        
        self.assertTrue(ratios['asset_turnover'] > 0)
    
    def test_calculate_valuation_ratios(self):
        """测试计算估值指标"""
        ratios = self.calculator.calculate_valuation_ratios()
        
        self.assertIn('pe_ratio', ratios)
        self.assertIn('pb_ratio', ratios)
        self.assertIn('ps_ratio', ratios)
        self.assertIn('dividend_yield', ratios)
        
        self.assertTrue(ratios['pe_ratio'] > 0)
        self.assertTrue(ratios['pb_ratio'] > 0)
    
    def test_calculate_all_indicators(self):
        """测试计算所有指标"""
        all_indicators = self.calculator.calculate_all_indicators()
        
        self.assertIn('moving_averages', all_indicators)
        self.assertIn('volatility', all_indicators)
        self.assertIn('momentum', all_indicators)
        self.assertIn('volume', all_indicators)
        self.assertIn('profitability', all_indicators)
        self.assertIn('liquidity', all_indicators)
        self.assertIn('solvency', all_indicators)
        self.assertIn('efficiency', all_indicators)
        self.assertIn('valuation', all_indicators)

if __name__ == '__main__':
    unittest.main()
